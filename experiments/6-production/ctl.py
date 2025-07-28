import sys, os, math, json, glob, re, requests, time, subprocess, heapq, yaml, shutil

from typing import List, Tuple

from datetime import datetime
from functools import wraps
from operator import itemgetter

import numpy as np
import pandas as pd

import shap
from sklearn.ensemble import IsolationForest

from kubernetes import client, config

from util.analysis import *
from util.sequence import *
from util.square import *

from pprint import pprint

# experimental code, remove security warning.
import warnings
warnings.filterwarnings('ignore', message="Unverified HTTPS request*")

from config import PATH, GEN_API_URL, SPAN_PROCESS_MAP

DATA_PROMPT = '''Please resolve the following anomaly: \n
Anomaly Start time: {start_time} \n
Anomaly duration: {duration} \n

this trace describes the time logged in various parts of the code paths of the microservices                          
<anomalous_trace_csv> \n 
{trace_df} \n
</anomalous_trace_csv> \n

The monitoring metrics describe what happend in the system during the anomalous trace.
<monitoring_metrics_csv> \n
{monitor_df} \n
</monitoring_metrics_csv> \n
'''

FILE_PROMPT = '''give one short and concise reasoning then answer with the corrected yaml file to mitigate the anomaly: \n
<yaml> \n
{service_file} \n
--- 
{global_file} \n
</yaml>'''

RESULT_PROMPT = '''the configuration produced the performance indicators: \n
<json> \n
{performance} \n
<json>'''


def report_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper


def locust_load_test(label, base_label, exp_time = "5m"): # todo add all loadtest configurations
    if len(glob.glob(PATH + "output/" + f"{base_label}/data/{label}_*")) != 0:
        print(f"loadtest using label: {label} already exists.", "skipping...")
        return

    cmd = [
        "/home/pager/Documents/closing-the-loop/venv/bin/locust",
        "--processes", "16",
        "-f", "locust/hotel-reservations.py",
        "-H", "http://spark.lab.uvalight.net:30505",
        "-t", exp_time,
        "--csv", label,
        "--headless"
    ]

    print(f"Test {label} started...\n")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1200  # 20 minutes
    )

    print(f"Test {label}, Finished")
    time.sleep(30)

    # if result.returncode != 0:
        # print(f"Error: {result.stderr}")

    experiment_files = glob.glob(f"./{label}_*")
    print("analysing the loadtest files...", experiment_files)

    os.makedirs(PATH + f"{base_label}/data/{label}", exist_ok=True)
    for file in experiment_files:
        shutil.move(file, PATH + f"{base_label}/data/{label}/")


# setting options for printing the anomaly data in the prompt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

@report_time
def generate_and_measure(label, loop, t="10m"):
    messages = []

    client = get_k8s_api_client()
    reset_k8s(client, PATH + f"/output/{label}/config" )

    try:
        history_df = read_history(f"{label}_{loop}", label)
        responce_df = read_response(f"{label}_{loop}", label)
        trace_df = read_traces(f"{label}_{loop}", label)
        metric_dfs = read_metrics(f"{label}_{loop}", label)
    except FileNotFoundError as fnfe:
        shutil.rmtree(PATH + f"/output/{base_label}/{label}_{loop}")
    
        print("reading the test data failed... retrying load test")
        locust_load_test(f"{label}_{loop}", label)


    # IMPROVEMENT: improve pipeline with the ELBD framework. will look good in the paper.
    # IMPROVEMENT: also perform outlier detection on the monitoring metrics to capture more than point anomalies.

    # Find outliers in the traces using an IsolationForest classifier.
    features = trace_df.select_dtypes(include=["number"]).drop(columns=["total"])
    iso_forest = IsolationForest(contamination="auto", random_state=42)
    trace_df["anomaly"] = iso_forest.fit_predict(features)

    # IMPROVEMENT: decision plot the shap values by clustering similar shap values...
    anomaly_indices = trace_df[(trace_df['anomaly'] == -1)].index.to_list()

    anom_features = features.iloc[anomaly_indices]
    shapes, names = shap_decisions(iso_forest, anom_features)
    
    # order the anomalies on their length.
    anomalies = []
    for s, ai in zip(shapes, anomaly_indices):
        duration = pd.to_timedelta(trace_df["total"][ai] + 1000000, unit="us")

        if duration < pd.Timedelta(seconds=2):
            print(f"Duration is less than one second. skipping anomaly {ai}")
            continue

        anomalies.append((s, ai, duration))
    
    sorted_list = sorted(anomalies, key=lambda x: -x[2])
    # print(sorted_list)

    print(f"Generating and Measuring configuration changes from {len(sorted_list)} anomalies")
    for s, ai, l in sorted_list:
        
        if ai in get_existing_seq(PATH + f"output/{label}/{label}_{loop}_prompts.json"):
            print(f"skipping... {ai}")
            continue

        if len(messages) > 12:
            messages = messages[:-12]

        # pprint(messages)

        values = heapq.nsmallest(2, enumerate(s), key=itemgetter(1))
        for v in values:
            # filter unsupported deployment files (ie. THAT DONT HAVE KNATIVE HORIZONTAL SCALING active)
            # if os.path.basename(SPAN_PROCESS_MAP[names[v[0]]])[:-5] in ["memcached-reservation-deployment", "memcached-rate-deployment", "frontend-deployment", "memcached-profile-deployment"]:
                # continue

            print("resolving anomaly for: ", names[v[0]])
            m_df = metric_dfs[os.path.basename(SPAN_PROCESS_MAP[names[v[0]]])[:-5]]
            
            # convert microseconds to timedelta
            start_time = trace_df["startTime"][ai]
            duration = pd.to_timedelta(trace_df["total"][ai] + 1000000, unit="us")  # 'us' for microseconds
            
            start_plus_duration = start_time + duration
            start_minus_5s = start_time - pd.Timedelta(seconds=10)

            time_mask = (m_df["index"] >= start_minus_5s) & (m_df["index"] <= start_plus_duration)
            time_m_df = m_df[time_mask]

            service_config = get_config_content(PATH + f"/output/{label}/config/{os.path.basename(SPAN_PROCESS_MAP[names[v[0]]])}")[-1]
            auto_config = get_config_content(PATH + f"/output/{label}/config/config-autoscaler.yaml")[-1]

            service_config = yaml.dump(service_config[1])
            auto_config = yaml.dump(auto_config[1])
            

            # IMPROVEMENTS ontology -> look at data and update. then use, to store the required knowledge for better generation and context.
            messages.append(["user", DATA_PROMPT.format(start_time=start_time, duration=duration, trace_df=trace_df.iloc[[ai]].to_string(), monitor_df=time_m_df.to_string())])
            messages.append(["user", FILE_PROMPT.format(service_file=service_config,global_file=auto_config)])

            payload = {
                "messages": messages,
                "max_new_tokens": 2000
            }

            response = requests.post(GEN_API_URL, json=payload)

            append_generation(PATH + f"output/{label}_{loop}_prompts.json", ai, response.json()["response"]) 

            messages.append(["assistant", response.json()["response"]])

            matches = re.findall(r"```yaml\n([\s\S]*?)\n```", response.json()["response"])

            yaml_str = ""
            for m in matches:
                yaml_str = m.strip()
        
            try:
                docs = yaml.safe_load_all(yaml_str)
                service_names = []
                for d in docs:
                    apply_yaml_configuration(d, client)
                    service_names = service_names + [d.get('metadata', {}).get('name')]

                print(f"applied updates to: {", ".join(service_names)}")
                print("waiting to update the configuration...")
                time.sleep(60)

            except Exception as e:
                for d in docs:
                    print(d)
                print(e)

                append_generation(PATH + "/output/" + f"{label}/{label}_{loop}_performance.json", ai, str(e)) 
                messages.append(["user", "the folowing error was produced: " + str(e)])

                break

            
            # Locust!
            experiment_label = f"{label}_{loop}-{ai}-{"-".join(service_names)}"
            locust_load_test(experiment_label, t)

            config_performance = "test error."
            # KPI's 
            try:
                config_performance = get_kpi_list(experiment_label, service_names[0])

            except FileNotFoundError as fnfe:
                print(fnfe)
                shutil.rmtree(PATH + "data/" + f"{experiment_label}")
                
                locust_load_test(experiment_label, t)
                config_performance = get_kpi_list(experiment_label, service_names[0])

            except Exception as e:
                config_performance = get_kpi_list(experiment_label, service_names[1])
                
            finally:
                print(config_performance)
                
                config_performance['service_names'] = service_names
                append_generation(PATH + "/output/" + f"{label}/{label}_{loop}_performance.json", ai, str(config_performance))
                
                messages.append(["user", RESULT_PROMPT.format(performance=str(config_performance))])

                reset_k8s(client, PATH + f"/output/{label}/config/*.yaml" )

            break


def clean_float(value):
    try:
        val = float(value)
        return val if not math.isnan(val) else float(1000)
    except (ValueError, TypeError):
        return float('inf')


def evolve(label : str, i: int):
    configs = load_generated_configurations(f"{label}_{i}")
    configs = [(i, yaml.safe_load_all(sol)) for i, sol in configs]

    trails = get_generation_content_perf(PATH + f"/output/{label}/{label}_{i}_performance.json")
    perfs = [(i, yaml.safe_load(p)) for i, p in trails]
    perfs = [(i, p) for i, p in perfs if "Audit-Id" not in p]

    perfs = [(i, p) for i, p in perfs if 140 <= p.get('max_throughput', 0)]


    perf_dict = dict(perfs)
    # Step 4: Filter configs to include only those with a corresponding perf entry
    merged = [
        (i, config, perf_dict[i])
        for i, config in configs
        if i in perf_dict
    ]

    # print(merged)
    sorted_data = sorted(
        merged,
        key=lambda item: (
            clean_float(item[2].get('total_anomaly_count')),
            clean_float(item[2].get('changed_service_anomaly_count')),
            clean_float(item[2].get('total_errors')),
            clean_float(item[2].get('service_max_cpu_usage')),
            clean_float(item[2].get('total_max_cpu_usage')),
        )
    )

    best = {}
    for p in reversed(sorted_data):
        # print(p[0], p[2])

        name = []
        docs = []
        for d in p[1]:
            name.append(d.get('metadata', {}).get('name'))
            docs.append(d)
            # print(d.get('metadata', {}).get('name'))
        
        for n in name:
            best[n] = (p[0], docs, p[2])

    for service_name, (s, c, p) in best.items():
        fp = f"{PATH}/output/{label}/config/{service_name}.yaml"
        for d in c:
            print("/n/n #### new config")
            print(d)
            if service_name == d.get('metadata', {}).get('name'):
                append_generation(fp, max(get_existing_seq(fp)) + 1, yaml.dump(d) + "\n---\n" + str(p)) 

    return best

@report_time
def main():
    LABEL = "alias"
    
    os.makedirs(PATH + f"output/{LABEL}", exist_ok=True)
    if(glob.glob(PATH + f"output/{LABEL}/config/*.yaml") == []):
        shutil.copytree(PATH + "base_config", PATH + f"/output/{LABEL}/config", dirs_exist_ok = True)
    else:
        print("base configuration files exist... skipping")

    print("Entering optimization loop:")
    for i in range(0, 5):
        print(f"########################### Loop: {i} ###########################")

        print("Load test to analyse present anomalies in the current configuration")
        client = get_k8s_api_client()
        reset_k8s(client, PATH + f"output/{LABEL}/config")
        
        locust_load_test(label=f"{LABEL}_{i}", base_label=LABEL, time='10m')

        print("Generate possible changes to mitigate then Apply and measure the changes")
        generate_and_measure(LABEL, i, '10m')

        # generate_configuration_updates(LABEL, i)
        # conf_list = load_generated_configurations(f"{LABEL}_{i}")
        # apply_and_measure(conf_list, LABEL, i, '10m')

        print("select the configurations that have performed the best and evolve")
        evolve(LABEL, i)

if __name__=="__main__":
    main()
