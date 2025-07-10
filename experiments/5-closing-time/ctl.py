import sys, os, json, glob, re, requests, time, subprocess, heapq, yaml, shutil

from typing import List, Tuple

from datetime import datetime
from functools import wraps
from operator import itemgetter

import numpy as np
import pandas as pd

import shap
from sklearn.ensemble import IsolationForest

from kubernetes import client, config

from ./util/analysis import *
from ./util/sequence import *
from ./util/square import *

# experimental code, remove security warning.
import warnings
warnings.filterwarnings('ignore', message="Unverified HTTPS request*")

PATH = "./experiments/5-closing-time/"

GEN_API_URL = "http://localhost:4242/generate"

SPAN_PROCESS_MAP = {'HTTP GET /hotels': './experiments/4-gen-doc/yaml/frontend-deployment.yaml',
'HTTP GET /recommendations': './experiments/4-gen-doc/yaml/frontend-deployment.yaml',
'HTTP POST /user': './experiments/4-gen-doc/yaml/frontend-deployment.yaml',
'HTTP POST /reservation': './experiments/4-gen-doc/yaml/frontend-deployment.yaml',

'memcached_get_profile': './experiments/4-gen-doc/yaml/memcached-profile-deployment.yaml',
'memcached_capacity_get_multi_number': './experiments/4-gen-doc/yaml/memcached-reservation-deployment.yaml',
'memcached_reserve_get_multi_number': './experiments/4-gen-doc/yaml/memcached-reservation-deployment.yaml',
'memcached_get_multi_rate': './experiments/4-gen-doc/yaml/memcached-rate-deployment.yaml',

'/profile.Profile/GetProfiles': './experiments/4-gen-doc/yaml/srv-profile.yaml',
'/search.Search/Nearby': './experiments/4-gen-doc/yaml/srv-search.yaml',
'/user.User/CheckUser': './experiments/4-gen-doc/yaml/srv-user.yaml',
'/geo.Geo/Nearby': './experiments/4-gen-doc/yaml/srv-geo.yaml',
'/recommendation.Recommendation/GetRecommendations': './experiments/4-gen-doc/yaml/srv-recommendation.yaml',
'/rate.Rate/GetRates': './experiments/4-gen-doc/yaml/srv-rate.yaml',
'/reservation.Reservation/CheckAvailability': './experiments/4-gen-doc/yaml/srv-reservation.yaml',
'/reservation.Reservation/MakeReservation': './experiments/4-gen-doc/yaml/srv-reservation.yaml'}

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


def report_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper


@report_time
def generate_configuration_updates(label):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    history_df = read_history(label)
    responce_df = read_response(label)
    trace_df = read_traces(label)
    metric_dfs = read_metrics(label)

    # IMPROVEMENT: improve pipeline with the ELBD framework. will look good in the paper.
    # IMPROVEMENT: also perform outlier detection on the monitoring metrics.

    # Find outliers in the traces using an IsolationForest classifier.
    features = trace_df.select_dtypes(include=["number"]).drop(columns=["total"])
    iso_forest = IsolationForest(contamination="auto", random_state=42)
    trace_df["anomaly"] = iso_forest.fit_predict(features)

    # IMPROVEMENT: decision plot the shap values by clustering similar shap values...
    anomaly_indices = trace_df[(trace_df['anomaly'] == -1)].index.to_list()

    if anomaly_indices == []: 
        return

    anom_features = features.iloc[anomaly_indices]
    shapes, names = shap_decisions(iso_forest, anom_features)
                
    for s, ai in zip(shapes, anomaly_indices):
        values = heapq.nsmallest(3, enumerate(s), key=itemgetter(1))
        for v in values:
        
            # IMRPOVEMENT: map to yaml by searching the entire space using an LLM or more complex NLP.
            # filter unsupported deployment files (ie. THAT DONT HAVE KNATIVE HORIZONTAL SCALING active)
            if os.path.basename(SPAN_PROCESS_MAP[names[v[0]]])[:-5] in ["memcached-reservation-deployment", "memcached-rate-deployment", "frontend-deployment", "memcached-profile-deployment"]:
                continue

            # time series of the selected service.
            metric_dfs[os.path.basename(SPAN_PROCESS_MAP[names[v[0]]])[:-5]]
            trace_df["startTime"][ai], pd.to_datetime(trace_df["total"][ai], unit='us')

            m_df = metric_dfs[os.path.basename(SPAN_PROCESS_MAP[names[v[0]]])[:-5]]

            # convert microseconds to timedelta
            start_time = trace_df["startTime"][ai]
            duration = pd.to_timedelta(trace_df["total"][ai] + 1000000, unit="us")  # 'us' for microseconds

            start_plus_duration = start_time + duration
            start_minus_5s = start_time - pd.Timedelta(seconds=10)

            print(start_time)
            print(duration)

            time_mask = (m_df["index"] >= start_minus_5s) & (m_df["index"] <= start_plus_duration)
            time_m_df = m_df[time_mask]

            with open(PATH + "yaml/" + f"{os.path.basename(SPAN_PROCESS_MAP[names[v[0]]])}") as f_yaml, open(PATH + "yaml/global.yaml") as f_glob:
                # IMPROVEMENTS ontology -> look at data and update. then use, to store the required knowledge for better generation and context.
                payload = {
                    "messages": [
                        # ["system", "give one short and concise reasoning then answer with the full yaml file including the fix."],
                        ["user", DATA_PROMPT.format(start_time=start_time, duration=duration, trace_df=trace_df.iloc[[ai]].to_string(), monitor_df=time_m_df.to_string())],
                        ["user", FILE_PROMPT.format(service_file=f_yaml.read(),global_file=f_glob.read())]
                    ],
                    "max_new_tokens": 2000
                }

                response = requests.post(GEN_API_URL, json=payload)

                seq = get_existing_seq(PATH + "output/" + f"{label}_prompts.json")
                append_generation(PATH + "output/" + f"{label}_prompts.json", max(seq) + 1,response.json()["response"]) 



def locust_load_test(label, time = "5m"): # todo add all loadtest configurations
    cmd = [
        "/home/pager/Documents/closing-the-loop/venv/bin/locust",
        "--processes", "16",
        "-f", "locust/hotel-reservations.py",
        "-H", "http://spark.lab.uvalight.net:30505",
        "-t", time,
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

    for file in experiment_files:
        shutil.move(file, PATH + "data")


@report_time
def apply_and_measure(configurations: List, label):
    client = get_k8s_api_client()
    reset_k8s(client)

    c_i = max(get_existing_seq(PATH + "output/" + f"{label}_performance.json"))

    for sol in configurations:
        if sol[0] <= c_i:
            print(f"already evaluated {sol[0]}, skipping...")
            continue

        try:
            docs = yaml.safe_load_all(sol[1])

            names = []
            for d in docs:
                apply_yaml_configuration(d, k8s_client)
                names = names + [d.get('metadata', {}).get('name')]

            print(f"applied updates to: {", ".join(names)}")
            print("waiting to update the configuration...")
            time.sleep(60)

            # Locust!
            experiment_label = f"{label}-{sol[0]}-{"-".join(names)}"
            locust_load_test(experiment_label, "10m")

            # KPI's 
            config_performance = get_kpi_list(experiment_label, names[0])
            print(config_performance)
            
            append_generation(PATH + "output/" + f"{label}_performance.json", sol[0], str(config_performance)) 

            reset_k8s(k8s_client)

        except Exception as e:
            print(e)
            continue

    return []
            
@report_time
def main():
    LABEL = "closed-5m"
    # for i in range(max_iterations):
    client = get_k8s_api_client()
    reset_k8s(client)

    locust_load_test(LABEL, "5m")  # label + "_round:" + str(i)

    generate_configuration_updates(LABEL)

    # apply and measure configuration updates()
    conf_list = load_generated_configurations(LABEL)
    apply_and_measure(conf_list, LABEL)


if __name__=="__main__":
    main()
