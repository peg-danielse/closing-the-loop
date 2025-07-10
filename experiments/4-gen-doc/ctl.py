import sys, os, json, glob, re, requests, time, subprocess, heapq, yaml, shutil

from typing import List, Tuple

from datetime import datetime
from functools import wraps
from operator import itemgetter

import numpy as np
import pandas as pd

import shap
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import seaborn as sns

from kubernetes import client, config

# experimental code, remove security warning.
import warnings
warnings.filterwarnings('ignore', message="Unverified HTTPS request*")

PATH = "./experiments/4-gen-doc/"
GEN_API_URL = "http://localhost:4242/generate"

KUBE_URL = "https://localhost:6443" 
KUBE_API_TOKEN =  ("eyJhbGciOiJSUzI1NiIsImtpZCI6ImtQQmZaNXU4all2UEVTMWg3Q0lwdmpNcEFLYVdZY2c4S0Y0d3NaRXZyMUkifQ"
                   ".eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uY"
                   "W1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImRlZmF"
                   "1bHQtdG9rZW4iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiZGVmY"
                   "XVsdCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6ImEwOTA0OGRiLTM"
                   "1OTItNGQ2OS1hMmY3LWU5MDM0Y2M1Y2U2NyIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpkZWZhdWx0OmRlZ"
                   "mF1bHQifQ.yXjIcIA4uL4gASdj5TE1v4hiHAKROka58RG7MAJOlXGQkekYVTkzc3rM0tmOMClEEED9kAZTb6Tdp1u2"
                   "2b9zBs4Qoil58wQ-ehlH4HMJG573lPRBVoq5kB3l3rKpwsWsfXpqm4xvWxlBxT6tZT9UZNTTcYUkLjnshGwsaE55NZ"
                   "M7TG4YIeslRj3CAT-gOGWFQxIt1QFhZUor3D6JHrDznLPYB5iAeqXbOuvPClILtlmXoVftp3hmOGtlYwH8uWox9YHI"
                   "_WECYrYEXp92a7yn7iq953hwD2lp22vozPDb_a5cTxC3AaSqv9FUTRby-fS8bpuXhKuyd-yLe9YqZTzk8Q")


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

# read stat history
def read_history(label):
    history_df = pd.read_csv(PATH + "data/" + f'{label}_stats_history.csv')
    history_df["Timestamp"] = pd.to_datetime(history_df['Timestamp'], unit='s')

    return history_df

# read response time data
def read_response(label):
    resp_df = pd.read_csv(PATH + "data/" + f'{label}_responce_log.csv')
    
    return resp_df

# read Jaeger trace data
def read_traces(label):
    data = {}
    with open(PATH + "data/" + f'{label}_traces.json', 'r') as file:
        data = json.load(file)

    rows = []
    for trace in data["data"]:
        row = {"id": trace['traceID']}
        
        total = 0
        st = sys.maxsize
        for s in trace["spans"]:
            st = min(st, s["startTime"])
            total += s["duration"]
            row[s["operationName"]] =  s["duration"]
        
        row["startTime"] = st
        row["total"] = total
        rows.append(row)

    # fill NaN
    trace_df = pd.DataFrame(rows).fillna(0)
    
    # create trace patterns.
    span_cols = trace_df.columns.difference(['id'])
    trace_df["pattern"] = trace_df[span_cols].gt(0).astype(int).astype(str).agg("".join, axis=1)

    # convert times
    trace_df["startTime"] = pd.to_datetime(trace_df['startTime'], unit='us')

    return trace_df

def read_metrics(label):
    metrics = {}
    for n in glob.glob(PATH + "data/" + f'{label}_*_metrics.csv'):
        match = re.search(r"[^_/]+-[^_]+(?=_metrics\.csv)", n)
        name = "unknown"
        if match:
            name=match.group()
        else:
            print("unregognized file:", n, "skipping")
            continue

        metric_df = pd.read_csv(n, index_col=False).drop('Unnamed: 0', axis=1)
        metric_columns = metric_df.columns.drop('index')
        metric_df["index"] = pd.to_datetime(metric_df['index'], unit='s')

        metrics[name] = metric_df
    
    return metrics

def shap_decisions(iso_forest, features, mark = "_"):
    # try to explain a specific data points. as a short cut for RCA.
    shap_values = shap.TreeExplainer(iso_forest).shap_values(features)

    return shap_values, features.columns.tolist()

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


def get_existing_seq(file_path):
    if not os.path.exists(file_path):
        return set([0])
    with open(file_path) as f:
        content = f.read()
    return map(int,set(re.findall(r"# --- START: seq=(.+?) ---", content)))


def append_generation(file_path, seq, content):
    if seq in get_existing_seq(file_path):
        print(f"Skipped {seq}: already exists.")
        return

    with open(file_path, "a") as f:
        f.write(f"# --- START: seq={seq} ---\n")
        f.write(content + "\n")
        f.write(f"# --- END: seq={seq} ---\n\n")


def get_generation_content(file_path):
    configurations = []
    with open(file_path, 'r') as f:
        content = f.read()

        matches = re.findall(r'(?s)# --- START: seq=(\d+) ---.*?```yaml(.*?)```.*?# --- END: seq=\1 ---', content)
        
        for seq, yaml in matches:
            configurations.append((seq, yaml.strip()))

    return configurations        
    

def load_generated_configurations(label):
    configurations = []

    for file in glob.glob(PATH + "output/" + f"{label}_prompts.json"):
        print(file, max(get_existing_seq(file)))
        confs = get_generation_content(file)
        print(len(confs))
        configurations = configurations + confs
    
    return configurations

def get_service_names(configurations: List):
    filtered = {}

    for a in configurations:
        try:
            docs = yaml.safe_load_all(a[1])
            # Extract metadata.name from each
            for i, doc in enumerate(docs, start=1):
                name = doc.get('metadata', {}).get('name')
                print(f"Document {i}: metadata.name = {name}")
                
        except Exception as e:
            print(e)
            continue

    return []

def update_globals(update, api_client):
    try:
        v1 = client.CoreV1Api(api_client)
        
        name = update["metadata"]["name"]
        namespace = update["metadata"].get("namespace", "knative-serving")

        # Fetch the existing ConfigMap to get the current resourceVersion
        existing_cm = v1.read_namespaced_config_map(name=name, namespace=namespace)
        update["metadata"]["resourceVersion"] = existing_cm.metadata.resource_version

        # Replace the ConfigMap
        v1.replace_namespaced_config_map(name=name, namespace=namespace, body=update)

        return True
    except Exception as e:
        print(e)
        return False

def update_knative_service(update, api_client):
    try:
        api = client.CustomObjectsApi(api_client)
        
        service_name = update["metadata"]["name"]
        namespace = update["metadata"].get("namespace", "default")

        existing = api.get_namespaced_custom_object(
            group="serving.knative.dev",
            version="v1",
            namespace=namespace,
            plural="services",
            name=service_name
        )

        update["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]

        # Copy immutable annotations to avoid webhook validation error
        existing_annotations = existing["metadata"].get("annotations", {})
        new_annotations = update["metadata"].setdefault("annotations", {})

        # Preserve immutable fields
        immutable_keys = [
            "serving.knative.dev/creator",
            "serving.knative.dev/lastModifier"
        ]

        for key in immutable_keys:
            if key in existing_annotations:
                new_annotations[key] = existing_annotations[key]

        # Replace the Knative Service
        api.replace_namespaced_custom_object(
            group="serving.knative.dev",
            version="v1",
            plural="services",    # this must match the CRD plural name
            namespace=namespace,
            name=service_name,
            body=update
        )

        return True

    except Exception as e:
        print(e)
        return False

def apply_yaml_configuration(doc, api_client):
    match (doc["kind"]):
        case "ConfigMap":
            print("Updated Config Map")
            update_globals(doc, api_client)
        case "Service":
            print("Updated Service")
            update_knative_service(doc, api_client)
        case _:
            print(doc['kind'], " not supported")


def locust_load_test(label, time = "5m"): # todo add all loadtest configurations
        # Locust!
    
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
        timeout=600  # 10 minutes
    )

    time.sleep(30)
    print(f"Test {label}, Finished")
    # if result.returncode != 0:
        # print(f"Error: {result.stderr}")


    experiment_files = glob.glob(f"./{label}_*")
    print("analysing the loadtest files...")
    print(experiment_files)
    for file in experiment_files:
        shutil.move(file, PATH + "data")
    


def reset_k8s(api_client):
    originals = glob.glob('./experiments/4-gen-doc/yaml/' + '*.yaml')
    for o in originals:
        with open(o) as f:
            doc = yaml.safe_load(f)
            apply_yaml_configuration(doc, api_client)

    print("waiting to fully accept initial configuration...")
    time.sleep(60)
    

def get_kpi_list(label:str, service) -> List:
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
    anom_features = features.iloc[anomaly_indices]
    shapes, names = shap_decisions(iso_forest, anom_features)

    service_anomaly_count = {}
    for s, ai in zip(shapes, anomaly_indices):
        values = heapq.nsmallest(3, enumerate(s), key=itemgetter(1))
        for v in values:
            service_name = os.path.basename(SPAN_PROCESS_MAP[names[v[0]]])[:-5] 

            if service in ["memcached-reservation-deployment", "memcached-rate-deployment", "frontend-deployment", "memcached-profile-deployment"]:
                continue

            service_anomaly_count[service_name] = service_anomaly_count.get(service_name, 0) + 1
            break

    service_max_cpu_usage = max(metric_dfs[service].get(f"{service}_container_cpu_usage_seconds_total", [0]))
    
    total_max_cpu_usage = 0.0
    for k_service, m_df in metric_dfs.items():
        total_max_cpu_usage = total_max_cpu_usage + max(m_df.get(f"{k_service}_container_cpu_usage_seconds_total",[0]))

    kpi = {
        "max_throughput": max(history_df["Requests/s"]),
        "total_errors" : max(history_df["Total Failure Count"]),
        "total_anomaly_count": len(trace_df[trace_df["anomaly"] == -1]),
        "changed_service_anomaly_count": service_anomaly_count.get(service, 0),
        "service_max_cpu_usage": service_max_cpu_usage,
        "total_max_cpu_usage": total_max_cpu_usage
    }

    return kpi


@report_time
def apply_and_measure(configurations: List, label):
    aConfiguration = client.Configuration()
    aConfiguration.host = KUBE_URL
    aConfiguration.verify_ssl = False
    aConfiguration.api_key = {"authorization": "Bearer " + KUBE_API_TOKEN}
    k8s_client = client.ApiClient(aConfiguration)

    reset_k8s(k8s_client)    

    for i, sol in enumerate(configurations):
        try:  
            docs = yaml.safe_load_all(sol[1])

            names = []
            for d in docs:
                apply_yaml_configuration(d, k8s_client)
                names = names + [d.get('metadata', {}).get('name')]


            print(f"applied updates to: {"-".join(names)}")
            print("waiting to update the configuration...")
            time.sleep(60)
            print("done")

            # Locust!
            experiment_label = f"{label}-{i}-{"-".join(names)}"
            locust_load_test(experiment_label, "5m")                   

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
    aConfiguration = client.Configuration()
    aConfiguration.host = KUBE_URL
    aConfiguration.verify_ssl = False
    aConfiguration.api_key = {"authorization": "Bearer " + KUBE_API_TOKEN}
    k8s_client = client.ApiClient(aConfiguration)
    
    reset_k8s(k8s_client)

    # locust_load_test(LABEL, "5m")  # label + "_round:" + str(i)
    # generate_configuration_updates(LABEL)
    conf_list = load_generated_configurations(LABEL)

    # apply and measure configuration updates()
    conf_perf = apply_and_measure(conf_list, LABEL)


if __name__=="__main__":
    main()
