import sys, os, json, glob, re, requests, time, subprocess, heapq, yaml, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import glob, re


import shap

from typing import List, Tuple


from datetime import datetime
from functools import wraps
from operator import itemgetter

from sklearn.ensemble import IsolationForest

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

PATH = "./experiments/6-production/"

LABEL_A = "general"
LABEL_B = "holdup"

data = {
    "x": [],
    "with": [],
    "without": [],
    }

"anomaly count"
"resource consumption"
"failures per iteration"


def read_jager(label: str, loop: int):
    data = {}
    with open(PATH + f"output/{label}/data/{label}_{loop}/{label}_{loop}_traces.json", 'r') as file:
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

    # Find outliers using an IsolationForest classifier.
    features = trace_df.select_dtypes(include=["number"]).drop(columns=["total", "startTime"])
    iso_forest = IsolationForest(contamination="auto", random_state=42)
    trace_df["anomaly"] = iso_forest.fit_predict(features)

    trace_df["duration"] = pd.to_timedelta(trace_df["total"], unit="us")


    filtered_df = trace_df[trace_df['duration'] > pd.Timedelta(seconds=2)]
    filtered_df = filtered_df[filtered_df['anomaly'] == -1]

    # print(trace_df["duration"])
    # print(filtered_df)

    return filtered_df


# read stat history
def read_history(label, base_label):
    history_df = pd.read_csv(PATH + f"/output/{base_label}/data/{label}/" + f'{label}_stats_history.csv')
    history_df["Timestamp"] = pd.to_datetime(history_df['Timestamp'], unit='s')


    history_df.fillna(0, inplace=True)

    return history_df

# read response time data
def read_response(label, base_label):
    resp_df = pd.read_csv(PATH + f"/output/{base_label}/data/{label}/" + f'{label}_responce_log.csv')
    
    return resp_df

# read Jaeger trace data
def read_traces(label, base_label):
    data = {}
    with open(PATH + f"/output/{base_label}/data/{label}/" + f'{label}_traces.json', 'r') as file:
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

def read_metrics(label, base_label):
    metrics = {}
    for n in glob.glob(PATH + f"/output/{base_label}/data/{label}/" + f'{label}_*_metrics.csv'):
        match = re.search(r"[^_/]+-[^_]+(?=_metrics\.csv)", n)
        name = "unknown"
        if match:
            name=match.group()
        else:
            # print("unregognized file:", n, "skipping")
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

def get_kpi_list(label: str, base_label: str, service) -> List:
    history_df = read_history(label, base_label)
    responce_df = read_response(label, base_label)
    trace_df = read_traces(label, base_label)
    metric_dfs = read_metrics(label, base_label)

    for key, m_df in metric_dfs.items():
        m_df.fillna(0, inplace=True)

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
        duration = pd.to_timedelta(trace_df["total"][ai], unit="us")

        if duration < pd.Timedelta(seconds=2):
            continue

        values = heapq.nsmallest(2, enumerate(s), key=itemgetter(1))
        for v in values:
            if names[v[0]] in ['mongo_rate']:
                continue

            service_name = os.path.basename(SPAN_PROCESS_MAP[names[v[0]]])[:-5] 
            service_anomaly_count[service_name] = service_anomaly_count.get(service_name, 0) + 1

            break

    service_max_cpu_usage = max(metric_dfs.get(service, {}).get(f"{service}_container_cpu_usage_seconds_total", [-1]))
    
    total_max_cpu_usage = 0.0
    for k_service, m_df in metric_dfs.items():
        total_max_cpu_usage = total_max_cpu_usage + max(m_df.get(f"{k_service}_container_cpu_usage_seconds_total",[0]))

    kpi = {
        "max_throughput": max(history_df["Requests/s"]),
        "total_errors" : max(history_df["Total Failure Count"]),
        "total_anomaly_count": sum(service_anomaly_count.values()),
        "changed_service_anomaly_count": service_anomaly_count.get(service, 0),
        "service_max_cpu_usage": service_max_cpu_usage,
        "total_max_cpu_usage": total_max_cpu_usage,
        "p99": sum(history_df["99%"])/len(history_df),
        "p50": sum(history_df["50%"])/len(history_df),
    }

    return kpi

def get_generation_content_perf(file_path):
    configurations = []
    with open(file_path, 'r') as f:
        content = f.read()

        matches = re.findall(r'(?s)# --- START: seq=(\d+) ---.*?({.*?}).*?# --- END: seq=\1 ---', content)

        for seq, yaml in matches:
            configurations.append((seq, yaml.strip()))

    return configurations

def get_error_rate(label, base_label):
    perf = get_generation_content_perf(PATH + f"/output/{base_label}/{label}_performance.json")

    perf = [(i, yaml.safe_load(p)) for i, p in perf]
    perf = [(i, p) for i, p in perf if "Audit-Id" in p]

    return len(perf)

data = []
for i in range(0, 3):
    
    label_a = f"{LABEL_A}_{i}"
    label_b = f"{LABEL_B}_{i}"

    tmp_a = get_kpi_list(label_a, LABEL_A, "srv-user")
    tmp_a['iteration'] = i
    tmp_a['config'] = "with domain knowledge"
    tmp_a['gen_error'] = get_error_rate(label_a, LABEL_A)
    data.append(tmp_a)


    tmp_b = get_kpi_list(label_b, LABEL_B, "srv-user")
    tmp_b['iteration'] = i
    tmp_b['config'] = "without domain knowledge"
    tmp_b['gen_error'] = get_error_rate(label_b, LABEL_B)
    data.append(tmp_b)

df = pd.DataFrame(data)

# # Plot max_throughput over iterations
# sns.lineplot(data=df, x='iteration', y='total_max_cpu_usage', hue='config', marker='o')
# plt.title('max CPU usage over Iterations')
# plt.ylabel('CPU seconds')
# plt.savefig("./metric.pdf")
# plt.clf()

# # Plot total_anomaly_count
# sns.lineplot(data=df, x='iteration', y='total_anomaly_count', hue='config', marker='o')
# plt.title('Total Anomaly Count over Iterations')
# plt.ylabel('Anomalies')
# plt.savefig("./anomaly.pdf")
# plt.clf()


# df["gen_error_rate"] = df["gen_error"] / df["total_anomaly_count"]
# sns.lineplot(data=df, x='iteration', y='gen_error', hue='config', marker='o')
# plt.title('Total generation error count over Iterations')
# plt.ylabel('Errors')
# plt.savefig("./errors.pdf")

from matplotlib.ticker import MaxNLocator


# Assuming df is already defined and includes 'gen_error' and 'total_anomaly_count'
df["gen_error_rate"] = df["gen_error"] / df["total_anomaly_count"]

# Create a single figure with 3 subplots (1 row x 3 columns)
fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharex=True)

axs = axs.flatten()
# Plot 1: Total Max CPU Usage
sns.lineplot(data=df, x='iteration', y='total_max_cpu_usage', hue='config', marker='o', ax=axs[0])
axs[0].set_title('Max CPU Usage over Iterations')
axs[0].set_ylabel('CPU seconds')

# Plot 2: Total Anomaly Count
sns.lineplot(data=df, x='iteration', y='total_anomaly_count', hue='config', marker='o', ax=axs[1])
axs[1].set_title('Total Anomaly Count over Iterations')
axs[1].set_ylabel('Anomalies')

# Plot 3: Generation Error Count
sns.lineplot(data=df, x='iteration', y='gen_error', hue='config', marker='o', ax=axs[2])
axs[2].set_title('Generation Errors over Iterations')
axs[2].set_ylabel('Errors')

# Plot 4: Generation request latenct Count
# sns.lineplot(data=df, x='iteration', y='p50', hue='config', marker='o', ax=axs[3])
sns.lineplot(data=df, x='iteration', y='p99', hue='config', marker='s', ax=axs[3])

axs[3].set_title('p99 execution time over Iterations')
axs[3].set_ylabel('Latency (s)')
axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))

# Move legend to bottom or remove from subplots if it's duplicated
handles, labels = axs[0].get_legend_handles_labels()
for ax in axs:
    ax.get_legend().remove()

fig.legend(handles, labels, loc='upper center', ncol=4)


# Save figure
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
plt.savefig("./combined_metrics.pdf")
plt.close()