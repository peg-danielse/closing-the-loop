import sys
import os as os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest

PATH = "./data/anomaly-detection/"
label = "anomaly-detection-weibull"

# read stat history
history_df = pd.read_csv(PATH + f'{label}_stats_history.csv')

fix, ax = plt.subplots(1,1)
ax.plot(history_df['Timestamp'], history_df['Requests/s'], label="Requests/s", color="blue")
ax2 = ax.twinx()

ax2.plot(history_df['Timestamp'], history_df['User Count'], label="user count", color="orange")

plt.title("Locust recorded Requests/s and User Count")
plt.xlabel("Time")
ax.set_ylabel("Request/s")
ax2.set_ylabel("User Count")
ax.legend(loc="lower left")
ax2.legend(loc="lower right")

plt.grid(True)

plt.savefig(PATH + "plots/" + f"{label}_requests_per_second.png")
plt.clf()

# read response time data
resp_df = pd.read_csv(PATH + f'{label}_responce_log.csv')

# read Jaeger trace data
data = {}
with open(PATH + f'{label}_traces.json', 'r') as file:
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

# make violin plot of the distributions
fig, ax = plt.subplots(1,5, figsize=(15,5))

sns.violinplot(resp_df["response_time"], ax = ax[0])
ax[0].set_xlabel(f'aggregated ({len(resp_df)})')
ax[0].set_ylabel('responce time [ms]')

for i, e in enumerate(resp_df['name'].unique(), 1):
    df = resp_df[resp_df['name'] == e]
        
    sns.violinplot(df["response_time"], ax = ax[i])
    ax[i].set_xlabel(f'{e} ({len(df)})')
    ax[i].set_ylabel('responce time [ms]')

plt.tight_layout()
plt.savefig(PATH + "plots/" + "resp_dist.pdf")
plt.clf()

# Find outliers using an IsolationForest classifier.
features = trace_df.select_dtypes(include=["number"]).drop(columns=["total", "startTime"])
iso_forest = IsolationForest(contamination="auto", random_state=42)
trace_df["anomaly"] = iso_forest.fit_predict(features)



anom_df = trace_df[trace_df["anomaly"]== -1]
print("#### Anomalies ####")
print(anom_df)
print(len(anom_df))

# make fig
color = {1: "steelblue", -1: "skyblue"}

fig, ax = plt.subplots(1, len(trace_df["pattern"].unique())+1, figsize=(60,5))
sns.violinplot(trace_df, y='total', ax=ax[0], split=True, hue='anomaly', palette=color, cut=0)
ax[0].set_xlabel(f'aggregated ({len(anom_df)}/{len(trace_df)})')
ax[0].set_ylabel('responce time [ms]')

handles, labels = ax[0].get_legend_handles_labels()
new_labels = ["Normal" if lbl == "1" else "Anomaly" for lbl in labels]
ax[0].legend(handles, new_labels, loc="upper right")

for i, p in enumerate(trace_df["pattern"].unique(), 1):
    df = trace_df[trace_df["pattern"] == p]
    if(len(df[df["anomaly"] == -1]) == 0 or len(df[df["anomaly"] == 1]) == 0):
        sns.violinplot(df, y='total', ax=ax[i], hue='anomaly', palette=color, inner='stick', cut=0)
    else:
        sns.violinplot(df, y='total', ax=ax[i], split=True, hue='anomaly', palette=color, inner='stick', cut=0)
    
    ax[i].set_xlabel(f'{p} ({len(df[df["anomaly"] == -1])}/{len(df)})')
    
    handles, labels = ax[i].get_legend_handles_labels()
    new_labels = ["Normal" if lbl == "1" else "Anomaly" for lbl in labels]
    ax[i].legend(handles, new_labels, loc="upper right")


plt.tight_layout()
plt.savefig(PATH + "plots/" + f"{label}_trace_dist.png")
plt.clf()

import shap
# feature importance for the prediction.
def shap_summary(iso_forest, features):
    all_shap_values = shap.TreeExplainer(iso_forest).shap_values(features)

    # try to explain a specific data points. as a short cut for RCA.
    anomaly_indices = [428, 399, 402, 434, 304, 321, 420]
    subset = features.iloc[anomaly_indices]

    shap_values = shap.TreeExplainer(iso_forest).shap_values(subset)

    shap.decision_plot(0, shap_values, feature_names=subset.columns.tolist())#, link='logit') #, feature_order="hclust",)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.tight_layout()
    plt.savefig(PATH + "plots/" + f"{label}_shap_decision_plot.png") 
    plt.clf()

    # explaination of the shap values themselves.
    shap.summary_plot(all_shap_values, features)
    plt.savefig(PATH + "plots/" + f"{label}_shap_feature_importance.png")
    plt.clf()

    shap.plots.violin(all_shap_values, features=features, plot_type="layered_violin")
    plt.savefig(PATH + "plots/" + f"{label}_shap_violin_feature_importance.png")
    plt.clf()






# make Requests per second x Anomalies per second :) 
_, ax = plt.subplots(1,1, figsize=(12,5))

history_df["Timestamp"] = pd.to_datetime(history_df['Timestamp'], unit='s')

sns.lineplot(x=history_df['Timestamp'], y=history_df['Requests/s'], label="Requests/s", color="steelblue", ax=ax)
ax2 = ax.twinx()
anom_df["startTime"] = pd.to_datetime(anom_df['startTime'], unit='us')
sns.histplot(data=anom_df, x="startTime", label="Anomaly/s", color="red", bins=500, ax=ax2)

# Optional: Combine legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax.legend(lines + lines2, labels + labels2, loc='upper left')
# ax2.get_legend().remove()

plt.savefig(PATH + "plots/" + f"{label}_anomaly_per_second.png")
plt.clf()
