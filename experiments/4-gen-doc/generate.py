import sys, os, json, glob, re, requests
import pylcs

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import shap
from sklearn.ensemble import IsolationForest
from datetime import datetime

PATH = "./experiments/4-gen-doc/"
API_URL = "http://localhost:4242/generate"
label = "llm-small"

# read stat history
def read_history():
    history_df = pd.read_csv(PATH + "data/" + f'{label}_stats_history.csv')
    history_df["Timestamp"] = pd.to_datetime(history_df['Timestamp'], unit='s')

    # history_df['Timestamp']
    # history_df['Requests/s']
    # history_df['Timestamp'] 
    # history_df['User Count']
    return history_df

# read response time data
def read_response():
    resp_df = pd.read_csv(PATH + "data/" + f'{label}_responce_log.csv')
    
    return resp_df

# read Jaeger trace data
def read_traces():
    data = {}
    with open(PATH + "data/" + f'{label}_traces.json', 
'r') as file:
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

def read_metrics():
    metrics = {}
    # make and investigate the newly create metrics in the metrics.csv(s)
    for n in glob.glob(PATH + "data/" + f'{label}_*_metrics.csv'):
        match = re.search(r"[^_/]+-[^_]+(?=_metrics\.csv)", n)
        name = "unknown"
        if match:
            name=match.group()

        metric_df = pd.read_csv(n, index_col=False).drop('Unnamed: 0', axis=1)
        metric_columns = metric_df.columns.drop('index')
        metric_df["index"] = pd.to_datetime(metric_df['index'], unit='s')

        # for ax, metric in zip(axes, metrics):
            # x=metric_df['index'], y=metric_df[metric], ax=ax)
            # set_title(metric)
        # print(metric_df)
        metrics[name] = metric_df
    
    return metrics

# feature importance for the prediction.
# all_shap_values = shap.TreeExplainer(iso_forest).shap_values(features)
def shap_decisions(iso_forest, features, mark = "_"):
    # try to explain a specific data points. as a short cut for RCA.
    shap_values = shap.TreeExplainer(iso_forest).shap_values(features)
    
    ### visualize the model features and their importance for the decision.
    # shap.decision_plot(0, shap_values, feature_names=features.columns.tolist())#, link='logit') #, feature_order="hclust",)

    # fig = plt.gcf()
    # fig.set_size_inches(18.5, 10.5)

    # plt.tight_layout()
    # plt.savefig(PATH + "output/" + f"{label}_{mark}_shap_decision_plot.png") 
    # plt.clf()

    return shap_values, features.columns.tolist()

def longest_common_substring(s1, s2):
   m = [[0] * (1 + len(s2)) for i in range (1 + len(s1))]
   longest, x_longest = 0, 0
   for x in range(1, 1 + len(s1)):
       for y in range(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
   return s1[x_longest - longest: x_longest]


# resp_df["response_time"]
# resp_df['name'].unique()
#   df = resp_df[resp_df['name'] == e]    
#   df["response_time"], ax = ax[i])
def main():
    history_df = read_history()
    responce_df = read_response()
    trace_df = read_traces()
    metric_dfs = read_metrics()    

    # Find outliers in the traces using an IsolationForest classifier.
    # IMPROVEMENT: improve pipeline with the ELBD framework. will look good in the paper.
    features = trace_df.select_dtypes(include=["number"]).drop(columns=["total"])
    iso_forest = IsolationForest(contamination="auto", random_state=42)
    trace_df["anomaly"] = iso_forest.fit_predict(features)

    from operator import itemgetter
    import heapq

    yaml = glob.glob('./experiments/4-gen-doc/yaml/' + '*.yaml')


    span_process_Map = {'HTTP GET /hotels': './experiments/4-gen-doc/yaml/frontend-deployment.yaml', 
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
    
    # IMPROVEMENT: decision plot the shap values by clustering similar shap values...
    for p in trace_df["pattern"].unique():
        anomaly_indices = trace_df[(trace_df['pattern'] == p) & (trace_df['anomaly'] == -1)].index.to_list()

        if anomaly_indices == []: continue

        anom_features = features.iloc[anomaly_indices]
        shapes, names = shap_decisions(iso_forest, anom_features, p)
        
        
        print("###", p, "###")
        for s, ai in zip(shapes, anomaly_indices):
            
            values = heapq.nsmallest(3, enumerate(s), key=itemgetter(1))
            # values = enumerate(s)

            print("time:",trace_df["startTime"][ai])
            print("duration:", trace_df["total"][ai] / 1000000)


            for v in values:
                print(v[1], names[v[0]])

                # IMRPOVEMENT: map to yaml by searching the entire space using an LLM or more complex NLP.
                with open(span_process_Map[names[v[0]]]) as f:
                    print(f.read())                 


                # filter unsupported deployment files (NO KNATIVE HORIZONTAL SCALING)
                if os.path.basename(span_process_Map[names[v[0]]])[:-5] in ["memcached-reservation-deployment", "memcached-rate-deployment", "frontend-deployment", "memcached-profile-deployment"]:
                    continue

                # time series of the selected service.
                metric_dfs[os.path.basename(span_process_Map[names[v[0]]])[:-5]]
                trace_df["startTime"][ai], pd.to_datetime(trace_df["total"][ai], unit='us')

                m_df = metric_dfs[os.path.basename(span_process_Map[names[v[0]]])[:-5]]
                
                time_mask = (m_df["index"].dt.second >= (trace_df["total"][ai] / 1000000) +  10) & \
                            (m_df["index"].dt.second <= 20)

                print(m_df[time_mask])


                f_yaml = open(PATH + "yaml/" + f"{os.path.basename(span_process_Map[names[v[0]]])}")
                    # print(f.read()) 
                f_glob = open(PATH + "yaml/global.yaml")

                payload = {
                    "messages": [
                        ["system","give one short and concise reasoning and on a new line the answer with only modifications of values."],
                        ["user", "Please resolve the following anomaly:"],
                        ["user", 
                        f'''Please resolve the following anomaly: 
                          Start time: {metric_dfs[os.path.basename(span_process_Map[names[v[0]]])[:-5]]}, 
                          Request length: {trace_df['startTime'][ai]} {pd.to_datetime(trace_df['total'][ai], unit='us')} 
                          <anomalous_trace_csv> 
                          {trace_df.iloc[[ai]]} 
                          </anomalous_trace_csv> 
                          <monitoring_metrics_csv> 
                          {m_df[time_mask]} 
                          </monitoring_metrics_csv> 
                        '''],
                        ["user", 
                        f"By updating the following configuration file: \
                          <yaml>{f_yaml.read()}</yaml> \
                          \
                          global configuration file: \
                          <yaml>{f_glob.read()}<yaml>"] 
                    ],
                    "max_new_tokens": 2000
                }

                print(payload)

                continue

                response = requests.post(API_URL, json=payload)
                print(response.json()["response"].splitlines()[-1])

    
                #---> apply candidates and celect them.
                # start population of candidates. measure.           


                # IMPROVEMENTS ontology -> look at data and update. then use, to store the required knowledge.

if __name__=="__main__":
    main()
