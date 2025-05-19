import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os as os



#import all data.
PATH = "./data/exploration/"
datafiles = [e for e in os.listdir(PATH) if '.csv' in e]

fig, ax = plt.subplots(1,3, figsize=(12,5))

for i, e in enumerate(["low", "mid2", "high2"]):
    resp_df = pd.read_csv(PATH + f'{e}_responce_log.csv')
    sns.violinplot(resp_df["response_time"], ax = ax[i])
    ax[i].set_title(f'resp of {e}')
    ax[i].set_xlabel('')
    ax[i].set_ylabel('responce time [ms]')

plt.savefig("box.png")

exit()

# 1. create an overview of the stats from the latency.
fig, axes = plt.subplots(1,4, figsize=(16,5), sharey=True)

stats = [e for e in datafiles if 'stats.csv' in e]
print("stats", stats)
stats_df = pd.DataFrame()
for i, e in enumerate(stats):
    df = pd.read_csv(PATH + e)
    df['experiment'] = e.split('_')[0]
    df = df[df['Name'] == "Aggregated"]

    box_data = [df["Median Response Time"], df[""] ]

    plt.boxplot([q1,median,q3],whis=[min,max],ax=axes[i], color=sns.color_palette("Set2")[i],)
    axes[i].set_title(f'Response time of {e.split('_')[0]}')



    print(df)

# 2. graph of the latency cumulative distributiond

# 3. Then lets create an overview of the failures

exit()

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

for i, col in enumerate(df.columns):
    sns.boxplot(y=df[col], ax=axes[i], color=sns.color_palette("Set2")[i])
    axes[i].set_title(f'the request latency of workload {col}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Value' if i == 0 else '')

plt.tight_layout()
plt.savefig('box.pdf')
