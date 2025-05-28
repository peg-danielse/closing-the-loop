import requests
import datetime

# Define time range
end = datetime.datetime.utcnow()
start = end - datetime.timedelta(days=2)  # past 1 hour

BASE_URL = "http://spark.lab.uvalight.net:31500"
METRIC_QUERIES = ["sum(activator_request_concurrency{namespace_name=\"default\", configuration_name=\"srv-user\", revision_name=\"srv-user-00001\"})", 
                  ""]

url = BASE_URL + '/api/v1/label/configuration_name/values'
response = requests.get(url)
data = response.json()
print("config names:", data['data'])

for c in data['data']:
    url = BASE_URL + '/api/v1/label/revision_name/values'
    params = {'match[]': f'autoscaler_desired_pods{{namespace_name="default",configuration_name="{c}"}}'}

    response = requests.get(url, params=params)

    data = response.json()
    print("Revision names:", data['data'])

    url = BASE_URL + '/api/v1/query_range'
    query = f'sum(autoscaler_actual_pods{{namespace_name="default", configuration_name="{c}", revision_name="{data['data'][0]}"}})'
    params = {'query': query,
              'start': start.isoformat() + 'Z',
              'end': end.isoformat() + 'Z',
              'step': '30s'}

    response = requests.get(url, params=params)

    result = response.json()
    print(result)
        