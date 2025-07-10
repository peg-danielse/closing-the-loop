import sys, os, json, glob, re, requests, time, subprocess, heapq, yaml, shutil

from typing import List, Tuple

from kubernetes import client, config

# experimental code, remove security warning.
import warnings
warnings.filterwarnings('ignore', message="Unverified HTTPS request*")

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
        raise
        

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
    
    return


def reset_k8s(api_client):
    originals = glob.glob('./experiments/4-gen-doc/yaml/' + '*.yaml')
    for o in originals:
        with open(o) as f:
            doc = yaml.safe_load(f)
            apply_yaml_configuration(doc, api_client)

    print("waiting to fully accept initial configuration...")
    time.sleep(60)

    return


def get_k8s_api_client():
    aConfiguration = client.Configuration()
    aConfiguration.host = KUBE_URL
    aConfiguration.verify_ssl = False
    aConfiguration.api_key = {"authorization": "Bearer " + KUBE_API_TOKEN}

    return client.ApiClient(aConfiguration)
