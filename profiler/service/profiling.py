from os import cpu_count

import pandas as pd
import os
import joblib
import sys
import requests
from werkzeug.http import parse_if_range_header

prometheus_ip = os.getenv("PROMETHEUS_IP", "0.0.0.0:9090")
prometheus_endpoint = 'http://' + prometheus_ip + '/api/v1/query?'
prometheus_queries =[{"cpu_usage_query":'100 - (avg by (instance) (rate(node_cpu_seconds_total[1m])) * 100)'},
                     {"device_type_query" : 'char_node_device_model'},
                     {"disk_usage_query": '(1 - (node_filesystem_avail_bytes{mountpoint="/",fstype!="rootfs"} / node_filesystem_size_bytes{mountpoint="/",fstype!="rootfs"})) * 100' }]
FEATURE_COLUMNS = [
    'conv_layers', 'device_load_percent', 'disk_io_read_bytes',
    'disk_io_write_bytes', 'device_disk_usage_percent',
    'filter_details', 'fully_conn_layers',
    'device',
    'pool_layers', 'total_parameters'
]

def perform_query():
    cpu_result =[]
    disk_result=[]
    device_result=[]
    for query in prometheus_queries:
        json_key = list(query.keys())
        response_status = requests.get(prometheus_endpoint,
                                   params={
                                       'query': query.get(json_key[0])
                                   })
        if response_status.status_code == 200 and response_status.json()['data']['result']:
            results = response_status.json()['data']['result']

            for result in results:
                metric_value = result.get('value')
                value = "% .1f" % float(metric_value[1])
                metric = result.get('metric')
                if json_key[0] == "cpu_usage_query":
                    hostname = metric.get('instance')
                    cpu_record = {'node': hostname, 'cpu_usage(percentage)': int(float(value.strip()))}
                    cpu_result.append(cpu_record)
                if json_key[0] == "disk_usage_query":
                    hostname = metric.get('instance')
                    disk_record = {'node': hostname, 'disk_usage(percentage)': float(value.strip())}
                    disk_result.append(disk_record)
                if json_key[0] == "device_type_query":
                    hostname = metric.get('char_node_name')
                    device_record = {'node': hostname, 'class':metric.get("char_node_class"), 'device_model':metric.get("char_node_device_model_name"),
                                    'node_uuid':metric.get("char_node_uuid")}
                    device_result.append(device_record)

    return cpu_result, disk_result, device_result





def run_profiling(characteristics,model_name):
    run_results =[]
    model_path = 'best_trained_xgboost_model.joblib'
    cpu_result, disk_result, device_result = perform_query()
    print(cpu_result)
    print(disk_result)
    print(device_result)
    size = len(device_result)
    for i in range(size):
        device_model = device_result[i].get("device_model")
        if "Raspberry" in device_model:
            device_type = 0
        elif "Jetson" in device_model:
            device_type = 1
        else:
            device_type = 0
        input ={
                'conv_layers': characteristics.get("conv_layers"),
                'device_load_percent': cpu_result[i].get("cpu_usage(percentage)"),
                'device': device_type,
                'disk_io_read_bytes': characteristics.get("model_input_size"),
                'disk_io_write_bytes': characteristics.get("model_output_size"),
                'device_disk_usage_percent': disk_result[i].get("disk_usage(percentage)"),
                'filter_details': characteristics.get("filter_details"),
                'fully_conn_layers': characteristics.get("fc_layers"),
                'pool_layers': characteristics.get("pool_layers"),
                'total_parameters': characteristics.get("total_parameters")
            }
        try:
            loaded_model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' not found.")
            sys.exit(1)
        features_df = pd.DataFrame([input])[FEATURE_COLUMNS]
        prediction = loaded_model.predict(features_df)

        run_record ={"model":model_name,
                       "predicted_execution_time": float(prediction[0])}

        merged_response =  device_result[i]| run_record
        run_results.append(merged_response)
        # Find fastest devices for this specific run
        #predicted_fastest = min(run_results, key=lambda x: x["predicted_execution_time"])["device"]'
    print(run_results)
    min_record = min(
        run_results,
        key=lambda record: record.get("predicted_execution_time", float("inf"))
    )

    return min_record

