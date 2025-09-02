# Profiler
The Profiler takes as input the model(s) specifications and the resource characteristics of the cluster (e.g., CPU current load, disk usage, and device type) that consists of a varying number of edge devices. 
Finally, the Profiler predicts in which device of the cluster the model will run faster and places the model(s) in that device.


![Profiler](./profiler.png)

## Training
To train the XGBoost model of the Profiler, run the following command:
```bash
cd training
python3 ten-fold-cross-validation
```
The output of the script would be the actual model. It will be saved with the name
`best_trained_xgboost_model.joblib` under the `profiler` directory.

## Web Service
To use the Profiler it is suggested to run its container version
by doing the following steps:

### Build the Docker image
The following command builds the image and stores it in Dockerhub
```bash 
docker buildx build --platform linux/amd64 -t gkorod/profiler:v1.0 --push --no-cache
```
### Run Docker Container
The following command initiates the Profiler inside a Docker container.
Profiler requires to interact with a Prometheus endpoint in order to retrieve metrics
regarding the devices.

The Prometheus monitoring mechanism 
should use [Node Exporter](https://github.com/prometheus/node_exporter) 
and [Characterization-Agent v1.2](https://github.com/Efficient-Computing-Lab/EdgeCloud-Mon/tree/main/char_agent)

During the development, we tested profiler with this [monitoring stack](https://github.com/Efficient-Computing-Lab/EdgeCloud-Mon/) that is able to monitor
every node of a Kubernetes cluster. Developers can use any Prometheus based monitoring stack that use the above mentioned agents
```bash
docker run -d -t -p 7001:7001 -e PROMETHEUS_IP=147.102.19.159:9090 gkorod/profiler:v1.1
```
### Send Request
Profiler expects this kind of input:
```bash 
curl --location 'http://147.102.19.159:7001/api/profiling' \
--header 'Content-Type: application/json' \
--data '{       "model" : "model",
        "conv_layers" : 66666666,
        "pool_layers" : 77777777,
        "fc_layers" : 34535353535,
        "filter_details" : 66,
        "total_parameters" : 10000000000000000000,
        "model_input_size": 345343543,
        "model_output_size": 123421435}'
```