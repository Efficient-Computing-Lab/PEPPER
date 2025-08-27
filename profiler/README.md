# Profiler
![Profiler](profiler.png)

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
docker buildx build --platform linux/amd64 -t gkorod/model_characteristics_extractor:v1.0 --push --no-cache
```
### Run Docker Container
The following command initiates the Profiler inside a Docker container.
Profiler requires to know a Prometheus endpoint in order to retrieve metrics
regarding the devices.

The Prometheus monitoring mechanism 
should use [Node Exporter](https://github.com/prometheus/node_exporter) 
and [Characterization-Agent v1.2](https://github.com/Efficient-Computing-Lab/EdgeCloud-Mon/tree/main/char_agent)
```bash
docker run -d -t -p 7001:7001 -e PROMETHEUS_IP=147.102.19.159:9090 gkorod/profiler:v1.0
```
### Send Request
Profiler expects this kind of input:
```bash 
curl --location 'http://147.102.19.159:7001/api/profiling' \
--header 'Content-Type: application/json' \
--data '{       "model_name" : "model",
        "conv_layers" : 66666666,
        "pool_layers" : 77777777,
        "fc_layers" : 34535353535,
        "filter_details" : 66,
        "total_parameters" : 10000000000000000000,
        "model_input_size": 345343543,
        "model_output_size": 123421435}'
```