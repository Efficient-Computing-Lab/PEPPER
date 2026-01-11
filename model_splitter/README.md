# Split Service
## Overview
The Model Splitter partitions a model that should be given in ONNX format using Tarjan's algorithm for detecting bridges in a graph. More specifically, the module:
- Preprocesses the ONNX graph (adds missing nodes and initializer nodes when necessary)
- Applies Tarjan’s algorithm for bridge detection to uncover critical connection points (graph bridges)
- Filters bridges according to their distance (`bridge_dist` argument) to prevent creation of trivial subgraphs
- Uses detected bridges as split points with the ONNX subgraph extraction utilities

The splitter is wrapped in a Flask service.

## Instructions

### Prerequisites
- x86 host with NVIDIA GPU
- Docker, Docker Compose
- NVIDIA container runtime (follow [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to set it up)

### Build the Docker image:
```
docker compose -f compose_split.yaml --profile model_splitter build
```

### Run the service:
- Create a `.env` file in the same folder as the Docker compose file (`compose_split.yaml`)defining the path to the ONNX models in the host (`${MODELS_PATH_HOST}`) and the container (`${MODELS_PATH_CONTAINER}`)
- Run 
```
docker compose -f compose_split.yaml --profile model_splitter up
```

### Send model and request splitting:
```
curl -X POST http://localhost:5000/jobs   -F "model=@<path to model>"   -F "bridge_dist=<integer number>"
```
Returns a json message with the job id and status ("pending")

### Download submodels:
```
curl -o submodels.zip -X POST http://localhost:5000/jobs/<job id>/submodels
```

Returns a zip file named `submodels.zip` with the model parts

### Request job status:
```
curl http://localhost:5000/jobs/<job_id>
```