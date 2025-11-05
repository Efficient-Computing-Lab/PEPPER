# PEPPER: Profiling-based Edge Placement and Partitioning for
 
This repository provides the code for a pipeline
is for profiling and partitioning ONNX models, to enhance
inference efficiency across heterogeneous hardware platforms. Optimal split points within the deep learning models are identified
through the application of Tarjan’s Bridge-Finding Algorithm, and
the inference times of the models are predicted per device based on
the respective characteristics and CPU load. For the prediction of
inference times, the XGBoost algorithm is employed. The effectiveness of the proposed approach is validated through experiments
conducted on real-world edge devices, demonstrating that highly efficient and adaptable deployment of complex deep learning models
can be achieved in such environments.


## Architecture

![Architecture](./architecture.png)

### Components

1. [Model Extractor Characteristics](/model_characteristics_extractor/README.md)
2. [Profiler](/profiler/README.md)
3. Model Splitter (to be added)
4. [EdgeCloud Mon as monitoring mechanism](https://github.com/Efficient-Computing-Lab/EdgeCloud-Mon)