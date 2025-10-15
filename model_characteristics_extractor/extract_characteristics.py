import numpy as np
import onnx
import requests
import json
import os

def load_model_as_graph(model_path):
    model = onnx.load(model_path)
    graph = model.graph
    return graph


def start_profiling(conv_layers, pool_layers, fc_layers, total_filters_details, total_parameters,filename, model_input_size,model_output_size):
    characteristics = {
        'conv_layers': conv_layers,
        'filter_details': total_filters_details,
        'fc_layers': fc_layers,
        'pool_layers': pool_layers,
        'total_parameters': total_parameters,
        'model': filename,
        'model_output_size': model_output_size,
        'model_input_size': model_input_size
    }

    # Convert to JSON
    data_json = json.dumps(characteristics)

    # Get URL from environment variable
    url = os.environ.get("PROFILING_SERVICE_URL")
    if not url:
        raise ValueError("Environment variable 'PROFILING_SERVICE_URL' is not set")

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=data_json, headers=headers)
    response.raise_for_status()  # Raise error if request fails

    return response.json()

def count_parameters(graph):
    total_params = 0
    param_shapes = {}
    for initializer in graph.initializer:
        shape = tuple(initializer.dims)
        total_params += np.prod(shape)
        param_shapes[initializer.name] = shape
    return int(total_params), param_shapes

def analyze_layers(graph):
    from collections import defaultdict

    conv_layers = []
    pool_layers = []
    fc_layers = []
    total_filters_details = 0

    # Map outputs to nodes
    output_to_node = {}
    for node in graph.node:
        for output in node.output:
            output_to_node[output] = node

    # Initializer map for weight shapes
    name_to_initializer = {init.name: init for init in graph.initializer}

    for node in graph.node:
        if node.op_type == "Conv":
            if len(node.input) > 1 and node.input[1] in name_to_initializer:
                weight_tensor = name_to_initializer[node.input[1]]
                if hasattr(weight_tensor, 'dims') and len(weight_tensor.dims) > 0:
                    num_filters = weight_tensor.dims[0]
                    total_filters_details += num_filters
                    conv_layers.append(node)

                    # Heuristic: Conv after GAP = classifier head
                    prev_tensor = node.input[0]
                    prev_node = output_to_node.get(prev_tensor)
                    if prev_node and prev_node.op_type == "GlobalAveragePool":
                        fc_layers.append(node)

        elif node.op_type in ["MaxPool", "AveragePool", "GlobalAveragePool"]:
            pool_layers.append(node)

        elif node.op_type == "Gemm":
            fc_layers.append(node)

        elif node.op_type == "MatMul":
            prev_tensor = node.input[0]
            prev_node = output_to_node.get(prev_tensor)

            # Heuristic 1: MatMul after Flatten
            if prev_node and prev_node.op_type == "Flatten":
                fc_layers.append(node)

            # Heuristic 2: MatMul with weight input from initializer
            elif len(node.input) > 1 and node.input[1] in name_to_initializer:
                fc_layers.append(node)

    print("conv_layers:", len(conv_layers))
    print("pool_layers:", len(pool_layers))
    print("fc_layers:", len(fc_layers))
    print("total_filters_details:", total_filters_details)

    return len(conv_layers), len(pool_layers), len(fc_layers), total_filters_details


model_as_graph = load_model_as_graph("/home/gkorod/Downloads/best.onnx")
conv_layers, pool_layers, fc_layers, total_filters_details = analyze_layers(model_as_graph)
total_parameters, param_shapes = count_parameters(model_as_graph)
print("conv_layers: "+ str(conv_layers))
print("pool_layers: "+str(pool_layers))
print("fc_layers: "+ str(fc_layers))
print("total_filters_details: "+ str(total_filters_details))
print("total_parameters: "+ str(total_parameters))
print("param_shapes: "+ str(param_shapes))