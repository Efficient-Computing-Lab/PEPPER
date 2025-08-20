import time
from os import write
import os
import onnx
import onnxruntime as ort
import numpy as np
import json
#from memory_profiler import memory_usage
import psutil
import threading
import datetime
import sys
import gpustat
import cv2
import os
import signal
import concurrent.futures
import argparse
import time
import queue
from collections import Counter
import csv
import random
from pathlib import Path
import subprocess
import re
import stresser

def load_tensors(input_name, input_data):
    dict_input ={}
    for input in input_name:
        dict_input[input]=input_data
    return dict_input

def write_distributed_results(model_name,result):
    os.makedirs("results", exist_ok=True)
    refactored_model_name = model_name.replace(".onnx","")
    path = "results/"+refactored_model_name+"_outputs_" + str(runs) + ".npy"
    np.save(path, result[0])
    write_bytes = os.path.getsize("results/"+refactored_model_name+"_outputs_" + str(runs) + ".npy")
    return write_bytes

def read_distributed_results(model_name):
    part_number = re.findall(r'\d+', model_name)
    input_number = int(part_number[0]) - 1
    refactored_model_name = model_name.replace(".onnx","")
    refactored_model_name = re.sub(r'\d+', str(input_number), refactored_model_name, count=1)
    path = "results/"+refactored_model_name+"_outputs_" + str(runs) + ".npy"
    read_bytes = os.path.getsize(path)
    input_data = np.load(path)
    delete_npy_file(path)
    return input_data, read_bytes



def count_parameters(graph):
    total_params = 0
    param_shapes = {}
    for initializer in graph.initializer:
        shape = tuple(initializer.dims)
        total_params += np.prod(shape)
        param_shapes[initializer.name] = shape
    return total_params, param_shapes


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





def infer_network_type(graph):
    node_types = [node.op_type for node in graph.node]
    if "Conv" in node_types:
        network_type = "Convolutional Neural Network (CNN)"
    elif any(op in node_types for op in ["LSTM", "RNN", "GRU"]):
        network_type = "Recurrent Neural Network (RNN)"
    elif all(op == "Gemm" for op in node_types):
        network_type = "Fully Connected Neural Network (MLP)"
    elif any("Attention" in op for op in node_types):
        network_type = "Transformer"
    else:
        network_type = "Unknown Network Type"
    return  network_type



# set image file dimensions to 224x224 by resizing and cropping image from center
def pre_process_edgetpu(image_path, dims):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

# resize the image with a proportional scale
def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

# crop the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def preprocess_image(img_path):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    center = 256 // 2
    img = img[center - 112:center + 112, center - 112:center + 112]  # Crop to 224x224
    img = img / 255.0  # Still float64 here
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std  # Still float64
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)  # ✅ Ensure final array is float32
    return img

def str2bool(value):
    if value.lower() in ('true', '1'):
        return True
    elif value.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def prompt_model_path(model_name):
    """ Prompt the user for model paths without providing a default. """
    user_input = input(f"Enter the path for {model_name}: ")
    while not user_input:  # Ensure the user provides a path
        print(f"Path for {model_name} cannot be empty!")
        user_input = input(f"Enter the path for {model_name}: ")
    model_path  = user_input
    return model_path

def save_output(model_name, device_type, metrics_list):
    # Gather all unique keys (sorted for consistent column order)
    all_keys = sorted({key for metrics in metrics_list for key in metrics})

    # Define the output filename
    output_file = f"inference-{model_name}-{device_type}.csv"

    # Check if file already exists
    file_exists = os.path.isfile(output_file)

    # Open file and write rows
    with open(output_file, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=all_keys)

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        # Write each metric dictionary as a row
        for metrics in metrics_list:
            # Ensure all keys are present (fill missing with None or 0)
            row = {key: metrics.get(key, "") for key in all_keys}
            writer.writerow(row)

    print(f"✅ Metrics saved to {output_file}")



def parse_gpu_stats(gpu_stats):
    gpus = gpu_stats.get("gpus")
    for gpu in gpus:
        temperature = gpu.get("temperature.gpu")
        utilization_percent = gpu.get("utilization.gpu")
        memory_used_mb = gpu.get("memory.used")
        memory_total_mb = gpu.get("memory.total")
        memory_percent = (memory_used_mb/memory_total_mb) * 100
    return temperature, utilization_percent, memory_percent

def count_execution_time(start_time,end_time):


    # Convert string timestamps to datetime objects
    dt_object_start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
    dt_object_end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f")

    # Calculate execution time
    execution_time = dt_object_end_time - dt_object_start_time
    # Extract total seconds and microseconds
    total_seconds = int(execution_time.total_seconds())
    microseconds = execution_time.microseconds

    # Convert total seconds to HH:MM:SS
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format with microseconds
    human_readable_string = f"{hours:02}:{minutes:02}:{seconds:02}.{microseconds:06}"
    return human_readable_string


def run_model(model_name,session,input_name,output_name,runs,image_path=None):
    read_bytes = 0
    write_bytes = 0
    if model_name == "deeplab_part1.onnx" or model_name == "deeplab.onnx" or model_name =="nasnet.onnx":
    #if len(input_name) == 1 and len(output_name) >=
        #disk_io_before = psutil.disk_io_counters
        img = cv2.imread(image_path)
        input_data = np.array(img).astype(np.float32)
        #input = input_name[0]
        if model_name != "nasnet.onnx":
            dict_input = load_tensors(input_name, input_data)
            result = session.run(output_name, dict_input)
        if model_name == "nasnet.onnx":
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32)
            img_preprocessed = (img_float / 127.5) - 1.0  # Same as nasnet_preprocess_input
            img_array = np.expand_dims(img_preprocessed, axis=0)
            result = session.run(None, {input_name[0]: img_array.astype(np.float32)})[0]
        if model_name == "deeplab_part1.onnx":
            write_bytes =write_distributed_results(model_name,result)
        #disk_io_after = psutil.disk_io_counters()
        read_bytes = os.path.getsize(image_path)
        if model_name == "deeplab.onnx" or model_name =="nasnet.onnx":
            write_bytes = sum(record.nbytes for record in result)
    if model_name == "deeplab_part2.onnx":
    #elif len(input_name) >1 and len(output_name)>=1:
        #disk_io_before = psutil.disk_io_counters
        input_data, read_bytes = read_distributed_results(model_name)
        dict_input = load_tensors(input_name, input_data)
        result = session.run(output_name, dict_input)
        #disk_io_after = psutil.disk_io_counters()
        write_bytes = sum(record.nbytes for record in result)
    if (model_name == "alexnet.onnx" or model_name == "resnet.onnx" or model_name == "vgg.onnx" or model_name == "googlenet.onnx"\
            or model_name == "densenet.onnx" or model_name == "mobilenet.onnx" or model_name == "efficientnet.onnx" or model_name == "alexnet_part1.onnx"\
            or model_name == "resnet_part1.onnx" or model_name == "vgg_part1.onnx" or model_name == "googlenet_part1.onnx"\
            or model_name == "densenet_part1.onnx" or model_name == "mobilenet_part1.onnx" or model_name == "efficientnet_part1.onnx"
            or model_name == "regnet.onnx" or model_name =="convnext.onnx"):
        with open("profiler/training/labels/synset.txt", "r") as f:
            labels = [l.strip() for l in f]
        input_name = session.get_inputs()[0].name
        read_bytes = os.path.getsize(image_path)
        #disk_io_before = psutil.disk_io_counters()
        if model_name != "efficientnet.onnx" and model_name!="efficientnet_part1.onnx":
            #print(model_name)
            img = preprocess_image(image_path)
            result = session.run(output_name, {input_name: img})
            #print(len(result))
            #googlenet result =1
            #mobilenet result =1
            #densenet result =1
            #alexnet result =1
            # disk_io_after = psutil.disk_io_counters()
            read_bytes = os.path.getsize(image_path)
            if "part" not in model_name:
                os.makedirs("results", exist_ok=True)
                np.save("results/" + model_name + "_outputs_" + str(runs) + ".npy", result[0])
                os.remove("results/" + model_name + "_outputs_" + str(runs) + ".npy")
            else:
                write_bytes = write_distributed_results(model_name, result)

        elif model_name == "efficientnet.onnx" or model_name =="efficientnet_part1.onnx":
            # pre-process the image like mobilenet and resize it to 224x224
            img = pre_process_edgetpu(image_path, (224, 224, 3))
            # create a batch of 1 (that batch size is buned into the saved_model)
            img_batch = np.expand_dims(img, axis=0)
            result = session.run(output_name, {input_name: img_batch})[0]
            #print(len(result))
            read_bytes = os.path.getsize(image_path)
            if model_name =="efficientnet.onnx":
                os.makedirs("results", exist_ok=True)
                np.save("results/"+model_name+"_outputs_" + str(runs) + ".npy", result[0])
                os.remove("results/"+model_name+"_outputs_" + str(runs) + ".npy")
            else:
                write_bytes = write_distributed_results(model_name, result)
    if model_name == "googlenet_part2.onnx" or model_name=="googlenet_part3.onnx" or model_name == "alexnet_part2.onnx":
        input_data, read_bytes = read_distributed_results(model_name)
        dict_input = load_tensors(input_name, input_data)
        result = session.run(output_name, dict_input)
        if model_name == "googlenet_part2.onnx":
            write_bytes = write_distributed_results(model_name, result)
        if model_name == "googlenet_part3.onnx" or model_name == "alexnet_part2.onnx":
            write_bytes = sum(record.nbytes for record in result)
        #disk_io_after = psutil.disk_io_counters()
        #read_bytes = disk_io_after.read_bytes - disk_io_before.read_bytes
        #write_bytes = disk_io_after.write_bytes - disk_io_before.write_bytes
    #scores = np.squeeze(result[0])
    #top5_idx = scores.argsort()[::-1][:5]
    #for idx in top5_idx:
      #print(f"class={labels[idx]} ; probability={scores[idx]:.4f}")

    return read_bytes,write_bytes





def load_onnx_model(model_path,device_type,model_name,runs,gpu_needed,metrics_list,random_cpu_cores,random_cpu_load, image_path=None):
    devices = ["raspberrypi","jetson","desktop"]
    if device_type not in devices:
        print("device_type should be either raspberrypi, jetson or desktop")
        sys.exit(1)
    if device_type in devices:
        # Load the ONNX model
        model = onnx.load(model_path)
        graph = model.graph


        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if gpu_needed == True and (device_type=="desktop" or device_type=="jetson"):
            providers = [
                ("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": "1", "cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider"]
        if gpu_needed == False or device_type=="raspberrypi":
            providers =[ "CPUExecutionProvider"]
        input_names = [inp.name for inp in model.graph.input]
        #print(input_names)  # List of input names
        output_names = [inp.name for inp in model.graph.output]
        #print(output_names)
        # Count the number of parameters
        #total_parameters = count_parameters(model)

        # Create an ONNX Runtime session
        session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
        # Prepare dynamic input data based on the input layer's shape
        #input_data = np.random.rand(*[dim if isinstance(dim, int) else 1 for dim in input_shape]).astype(np.float32)




        start_timestamp = datetime.datetime.now()
        read_bytes, write_bytes = run_model(model_name,session,input_names,output_names,runs,image_path)
        end_timestamp = datetime.datetime.now()
        execution_time = count_execution_time(str(start_timestamp),str(end_timestamp))
        cpu_usage = random_cpu_load * random_cpu_cores / 4
        disk_usage = psutil.disk_usage("/").percent
        output ={"model_name": model_name,
        "execution_time": execution_time,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "cpu_usage": cpu_usage,
        "disk_usage": disk_usage}
        print(output)
        metrics_list.append(output)




def delete_npy_file(path):
    os.remove(path)
    print(f"Deleted: {path}")


def show_menu():
    profiling_choice = input(
        "Select profiling mode:\n"
        "  0 - Profile main models\n"
        "  1 - Profile distributed models\n"
        "Enter your choice (0 or 1): "
    )

    model_name = None

    if profiling_choice == "0":
        while True:
            print("\n=== MAIN MODELS MENU ===")
            print("1. VGGNet")
            print("2. ResNet")
            print("3. GoogLeNet")
            print("4. DenseNet")
            print("5. MobileNet")
            print("6. EfficientNet")
            print("7. AlexNet")
            print("8. DeepLab")
            print("9. DeepLab part 1")
            print("10. DeepLab part 2")
            print("11 Regnet")
            print("12 ConvNext")
            print("13 Nasnet")
            print("14. EXIT")

            choice = input("Enter your choice (1–14): ")

            main_models = {
                '1': "vgg.onnx",
                '2': "resnet.onnx",
                '3': "googlenet.onnx",
                '4': "densenet.onnx",
                '5': "mobilenet.onnx",
                '6': "efficientnet.onnx",
                '7': "alexnet.onnx",
                '8': "deeplab.onnx",
                '9': "deeplab_part1.onnx",
                '10': "deeplab_part2.onnx",
                '11': "regnet.onnx",
                '12': "convnext.onnx",
                '13': "nasnet.onnx"
            }

            if choice in main_models:
                return main_models[choice]
            elif choice == '14':
                print("Exiting the menu.")
                break
            else:
                print("Invalid choice. Try again.")

    elif profiling_choice == "1":
        while True:
            print("\n=== DISTRIBUTED MODELS MENU ===")
            print("1. VGGNet part 1")
            print("2. VGGNet part 2")
            print("3. VGGNet part 3")
            print("4. ResNet part 1")
            print("5. ResNet part 2")
            print("6. ResNet part 3")
            print("7. ResNet part 4")
            print("8. ResNet part 5")
            print("9. ResNet part 6")
            print("10. ResNet part 7")
            print("11. ResNet part 8")
            print("12. ResNet part 9")
            print("13. GoogLeNet part 1")
            print("14. GoogLeNet part 2")
            print("15. GoogLeNet part 3")
            print("16. DenseNet part 1")
            print("17. DenseNet part 2")
            print("18. DenseNet part 3")
            print("19. DenseNet part 4")
            print("20. MobileNet part 1")
            print("21. MobileNet part 2")
            print("22. MobileNet part 3")
            print("23. MobileNet part 4")
            print("24. MobileNet part 5")
            print("25. MobileNet part 6")
            print("26. EfficientNet part 1")
            print("27. EfficientNet part 2")
            print("28. EfficientNet part 3")
            print("29. EfficientNet part 4")
            print("30. EfficientNet part 5")
            print("31. EfficientNet part 6")
            print("32. AlexNet part 1")
            print("33. AlexNet part 2")
            print("34. DeepLab part 1")
            print("35. DeepLab part 2")
            print("36. EXIT")

            choice = input("Enter your choice (1–36): ")

            distributed_models = {
                '1': "vgg_part1.onnx",   '2': "vgg_part2.onnx",   '3': "vgg_part3.onnx",
                '4': "resnet_part1.onnx",'5': "resnet_part2.onnx",'6': "resnet_part3.onnx",
                '7': "resnet_part4.onnx",'8': "resnet_part5.onnx",'9': "resnet_part6.onnx",
                '10': "resnet_part7.onnx",'11': "resnet_part8.onnx",'12': "resnet_part9.onnx",
                '13': "googlenet_part1.onnx",'14': "googlenet_part2.onnx",'15': "googlenet_part3.onnx",
                '16': "densenet_part1.onnx",'17': "densenet_part2.onnx",'18': "densenet_part3.onnx",'19': "densenet_part4.onnx",
                '20': "mobilenet_part1.onnx",'21': "mobilenet_part2.onnx",'22': "mobilenet_part3.onnx",
                '23': "mobilenet_part4.onnx",'24': "mobilenet_part5.onnx",'25': "mobilenet_part6.onnx",
                '26': "efficientnet_part1.onnx",'27': "efficientnet_part2.onnx",'28': "efficientnet_part3.onnx",
                '29': "efficientnet_part4.onnx",'30': "efficientnet_part5.onnx",'31': "efficientnet_part6.onnx",
                '32': "alexnet_part1.onnx",'33': "alexnet_part2.onnx",
                '34': "deeplab_part1.onnx",'35': "deeplab_part2.onnx"
            }

            if choice in distributed_models:
                return distributed_models[choice]
            elif choice == '36':
                print("Exiting the menu.")
                break
            else:
                print("Invalid choice. Try again.")
    else:
        print("Invalid profiling mode. Exiting.")

    return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Profile ONNX models.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of profiling iterations.")
    parser.add_argument("--device_type", type=str, default="desktop", help="Device type (desktop, raspberrypi, jetson).")
    parser.add_argument("--image_path", type=str,
                        help="Path to the image file.")
    parser.add_argument("--inference_mode", type=str, default="simple", help="Inference mode(simple, distributed)")

    #parser.add_argument("--model_name", type=str, default="deeplab.onnx",help="ONNX model name.")
    parser.add_argument("--gpu_needed", type=str2bool, default=[False],
                        help="Inference with GPU or not (true/false")
    args = parser.parse_args()

    device_type = args.device_type
    devices = ["raspberrypi","jetson","desktop"]
    if device_type not in devices:
        print("device_type should be either raspberrypi, jetson or desktop")
        sys.exit(1)
    image = args.image_path


    model_name = show_menu()
    complete_metrics_list = []
    gpu_needed = args.gpu_needed
    # Prompt the user for model paths before starting the profiling loop
    model_path = prompt_model_path(model_name)
    for iteration in range(args.iterations):

        runs = iteration + 1
        print(f"Profiling {model_name}")
        print(f"Iteration: {runs}")
        print('-' * 50)
        random_cpu_load = random.randint(20, 80)
        random_cpu_cores = random.randint(1, 4)  # Use 1–4 to avoid cpu=0
        stresser.start_stresser(random_cpu_cores, random_cpu_load)
        time.sleep(15)
        stresser.get_metrics(random_cpu_cores, random_cpu_load)
        if image:
            load_onnx_model(model_path, device_type, model_name, runs, gpu_needed, complete_metrics_list, random_cpu_cores, random_cpu_load, image)
        else:
            load_onnx_model(model_path, device_type, model_name, runs, gpu_needed, complete_metrics_list, random_cpu_cores, random_cpu_load)
        stresser.end_stresser()
    #if model_name == "deeplab_part2.onnx":

    save_output(model_name, device_type, complete_metrics_list)
