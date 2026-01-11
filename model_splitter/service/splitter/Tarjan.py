from .SplitAlgo import SplitAlgo
import re 
import json 
import networkx as nx 
#import matplotlib.pyplot as plt 

class Tarjan(SplitAlgo):
    def __init__(self, mix_distance):
        self.mix_distance = mix_distance
        #self.onnx_model = onnx_model
        self.split_points = []

    def get_key_names(self, pattern, dict_graph):
        key_names = []
        for key in dict_graph.keys():
            check = re.search(pattern, str(key))
            if check is not None:
                key_names.append(check.string)
        return key_names

    def is_operator_node(self, dict_graph, node_key):
        node_data = dict_graph.get(node_key, {})
        return node_data.get("op_type") not in ["Input", "Output", "Initializer"]
    
    def is_valid_split_output(self, tensor_name):
        return (
            "raw_output" not in tensor_name
            and tensor_name.endswith(":0")
            and "Identity" not in tensor_name
        )

    # def generate_nx_graph # generate_graph_multi_input
    def onnx_to_networkx(self, model):
    # Load the ONNX model
        #model = onnx.load(onnx_model_path)
        graph = model.graph

        # Create a new directed graph
        G = nx.DiGraph()
        initializer_names = {init.name for init in graph.initializer}
        # Add nodes to the graph
        for node in graph.node:
            # Convert attributes to a serializable format (e.g., JSON string)
            # attributes = {}
            # for attr in node.attribute:
            #     try:
            #         # Serialize the attribute to a string
            #         attributes[attr.name] = json.dumps(attr.SerializeToString().decode("utf-8", errors="ignore")) 
            #     except Exception as e:
            #         # Skip attributes that can't be converted to a string
            #         print(f"Skipping attribute {attr.name}: {e}")

            # Convert the attributes dictionary to a JSON string
            #attrs=json.dumps(attributes),
            G.add_node(
        node.name,
        op_type=node.op_type,
        
        inputs=list(node.input),
        outputs=list(node.output)
    )

        # Add edges to the graph
        for node in graph.node:
            for input_name in node.input:
                for output_node in graph.node:
                    if input_name in output_node.output:
                        G.add_edge(output_node.name, node.name)

        #import pdb; pdb.set_trace()
        # Optionally, add information about inputs and outputs
        i = 0
        for input_tensor in graph.input:
            is_initializer = input_tensor.name in initializer_names
            G.add_node(f"input_{i}", exact_name = input_tensor.name, op_type='Initializer' if is_initializer else 'Input', attrs=json.dumps({}))
            for node in graph.node:
                if input_tensor.name in node.input:
                    G.add_edge(f"input_{i}", node.name)
            i = i + 1

        i = 0
        for output_tensor in graph.output:
            G.add_node(f"output_{i}", exact_name = output_tensor.name, op_type='Output', attrs=json.dumps({}))
            #G.add_node(output_tensor.name, op_type='Output', attrs=json.dumps({}))
            
            for node in graph.node:
                if output_tensor.name in node.output:
                    G.add_edge(node.name,f"output_{i}")
            i = i + 1

        
        #import pdb; pdb.set_trace()
        # dict_graph = dict(G.nodes.data())
        # for node in dict_graph:
        #     if 'outputs' in node.keys():
        #         if len(node['outputs']) > 1:
        #             print(node)

        return G
    
    def find_split_points(self, model):
        graph = self.onnx_to_networkx(model)
        
        dict_graph = dict(graph.nodes.data())
        # Plot the graph
        # plt.figure(figsize=(12, 12))

        # # Draw the graph using spring layout for better visualization
        # pos = nx.spring_layout(graph)

        # # Draw nodes and edges with labels
        # nx.draw(graph, pos, with_labels=True, node_size=500, font_size=10, node_color='skyblue', edge_color='gray', linewidths=1.5, font_weight='bold')

        # # Display the plot
        # plt.title('ONNX Model Graph')
        # plt.show()

        # Convert the directed graph to an undirected graph
        undirected_graph = graph.to_undirected()

        # Find all the bridges in the undirected graph using Tarjan's algorithm
        bridges = list(nx.bridges(undirected_graph))

        # filter identified bridges to exclude input and output nodes. Only ops nodes are meaningful
        filtered_bridges = [
        b for b in bridges
        if self.is_operator_node(dict_graph, b[0]) and self.is_operator_node(dict_graph, b[1])
    ]

        bridges = filtered_bridges
        # Output the bridges
        print("Bridges found in the graph:")
        for bridge in bridges:
            print(bridge)

        print(len(bridges))

        result = bridges 


        inputs = self.get_key_names("input_[0-9]", dict_graph)
        outputs = self.get_key_names("output_[0-9]", dict_graph)
        #import pdb; pdb.set_trace()
        root_node = inputs[0]
        depths_start = nx.single_source_shortest_path_length(graph, root_node)
        depths_end = nx.single_source_shortest_path_length(graph, outputs[0])
        # for bridge in result:
        #     print(nx.single_source_shortest_path_length(,)

        found = []
        for bridge in result:
            if bridge[0] in list(depths_start.keys()): #or bridge[1] in list(depths_end.keys()):
                found.append(bridge)


        all_info = []
        for bridge in found:
            info = {}
            info['bridge'] = bridge
            info['depth_start'] = depths_start[bridge[0]]
            try:
                info['depth_end'] = depths_end[bridge[1]]
            except:
                info['depth_end'] = -1
            all_info.append(info)

        end_nodes = [dict_graph[key]["exact_name"] for key in outputs]

        split_points = []
        last_split_point = 0
    
        for info in all_info:
            if info['depth_start'] - last_split_point > self.mix_distance:
                split_points.append(info)
                last_split_point = info['depth_start']
        if not split_points:
            raise ValueError("No split points found")
        #import pdb; pdb.set_trace()
        target_node_name = split_points[0]['bridge'][1]
        target_splits = [([dict_graph[root_node]["exact_name"]],dict_graph[target_node_name]['outputs'])]
        last_split = dict_graph[target_node_name]['outputs']
        for bridge in split_points[1:-1]:


            target_splits.append((last_split, dict_graph[bridge['bridge'][1]]['outputs']))
            last_split = dict_graph[bridge['bridge'][1]]['outputs']
        target_splits.append((last_split, end_nodes))

        for i, (input_name, output_name) in enumerate(target_splits):
                #input_names = [input_name]
                #output_names = [output_name]
                print(input_name, output_name)

        return target_splits, graph