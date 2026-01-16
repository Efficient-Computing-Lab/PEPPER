# print(__file__)
# import sys
# sys.path.append(__file__)
from .SplitAlgo import SplitAlgo
import onnx 
import json
import os 
import networkx as nx

""" Receive the model as is, how you receive the model is independent of the model splitter logic 
(so it can be plugged in any pipeline ideally, whether it is a Flask app or a simple script which loads the model)."""
class ModelSplitter:
    def __init__(self, model, algo: SplitAlgo):
        self.model = model
        self.algo = algo

    def add_initializers_to_graph_input(self): # add initializer nodes to the full model
        # Patch initializer nodes to ONNX graph input because it's required by the ONNX submodel extraction routine
        initializer_names = {init.name for init in self.model.graph.initializer}
        input_names = {inp.name for inp in self.model.graph.input}

        # Find initializers that are used but not listed as inputs
        missing_inputs = initializer_names - input_names

        for init in self.model.graph.initializer:
            if init.name in missing_inputs:
                value_info = onnx.helper.make_tensor_value_info(
                    name=init.name,
                    elem_type=init.data_type,
                    shape=init.dims
                )
                self.model.graph.input.append(value_info)
        #return model

    """
    Add initializers to input names of target split points to be used with the onnx submodel extraction routine
    """
    def add_missing_initializers(self, input_names):
        initializer_names = {init.name for init in self.model.graph.initializer}
        used_inputs = set()
        for node in self.model.graph.node:
            used_inputs.update(node.input)

        # Add missing initializers that are used but not already in input_names
        for name in initializer_names:
            if name in used_inputs and name not in input_names:
                input_names.append(name)
        return input_names
    
    # Patch initializer nodes to extracted submodel
    def patch_missing_initializers_submodel(self, sub_model):
        #full_model = onnx.load(original_model_path)
        #sub_model = onnx.load(submodel_path)

        full_initializers = {init.name: init for init in self.model.graph.initializer}
        sub_input_names = {inp.name for inp in sub_model.graph.input}
        sub_initializer_names = {init.name for init in sub_model.graph.initializer}
        
        # Find which required initializers are missing
        missing = [
            name for name in sub_input_names
            if name in full_initializers and name not in sub_initializer_names
        ]

        for name in missing:
            sub_model.graph.initializer.append(full_initializers[name])
        
        return sub_model
    
    #f Fill missing node names in the full model with dummy names
    def fill_missing_node_names(self): 
        graph = self.model.graph
        i=0
        for node in graph.node:
            if node.name=='':
                node.name='dummy_'+str(i)
                i = i + 1

    def preprocess(self, model_path, output_dir):
        self.fill_missing_node_names()
        self.add_initializers_to_graph_input()

        return 


    # get the split points produced by the splitting algorithm implementing the abstract SplitAlgo class
    
    def extract_sub_models(self, model_path, split_points, output_dir, nx_graph):
        # Load the ONNX model
        #model = onnx.load(model_path)
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract sub-models
        prev_output_names = None
        model_parts = []
        for i, (input_names, output_names) in enumerate(split_points): 
            
            # Define the output path for the sub-model
            part_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(model_path))[0]}_part{i+1}.onnx")
            input_names = self.add_missing_initializers(input_names)

            onnx.utils.extract_model(model_path, part_filename, input_names, output_names)
            model_part = onnx.load(part_filename)
            model_part = self.patch_missing_initializers_submodel(model_part)
            onnx.checker.check_model(model_part)  
            model_parts.append(model_part)
            onnx.save(model_part, part_filename)

            # Update previous output names for next sub-model input
            output_names = []
            input_names = []
            
            
        return model_parts
            
    
    def split(self, model_path, output_dir):
        self.preprocess(model_path, output_dir)
        patched_model_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(model_path))[0]}_patched.onnx")
        onnx.save(self.model, patched_model_path)
        self.model = onnx.load(patched_model_path)
        split_points, nx_graph = self.algo.find_split_points(self.model)
        print(type(nx_graph))
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(patched_model_path))[0]+".graphml")
        nx.write_graphml(nx_graph, output_file)
        return self.extract_sub_models(patched_model_path, split_points, output_dir, nx_graph)


