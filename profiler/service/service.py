from flask import Flask, request, jsonify
from profiling import run_profiling
app = Flask(__name__)
import logging

@app.route('/api/profiling', methods=['POST'])
def profiling():
        # Get the JSON payload
        data = request.get_json()

        # Extract fields (optional validation)
        model_name = data.get('model_name')
        model_input_size = data.get('model_input_size')
        model_output_size = data.get('model_output_size')
        conv_layers = data.get('conv_layers')
        pool_layers = data.get('pool_layers')
        fc_layers = data.get('fc_layers')
        filter_details = data.get('filter_details')
        total_parameters = data.get('total_parameters')


        # Example: process / log / store the profiling info
        print(f"Model: {model_name}")
        print(f"Model Input Size: {model_input_size}")
        print(f"Model Output Size: {model_output_size}")
        print(f"Conv Layers: {conv_layers}")
        print(f"Pool Layers: {pool_layers}")
        print(f"Fully Connected Layers: {fc_layers}")
        print(f"Filter Details: {filter_details}")
        print(f"Total Parameters: {total_parameters}")
        data.pop("model_name", None)
        profiling_prediction = run_profiling(data,model_name)
        # Build response
        response = {
            "status": "success",
            "message": "Profiling data received",
            "received_data": data,
            "profiling_prediction": profiling_prediction
        }
        logging.info(response)
        return jsonify(response), 200


if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=7001, debug=True)
