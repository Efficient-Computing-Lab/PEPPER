import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import extract_characteristics
# 1. Create the Flask app instance
app = Flask(__name__)

# 2. Configure the upload folder and allowed extensions
# This is where uploaded files will be saved
UPLOAD_FOLDER = '/app/uploads'
ALLOWED_EXTENSIONS = {'onnx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    """Checks if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 3. Define the upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles POST requests to upload a file."""
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty part without a filename.
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 4. Save the file if it's valid
    if file and allowed_file(file.filename):
        # Use secure_filename to prevent security vulnerabilities
        filename = secure_filename(file.filename)

        # Construct the full path to save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the file to the specified path
        file.save(filepath)
        model_as_graph= extract_characteristics.load_model_as_graph(filepath)
        conv_layers, pool_layers, fc_layers, total_filters_details = extract_characteristics.analyze_layers(model_as_graph)
        total_parameters, param_shapes = extract_characteristics.count_parameters(model_as_graph)
        extract_characteristics.start_profiling(conv_layers,pool_layers,fc_layers, total_filters_details, total_parameters, filename)
        return jsonify({
            "message": "File uploaded successfully",
            "filename": filename,
            "filepath": filepath
        }), 201
    else:
        return jsonify({"error": "File type not allowed"}), 400


# 5. Run the application
if __name__ == '__main__':
    # Start the Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)