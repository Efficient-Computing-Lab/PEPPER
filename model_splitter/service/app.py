# app.py

from flask import Flask, request, jsonify, send_file
import os
import uuid
import threading
import sys 
from splitter.ModelSplitter import ModelSplitter
from splitter.Tarjan import Tarjan


import onnx
import re 
import base64 
import zipfile
#sys.path.append(__file__)
app = Flask(__name__)

UPLOAD_FOLDER = "/root/models"
OUTPUT_FOLDER = "/root/outputs"
JOB_STATUS = {}  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def sanitize_filename(name: str) -> str:
    # Only allow alphanumeric, dash, underscore, dot
    name = os.path.basename(name)
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

def sanitize_bridge_dist(bridge_dist):
    if not isinstance(bridge_dist, int) or bridge_dist < 0:
        bridge_dist = 6
    return bridge_dist


def process_model(job_id, filename, bridge_dist):
    print(f"[PROCESS] Starting model processing for job: {job_id}")
    try:
        JOB_STATUS[job_id]['status'] = "processing"
        model_path=filename
        
        # model_path = os.path.join(UPLOAD_FOLDER, filename)
        output_dir = os.path.join(OUTPUT_FOLDER, job_id)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[PROCESS] Loading model from {model_path}")
   
        try:
            
            onnx.checker.check_model(model_path)
        except Exception as e:
            JOB_STATUS[job_id]['status'] = "error"
            JOB_STATUS[job_id]['error'] = f"Invalid ONNX file: {str(e)}"
        
        model = onnx.load(model_path)
        tarjan = Tarjan(bridge_dist)
        print("you are here")
        md_splitter = ModelSplitter(model, tarjan)
        model_parts = md_splitter.split(model_path, output_dir)
        if not model_parts:
            raise ValueError("Model split returned no submodels.")
        print("[PROCESS] Finished processing")
        JOB_STATUS[job_id]['status'] = "done"
        JOB_STATUS[job_id]['output_dir'] = output_dir

    except Exception as e:
        import traceback
        traceback.print_exc()
        JOB_STATUS[job_id]['status'] = "error"
        JOB_STATUS[job_id]['error'] = str(e)
        print(f"[ERROR] Job error for job {job_id}: {e}")

@app.route("/jobs/upload", methods=["POST"])
def create_job():
    
    print("Headers:", dict(request.headers))
    print("Files:", request.files)
    print("Form:", request.form)
    if "model" not in request.files:
        return jsonify({"error": "No model file"}), 400

    model_file = request.files["model"]
    print("Client filename: ", model_file)
    bridge_dist = sanitize_bridge_dist(int(request.form.get("bridge_dist")))
    print(bridge_dist)
    job_id = str(uuid.uuid4())

    model_name_clean = sanitize_filename(model_file.filename)
    filename = os.path.splitext(os.path.basename(model_name_clean))[0]+".onnx"
    model_folder = os.path.join(UPLOAD_FOLDER, job_id)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_path = os.path.join(model_folder, filename)
    JOB_STATUS[job_id] = {
        "status": "pending",
        "model_path": model_path,
        "model_name": model_name_clean
    }
    model_file.save(model_path)

    thread = threading.Thread(
        target=process_model,
        args=(job_id, model_path, bridge_dist),
        daemon=True
    )
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "pending"
    }), 202


@app.route("/jobs/<job_id>/status", methods=["GET"])
def job_status(job_id):
    job = JOB_STATUS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({"status": job["status"], "model_name": job["model_name"]})


@app.route("/jobs/<job_id>/submodels", methods=["POST"])
def download_submodels(job_id):
    job = JOB_STATUS.get(job_id)

    if not job or job["status"] != "done":
        return jsonify({"error": "Job not ready"}), 400

    output_dir = job["output_dir"]
    zip_path = os.path.join(output_dir, "submodels_"+job_id+".zip")
    print('!! output dir', output_dir, 'zip path', zip_path)
    #import pdb; pdb.set_trace()
    
    if not os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "w") as zipf:
            
            print(os.listdir(output_dir))
            for file in os.listdir(output_dir):
                if file.endswith(".onnx") and file.find("part") > 0:
                    print("Sending ", file)
                    zipf.write(
                        os.path.join(output_dir, file),
                        arcname=file
                    )

    return send_file(zip_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
