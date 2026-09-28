from flask import Flask, request, render_template_string, redirect, url_for, jsonify
from pathlib import Path
import subprocess
import threading
import uuid
import os
import re

app = Flask(__name__)

INPUT_DIR = Path("/data/input")
OUTPUT_DIR = Path("/data/output")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs = {}

HTML = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Video Slide Extractor</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f5f5f7;
            margin: 0;
            padding: 40px;
        }
        .box {
            max-width: 700px;
            margin: auto;
            background: white;
            padding: 40px;
            border-radius: 18px;
            box-shadow: 0 8px 30px rgba(0,0,0,.08);
        }
        h1 { margin-top: 0; }
        .upload {
            border: 2px dashed #aaa;
            padding: 40px;
            text-align: center;
            border-radius: 14px;
            margin: 25px 0;
        }
        button {
            background: #111;
            color: white;
            border: 0;
            padding: 13px 24px;
            border-radius: 9px;
            font-size: 16px;
            cursor: pointer;
        }
        button:disabled {
            background: #999;
        }
        #filename {
            margin: 15px 0;
            font-weight: 600;
        }
        #status {
            margin-top: 25px;
            white-space: pre-wrap;
            background: #f0f0f0;
            padding: 18px;
            border-radius: 10px;
            min-height: 80px;
        }
    </style>
</head>
<body>
<div class="box">
    <h1>Video Slide Extractor</h1>
    <p>
        Upload a lecture video. The extractor will process it chronologically
        and keep the final annotated state of each logical slide.
    </p>

    <form id="uploadForm">
        <div class="upload">
            <input id="video" type="file" name="video"
                   accept="video/*" required>
            <div id="filename">No video selected</div>
        </div>

        <button id="start" type="submit">Upload & Start Extraction</button>
    </form>

    <div id="status">Waiting for video...</div>
</div>

<script>
const video = document.getElementById("video");
const filename = document.getElementById("filename");
const form = document.getElementById("uploadForm");
const statusBox = document.getElementById("status");
const button = document.getElementById("start");

video.addEventListener("change", () => {
    filename.textContent =
        video.files.length ? video.files[0].name : "No video selected";
});

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!video.files.length) return;

    button.disabled = true;
    statusBox.textContent = "Uploading video...";

    const data = new FormData();
    data.append("video", video.files[0]);

    const response = await fetch("/upload", {
        method: "POST",
        body: data
    });

    const result = await response.json();

    if (!result.job_id) {
        statusBox.textContent = result.error || "Upload failed.";
        button.disabled = false;
        return;
    }

    poll(result.job_id);
});

async function poll(jobId) {
    const response = await fetch("/status/" + jobId);
    const result = await response.json();

    statusBox.textContent = result.message || "";

    if (result.done) {
        button.disabled = false;

        if (result.success) {
            statusBox.textContent +=
                "\\n\\nCompleted successfully.\\n\\n" +
                "Output folder:\\n" + result.output;
        }

        return;
    }

    setTimeout(() => poll(jobId), 2000);
}
</script>
</body>
</html>
"""


def clean_filename(name):
    name = Path(name).stem
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    name = name.strip(" .")
    return name or "video"


def run_extraction(job_id, video_path, output_dir):
    jobs[job_id]["message"] = "Starting extraction..."

    try:
        process = subprocess.Popen(
            [
                "python",
                "extract_slides_final.py",
                str(video_path),
                str(output_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        lines = []

        for line in process.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                jobs[job_id]["message"] = line

        return_code = process.wait()

        if return_code == 0:
            jobs[job_id]["done"] = True
            jobs[job_id]["success"] = True
            jobs[job_id]["output"] = str(output_dir)
            jobs[job_id]["message"] = (
                "Extraction complete.\\n" +
                "\\n".join(lines[-8:])
            )
        else:
            jobs[job_id]["done"] = True
            jobs[job_id]["success"] = False
            jobs[job_id]["message"] = (
                "Extraction failed.\\n" +
                "\\n".join(lines[-15:])
            )

    except Exception as e:
        jobs[job_id]["done"] = True
        jobs[job_id]["success"] = False
        jobs[job_id]["message"] = f"Error: {e}"


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video selected."}), 400

    file = request.files["video"]

    if not file.filename:
        return jsonify({"error": "No video selected."}), 400

    original_stem = clean_filename(file.filename)
    suffix = Path(file.filename).suffix.lower()

    # Keep every uploaded video.
    # If the same filename is uploaded again, create:
    #   Lecture
    #   Lecture_2
    #   Lecture_3
    #   ...
    stem = original_stem
    counter = 2

    while (
        (INPUT_DIR / (stem + suffix)).exists()
        or (OUTPUT_DIR / stem).exists()
    ):
        stem = f"{original_stem}_{counter}"
        counter += 1

    video_path = INPUT_DIR / (stem + suffix)
    output_dir = OUTPUT_DIR / stem

    video_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    file.save(video_path)

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "done": False,
        "success": False,
        "message": "Video uploaded. Preparing extraction...",
        "output": str(output_dir),
    }

    thread = threading.Thread(
        target=run_extraction,
        args=(job_id, video_path, output_dir),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found."}), 404

    return jsonify(job)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
