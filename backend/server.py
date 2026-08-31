import asyncio
import os
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# PyInstaller places bundled data in _MEIPASS.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    FRONTEND_DIR = Path(sys._MEIPASS) / "frontend"
else:
    FRONTEND_DIR = PROJECT_DIR / "frontend"

EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

from backend import (
    run_verification_system,
    run_batch_verification,
    extract_references_from_text,
    extract_references_from_pdf,
    export_results_to_excel,
)

app = Flask(__name__, static_folder=None)
ALLOWED_EXTENSIONS = {"pdf"}

def run_async(coro):
    return asyncio.run(coro)

def export_and_url(results, filename):
    path = EXPORT_DIR / filename
    export_results_to_excel(results, filename=str(path))
    return "/api/download/" + path.name

@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.get("/<path:path>")
def frontend_files(path):
    candidate = FRONTEND_DIR / path
    if candidate.exists() and candidate.is_file():
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.post("/api/verify/single")
def verify_single_api():
    data = request.get_json(silent=True) or {}
    reference = (data.get("reference") or "").strip()
    if not reference:
        return jsonify(message="Please enter a reference."), 400

    result = run_async(run_verification_system(reference))
    results = [result]
    return jsonify(results=results,
                   download_url=export_and_url(results, "single_reference_result.xlsx"))

@app.post("/api/verify/list")
def verify_list_api():
    data = request.get_json(silent=True) or {}
    text = (data.get("reference_text") or "").strip()
    if not text:
        return jsonify(message="Please enter a reference list."), 400

    references = extract_references_from_text(text)
    if not references:
        return jsonify(message="No references could be detected."), 400

    results = run_async(run_batch_verification(references))
    return jsonify(results=results,
                   download_url=export_and_url(results, "reference_verification_results.xlsx"))

@app.post("/api/verify/pdf")
def verify_pdf_api():
    if "file" not in request.files:
        return jsonify(message="Please select a PDF."), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify(message="Please select a PDF."), 400

    if "." not in file.filename or file.filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify(message="Please upload a PDF file."), 400

    temp_path = EXPORT_DIR / secure_filename(file.filename)
    file.save(temp_path)
    try:
        references = extract_references_from_pdf(str(temp_path))
        if not references:
            return jsonify(message="No references could be extracted from this PDF."), 400
        results = run_async(run_batch_verification(references))
        return jsonify(results=results,
                       download_url=export_and_url(results, "pdf_reference_verification_results.xlsx"))
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

@app.get("/api/download/<filename>")
def download(filename):
    return send_from_directory(EXPORT_DIR, filename, as_attachment=True)

def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("Reference Authenticator running at http://127.0.0.1:8000", flush=True)
    if os.environ.get("REFERENCE_AUTH_NO_BROWSER") != "1":
        threading.Timer(1.2, open_browser).start()
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
