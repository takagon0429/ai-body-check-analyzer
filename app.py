import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get("/")
def root():
    return "ok", 200

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.post("/analyze")
def analyze():
    if 'front' not in request.files or 'side' not in request.files:
        return jsonify({"error": "front and side images are required"}), 400

    front_file = request.files['front']
    side_file = request.files['side']

    result = {
        "status": "ok",
        "message": "files received",
        "front_filename": front_file.filename,
        "side_filename": side_file.filename,
        "scores": {"overall": 7.3, "posture": 6.0, "balance": 7.0, "muscle_fat": 8.2, "fashion": 8.0},
        "front_metrics": {"shoulder_angle": "178.3°", "pelvis_tilt": "179.9°"},
        "side_metrics": {"forward_head": "2.9cm", "kyphosis": "軽度"},
        "advice": ["肩の高さ差に注意。片側だけ荷物を持たない。", "胸椎伸展ストレッチと顎引きエクササイズを1日2回。"]
    }
    return jsonify(result), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
