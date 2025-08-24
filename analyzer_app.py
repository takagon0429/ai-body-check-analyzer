# analyzer_app.py
import io
import json
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# -------- Health / Version --------
@app.get("/healthz")
def healthz():
    return "ok", 200

@app.get("/version")
def version():
    return jsonify({"service": "analyzer", "rev": "img-feats-v1-stable"}), 200


# -------- Helpers --------
def _safe_open_rgb(file_storage):
    """
    FileStorage -> (np.ndarray RGB, raw bytes)
    どんな画像でも極力読み込んでRGB np.uint8にする。失敗時は (None, None)。
    """
    try:
        b = file_storage.read()
        img = Image.open(io.BytesIO(b)).convert("RGB")
        arr = np.array(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None, None
        return arr, b
    except Exception:
        return None, None


def _metrics_from_image(arr: np.ndarray):
    """
    画像から簡易特徴量を安定取得（明るさ・コントラスト・エッジ率）
    """
    try:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    except Exception:
        # すでにgrayの可能性など
        gray = arr if arr.ndim == 2 else cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)

    mean = float(np.mean(gray))
    std = float(np.std(gray))

    try:
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.mean(edges > 0))
    except Exception:
        edge_ratio = 0.0

    return dict(mean=mean, std=std, edge=edge_ratio)


def _norm10(x, lo, hi):
    if hi <= lo:
        return 5.0
    t = 10.0 * (float(x) - lo) / (hi - lo)
    return max(0.0, min(10.0, t))


def _scores_from_feats(f_front, f_side):
    """
    0-10に正規化。差が出やすいようにレンジは少し広めに。
    """
    # 多少のバリエーションが出るようレンジを調整
    balance   = 10.0 - abs(_norm10(f_front["std"], 10, 90) - _norm10(f_side["std"], 10, 90))
    fashion   = _norm10(f_front["mean"], 50, 220)
    posture   = 10.0 - _norm10(f_side["edge"], 0.005, 0.25)   # エッジ多い→猫背寄り仮説
    musclefat = _norm10(f_front["std"], 10, 100)
    overall   = round((balance + fashion + posture) / 3.0, 1)

    return {
        "balance": round(balance, 1),
        "fashion": round(fashion, 1),
        "muscle_fat": round(musclefat, 1),
        "overall": overall,
        "posture": round(posture, 1),
    }


def _angles_from_feats(f_front, f_side):
    """
    角度/距離のダミー値を画像依存で安定生成（表記一貫）
    """
    pelvis_tilt     = round(170.0 + (f_front["std"]  % 10.0), 1)  # 170〜180°
    shoulder_angle  = round(170.0 + (f_front["mean"] % 10.0), 1)  # 170〜180°
    forward_head_cm = round(1.0 + (f_side["edge"] * 12.0), 1)     # 1.0〜3.0台が多い想定
    if forward_head_cm < 2.0:
        kyphosis = "軽度"
    elif forward_head_cm < 3.0:
        kyphosis = "中等度"
    else:
        kyphosis = "やや強い"

    return (
        {"pelvis_tilt": f"{pelvis_tilt}°", "shoulder_angle": f"{shoulder_angle}°"},
        {"forward_head": f"{forward_head_cm}cm", "kyphosis": kyphosis},
    )


def _advice_from_scores(scores, front_metrics):
    advice = []
    try:
        if scores.get("posture", 10) < 6.5:
            advice.append("胸椎伸展ストレッチと顎引きエクササイズを1日2回。")
        shoulder_deg = float(front_metrics["shoulder_angle"].rstrip("°"))
        if abs(shoulder_deg - 175.0) > 3.0:
            advice.append("肩の高さ差に注意。片側だけ荷物を持たない。")
    except Exception:
        pass
    return advice


# -------- API --------
@app.post("/analyze")
def analyze():
    # 返却雛形（例外時でも必ず返す）
    ret = {
        "status": "error",
        "message": "",
        "scores": {},
        "front_metrics": {},
        "side_metrics": {},
        "advice": [],
        "front_filename": None,
        "side_filename": None,
    }

    try:
        front_fs = request.files.get("front")
        side_fs  = request.files.get("side")
        if not front_fs or not side_fs:
            ret["message"] = "front and side required"
            return jsonify(ret), 400

        ret["front_filename"] = front_fs.filename
        ret["side_filename"]  = side_fs.filename

        front_arr, _ = _safe_open_rgb(front_fs)
        side_arr,  _ = _safe_open_rgb(side_fs)
        if front_arr is None or side_arr is None:
            ret["message"] = "failed to read image(s)"
            return jsonify(ret), 400

        f_front = _metrics_from_image(front_arr)
        f_side  = _metrics_from_image(side_arr)

        scores = _scores_from_feats(f_front, f_side)
        front_metrics, side_metrics = _angles_from_feats(f_front, f_side)
        advice = _advice_from_scores(scores, front_metrics)

        ret.update({
            "status": "ok",
            "message": "files received",
            "scores": scores,
            "front_metrics": front_metrics,
            "side_metrics": side_metrics,
            "advice": advice,
        })
        return jsonify(ret), 200

    except Exception as e:
        # ここを通っても JSON で返す（UnboundLocalError などを完全回避）
        ret["message"] = f"exception: {type(e).__name__}: {e}"
        return jsonify(ret), 200
