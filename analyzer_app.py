# analyzer_app.py
import io
import math
import numpy as np
import cv2
from PIL import Image, ImageOps
from flask import Flask, request, jsonify

# ========= 可用なら MediaPipe を使う（無ければ自動フォールバック）=========
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False
    mp = None  # type: ignore

app = Flask(__name__)

# ==========================
# Health & Version
# ==========================
@app.get("/healthz")
def healthz():
    return "ok", 200

@app.get("/version")
def version():
    return jsonify({"service": "analyzer", "rev": "img-feats+mp-v2"}), 200


# ==========================
# 画像ロード（EXIF回転補正）
# ==========================
def _load_img(file_storage):
    """werkzeug FileStorage -> (np.ndarray RGB, raw bytes)"""
    b = file_storage.read()
    img = Image.open(io.BytesIO(b))
    # EXIFの向きを反映
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img)
    return arr, b


# ==========================
# 前処理：露出標準化（Gray-World + CLAHE）
# ==========================
def _normalize_rgb(arr: np.ndarray) -> np.ndarray:
    img = arr.astype(np.float32) + 1e-6
    mean = img.reshape(-1, 3).mean(axis=0)
    img = img * (128.0 / mean)  # Gray-World
    img = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out


# ==========================
# 人物セグメンテーション & クロップ
# ==========================
def _person_crop(arr: np.ndarray) -> np.ndarray:
    if not MP_AVAILABLE:
        return arr
    try:
        with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
            res = seg.process(arr)
        mask = (res.segmentation_mask > 0.4).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) < 1000:
            return arr
        h, w = arr.shape[:2]
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        pad = int(0.08 * max(h, w))
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        crop = arr[y1:y2, x1:x2]
        if crop.size == 0:
            return arr
        # 統一スケール（Pose安定化）
        target_h = 960
        r = target_h / crop.shape[0]
        crop = cv2.resize(crop, (int(crop.shape[1] * r), target_h))
        return crop
    except Exception:
        return arr


# ==========================
# 顔向き（Yaw簡易指標）
# 0=正面、1=横顔寄り（側面向き向け）
# ==========================
def _estimate_yaw_conf(arr: np.ndarray) -> float:
    if not MP_AVAILABLE:
        return 0.5
    try:
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=False) as fm:
            res = fm.process(arr)
        if not res.multi_face_landmarks:
            return 0.5
        lm = res.multi_face_landmarks[0].landmark
        # 目尻の代表点（右目外端=33, 左目外端=263）
        R_OUT, L_OUT = 33, 263
        h, w = arr.shape[:2]
        rx, ry = lm[R_OUT].x * w, lm[R_OUT].y * h
        lx, ly = lm[L_OUT].x * w, lm[L_OUT].y * h
        eye_w = abs(lx - rx) + 1e-6
        # eye_w が小さいほど横向きとみなす（超簡易）
        # 0（正面）〜1（横顔）
        yaw = float(np.clip(60.0 / eye_w, 0.0, 1.0))
        return yaw
    except Exception:
        return 0.5


# ==========================
# 幾何ユーティリティ
# ==========================
def _dist(p1, p2):
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

def _px(lm, w, h):
    return (lm[0] * w, lm[1] * h)

def _angle_deg(p1, p2):
    """p1->p2 の線の水平角度（度）。水平=180/0付近。"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = math.degrees(math.atan2(dy, dx))
    # 0〜180 に寄せる（左右どちらでも同じ角度に寄せる）
    ang = (ang + 360.0) % 360.0
    if ang > 180:
        ang = 360.0 - ang
    return ang


# ==========================
# 画像特徴（明るさ・コントラスト・エッジ量）
# ==========================
def _img_feats(arr: np.ndarray):
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.mean(edges > 0))
    return dict(mean=mean, std=std, edge=edge_ratio)


# ==========================
# Pose 解析（正面）: 肩/骨盤角度 + 肩座標
# ==========================
def _safe_get_landmark(lms, idx):
    pt = lms[idx]
    # (x, y, visibility)
    return (pt.x, pt.y, getattr(pt, "visibility", 1.0))

def _analyze_front_with_pose(arr: np.ndarray):
    if not MP_AVAILABLE:
        return None
    h, w = arr.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        res = pose.process(arr)
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks.landmark
        try:
            L_SH, R_SH = _safe_get_landmark(lms, 11), _safe_get_landmark(lms, 12)
            L_HP, R_HP = _safe_get_landmark(lms, 23), _safe_get_landmark(lms, 24)
        except Exception:
            return None
        # 可視性が低すぎると却下
        if min(L_SH[2], R_SH[2], L_HP[2], R_HP[2]) < 0.3:
            return None
        L_SH_px, R_SH_px = _px(L_SH, w, h), _px(R_SH, w, h)
        L_HP_px, R_HP_px = _px(L_HP, w, h), _px(R_HP, w, h)
        shoulder_angle = _angle_deg(L_SH_px, R_SH_px)
        pelvis_tilt = _angle_deg(L_HP_px, R_HP_px)
        return {
            "shoulder_angle": round(shoulder_angle, 1),
            "pelvis_tilt": round(pelvis_tilt, 1),
            # スケール推定用
            "L_SH": (L_SH[0], L_SH[1]),
            "R_SH": (R_SH[0], R_SH[1]),
        }


# ==========================
# Pose 解析（側面）: 耳と肩の水平距離 → 前方頭位px
# ==========================
def _analyze_side_with_pose(arr: np.ndarray):
    if not MP_AVAILABLE:
        return None
    h, w = arr.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        res = pose.process(arr)
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks.landmark
        # 右耳:8 / 左耳:7, 右肩:12 / 左肩:11
        cand = []
        for ear_idx, sh_idx in [(8, 12), (7, 11)]:
            e = _safe_get_landmark(lms, ear_idx)
            s = _safe_get_landmark(lms, sh_idx)
            if min(e[2], s[2]) < 0.3:
                continue
            e_px, s_px = _px(e, w, h), _px(s, w, h)
            horiz = abs(e_px[0] - s_px[0])   # 水平方向の差
            vert = abs(e_px[1] - s_px[1])    # 縦方向の差（参考）
            cand.append((horiz, vert))
        if not cand:
            return None
        # 一番大きい水平差を採用（横向きなら片側がはっきり出る）
        cand.sort(key=lambda t: t[0], reverse=True)
        forward_head_px, vert_px = cand[0]
        # 胸の厚みやスケールの proxy として肩-腰距離（トルソ高さpx）も取る
        try:
            sh = _safe_get_landmark(lms, 12)
            hip = _safe_get_landmark(lms, 24)
            torso_px = _dist(_px(sh, w, h), _px(hip, w, h))
        except Exception:
            torso_px = None
        return {
            "forward_head_px": float(forward_head_px),
            "torso_px": float(torso_px) if torso_px else None,
        }


# ==========================
# スケール推定（cm/px）: 正面の肩幅
# ==========================
def _estimate_shoulder_cm_per_px(front_pose, img_w, img_h) -> float | None:
    try:
        L_SH = (front_pose["L_SH"][0] * img_w, front_pose["L_SH"][1] * img_h)
        R_SH = (front_pose["R_SH"][0] * img_w, front_pose["R_SH"][1] * img_h)
        px = _dist(L_SH, R_SH)
        if px < 10:
            return None
        assumed_cm = 40.0  # 平均肩幅（男女混合のざっくり）
        return assumed_cm / px
    except Exception:
        return None


# ==========================
# 品質重み（顔向き）
# ==========================
def _quality_weight(yaw_front, yaw_side):
    # 正面：小さいほど良い、側面：大きいほど良い
    wf = float(np.clip(1.0 - yaw_front, 0.4, 1.0))
    ws = float(np.clip(yaw_side,       0.4, 1.0))
    return round((wf + ws) / 2.0, 2)


# ==========================
# 既存の簡易特徴→スコア
# ==========================
def _score_from_feats(f_front, f_side, quality=1.0):
    def norm(x, lo, hi):
        return max(0.0, min(10.0, 10.0 * (x - lo) / (hi - lo))) if hi > lo else 5.0

    balance = 10.0 - abs(norm(f_front["std"], 20, 80) - norm(f_side["std"], 20, 80))
    fashion = norm(f_front["mean"], 60, 200)
    posture = 10.0 - norm(f_side["edge"], 0.01, 0.20)
    muscle_fat = norm(f_front["std"], 15, 90)
    overall = ((balance + fashion + posture) / 3.0) * float(np.clip(quality, 0.5, 1.0))
    return dict(
        balance=round(balance, 1),
        fashion=round(fashion, 1),
        muscle_fat=round(muscle_fat, 1),
        posture=round(posture, 1),
        overall=round(float(np.clip(overall, 0.0, 10.0)), 1),
    )


# ==========================
# 角度/cm 表示（フォールバックあり）
# ==========================
def _front_side_metrics_from_pose(front_pose, side_pose, cm_per_px, f_front_feats, f_side_feats):
    # --- 正面 ---
    if front_pose:
        pelvis_tilt = f'{front_pose["pelvis_tilt"]}°'
        shoulder_angle = f'{front_pose["shoulder_angle"]}°'
    else:
        # 簡易フォールバック（画像特徴で擬似角度）
        pelvis_tilt = f'{round(170 + (f_front_feats["std"] % 10), 1)}°'
        shoulder_angle = f'{round(170 + (f_front_feats["mean"] % 10), 1)}°'

    # --- 側面 ---
    if side_pose and cm_per_px:
        forward_head_cm = float(side_pose["forward_head_px"]) * float(cm_per_px)
        forward_head_cm = round(forward_head_cm, 1)
    elif side_pose and side_pose.get("torso_px"):
        # torso 比から 40cm 仮定
        forward_head_cm = 40.0 * float(side_pose["forward_head_px"]) / float(side_pose["torso_px"])
        forward_head_cm = round(forward_head_cm, 1)
    else:
        # フォールバック（エッジ比）
        forward_head_cm = round(1.0 + (f_side_feats["edge"] * 10), 1)

    kyphosis = "軽度" if forward_head_cm < 2.0 else ("中等度" if forward_head_cm < 3.0 else "やや強い")

    return (
        {"pelvis_tilt": pelvis_tilt, "shoulder_angle": shoulder_angle},
        {"forward_head": f"{forward_head_cm}cm", "kyphosis": kyphosis},
    )


# ==========================
# /analyze
# ==========================
@app.post("/analyze")
def analyze():
    front_fs = request.files.get("front")
    side_fs = request.files.get("side")
    if not front_fs or not side_fs:
        return jsonify({"status": "error", "message": "front and side required"}), 400

    # --- Load & preprocess ---
    front_arr, _ = _load_img(front_fs)
    side_arr, _ = _load_img(side_fs)

    front_arr = _normalize_rgb(front_arr)
    side_arr = _normalize_rgb(side_arr)

    front_arr = _person_crop(front_arr)
    side_arr = _person_crop(side_arr)

    # 基本特徴（常に計測）
    f_front = _img_feats(front_arr)
    f_side = _img_feats(side_arr)

    # 顔向き（品質重み）
    yaw_front = _estimate_yaw_conf(front_arr)
    yaw_side = _estimate_yaw_conf(side_arr)
    quality = _quality_weight(yaw_front, yaw_side)

    # Pose（あれば使う）
    front_pose = _analyze_front_with_pose(front_arr) if MP_AVAILABLE else None
    side_pose = _analyze_side_with_pose(side_arr) if MP_AVAILABLE else None

    # スケール cm/px
    cm_per_px = None
    if front_pose:
        fh, fw = front_arr.shape[0], front_arr.shape[1]
        cm_per_px = _estimate_shoulder_cm_per_px(front_pose, fw, fh)

    # メトリクス（角度・cm）
    front_metrics, side_metrics = _front_side_metrics_from_pose(
        front_pose, side_pose, cm_per_px, f_front, f_side
    )

    # スコア
    scores = _score_from_feats(f_front, f_side, quality=quality)

    # アドバイス
    advice = []
    try:
        shoulder_deg = float(str(front_metrics["shoulder_angle"]).rstrip("°"))
        if abs(shoulder_deg - 175.0) > 3.0:
            advice.append("肩の高さ差に注意。片側だけ荷物を持たない。")
    except Exception:
        pass
    try:
        fwd_cm = float(str(side_metrics["forward_head"]).rstrip("cm"))
        if fwd_cm >= 2.0:
            advice.append("胸椎伸展ストレッチと顎引きエクササイズを1日2回。")
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "message": "files received",
        "scores": scores,
        "front_metrics": front_metrics,
        "side_metrics": side_metrics,
        "advice": advice,
        "confidence": {
            "front_yaw": yaw_front,
            "side_yaw": yaw_side,
            "quality": quality,
            "used_pose_front": bool(front_pose is not None),
            "used_pose_side": bool(side_pose is not None),
            "cm_per_px": cm_per_px,
        },
        "front_filename": front_fs.filename,
        "side_filename": side_fs.filename,
    })
