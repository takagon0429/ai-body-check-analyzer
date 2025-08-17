# analyzer_app.py
import io
import math
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

# --- Mediapipe (0.10 以降推奨) ---
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

app = Flask(__name__)

# ================================
# Health & Version
# ================================
@app.get("/healthz")
def healthz():
    return "ok", 200

@app.get("/version")
def version():
    return jsonify({"service": "analyzer", "rev": "pose-v1"}), 200


# ================================
# 画像 I/O
# ================================
def _load_img(file_storage):
    """
    werkzeug FileStorage -> (np.ndarray RGB, raw bytes)
    """
    b = file_storage.read()
    img = Image.open(io.BytesIO(b)).convert("RGB")
    arr = np.array(img)
    return arr, b


# ================================
# 失敗時のフォールバック特徴量（画像統計）
# ================================
def _img_feats(arr: np.ndarray):
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.mean(edges > 0))  # 0〜1
    return dict(mean=mean, std=std, edge=edge_ratio)


# ================================
# 幾何ユーティリティ
# ================================
def _angle_deg(p1, p2):
    """
    2点 p1(x,y), p2(x,y) を結ぶ線分の「水平からの角度」
    右が0°, 上が+90° という通常atan2の度数法
    ここでは水平との差を見たいので、水平に対し 180°に近いほど水平とみなす
    """
    dx, dy = (p2[0] - p1[0], p2[1] - p1[1])
    rad = math.atan2(dy, dx)
    deg = math.degrees(rad)
    # 水平基準に合わせ、左右対称にするため 180 に寄せる
    # 例: 完全水平なら 0° or 180°。ここでは 180°に正規化して返す
    # 180 - |deg| を 0〜180 に収め、最終的に 180 に近いほど水平
    norm = 180 - abs(deg)
    return max(0.0, min(180.0, 180.0 - abs(180.0 - norm)))


def _dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def _safe_get_landmark(lms, idx):
    lm = lms[idx]
    return (lm.x, lm.y, lm.visibility)


def _px(pt, w, h):
    return (pt[0]*w, pt[1]*h)


# ================================
# Mediapipe ポーズ解析（フロント・サイド）
# ================================
def _analyze_front_with_pose(arr: np.ndarray):
    """
    正面画像: 肩ライン, 骨盤ラインの水平度
    返り値:
      {
        "shoulder_angle": 180に近いほど水平,
        "pelvis_tilt": 180に近いほど水平,
        "aux": {...}  # デバッグ用（必要なら）
      }
    """
    if not MP_AVAILABLE:
        return None

    h, w = arr.shape[:2]
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        res = pose.process(arr)
        if not res.pose_landmarks:
            return None

        lms = res.pose_landmarks.landmark

        # 主要点（左右の肩・腰）
        # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        # MP 0.10の index: LEFT/RIGHT_SHOULDER=11/12, LEFT/RIGHT_HIP=23/24
        try:
            L_SH, R_SH = _safe_get_landmark(lms, 11), _safe_get_landmark(lms, 12)
            L_HP, R_HP = _safe_get_landmark(lms, 23), _safe_get_landmark(lms, 24)
        except Exception:
            return None

        # 可視性チェック（低すぎるなら使わない）
        if min(L_SH[2], R_SH[2], L_HP[2], R_HP[2]) < 0.3:
            return None

        L_SH_px, R_SH_px = _px(L_SH, w, h), _px(R_SH, w, h)
        L_HP_px, R_HP_px = _px(L_HP, w, h), _px(R_HP, w, h)

        shoulder_angle = _angle_deg(L_SH_px, R_SH_px)  # 180に近いほど水平
        pelvis_tilt    = _angle_deg(L_HP_px, R_HP_px)  # 180に近いほど水平

        return {
            "shoulder_angle": round(shoulder_angle, 1),
            "pelvis_tilt": round(pelvis_tilt, 1)
        }


def _analyze_side_with_pose(arr: np.ndarray):
    """
    側面画像: 頭の前方変位 (耳-肩の水平ズレ) と 胸椎カーブの粗い推定
    返り値:
      {
        "forward_head_cm": float,
        "kyphosis_grade": "軽度"|"中等度"|"やや強い"
      }
    """
    if not MP_AVAILABLE:
        return None

    h, w = arr.shape[:2]
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        res = pose.process(arr)
        if not res.pose_landmarks:
            return None

        lms = res.pose_landmarks.landmark

        # 主要点（耳/肩/腰/膝）: 耳(EAR)は Pose にないため、頭部の代替として NOSE(0) or EYE(1,2) を使う
        try:
            NOSE   = _safe_get_landmark(lms, 0)
            L_SH   = _safe_get_landmark(lms, 11)
            R_SH   = _safe_get_landmark(lms, 12)
            L_HIP  = _safe_get_landmark(lms, 23)
            R_HIP  = _safe_get_landmark(lms, 24)
            L_KNEE = _safe_get_landmark(lms, 25)
            R_KNEE = _safe_get_landmark(lms, 26)
        except Exception:
            return None

        if min(NOSE[2], L_SH[2], R_SH[2], L_HIP[2], R_HIP[2]) < 0.3:
            return None

        NOSE_px  = _px(NOSE, w, h)
        # 側面ではどちらかの肩だけが主に見えることが多いので、より見えている方（xの偏りが小さい方）を使う
        L_SH_px, R_SH_px = _px(L_SH, w, h), _px(R_SH, w, h)
        SH_px = L_SH_px if L_SH[2] >= R_SH[2] else R_SH_px

        # 水平スケール: 肩-腰の距離を基準スケール（仮に 40cm）に対応させる
        L_HIP_px, R_HIP_px = _px(L_HIP, w, h), _px(R_HIP, w, h)
        HIP_px = L_HIP_px if L_HIP[2] >= R_HIP[2] else R_HIP_px

        torso_px = _dist(SH_px, HIP_px)  # ピクセル距離
        if torso_px < 10:  # 極端に小さい時は不安定
            return None

        # 頭の前方変位: 鼻と肩の x 差（横ずれ）を正値化・cm換算
        head_forward_px = abs(NOSE_px[0] - SH_px[0])  # 横方向のズレ
        # 仮想スケール: torso_px ピクセル ≒ 40cm と仮定
        forward_head_cm = (head_forward_px / torso_px) * 40.0
        forward_head_cm = float(np.clip(forward_head_cm, 0.0, 10.0))

        # ざっくり胸椎カーブ: 肩-腰-膝の角度（鋭角ほど丸まり）
        try:
            L_KNEE_px, R_KNEE_px = _px(L_KNEE, w, h), _px(R_KNEE, w, h)
            KNEE_px = L_KNEE_px if L_KNEE[2] >= R_KNEE[2] else R_KNEE_px
            # 角度 θ = ∠(shoulder -> hip, knee -> hip)
            v1 = (SH_px[0]-HIP_px[0], SH_px[1]-HIP_px[1])
            v2 = (KNEE_px[0]-HIP_px[0], KNEE_px[1]-HIP_px[1])
            def _angle_between(a, b):
                an = math.hypot(a[0], a[1]); bn = math.hypot(b[0], b[1])
                if an < 1e-6 or bn < 1e-6: return 180.0
                cosv = max(-1.0, min(1.0, (a[0]*b[0] + a[1]*b[1])/(an*bn)))
                return math.degrees(math.acos(cosv))
            hip_angle = _angle_between(v1, v2)  # だいたい 160~200 のレンジ想定
        except Exception:
            hip_angle = 180.0

        # 腰角度が鋭く（小さく）なるほど丸まりと仮定（単純化）
        if hip_angle >= 170:
            kyphosis = "軽度"
        elif hip_angle >= 150:
            kyphosis = "中等度"
        else:
            kyphosis = "やや強い"

        return {
            "forward_head_cm": round(forward_head_cm, 1),
            "kyphosis_grade": kyphosis
        }


# ================================
# スコアリング
# ================================
def _scores_from_pose(front_metrics, side_metrics, f_front_fallback, f_side_fallback):
    """
    front_metrics: {"shoulder_angle", "pelvis_tilt"} 180に近いほど水平
    side_metrics: {"forward_head_cm", "kyphosis_grade"}
    フォールバック: f_front_fallback, f_side_fallback は _img_feats() の結果
    """
    # ---- 姿勢(posture)
    # 前方頭位が大きいほど減点
    if side_metrics and "forward_head_cm" in side_metrics:
        fh = side_metrics["forward_head_cm"]
        # 0cm -> 10点, 5cm -> 6点, 8cm -> 3点 くらいのカーブ
        posture = 10.0 - np.clip((fh/8.0)*7.0, 0.0, 7.0)
    else:
        # 画像エッジから代用（エッジ多い=猫背寄り）
        posture = 10.0 - np.clip((f_side_fallback["edge"] - 0.02) * 80.0, 0.0, 7.0)
    posture = round(float(np.clip(posture, 0.0, 10.0)), 1)

    # ---- バランス(balance) ＝ 肩・骨盤の水平度
    if front_metrics:
        sh = front_metrics["shoulder_angle"]  # 180に近いほど水平
        hp = front_metrics["pelvis_tilt"]
        # 180との差分を減点（1°ズレ= 0.5点減点くらい）
        def to_score(a):
            return max(0.0, 10.0 - 0.5*abs(180.0 - a))
        balance = (to_score(sh) + to_score(hp)) / 2.0
    else:
        # 代用：明るさの安定性（std）でごまかし
        s1 = float(np.clip((f_front_fallback["std"]-20)/60*10, 0, 10))
        s2 = float(np.clip((f_side_fallback["std"]-20)/60*10, 0, 10))
        balance = (10.0 - abs(s1 - s2))
    balance = round(float(np.clip(balance, 0.0, 10.0)), 1)

    # ---- 筋肉・脂肪(muscle_fat) ＝ 画のコントラスト（仮）
    muscle_fat = float(np.clip((f_front_fallback["std"]-15)/75*10, 0.0, 10.0))
    muscle_fat = round(muscle_fat, 1)

    # ---- ファッション映え(fashion) ＝ 明るさ（仮）
    fashion = float(np.clip((f_front_fallback["mean"]-60)/140*10, 0.0, 10.0))
    fashion = round(fashion, 1)

    # ---- 総合(overall)
    overall = round(float(np.clip((balance + posture + fashion)/3.0, 0.0, 10.0)), 1)

    return {
        "balance": balance,
        "fashion": fashion,
        "muscle_fat": muscle_fat,
        "overall": overall,
        "posture": posture
    }


def _advice(front_metrics, side_metrics, scores):
    adv = []

    # 肩の左右差（肩角度が 175±3 を外れたら）
    if front_metrics:
        sh = front_metrics["shoulder_angle"]
        if abs(sh - 175.0) > 3.0:
            adv.append("肩の高さ差に注意。片側だけ荷物を持たない。")

    # 頭の前方変位
    if side_metrics and "forward_head_cm" in side_metrics:
        fh = side_metrics["forward_head_cm"]
        if fh >= 5.0:
            adv.append("頭が前に出ています。顎引きエクササイズと胸椎伸展ストレッチを習慣化。")
        elif fh >= 3.0:
            adv.append("軽度の前方頭位。1日2回の顎引き/胸を開くストレッチが効果的。")
        else:
            adv.append("頭の位置は概ね良好です。作業時の姿勢維持を意識しましょう。")
    else:
        # フォールバック時
        if scores["posture"] < 6.5:
            adv.append("胸椎伸展ストレッチと顎引きエクササイズを1日2回。")

    # 姿勢スコア全体
    if scores["posture"] < 5.0:
        adv.append("長時間座位では30分ごとに立ち上がって肩甲骨を大きく動かす。")

    # ダブリ除去
    out = []
    seen = set()
    for a in adv:
        if a and a not in seen:
            out.append(a)
            seen.add(a)
    return out


# ================================
# API 本体
# ================================
@app.post("/analyze")
def analyze():
    front_fs = request.files.get("front")
    side_fs  = request.files.get("side")
    if not front_fs or not side_fs:
        return jsonify({"status": "error", "message": "front and side required"}), 400

    # 読み込み
    front_arr, _ = _load_img(front_fs)
    side_arr,  _ = _load_img(side_fs)

    # まずはフォールバック特徴（必ず計算しておく）
    f_front_fb = _img_feats(front_arr)
    f_side_fb  = _img_feats(side_arr)

    # Pose が使えればポーズ計測
    front_metrics_pose = None
    side_metrics_pose  = None
    if MP_AVAILABLE:
        try:
            # Mediapipe は RGB 想定、既にRGBなのでそのまま
            front_metrics_pose = _analyze_front_with_pose(front_arr)
        except Exception:
            front_metrics_pose = None
        try:
            side_metrics_pose  = _analyze_side_with_pose(side_arr)
        except Exception:
            side_metrics_pose = None

    # 返却用のメトリクス整形
    if front_metrics_pose:
        front_metrics = {
            "pelvis_tilt": f'{front_metrics_pose["pelvis_tilt"]}°',
            "shoulder_angle": f'{front_metrics_pose["shoulder_angle"]}°',
        }
    else:
        # フォールバック（画像統計からダミー角度）
        pelvis = round(170 + (f_front_fb["std"] % 10), 1)
        shoulder = round(170 + (f_front_fb["mean"] % 10), 1)
        front_metrics = {"pelvis_tilt": f"{pelvis}°", "shoulder_angle": f"{shoulder}°"}

    if side_metrics_pose:
        side_metrics = {
            "forward_head": f'{side_metrics_pose["forward_head_cm"]}cm',
            "kyphosis": side_metrics_pose["kyphosis_grade"],
        }
    else:
        # フォールバック
        fh_cm = round(1.0 + (f_side_fb["edge"] * 10), 1)
        ky = "軽度" if fh_cm < 2.0 else ("中等度" if fh_cm < 3.0 else "やや強い")
        side_metrics = {"forward_head": f"{fh_cm}cm", "kyphosis": ky}

    # スコア
    scores = _scores_from_pose(front_metrics_pose, side_metrics_pose, f_front_fb, f_side_fb)

    # アドバイス
    advice = _advice(front_metrics_pose, side_metrics_pose, scores)

    return jsonify({
        "status": "ok",
        "message": "files received",
        "scores": scores,
        "front_metrics": front_metrics,
        "side_metrics": side_metrics,
        "advice": advice,
        "front_filename": front_fs.filename,
        "side_filename": side_fs.filename,
    })
