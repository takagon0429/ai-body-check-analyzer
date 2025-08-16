# analyzer_app.py
import io
import math
from typing import Tuple, Dict, Any

import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# ========== Health / Version ==========
@app.get("/healthz")
def healthz():
    return "ok", 200

@app.get("/version")
def version():
    return jsonify({"service": "analyzer", "rev": "img-feats-v2-symmetry-centroids"}), 200


# ========== Utils ==========
def _safe_numpy(arr):
    """NaNやinfが出ても落ちないようにクランプ"""
    a = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _read_rgb(file_storage) -> Tuple[np.ndarray, bytes]:
    """Werkzeug FileStorage -> (RGB ndarray[h,w,3], raw bytes)"""
    b = file_storage.read()
    img = Image.open(io.BytesIO(b)).convert("RGB")
    arr = np.array(img)
    return arr, b


def _resize_keep(arr: np.ndarray, max_side: int = 720) -> np.ndarray:
    """計算軽量化のため最大辺を縮小"""
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 0.999:
        return arr
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)


def _gray_edge(arr_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 160)
    return gray, edges


def _norm01(x, lo, hi):
    if hi <= lo:
        return 0.5
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))


def _to10(x01):
    return float(round(10.0 * max(0.0, min(1.0, x01)), 1))


def _center_of_edges(edges: np.ndarray, mask: np.ndarray = None) -> Tuple[float, float]:
    """エッジ点の重心 (y, x)。無ければ画像中心を返す。"""
    if mask is not None:
        sel = (edges > 0) & (mask > 0)
    else:
        sel = edges > 0
    ys, xs = np.where(sel)
    if len(xs) == 0:
        h, w = edges.shape[:2]
        return h / 2.0, w / 2.0
    return float(np.mean(ys)), float(np.mean(xs))


def _upper_lower_masks(h, w, upper_rate=0.4, lower_rate=0.4):
    upper = np.zeros((h, w), np.uint8)
    lower = np.zeros((h, w), np.uint8)
    upper[: int(h * upper_rate), :] = 255
    lower[int(h * (1.0 - lower_rate)) :, :] = 255
    return upper, lower


def _left_right_masks(h, w, left_rate=0.5):
    left = np.zeros((h, w), np.uint8)
    right = np.zeros((h, w), np.uint8)
    left[:, : int(w * left_rate)] = 255
    right[:, int(w * left_rate) :] = 255
    return left, right


# ========== 画像特徴（ベース） ==========
def _basic_feats(gray: np.ndarray, edges: np.ndarray) -> Dict[str, float]:
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    edge_ratio = float(np.mean(edges > 0))
    # 横/縦のエッジ傾向（Sobelの方向性）
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gx = _safe_numpy(gx)
    gy = _safe_numpy(gy)
    horiz = float(np.mean(np.abs(gx)))
    verti = float(np.mean(np.abs(gy)))
    hv_ratio = float(horiz / (verti + 1e-6))  # >1 で横成分が強い

    return dict(mean=mean, std=std, edge=edge_ratio, hv_ratio=hv_ratio)


# ========== FRONT: 肩/骨盤の“傾き”を近似 ==========
def _front_metrics(arr_rgb: np.ndarray) -> Dict[str, Any]:
    """
    片側の肩が下がっていると、左右上部のエッジ重心のy差が出やすい。
    骨盤も下部の左右エッジ重心y差で近似。差分を角度換算（小振幅）。
    """
    arr = _resize_keep(arr_rgb, 720)
    gray, edges = _gray_edge(arr)
    h, w = gray.shape[:2]

    # 上部・下部マスク
    upper_mask, lower_mask = _upper_lower_masks(h, w, 0.42, 0.42)
    # 左右マスク
    left_mask, right_mask = _left_right_masks(h, w, 0.5)

    # 肩：上部 × 左右 重心のy差
    ul = cv2.bitwise_and(upper_mask, left_mask)
    ur = cv2.bitwise_and(upper_mask, right_mask)
    y_ul, _ = _center_of_edges(edges, ul)
    y_ur, _ = _center_of_edges(edges, ur)
    shoulder_y_diff = float(y_ul - y_ur)  # +で左が下がり気味

    # 骨盤：下部 × 左右 重心のy差
    ll = cv2.bitwise_and(lower_mask, left_mask)
    lr = cv2.bitwise_and(lower_mask, right_mask)
    y_ll, _ = _center_of_edges(edges, ll)
    y_lr, _ = _center_of_edges(edges, lr)
    pelvis_y_diff = float(y_ll - y_lr)

    # 角度化（画面幅でスケール、小さく安定させる）
    # 1pxのy差 ~ 0.15° くらいの小さめ傾斜で換算（経験値的に控えめ）
    px2deg = 0.15
    shoulder_angle = 180.0 - float(px2deg * shoulder_y_diff)
    pelvis_tilt = 180.0 - float(px2deg * pelvis_y_diff)

    # 常識的範囲にクランプ
    shoulder_angle = float(round(max(160.0, min(180.0, shoulder_angle)), 1))
    pelvis_tilt = float(round(max(160.0, min(180.0, pelvis_tilt)), 1))

    return {
        "pelvis_tilt": f"{pelvis_tilt}°",
        "shoulder_angle": f"{shoulder_angle}°",
        "debug": {
            "shoulder_y_diff": round(shoulder_y_diff, 2),
            "pelvis_y_diff": round(pelvis_y_diff, 2),
        },
    }


# ========== SIDE: 頭の前方/胸椎丸まりの“傾向”を近似 ==========
def _side_metrics(arr_rgb: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    頭が前に出ると、上部のエッジ重心xが胴体より前方に寄る。
    また、横方向エッジが相対的に強いと背中の丸まり（後弯）傾向が出やすい。
    """
    arr = _resize_keep(arr_rgb, 720)
    gray, edges = _gray_edge(arr)
    h, w = gray.shape[:2]

    # 上部＝頭/頸部領域、中央＝胸郭領域の想定
    upper_mask, lower_mask = _upper_lower_masks(h, w, 0.35, 0.35)
    mid_mask = np.zeros((h, w), np.uint8)
    mid_mask[int(h * 0.35) : int(h * 0.70), :] = 255

    yu, xu = _center_of_edges(edges, upper_mask)  # 頭/首（上部）の重心
    ym, xm = _center_of_edges(edges, mid_mask)    # 胸郭（中部）の重心

    # 前方変位の近似（右向きでも左向きでも「上部xが胴体xより前」だと大きくなるよう、差の絶対値）
    x_shift_px = float(abs(xu - xm))

    # 画素→cm換算の疑似（画像幅 600px を 25cm相当としてスケール）
    #  → 数cmオーダーで出す。最大で ~5cm程度に抑える。
    px_per_cm = max(80.0, min(200.0, w / 25.0))  # だいたい25cmが画像幅の1/2〜1倍想定
    forward_head_cm = float(round(max(0.5, min(5.0, x_shift_px / px_per_cm * 25.0 / 12.0)), 1))

    # 後弯（kyphosis）の近似：横方向エッジの強さ（horiz/verti比）で判定
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    horiz = float(np.mean(np.abs(_safe_numpy(gx))))
    verti = float(np.mean(np.abs(_safe_numpy(gy))))
    hv_ratio = float(horiz / (verti + 1e-6))

    # ラベル化（控えめ）
    if hv_ratio < 0.9:
        kyphosis = "軽度"
    elif hv_ratio < 1.2:
        kyphosis = "中等度"
    else:
        kyphosis = "やや強い"

    metrics = {
        "forward_head": f"{forward_head_cm}cm",
        "kyphosis": kyphosis,
    }
    debug = {
        "x_shift_px": round(x_shift_px, 2),
        "hv_ratio": round(hv_ratio, 2),
    }
    return metrics, debug


# ========== 総合スコア ==========
def _scores(front_feats: Dict[str, float], side_feats: Dict[str, float],
            front_m: Dict[str, Any], side_m: Dict[str, Any]) -> Dict[str, float]:
    """
    ・balance: 前面の左右上/下の重心差が小さいほど高得点
    ・fashion: 明るさ（mean）が適正域に近いほど高い（露出良好の仮指標）
    ・posture: forward_headが小さい＋横エッジ優位すぎないほど高い
    ・muscle_fat: コントラスト(std)が“ほどよい”ほど高い
    ・overall: 上の平均
    """
    # front_mのdebug値を使って左右差を反映
    sh_diff = abs(float(front_m.get("debug", {}).get("shoulder_y_diff", 0.0)))
    pv_diff = abs(float(front_m.get("debug", {}).get("pelvis_y_diff", 0.0)))
    # 差が小さいほど良い → 0差で1.0、5pxで0.6、10pxで0.2くらいに落ちるよう設計
    bal_sh = 1.0 - _norm01(sh_diff, 2.5, 10.0)
    bal_pv = 1.0 - _norm01(pv_diff, 2.5, 10.0)
    balance = _to10(0.5 * bal_sh + 0.5 * bal_pv)

    # fashion: 明るさ適正 80〜170 を高評価域とし、そこから外れると減点
    mean = float(front_feats["mean"])
    if mean <= 80:
        fashion = _to10(_norm01(mean, 30, 80))
    elif mean >= 170:
        fashion = _to10(1.0 - _norm01(mean, 170, 230))
    else:
        fashion = _to10(1.0)  # 80~170は満点寄り

    # posture: forward_head と hv_ratio を使う（小さい/低いほど良い）
    try:
        fwd_cm = float(str(side_m["forward_head"]).rstrip("cm"))
    except Exception:
        fwd_cm = 2.0
    hv = float(side_feats["hv_ratio"])
    # 2cm以下なら高得点、5cmで低得点
    p1 = 1.0 - _norm01(fwd_cm, 2.0, 5.0)
    # 横方向エッジ優位（>1.2）はやや減点、0.8~1.0くらいはOK
    p2 = 1.0 - _norm01(hv, 1.0, 1.5)
    posture = _to10(0.6 * p1 + 0.4 * p2)

    # muscle_fat: stdが“ほどよい” 20~80 を高評価域
    std = float(front_feats["std"])
    if std <= 20:
        mfat = _to10(_norm01(std, 5, 20))
    elif std >= 80:
        mfat = _to10(1.0 - _norm01(std, 80, 120))
    else:
        mfat = _to10(1.0)
    muscle_fat = mfat

    overall = float(round((balance + fashion + posture + muscle_fat) / 4.0, 1))

    return dict(
        balance=balance,
        fashion=fashion,
        muscle_fat=muscle_fat,
        overall=overall,
        posture=posture,
    )


# ========== アドバイス生成 ==========
def _advice(front_m: Dict[str, Any], side_m: Dict[str, Any], scores: Dict[str, float]) -> list:
    adv = []
    # 肩角度 175°から大きくズレたら注意
    try:
        shoulder_deg = float(str(front_m["shoulder_angle"]).rstrip("°"))
        if abs(shoulder_deg - 175.0) > 3.0:
            adv.append("肩の高さ差に注意。片側だけ荷物を持たない。")
    except Exception:
        pass
    # 前方頭位
    try:
        fwd = float(str(side_m["forward_head"]).rstrip("cm"))
        if fwd >= 2.0:
            adv.append("胸椎伸展ストレッチと顎引きエクササイズを1日2回。")
    except Exception:
        pass
    # 姿勢スコアが低めなら一般アドバイス
    if scores.get("posture", 10.0) < 6.5 and "胸椎伸展ストレッチと顎引きエクササイズを1日2回。" not in adv:
        adv.append("胸椎伸展ストレッチと顎引きエクササイズを1日2回。")
    return adv


# ========== API ==========
@app.post("/analyze")
def analyze():
    front_fs = request.files.get("front")
    side_fs = request.files.get("side")
    if not front_fs or not side_fs:
        return jsonify({"status": "error", "message": "front and side required"}), 400

    # 読み込み
    front_rgb, _ = _read_rgb(front_fs)
    side_rgb, _ = _read_rgb(side_fs)

    # 特徴抽出
    f_gray, f_edges = _gray_edge(_resize_keep(front_rgb, 720))
    s_gray, s_edges = _gray_edge(_resize_keep(side_rgb, 720))

    front_feats = _basic_feats(f_gray, f_edges)
    side_feats = _basic_feats(s_gray, s_edges)

    # メトリクス
    front_metrics = _front_metrics(front_rgb)
    side_metrics, _side_dbg = _side_metrics(side_rgb)

    # スコア
    scores = _scores(front_feats, side_feats, front_metrics, side_metrics)

    # アドバイス
    advice = _advice(front_metrics, side_metrics, scores)

    return jsonify({
        "status": "ok",
        "message": "files received",
        "scores": scores,
        "front_metrics": {k: v for k, v in front_metrics.items() if k != "debug"},
        "side_metrics": side_metrics,
        "advice": advice,
        "front_filename": request.files["front"].filename,
        "side_filename": request.files["side"].filename,
        # デバッグを見たいときはコメント解除
        # "debug": {"front_feats": front_feats, "side_feats": side_feats, "front_debug": front_metrics.get("debug"), "side_debug": _side_dbg},
    })
