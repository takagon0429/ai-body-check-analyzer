import os
import io
import json
import tempfile
import logging
import threading
from typing import Dict, Optional
from datetime import datetime

import requests
from flask import Flask, request

# === LINE SDK v3 ===
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
)

# ----------------------------------
# 環境変数
# ----------------------------------
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# Analyzer 側（Render 側の /analyze を指す）
ANALYZER_URL = os.getenv("ANALYZER_URL", "https://ai-body-check-analyzer.onrender.com/analyze")

PORT = int(os.getenv("PORT", "10000"))

if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    raise RuntimeError("LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN を設定してください。")

# ----------------------------------
# Flask & Logger
# ----------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("line-bot")

# ----------------------------------
# LINE API クライアント
# ----------------------------------
config = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration=config)
messaging_api = MessagingApi(api_client)
blob_api = MessagingApiBlob(api_client)
handler = WebhookHandler(CHANNEL_SECRET)

# ----------------------------------
# ユーザーごとの一時収納（front/side）
# 実運用は Redis/DB 推奨
# ----------------------------------
# user_temp[user_id] = {"front": "/tmp/xxx.jpg", "side": "/tmp/yyy.jpg"}
user_temp: Dict[str, Dict[str, str]] = {}

def _tmp_path(prefix: str, suffix: str = ".jpg") -> str:
    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=suffix)
    os.close(fd)
    return path

def _save_bytes_to_file(data: bytes, path: str):
    with open(path, "wb") as f:
        f.write(data)

def download_line_image_to_temp(message_id: str) -> str:
    """
    LINEサーバーから画像バイナリを取得して /tmp に保存し、保存パスを返す。
    """
    try:
        # v3 SDK の Blob API はストリームを返す
        resp = blob_api.get_message_content(message_id)
        if hasattr(resp, "read"):
            content = resp.read()  # type: ignore
        elif isinstance(resp, (bytes, bytearray)):
            content = bytes(resp)
        elif hasattr(resp, "data"):
            content = resp.data  # type: ignore
        else:
            content = bytes(resp) if resp is not None else b""
        path = _tmp_path("lineimg", ".jpg")
        _save_bytes_to_file(content, path)
        return path
    except Exception as e:
        log.error(f"download error: {e}", exc_info=True)
        raise

def call_analyzer(front_path: str, side_path: str, timeout_sec: int = 110) -> dict:
    """
    Analyzer API を叩いて JSON を返す
    - Render Free のスリープ復帰などを考慮して timeout を長めに
    """
    with open(front_path, "rb") as f1, open(side_path, "rb") as f2:
        files = {
            "front": ("front.jpg", f1, "image/jpeg"),
            "side": ("side.jpg", f2, "image/jpeg"),
        }
        # connect/read を分けたい場合は tuple でも OK: timeout=(15, timeout_sec)
        r = requests.post(ANALYZER_URL, files=files, timeout=timeout_sec)
    r.raise_for_status()
    return r.json()

def format_result(result: dict) -> str:
    """
    Analyzer の戻り JSON をユーザー向けに整形
    期待例:
      {
        "scores": {"overall":8.8,"posture":9.8,"balance":7.3,"fashion":8.8,"muscle_fat":9.8},
        "front_metrics": {...}, "side_metrics": {...},
        "advice": ["...","..."]
      }
    """
    parts = []
    scores = result.get("scores")
    if scores:
        parts.append("■スコア")
        for k in ["overall", "posture", "balance", "fashion", "muscle_fat"]:
            if k in scores:
                parts.append(f"・{k}: {scores[k]}")
        parts.append("")

    fm = result.get("front_metrics")
    if fm:
        parts.append("■正面メトリクス")
        for k, v in fm.items():
            parts.append(f"・{k}: {v}")
        parts.append("")

    sm = result.get("side_metrics")
    if sm:
        parts.append("■側面メトリクス")
        for k, v in sm.items():
            parts.append(f"・{k}: {v}")
        parts.append("")

    adv = result.get("advice")
    if adv and isinstance(adv, list) and adv:
        parts.append("■アドバイス")
        for a in adv:
            parts.append(f"・{a}")

    if not parts:
        parts.append("診断結果を受け取りました。詳細は後ほどご案内します。")

    return "\n".join(parts)

def reply_text(reply_token: str, text: str):
    try:
        messaging_api.reply_message(
            ReplyMessageRequest(
                replyToken=reply_token,
                messages=[TextMessage(text=text)],
            )
        )
    except Exception as e:
        log.error(f"reply text error: {e}", exc_info=True)

def push_text(to_user_id: str, text: str):
    try:
        messaging_api.push_message(
            PushMessageRequest(
                to=to_user_id,
                messages=[TextMessage(text=text)],
            )
        )
    except Exception as e:
        log.error(f"push text error: {e}", exc_info=True)

# ----------------------------------
# 非同期ワーカー
# ----------------------------------
def analyze_and_push(user_id: str, front_path: str, side_path: str):
    """
    画像2枚が揃ったあと、別スレッドで Analyzer を叩き、結果を push で返す。
    """
    try:
        result = call_analyzer(front_path, side_path, timeout_sec=110)
        pretty = format_result(result)
        push_text(user_id, f"診断が完了しました。\n\n{pretty}")
    except requests.HTTPError as he:
        log.error(f"analyzer HTTP error: {he}", exc_info=True)
        push_text(user_id, "診断API呼び出しでエラーが発生しました。（HTTP）\n時間をおいて再度お試しください。")
    except requests.Timeout:
        log.error("analyzer timeout", exc_info=True)
        push_text(user_id, "診断がタイムアウトしました。サーバがスリープ復帰直後の可能性があります。もう一度お試しください。")
    except Exception as e:
        log.error(f"analyzer post error: {e}", exc_info=True)
        push_text(user_id, "診断API呼び出しでエラーが発生しました。時間をおいて再度お試しください。")
    finally:
        # 一時ファイルの掃除
        for p in [front_path, side_path]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        # 状態クリア
        user_temp[user_id] = {}

# ----------------------------------
# ルーティング
# ----------------------------------
@app.get("/")
def index():
    return "LINE Bot is running. Health: /healthz", 200

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.post("/callback")
def callback():
    # LINE 署名検証
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception as e:
        log.error(f"callback handle error: {e}", exc_info=True)
        return "Bad Request", 400
    return "OK", 200

# ----------------------------------
# ハンドラ（テキスト）
# ----------------------------------
@handler.add(MessageEvent, message=TextMessageContent)
def on_text_message(event: MessageEvent):
    text = event.message.text.strip() if event.message and hasattr(event.message, "text") else ""
    user_id = getattr(event.source, "user_id", None) or getattr(event.source, "userId", None)

    if text in ("開始", "スタート", "はじめる", "診断"):
        reply_text(
            event.reply_token,
            "姿勢診断を始めます。\n① 正面の全身写真を送ってください。\n② 次に側面の全身写真を送ってください。\n（顔は写ってもOK／服は体の線が分かるもの推奨）",
        )
        if user_id:
            user_temp[user_id] = {}
        return

    if text.lower() in ("ping", "health", "status"):
        reply_text(event.reply_token, "pong / bot alive")
        return

    reply_text(
        event.reply_token,
        "テキストありがとうございます。姿勢診断を行うには、\n正面→側面 の順に全身写真を2枚お送りください。\n（先に「開始」と送ると案内が表示されます）",
    )

# ----------------------------------
# ハンドラ（画像）
# ----------------------------------
@handler.add(MessageEvent, message=ImageMessageContent)
def on_image_message(event: MessageEvent):
    # 画像を一時保存
    try:
        img_path = download_line_image_to_temp(event.message.id)
    except Exception:
        reply_text(event.reply_token, "画像の取得に失敗しました。少し待って再送してください。")
        return

    user_id = getattr(event.source, "user_id", None) or getattr(event.source, "userId", None)
    if not user_id:
        reply_text(event.reply_token, "ユーザー識別に失敗しました。もう一度お試しください。")
        return

    entry = user_temp.get(user_id) or {}
    front_path = entry.get("front")
    side_path = entry.get("side")

    if not front_path:
        entry["front"] = img_path
        user_temp[user_id] = entry
        reply_text(event.reply_token, "正面の写真を受け取りました。次に『側面の全身写真』を送ってください。")
        return

    if not side_path:
        entry["side"] = img_path
        user_temp[user_id] = entry
        # ここで “返信” は短文で即時返し、解析はバックグラウンドへ
        reply_text(event.reply_token, "側面の写真を受け取りました。診断を開始します。結果はこのトークに届きます。")
    else:
        # 3枚目以降は最後の2枚で診断
        entry["front"] = entry["side"]
        entry["side"] = img_path
        user_temp[user_id] = entry
        reply_text(event.reply_token, "画像を更新しました。最新2枚で診断します。結果はこのトークに届きます。")

    # 最新 front/side を確認
    entry = user_temp[user_id]
    front_path = entry.get("front")
    side_path = entry.get("side")
    if not front_path or not side_path:
        reply_text(event.reply_token, "画像が2枚揃っていません。正面→側面の順に送ってください。")
        return

    # 別スレッドで解析 → push 返信
    threading.Thread(
        target=analyze_and_push,
        kwargs={"user_id": user_id, "front_path": front_path, "side_path": side_path},
        daemon=True,
    ).start()

# ----------------------------------
# main
# ----------------------------------
if __name__ == "__main__":
    # Render では Procfile で gunicorn 起動を推奨
    app.run(host="0.0.0.0", port=PORT)
