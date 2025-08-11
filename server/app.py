from __future__ import annotations

import os
from typing import Any, Dict, List

from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

@app.post("/api/chat")
def chat() -> Any:
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 500

    body = request.get_json(force=True, silent=True) or {}
    system = body.get("system", "You are a helpful assistant.")
    messages = body.get("messages", [])
    context = body.get("context")

    if context:
        messages = messages + [{"role": "system", "content": f"Context: URL={context.get('url')}, PATH={context.get('path')}, TITLE={context.get('title')}"}]

    # Compose OpenAI payload
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.3,
    }

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            return jsonify({"error": f"OpenAI error {resp.status_code}", "detail": resp.text}), 500
        data = resp.json()
        reply = data["choices"][0]["message"]["content"].strip()
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": "Request failed", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5057))
    app.run(host="0.0.0.0", port=port, debug=True)
