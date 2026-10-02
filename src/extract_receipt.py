import json
import os
import sys
from pathlib import Path

import lmstudio as lms

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "data" / "Raw" / "nota-sample.png"
OUT_PATH = ROOT / "reports" / "receipt_extraction.json"
MODEL_NAME = os.environ.get("LM_STUDIO_MODEL", "qwen2.5-vl-7b-instruct")

SCHEMA = {
    "type": "object",
    "properties": {
        "merchant": {"type": "string"},
        "tanggal": {"type": "string"},
        "item": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nama": {"type": "string"},
                    "qty": {"type": "number"},
                    "harga": {"type": "number"},
                },
                "required": ["nama"],
            },
        },
        "subtotal": {"type": "number"},
        "pajak": {"type": "number"},

        "total": {"type": "number"},
    },
    "required": ["merchant", "tanggal", "item", "total"],
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if not IMAGE_PATH.exists():
    print(f"❌ Gambar tidak ditemukan: {IMAGE_PATH}")
    sys.exit(1)

print(f"📸 Memproses gambar: {IMAGE_PATH}")
print(f"🤖 Menggunakan model: {MODEL_NAME}")

try:
    image = lms.prepare_image(str(IMAGE_PATH))
    model = lms.llm(MODEL_NAME)

    chat = lms.Chat()
    chat.add_user_message(
        "Baca nota ini. Ekstrak: merchant, tanggal, item, subtotal, pajak, dan total.",
        images=[image],
    )

    result = model.respond(chat, response_format=SCHEMA)
    data = result.parsed if hasattr(result, "parsed") else json.loads(result.content)

    OUT_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"✅ Tersimpan di {OUT_PATH}")

except Exception as e:
    print(f"❌ Terjadi kesalahan: {e}")
    sys.exit(1)
