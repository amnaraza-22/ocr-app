import io
import os

import requests
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

app = Flask(__name__)

OCR_URL = "https://api.ocr.space/parse/image"
API_KEY = os.environ.get("OCR_SPACE_API_KEY", "")
MAX_BYTES = 950_000  # stay under OCR.space's 1 MB free limit


def enhance(raw: bytes) -> bytes:
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)          # fix phone-photo rotation
    img = img.convert("L")                      # grayscale

    # Upscale small/low-res images so letters have more pixels
    w, h = img.size
    longest = max(w, h)
    if longest < 1600:
        scale = 1600 / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif longest > 2600:
        scale = 2600 / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img = ImageOps.autocontrast(img, cutoff=1)  # stretch faded/washed-out text
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    img = ImageEnhance.Contrast(img).enhance(1.3)

    # Compress until under the size limit
    while True:
        for quality in (88, 78, 68, 58, 48):
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=quality)
            if buf.tell() <= MAX_BYTES:
                return buf.getvalue()
        w, h = img.size
        img = img.resize((int(w * 0.85), int(h * 0.85)), Image.LANCZOS)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/ocr", methods=["POST"])
def ocr():
    file = request.files.get("image")
    if not file:
        return jsonify(error="No image uploaded."), 400
    if not API_KEY:
        return jsonify(error="Server is missing the OCR API key."), 500

    try:
        processed = enhance(file.read())
    except Exception:
        return jsonify(error="Could not read that image file."), 400

    try:
        resp = requests.post(
            OCR_URL,
            headers={"apikey": API_KEY},
            files={"file": ("image.jpg", processed, "image/jpeg")},
            data={
                "language": "eng",
                "OCREngine": "2",
                "scale": "true",
                "detectOrientation": "true",
            },
            timeout=30,
        )
        result = resp.json()
    except Exception:
        return jsonify(error="OCR service is unreachable. Try again."), 502

    if result.get("IsErroredOnProcessing"):
        msg = result.get("ErrorMessage") or "OCR failed."
        if isinstance(msg, list):
            msg = " ".join(msg)
        return jsonify(error=msg), 422

    text = "\n".join(
        p.get("ParsedText", "") for p in result.get("ParsedResults", [])
    ).strip()
    return jsonify(text=text or "(No text found in this image.)")
