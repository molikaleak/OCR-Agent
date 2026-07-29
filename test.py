"""
Send a local file (image / PDF / docx) to the Local FastAPI OCR Server.
Supports interactive mode & category endpoint selection.

Interactive Mode:
    python send_to_n8n_webhook.py

Command Line Mode:
    python send_to_n8n_webhook.py khmer_id_card.png -c khmer_id
    python send_to_n8n_webhook.py passport.jpg -c passport
    python send_to_n8n_webhook.py khmercv.png -c cv
    python send_to_n8n_webhook.py mlika.pdf -c certificate
    python send_to_n8n_webhook.py invoice.png -c invoice

Categories & Endpoints:
    1. khmer_id    : Cambodian National ID Card -> http://localhost:8080/khmer-id
    2. passport    : International Passport    -> http://localhost:8080/passport
    3. cv          : Resume / CV Document      -> http://localhost:8080/cv
    4. certificate : Certificate, Diploma      -> http://localhost:8080/certificate
    5. invoice     : Invoices & Receipts       -> http://localhost:8080/invoice

Requires: Standard Python 3 (Optional optimization modules: requests, pymupdf, pillow)
"""

import sys
import base64
import mimetypes
import json
import os
import io
import time
import argparse
import csv
from datetime import datetime

# Base URL for local FastAPI server
LOCAL_BASE_URL = "http://localhost:8080"

# Endpoint Mappings
LOCAL_ENDPOINTS = {
    "khmer_id": f"{LOCAL_BASE_URL}/khmer-id",
    "passport": f"{LOCAL_BASE_URL}/passport",
    "cv": f"{LOCAL_BASE_URL}/cv",
    "certificate": f"{LOCAL_BASE_URL}/certificate",
    "invoice": f"{LOCAL_BASE_URL}/invoice",
    "auto": f"{LOCAL_BASE_URL}/document-ocr"
}

MAX_UNCOMPRESSED_SIZE = 2 * 1024 * 1024  # 2 MB threshold


def infer_category_from_filename(filename: str) -> str:
    """Infer category automatically from filename hints if category is not provided."""
    name_lower = filename.lower()
    if any(k in name_lower for k in ["id", "khmer_id", "national_id", "card"]):
        return "khmer_id"
    elif any(k in name_lower for k in ["passport", "pass"]):
        return "passport"
    elif any(k in name_lower for k in ["cv", "resume", "biodata"]):
        return "cv"
    elif any(k in name_lower for k in ["cert", "certificate", "diploma", "degree"]):
        return "certificate"
    elif any(k in name_lower for k in ["invoice", "bill", "receipt", "statement"]):
        return "invoice"
    return "auto"


def guess_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type
    ext = file_path.lower().rsplit(".", 1)[-1]
    fallback = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    return fallback.get(ext, "application/octet-stream")


def prepare_payload(file_path: str, raw_bytes: bytes, mime_type: str):
    """
    Prepares payload bytes & MIME type for Gemini Vision OCR:
    - Scanned PDFs (no text layer) are rendered into high-clarity JPEG images (if PyMuPDF installed).
    - Large images (>2MB) are downsampled to prevent payload errors (if Pillow installed).
    """
    file_name = os.path.basename(file_path)

    # Handle PDF files
    if mime_type == "application/pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            text = "".join([page.get_text() for page in doc]).strip()

            if len(text) < 10 or len(raw_bytes) > MAX_UNCOMPRESSED_SIZE:
                print(f"📄 Scanned/Image PDF detected ({len(text)} text chars). Rendering page to high-quality JPEG for Vision OCR...")
                page = doc[0]
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("jpeg", jpg_quality=90)
                out_name = file_name.rsplit(".", 1)[0] + ".jpg"
                print(f"✅ Rendered scanned PDF to JPEG ({len(raw_bytes):,} bytes -> {len(img_bytes):,} bytes)")
                return img_bytes, "image/jpeg", out_name
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️ PDF rendering note: {e}")

    # Handle Image files
    elif mime_type.startswith("image/") and len(raw_bytes) > MAX_UNCOMPRESSED_SIZE:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(raw_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85, optimize=True)
            optimized_bytes = buffer.getvalue()
            print(f"✅ Image optimized ({len(raw_bytes):,} bytes -> {len(optimized_bytes):,} bytes)")
            return optimized_bytes, "image/jpeg", file_name
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️ Image compression note: {e}")

    return raw_bytes, mime_type, file_name


def make_http_post(url: str, payload: dict, timeout: int = 120):
    """Performs HTTP POST request using requests library or built-in urllib with API auth headers."""
    json_bytes = json.dumps(payload).encode("utf-8")
    
    # Load auth token dynamically if it is configured in server/.env
    auth_token = ""
    try:
        env_path = os.path.join(os.path.dirname(__file__), "server", ".env")
        if os.path.isfile(env_path):
            with open(env_path, "r") as ef:
                for line in ef:
                    if line.startswith("API_AUTH_TOKEN="):
                        auth_token = line.split("=", 1)[1].strip().strip("'\"").strip()
    except Exception as e:
        print(f"⚠️ Note: Failed to read auth token from .env: {e}")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ocr-client/1.0"
    }
    if auth_token:
        headers["X-API-Key"] = auth_token
    
    try:
        import requests
        start_time = time.time()
        res = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed = time.time() - start_time
        return res.status_code, res.text, elapsed
    except ImportError:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            url,
            data=json_bytes,
            headers=headers,
            method="POST"
        )
        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.status
                res_text = response.read().decode("utf-8")
                elapsed = time.time() - start_time
                return status_code, res_text, elapsed
        except urllib.error.HTTPError as e:
            elapsed = time.time() - start_time
            res_text = e.read().decode("utf-8") if e.fp else str(e)
            return e.code, res_text, elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            return 500, f"Error: {e}", elapsed


def send_file(file_path: str, category: str = None, custom_url: str = None):
    file_path = file_path.strip("'\"").strip()

    if not os.path.isfile(file_path):
        print(f"\n❌ Error: File not found: {file_path}")
        sys.exit(1)

    file_name = os.path.basename(file_path)
    mime_type = guess_mime_type(file_path)

    if not category:
        inferred = infer_category_from_filename(file_name)
        print(f"ℹ️ Category not specified. Inferred '{inferred}' from filename '{file_name}'")
        category = inferred

    category = category.lower()

    if custom_url:
        target_url = custom_url
    else:
        target_url = LOCAL_ENDPOINTS.get(category, LOCAL_ENDPOINTS["auto"])

    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    payload_bytes, final_mime, final_name = prepare_payload(file_path, raw_bytes, mime_type)
    encoded = base64.b64encode(payload_bytes).decode("utf-8")

    payload = {
        "file": encoded,
        "mimeType": final_mime,
        "fileName": final_name,
        "category": category if category != "auto" else None
    }

    print("\n" + "=" * 60)
    print(f"🚀 Target Environment: Local FastAPI Server")
    print(f"📄 Sending File       : {final_name}")
    print(f"📦 MIME Type         : {final_mime}")
    print(f"🎯 Document Category  : {category.upper()}")
    print(f"🌐 Target Endpoint    : {target_url}")
    print(f"📊 Base64 Size       : {len(encoded):,} chars")
    print("=" * 60 + "\n")

    status_code, res_text, elapsed = make_http_post(target_url, payload, timeout=120)

    print(f"⚡ Status Code: {status_code} (Took {elapsed:.2f} seconds)")
    print("\n📥 Response JSON:")
    try:
        parsed_json = json.loads(res_text)
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
    except ValueError:
        print(res_text)

    log_cost_metrics(file_name, category, status_code, elapsed, res_text)


def log_cost_metrics(file_name: str, category: str, status_code: int, elapsed: float, res_text: str):
    log_file = os.path.join(os.path.dirname(__file__), "ocr_cost_tracker.csv")
    file_exists = os.path.isfile(log_file)
    
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated_cost = 0.0
    confidence = None
    classifier_model = None
    ocr_model = None
    
    try:
        data = json.loads(res_text)
        confidence = data.get("confidence")
        classifier_model = data.get("classifier_model")
        ocr_model = data.get("ocr_model")
        usage = data.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            estimated_cost = usage.get("estimated_cost_usd", 0.0)
    except Exception:
        pass
        
    confidence_str = f"{confidence:.2%}" if confidence is not None else "N/A"
    classifier_str = classifier_model if classifier_model else "N/A"
    ocr_str = ocr_model if ocr_model else "N/A"
        
    try:
        with open(log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "File Name", "Category", "Status Code", 
                    "Classifier Model", "Confidence", "OCR Model",
                    "Prompt Tokens", "Completion Tokens", "Total Tokens", 
                    "Estimated Cost (USD)", "Duration (Seconds)"
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                file_name,
                category.upper(),
                status_code,
                classifier_str,
                confidence_str,
                ocr_str,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                f"${estimated_cost:.6f}",
                f"{elapsed:.2f}"
            ])
        print(f"\n📊 Metrics recorded to log: [ocr_cost_tracker.csv](file:///Users/molika/Desktop/n8n%20ocr/ocr_cost_tracker.csv)")
    except Exception as e:
        print(f"⚠️ Warning: Failed to write to log file: {e}")



def interactive_mode():
    print("\n" + "=" * 60)
    print("📄 Local FastAPI OCR Client — Interactive Mode")
    print("=" * 60)

    print("\nSelect Document Type:")
    print("  1. Khmer ID Card ( Cambodian National ID )")
    print("  2. Passport ( Any International Passport )")
    print("  3. CV / Resume")
    print("  4. Certificate / Diploma / Award")
    print("  5. Invoice / Bill / Receipt")
    print("  6. Auto-Detect ( Classifier Router )")

    choice_map = {
        "1": "khmer_id",
        "2": "passport",
        "3": "cv",
        "4": "certificate",
        "5": "invoice",
        "6": "auto"
    }

    category_choice = ""
    while category_choice not in choice_map:
        category_choice = input("\nEnter choice [1-6]: ").strip()
        if category_choice not in choice_map:
            print("Invalid choice. Please enter a number between 1 and 6.")

    selected_category = choice_map[category_choice]

    file_path = ""
    while not file_path:
        file_path = input("\nEnter file path (or drag & drop file from Finder): ").strip()
        file_path = file_path.strip("'\"").strip()
        if not file_path:
            print("File path cannot be empty.")
        elif not os.path.isfile(file_path):
            print(f"File not found: '{file_path}'. Please try again.")
            file_path = ""

    send_file(file_path, category=selected_category)


def main():
    if len(sys.argv) == 1:
        interactive_mode()
        return

    parser = argparse.ArgumentParser(description="Send documents to Local FastAPI OCR Server.")
    parser.add_argument("file_path", nargs="?", help="Path to document file (JPG, PNG, PDF, DOCX)")
    parser.add_argument(
        "-c", "--category",
        choices=["khmer_id", "passport", "cv", "certificate", "invoice", "auto"],
        help="Target document category endpoint."
    )
    parser.add_argument(
        "-u", "--url",
        help="Custom webhook URL override."
    )

    args = parser.parse_args()

    if not args.file_path:
        interactive_mode()
    else:
        send_file(args.file_path, category=args.category, custom_url=args.url)


if __name__ == "__main__":
    main()
