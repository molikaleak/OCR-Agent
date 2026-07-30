import os
import subprocess
import sys

# Directory containing test data files
DATA_DIR = "./data"

# Mapping of file name keywords to their expected category
CATEGORY_MAPPING = {
    # Certificates
    "c1.png": "certificate",
    "c2.png": "certificate",
    "c3.png": "certificate",
    "c4.png": "certificate",
    "c5.png": "certificate",
    "c6.png": "certificate",
    "c7.png": "certificate",
    "technovation": "certificate",
    "certificate": "certificate",
    
    # CVs
    "cv1": "cv",
    "cv2": "cv",
    "cv3": "cv",
    "cv4": "cv",
    "cv5": "cv",
    "cv6": "cv",
    "cv7": "cv",
    "cvkh1": "cv",
    "cvkh2": "cv",
    "molika": "cv",
    
    # Khmer IDs
    "id.png": "khmer_id",
    "id2": "khmer_id",
    "id3": "khmer_id",
    "id4": "khmer_id",
    "id6": "khmer_id",
    
    # Passports
    "passport": "passport"
}

def get_category_for_file(filename):
    filename_lower = filename.lower()
    for keyword, category in CATEGORY_MAPPING.items():
        if keyword in filename_lower:
            return category
    return None

def run_tests():
    if not os.path.isdir(DATA_DIR):
        print(f"❌ Error: Data directory '{DATA_DIR}' not found.")
        sys.exit(1)

    files = sorted([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
    
    print("\n" + "=" * 60)
    print(f"🚀 Starting Batch Test Runner - Found {len(files)} files")
    print("=" * 60)

    import urllib.request
    import time
    
    print("⏳ Waiting for FastAPI server to become healthy (downloading model weights)...")
    start_time = time.time()
    server_ready = False
    while time.time() - start_time < 1800:  # 30 minutes timeout
        try:
            with urllib.request.urlopen("http://localhost:8080/health", timeout=2) as response:
                if response.status == 200:
                    server_ready = True
                    break
        except Exception:
            pass
        time.sleep(5)
        
    if not server_ready:
        print("❌ Timeout: FastAPI server did not become ready.")
        sys.exit(1)
    print("✅ Server is ready! Starting tests...")

    python_bin = "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
    
    success_count = 0
    total_count = 0

    for file in files:
        if file.startswith("."):
            continue
            
        file_path = os.path.join(DATA_DIR, file)
        category = get_category_for_file(file)
        
        if not category:
            print(f"⚠️ Skipping '{file}': Could not auto-detect category.")
            continue
            
        total_count += 1
        print(f"\n[{total_count}/{len(files)}] Running test: {file} ➡️ category: {category.upper()}")
        
        # Invoke test.py for this file
        try:
            cmd = [python_bin, "test.py", file_path, "-c", category]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Print brief status summary
            if "Status Code: 200" in result.stdout:
                print(f"✅ Success! Response logged to ocr_cost_tracker.csv")
                success_count += 1
            else:
                print(f"❌ Failed! Output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error executing test.py for {file}: {e}")
            print(f"Stdout:\n{e.stdout}")
            print(f"Stderr:\n{e.stderr}")

    print("\n" + "=" * 60)
    print(f"🎯 Batch Testing Completed: {success_count}/{total_count} passed successfully!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_tests()
