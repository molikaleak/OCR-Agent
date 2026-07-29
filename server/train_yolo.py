"""
YOLOv11 Classification Training Script.
This script guides you on how to organize your document dataset and train
a local YOLO11m classifier model to filter out invalid or useless documents
before calling the Gemini API.

Dataset Directory Structure:
    dataset/
    ├── train/
    │   ├── khmer_id/      <- images of Khmer ID cards
    │   ├── passport/      <- images of passports
    │   ├── cv/            <- images of CV pages
    │   ├── certificate/   <- images of certificates
    │   ├── invoice/       <- images of invoices
    │   └── useless/       <- images of random, invalid, or blurry objects
    └── val/
        ├── khmer_id/
        └── ... (same classes as train)

Usage:
    python train_yolo.py --data ./dataset --epochs 50 --imgsz 224
"""

import os
import argparse
from ultralytics import YOLO

def train_model(data_path: str, epochs: int, imgsz: int):
    # Verify dataset directories exist
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"❌ Error: Dataset path must contain 'train' and 'val' subdirectories.")
        print(f"Expected path: {data_path}")
        print("Please organize your images into class subdirectories (e.g., dataset/train/useless/).")
        return

    print("=" * 60)
    print("🚀 Starting YOLO11n Image Classifier Training")
    print("=" * 60)
    print(f"📂 Dataset Path: {os.path.abspath(data_path)}")
    print(f"⏳ Epochs:       {epochs}")
    print(f"🖼️ Image Size:   {imgsz}x{imgsz}")
    print("=" * 60 + "\n")

    # Load pre-trained YOLO11n classification model
    # (will auto-download 'yolo11n-cls.pt' from Ultralytics repo if not present)
    model = YOLO("yolo11n-cls.pt")

    # Start training
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        workers=4,
        device="cpu"  # Set to 0 or 'mps' on Apple Silicon Mac for GPU acceleration
    )

    print("\n" + "=" * 60)
    print("🏆 Training Complete!")
    print("=" * 60)
    print(f"Best model weights saved to: {results.save_dir}/weights/best.pt")
    print("Please copy 'best.pt' and update your server/main.py model configuration.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO11m document classifier.")
    parser.add_argument(
        "--data",
        default="./dataset",
        help="Path to dataset directory containing 'train' and 'val' folders."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Target image size for classification input."
    )
    
    args = parser.parse_args()
    train_model(args.data, args.epochs, args.imgsz)
