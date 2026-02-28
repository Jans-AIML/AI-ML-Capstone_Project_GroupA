"""
Prerequisites:
    pip install ultralytics tqdm

Steps before running:
    1. Download COCO 2017 dataset from https://cocodataset.org/#download
       - 2017 Train images [118K/18GB]
       - 2017 Val images [5K/1GB]
       - 2017 Train/Val annotations [241MB]
    2. Run convert_coco_to_yolo.py to convert annotations
    3. Update paths in coco.yaml
    4. Run this script
"""

from ultralytics import YOLO


def main():
    # ============================================================
    # CONFIGURATION
    # ============================================================

    # Path to your coco.yaml config
    DATA_YAML = "./images/coco.yaml"

    # Model variant - using yolo11s
    # Options: yolo11n.yaml, yolo11s.yaml, yolo11m.yaml, yolo11l.yaml, yolo11x.yaml
    #
    # Two choices:
    #   1. Train from scratch:      YOLO("yolo11m.yaml")   <-- builds model with random weights
    #   2. Fine-tune pretrained:    YOLO("yolo11m.pt")     <-- loads COCO-pretrained weights
    #
    # Training from scratch on COCO will take MUCH longer to converge.
    # If you want true from-scratch training, use the .yaml config:
    MODEL = "yolo11s.yaml"  # from scratch (random initialization)
    # MODEL = "yolo11s.pt"  # pretrained (uncomment for fine-tuning)

    # ============================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================
    model = YOLO(MODEL)

    results = model.train(
        # --- Data ---
        data=DATA_YAML,
        imgsz=640,

        # --- Training schedule ---
        epochs=300,              # Standard COCO training schedule
        patience=50,             # Early stopping patience (epochs with no improvement)

        # --- Batch size ---
        # For 12GB VRAM with yolo11m at imgsz=640:
        #   batch=8  -> safe, ~8-9GB VRAM
        #   batch=12 -> tight, ~10-11GB VRAM
        #   batch=-1 -> auto-detect (recommended first run)
        batch=-1,                # Auto batch size detection

        # --- Optimizer ---
        optimizer="SGD",         # SGD is standard for COCO training
        lr0=0.01,                # Initial learning rate
        lrf=0.01,                # Final learning rate (lr0 * lrf)
        momentum=0.937,          # SGD momentum
        weight_decay=0.0005,     # L2 regularization

        # --- Warmup ---
        warmup_epochs=3.0,       # Warmup period
        warmup_momentum=0.8,     # Warmup momentum
        warmup_bias_lr=0.1,      # Warmup bias learning rate

        # --- Augmentation ---
        hsv_h=0.015,             # HSV-Hue augmentation
        hsv_s=0.7,               # HSV-Saturation augmentation
        hsv_v=0.4,               # HSV-Value augmentation
        degrees=0.0,             # Rotation (+/- degrees)
        translate=0.1,           # Translation (+/- fraction)
        scale=0.5,               # Scale (+/- gain)
        shear=0.0,               # Shear (+/- degrees)
        perspective=0.0,         # Perspective (+/- fraction)
        flipud=0.0,              # Flip up-down probability
        fliplr=0.5,              # Flip left-right probability
        mosaic=1.0,              # Mosaic augmentation probability
        mixup=0.0,               # Mixup augmentation probability
        copy_paste=0.0,          # Copy-paste augmentation probability
        close_mosaic=10,         # Disable mosaic for last N epochs

        # --- Performance ---
        amp=True,                # Mixed precision (crucial for 12GB)
        workers=8,               # Dataloader workers (adjust to your CPU cores)
        cache=False,             # Don't cache images in RAM (set True if you have 64GB+ RAM)
        device=0,                # GPU device (0 for single GPU)

        # --- Logging & Saving ---
        project="runs/coco",     # Save directory
        name="yolo11m_scratch",  # Experiment name
        save_period=10,          # Save checkpoint every N epochs
        val=True,                # Run validation each epoch
        plots=True,              # Generate training plots

        # --- Other ---
        seed=42,                 # Reproducibility
        verbose=True,
    )

    # ============================================================
    # VALIDATION
    # ============================================================
    print("\n" + "=" * 60)
    print("Running final validation...")
    print("=" * 60)

    # Validate best model
    metrics = model.val(
        data=DATA_YAML,
        imgsz=640,
        batch=8,
        device=0,
    )

    print(f"\nmAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    # ============================================================
    # EXPORT (Optional)
    # ============================================================
    # Uncomment to export to different formats after training:

    # ONNX export
    # model.export(format="onnx", imgsz=640, dynamic=True)

    # TensorRT export (for NVIDIA deployment)
    # model.export(format="engine", imgsz=640, half=True)

    # CoreML export (for Apple devices)
    # model.export(format="coreml", imgsz=640)


if __name__ == "__main__":
    main()