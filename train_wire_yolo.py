from ultralytics import YOLO

def train():
    # Load the model
    # Switched back to x model but with smaller batch to avoid OOM
    model = YOLO("yolo11x.pt")  
    
    # Train the model
    result = model.train(
        data="wire_data.yaml",
        epochs=10000,
        imgsz=1024,
        batch=4,   # 减少 batch size 以适配 16GB 显存
        device=0,
        project="runs/detect",
        name="wire_detection",
        exist_ok=True,
        optimizer="AdamW",  # 强制使用 AdamW 优化器，避免小 Batch 下 SGD 梯度爆炸
        lr0=0.001,         # 降低初始学习率
        scale=0.0,         # 禁止随机缩放，保护细小导线
        mosaic=0.0,        # 导线检测建议关闭 Mosaic 以保持图像细节
        flipud=0.0,        # 禁止上下翻转
        mixup=0.0          # 禁止 Mixup 防止线条模糊
    )

if __name__ == "__main__":
    train()
