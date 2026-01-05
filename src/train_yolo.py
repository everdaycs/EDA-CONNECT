from ultralytics import YOLO
import os

def train():
    # Load a model
    # Switching to the largest YOLOv11 model (yolo11x.pt) for maximum accuracy
    model = YOLO('yolo11x.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # We use the data.yaml created in the previous step
    data_path = '/home/kaga/Desktop/EDA-Connect/data.yaml'
    
    print("Starting training...")
        # 建议修改后的 train_yolo.py 关键部分
    results = model.train(
        data=data_path,
        epochs=200,
        imgsz=1024,        # 增加分辨率，对提升 Recall 效果最明显
        batch=4,           # 显存压力增加，调小 batch
        cls=1.5,           # 增加分类损失权重，减少漏检
        copy_paste=0.5,    # 开启 Copy-Paste 增强
        mixup=0.2,         # 开启 Mixup 增强
        fraction=1.0,      # 确保使用全部数据
        box=7.5,           # 保持边界框回归权重
        augment=True,      # 开启增强
        fliplr=0.5,        # 增加水平翻转
        project='/home/kaga/Desktop/EDA-Connect/runs/detect',
        name='train_recall_opt',
        exist_ok=True
    )
    # results = model.train(
    #     data=data_path,
    #     epochs=200,
    #     imgsz=640,
    #     batch=8,
    #     project='/home/kaga/Desktop/EDA-Connect/runs/detect',
    #     name='train_demo',
    #     exist_ok=True
    # )
    
    print("Training completed.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train()
