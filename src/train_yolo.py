from ultralytics import YOLO
import os

def train():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # We use the data.yaml created in the previous step
    data_path = '/home/kaga/Desktop/EDA-Connect/data.yaml'
    
    print("Starting training...")
    results = model.train(
        data=data_path,
        epochs=10,
        imgsz=640,
        batch=16,
        project='/home/kaga/Desktop/EDA-Connect/runs/detect',
        name='train_demo',
        exist_ok=True
    )
    
    print("Training completed.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train()
