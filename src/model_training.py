import os

from ultralytics import YOLO

BASE_DIR = os.path.dirname(__file__)  # Project Dir/src
ROOT = os.path.dirname(BASE_DIR)  # Project_Dir/
DATA_PATH = os.path.join(ROOT, 'data')  # Project_Dir/data
MODELS_PATH = os.path.join(ROOT, 'models')  # Project_Dir/models


def main():
    model = YOLO('yolov8n.pt')

    model.train(
        data=os.path.join(DATA_PATH, 'sku110k.yaml'),
        epochs=40,
        patience=8,  # 1 / 5
        imgsz=416,
        batch=16,
        workers=4,
        project=MODELS_PATH,
        name='yolo_sku110k_model',
        device=0
    )

    metrics = model.val()
    print(metrics)


if __name__ == '__main__':
    main()
