from ultralytics import YOLO
import torch
from datetime import datetime
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class YOLOTrainer:
    def __init__(
        self, model_size="s", yaml_path="dataset/data.yaml", epochs=100, batch_size=None
    ):
        """
        Initialize the YOLOTrainer class.

        Args:
            model_size (str): Size of the model ('n', 's', 'm', etc.)
            yaml_path (str): Path to the dataset YAML file
            epochs (int): Number of training epochs
            batch_size (int): Batch size (default is determined by model size)
        """
        if yaml_path is None:
            raise ValueError("Please provide the path to your YAML file")

        self.model_size = model_size
        self.yaml_path = yaml_path
        self.epochs = epochs
        self.batch_size = batch_size or self._get_default_batch_size()
        self.device = 0 if torch.cuda.is_available() else -1
        self.project_name = f"yolov8_{self.model_size}_custom"
        self.run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.model = YOLO(f"yolov8{self.model_size}.pt")

    def _get_default_batch_size(self):
        """
        Get the default batch size based on the model size.

        Returns:
            int: Default batch size
        """
        default_batch_sizes = {"n": 16, "s": 16, "m": 14}
        return default_batch_sizes.get(self.model_size, 16)

    def train(self):
        """
        Train the YOLO model.

        Returns:
            tuple: Training results and validation metrics
        """
        print(f"Starting training for YOLOv8-{self.model_size} model...")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")

        results = self.model.train(
            data=self.yaml_path,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=640,
            device=self.device,
            project=self.project_name,
            name=self.run_name,
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            verbose=True,
        )

        print("Running validation...")
        metrics = self.model.val()

        save_path = f"{self.project_name}/{self.run_name}/best.pt"
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

        return results, metrics


if __name__ == "__main__":
    YAML_PATH = "dataset/data.yaml"

    # Train YOLOv8

    trainer = YOLOTrainer(
        model_size="n", yaml_path=YAML_PATH, epochs=250, batch_size=16
    )
    results, metrics = trainer.train()
