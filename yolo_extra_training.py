import os
from ultralytics import YOLO


class YOLOv8ExtraTrainer:
    def __init__(
        self,
        model_path,
        dataset_path,
        output_path,
        new_epochs=300,
        batch_size=16,
        image_size=640,
    ):
        """
        Initialize the YOLOv8ExtraTrainer with model, dataset, and training configurations.

        :param model_path: Path to the pre-trained model weights.
        :param dataset_path: Path to the dataset YAML file.
        :param output_path: Path to save the training outputs.
        :param new_epochs: Number of additional epochs to train.
        :param batch_size: Batch size for training.
        :param image_size: Image size for training.
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.new_epochs = new_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.model = YOLO(model_path)  # Load the YOLO model

    def resume_training(self):
        """
        Resume training from the last checkpoint or start fresh.
        """
        # Check if a previous run exists to resume training
        last_run_path = os.path.join(self.output_path, "last")
        if os.path.exists(last_run_path):
            print("Resuming training from the last checkpoint...")
        else:
            print("Starting new training from scratch...")

        # Training configuration
        # train_config = {
        #     "data": self.dataset_path,
        #     "epochs": self.new_epochs,
        #     "batch": self.batch_size,
        #     "imgsz": self.image_size,
        #     "project": self.output_path,
        #     "name": "yolov8_extra_train",
        #     "exist_ok": True,  # Overwrite existing run
        #     "optimizer": "SGD",  # Specify the optimizer
        #     "lr0": 0.0001,  # Reduced learning rate
        # }
        train_config = {
            "data": self.dataset_path,
            "epochs": self.new_epochs,
            "batch": self.batch_size,
            "imgsz": self.image_size,
            "project": self.output_path,
            "name": "yolov8_extra_train",
            "exist_ok": True,  # Overwrite existing run
            "optimizer": "AdamW",  # Specify AdamW optimizer
            "augment": True,  # Enable data augmentation
            "lr0": 0.001,  # Initial learning rate
        }

        # Train the model
        self.model.train(**train_config)

    def save_final_model(self):
        """
        Save the final trained model weights.
        """
        self.model.save()
        print(f"Model weights saved to {self.model_path}")


# Example Usage
if __name__ == "__main__":
    extra_trainer = YOLOv8ExtraTrainer(
        model_path="yolov8_n_custom/train_20241227_0947/weights/best.pt",
        dataset_path="dataset/data.yaml",
        output_path="extra_training",
        new_epochs=300,
        batch_size=24,
        image_size=640,
    )
    extra_trainer.resume_training()
    extra_trainer.save_final_model()
