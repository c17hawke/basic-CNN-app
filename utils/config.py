import os

DATA_DIR = os.path.join("data", "PetImages")
IMAGE_SIZE = (224, 224, 3)
BATCH_SIZE = 16
EPOCHS = 10
CLASSES = 2
TRAINED_MODEL_DIR = os.path.join("VGGmodel", "models")
CHECKPOINT_DIR = os.path.join("VGGmodel", "checkpoints")
BASE_LOG_DIR = "base_log_dir"
TENSORBOARD_ROOT_LOG_DIR = os.path.join(BASE_LOG_DIR, "tensorboard_log_dir")
AUGMENTATION = True