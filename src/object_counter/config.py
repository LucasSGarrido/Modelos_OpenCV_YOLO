from __future__ import annotations

from pathlib import Path

DEFAULT_DETECTION_MODEL_PATH = "yolov8n.pt"
DEFAULT_SEGMENTATION_MODEL_PATH = "yolov8n-seg.pt"
DEFAULT_POSE_MODEL_PATH = "yolov8n-pose.pt"
DEFAULT_MODEL_PATH = DEFAULT_DETECTION_MODEL_PATH
DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.5
SEGMENTATION_MODEL_OPTIONS = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt"]
POSE_MODEL_OPTIONS = ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt"]
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
PEOPLE_VEHICLE_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
DEFAULT_IMAGE_OUTPUT_DIR = Path("reports/figures")
DEFAULT_VIDEO_OUTPUT_DIR = Path("reports/videos")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
