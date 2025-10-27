import torch
from ultralytics.nn.tasks import DetectionModel
import ultralytics.nn.modules as yolo_modules
from torch import nn
from torch.nn.modules.container import Sequential
import numpy as np  

torch.serialization.add_safe_globals([
    DetectionModel,
    Sequential,
    nn.Module,
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.SiLU,
    nn.MaxPool2d,
    nn.Upsample,
    nn.ConvTranspose2d,
    yolo_modules.C2f,
    yolo_modules.SPPF,
    yolo_modules.Concat,
    yolo_modules.Detect,
    yolo_modules.Conv,  
])

from utils import read_video, save_video
from tracker import Tracker

video_path = r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\input_video.mp4"
video = read_video(video_path)

model_path = r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\models\best.pt"
tracker = Tracker(model_path)

# tracker.get_tracks(video)
tracks = tracker.get_tracks(video)





# save_video(frames, r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\Output-Video2.avi")