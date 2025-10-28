import torch
from ultralytics.nn.tasks import DetectionModel
import ultralytics.nn.modules as yolo_modules
from torch import nn
from torch.nn.modules.container import Sequential
import numpy as np  
import os
import cv2

# torch.serialization.add_safe_globals([
#     DetectionModel,
#     Sequential,f
#     nn.Module,
#     nn.Conv2d,
#     nn.BatchNorm2d,
#     nn.ReLU,
#     nn.SiLU,
#     nn.MaxPool2d,
#     nn.Upsample,
#     nn.ConvTranspose2d,
#     yolo_modules.C2f,
#     yolo_modules.SPPF,
#     yolo_modules.Concat,
#     yolo_modules.Detect,
#     yolo_modules.Conv,  
# ])

from utils import read_video, save_video
from tracker import Tracker
from assign_players_teams import PlayerTeamAssigner

video_path = r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\input_video.mp4"

video = read_video(video_path)

model_path = r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\models\best.pt"
tracker = Tracker(model_path)

tracks = tracker.get_tracks(video , stub=True , stub_path=r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\tracks_stub.pkl")

tracks = tracker.find_position_tracks(tracks)

tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

player_team_assigner = PlayerTeamAssigner()

player_team_assigner.assign_colors(video[0],tracks['players'][0])

for frame_num , player in enumerate(tracks['players']):
    for track_id , track in player.items():
        bbox = track['bbox']
        team_id = player_team_assigner.assign_teams(video[frame_num] , bbox , track_id)
        tracks['players'][frame_num][track_id]["team_id"] = team_id
        color = player_team_assigner.teams_dict[team_id]
        tracks['players'][frame_num][track_id]["color"] = color  



frames = tracker.draw_annotations(video , tracks)

# print(tracks['players'][0])

## croping image and save it 

# frame = frames[0]
# for track_id , player_track in tracks['players'][0].items():
#         bbox = player_track['bbox']
#         image = frame[int(bbox[1]):int(bbox[3]) , int(bbox[0]):int(bbox[2])]
#         cv2.imwrite(r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\cropped_image.jpg" , image)
#         break

save_video(frames, r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\Output-Video.avi")