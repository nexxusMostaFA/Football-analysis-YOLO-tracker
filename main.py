import torch
from ultralytics.nn.tasks import DetectionModel
import ultralytics.nn.modules as yolo_modules
from torch import nn
from torch.nn.modules.container import Sequential
import numpy as np  
import os
import cv2
from camera_movment_calc import CameraMovmentCalculator

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
from ball_assigner import BallAssigner

video_path = r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\input_video.mp4"

video = read_video(video_path)

model_path = r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\models\best.pt"
tracker = Tracker(model_path)

tracks = tracker.get_tracks(video , stub=True , stub_path=r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\tracks_stub.pkl")

tracks = tracker.find_position_tracks(tracks)

# print(tracks['ball'][0])

# print("#" *50)

tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

# print(tracks['ball'][0])

player_team_assigner = PlayerTeamAssigner()

player_team_assigner.assign_colors(video[0],tracks['players'][0])

for frame_num , player in enumerate(tracks['players']):
    for track_id , track in player.items():
        bbox = track['bbox']
        team_id = player_team_assigner.assign_teams(video[frame_num] , bbox , track_id)
        tracks['players'][frame_num][track_id]["team_id"] = team_id
        color = player_team_assigner.teams_dict[team_id]
        tracks['players'][frame_num][track_id]["color"] = color  


ball_assigner = BallAssigner()

for frame_num  , player_track in enumerate(tracks['players']):
    ball_bbox = tracks['ball'][frame_num][1]['bbox']
    assigned_player = ball_assigner.ball_assigner(player_track , ball_bbox)
    # print(assigned_player)

    if assigned_player != -1:
        tracks['players'][frame_num][assigned_player]['has_ball'] = True




camera_movment_calculator = CameraMovmentCalculator(video[0])
camera_movments = camera_movment_calculator.calculate_camera_movment(video , stub=True , stub_path=r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\camera_movments_stub.pkl")

# print(tracks['referees'][0])
# print("#" *50)
# print(tracks['ball'][0])




tracks = camera_movment_calculator.add_camera_movments_to_tracks(tracks , camera_movments)


frames = camera_movment_calculator.draw_camera_movement(video , camera_movments)

frames = tracker.draw_annotations(frames , tracks)


# print(tracks['players'][0])

## croping image and save it 

# frame = frames[0]
# for track_id , player_track in tracks['players'][0].items():
#         bbox = player_track['bbox']
#         image = frame[int(bbox[1]):int(bbox[3]) , int(bbox[0]):int(bbox[2])]
#         cv2.imwrite(r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\cropped_image.jpg" , image)
#         break

save_video(frames, r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\Output-Video.avi")