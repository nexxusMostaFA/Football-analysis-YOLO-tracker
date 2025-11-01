import torch
from ultralytics.nn.tasks import DetectionModel
import ultralytics.nn.modules as yolo_modules
from torch import nn
from torch.nn.modules.container import Sequential
import numpy as np  
import os
import cv2
from camera_movment_calc import CameraMovmentCalculator

from utils import read_video, save_video
from tracker import Tracker
from assign_players_teams import PlayerTeamAssigner
from ball_assigner import BallAssigner
from transformer import Transformer
from speed_and_distance import SpeedAndDistance

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

team_ball_control= []

for frame_num  , player_track in enumerate(tracks['players']):
    ball_bbox = tracks['ball'][frame_num][1]['bbox']
    assigned_player = ball_assigner.ball_assigner(player_track , ball_bbox)
    # print(assigned_player)

    if assigned_player != -1:
        tracks['players'][frame_num][assigned_player]['has_ball'] = True
        team_ball_control.append(tracks['players'][frame_num][assigned_player]['team_id'])
    else:
        if len(team_ball_control) >0:
            team_ball_control.append(-1)
        else:
            team_ball_control.append(None)

team_ball_control= np.array(team_ball_control)





camera_movment_calculator = CameraMovmentCalculator(video[0])
camera_movments = camera_movment_calculator.calculate_camera_movment(video , stub=True , stub_path=r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\camera_movments_stub.pkl")

# print(tracks['referees'][0])
# print("#" *50)
# print(tracks['ball'][0])


tracks = camera_movment_calculator.add_camera_movments_to_tracks(tracks , camera_movments)


frames = camera_movment_calculator.draw_camera_movement(video , camera_movments)

transformer = Transformer()

tracks = transformer.add_transformed_points_to_tracks(tracks)
# print(tracks['players'][0])

speed_and_distance_calculator = SpeedAndDistance()

tracks = speed_and_distance_calculator.calculate_speed_and_distance(tracks)




frames = tracker.draw_annotations(frames , tracks , team_ball_control)

frames = speed_and_distance_calculator.draw_speed_and_distance(frames , tracks)



save_video(frames, r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\Output-Video.avi")