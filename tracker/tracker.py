from ultralytics import YOLO
import supervision as sv
import pickle
import os
from utils.bbox_utils import center_of_bbox , foot_position_of_bbox ,get_bbox_width
import cv2
import numpy as np
import pandas as pd



class Tracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
       
    def yolo_prediction(self, frames):
        batch_size = 100
        predictions = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i+batch_size]
            results = self.model.predict(batch_frames)
            predictions += results
        return predictions
    
    def get_tracks(self , frames , stub = False , stub_path = None):

        if stub and os.path.exists(stub_path):
            with open(stub_path , 'rb') as f:
                tracks = pickle.load(f)
            return tracks


        yolo_predictions = self.yolo_prediction(frames)

        tracks = {
            'players':[]
            ,'referees':[]
            ,'ball':[]
        }

        for frame_num , prediction_frame in enumerate(yolo_predictions):

            class_names = prediction_frame.names
            class_names_inv = { v:k for k , v in class_names.items()}
            # print("#"*50)
            # print("class_names",class_names)
            # print("#"*50)
            # print( "class_names inverse", class_names_inv)
            # print("#"*50)
            supervision_detiction = sv.Detections.from_ultralytics(prediction_frame)
            # print("#"*50)
            # print("sv detiction" , supervision_detiction)
            # print("#"*50)
            # print("class_id " , supervision_detiction.class_id)
            # print("#"*50)
            # convert goalkeeper to player
            for index , class_id in enumerate(supervision_detiction.class_id):
                if class_id == class_names_inv['goalkeeper']:
                    supervision_detiction.class_id[index] = class_names_inv['player']

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            detection_with_tracks = self.tracker.update_with_detections(supervision_detiction)
            # print("#"*50)
            # print( 'detection_with_tracks' , detection_with_tracks)
            # print("#"*50)

            for object in detection_with_tracks:
                bbox = object[0].tolist()
                class_id = object[3]
                # print("#"*50)
                # print( "type of class id", class_id.dtype)
                # print("#"*50)
                track_id = object[4]

                if class_id == class_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox' : bbox}

                
                if class_id == class_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox' : bbox}

            for object in supervision_detiction:
                bbox = object[0].tolist()
                class_id = object[3]
                if class_id == class_names_inv['ball']:
                    track_id = 1
                    tracks['ball'][frame_num][track_id] = {'bbox' : bbox}

        if stub_path is not None:
            with open(stub_path , 'wb') as f:
                pickle.dump(tracks , f)

        return tracks
        

    def find_position_tracks(self , tracks):

        for object , track_info in tracks.items():
            for index , frame_info in enumerate(track_info):
                for track_id , info in frame_info.items():
                    bbox = info['bbox']
                    if object == 'ball':
                        position = center_of_bbox(bbox)
                    else:
                        position = foot_position_of_bbox(bbox)
                    tracks[object][index][track_id]['position'] = position

        return tracks
    

    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # rectangle_width = 40
        # rectangle_height=20
        # x1_rect = x_center - rectangle_width//2
        # x2_rect = x_center + rectangle_width//2
        # y1_rect = (y2- rectangle_height//2) +15
        # y2_rect = (y2+ rectangle_height//2) +15

        # if track_id is not None:
        #     cv2.rectangle(frame,
        #                 (int(x1_rect),int(y1_rect) ),
        #                 (int(x2_rect),int(y2_rect)),
        #                 color,
        #                 cv2.FILLED)
            
        #     x1_text = x1_rect+12
        #     if track_id > 99:
        #         x1_text -=10
            
        #     cv2.putText(
        #         frame,
        #         f"{track_id}",
        #         (int(x1_text),int(y1_rect+15)),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.6,
        #         (0,0,0),
        #         2
        #     )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    

    def interpolate_ball_positions(self, ball_tracks):

        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_tracks]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_tracks = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        for frame_num , track in enumerate(ball_tracks):
            for _ , info in track.items():
                ball_bbox = info['bbox']
            ball_position = center_of_bbox(ball_bbox)
            ball_tracks[frame_num][1]['position'] = ball_position
            ball_tracks[frame_num][1]['bbox'] = ball_bbox

        return ball_tracks

    

    def draw_annotations(self,video_frames, tracks ,team_ball_control):

        output_video_frames= []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player['color']
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)
                if 'has_ball' in player and player['has_ball'] == True:
                    frame = self.draw_traingle(frame , player['bbox'] , (0 ,0 , 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
    

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        total_frames = team_1_num_frames + team_2_num_frames
        
        if total_frames == 0:
            team_1 = 0
            team_2 = 0
        else:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        
        panel_width = 400
        panel_height = 120
        panel_x = frame.shape[1] - panel_width - 30
        panel_y = frame.shape[0] - panel_height - 30
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x + 4, panel_y + 4),
                    (panel_x + panel_width + 4, panel_y + panel_height + 4),
                    (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                    (panel_x + panel_width, panel_y + panel_height),
                    (250, 250, 250), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        cv2.rectangle(frame, (panel_x, panel_y),
                    (panel_x + panel_width, panel_y + panel_height),
                    (200, 200, 200), 2, cv2.LINE_AA)
        
        cv2.putText(frame, "BALL CONTROL", (panel_x + 20, panel_y + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
        
        team1_y = panel_y + 60
        cv2.circle(frame, (panel_x + 25, team1_y - 6), 6, (255, 100, 100), -1, cv2.LINE_AA)
        cv2.putText(frame, "Team 1", (panel_x + 40, team1_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{team_1 * 100:.1f}%", (panel_x + 120, team1_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 100, 100), 2, cv2.LINE_AA)
        
        bar_x = panel_x + 200
        bar_width = 170
        bar_height = 12
        cv2.rectangle(frame, (bar_x, team1_y - 10),
                    (bar_x + bar_width, team1_y - 10 + bar_height),
                    (220, 220, 220), -1, cv2.LINE_AA)
        team1_bar_width = int(bar_width * team_1)
        if team1_bar_width > 0:
            cv2.rectangle(frame, (bar_x, team1_y - 10),
                        (bar_x + team1_bar_width, team1_y - 10 + bar_height),
                        (255, 100, 100), -1, cv2.LINE_AA)
        
        team2_y = panel_y + 95
        cv2.circle(frame, (panel_x + 25, team2_y - 6), 6, (100, 150, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "Team 2", (panel_x + 40, team2_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{team_2 * 100:.1f}%", (panel_x + 120, team2_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 150, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(frame, (bar_x, team2_y - 10),
                    (bar_x + bar_width, team2_y - 10 + bar_height),
                    (220, 220, 220), -1, cv2.LINE_AA)
        team2_bar_width = int(bar_width * team_2)
        if team2_bar_width > 0:
            cv2.rectangle(frame, (bar_x, team2_y - 10),
                        (bar_x + team2_bar_width, team2_y - 10 + bar_height),
                        (100, 150, 255), -1, cv2.LINE_AA)
        
        return frame