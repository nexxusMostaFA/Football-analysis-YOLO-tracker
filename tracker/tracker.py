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

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                        (int(x1_rect),int(y1_rect) ),
                        (int(x2_rect),int(y2_rect)),
                        color,
                        cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

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
    

    def interpolate_ball_positions(self, ball_positions):

        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    

    def draw_annotations(self,video_frames, tracks):

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

            output_video_frames.append(frame)

        return output_video_frames