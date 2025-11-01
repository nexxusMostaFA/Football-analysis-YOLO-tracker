import cv2
from utils import measure_distance ,calc_xy_distance
import numpy as np
import pickle
import os


class CameraMovmentCalculator():
    def __init__(self , frame):
        
        self.min_distance = 7

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features_params = dict(
            maxCorners=100,
            qualityLevel=0.3,  
            minDistance=7,
            blockSize=7, 
            mask = mask_features        
        )

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

   

    def add_camera_movments_to_tracks(self , tracks , camera_movments):
        for track_type , track_dict in tracks.items():
            for frame_num , frame_dict in enumerate(track_dict):
                for track_id , track_info in frame_dict.items():
                    immeditely_player_position = track_info['position']
                    camera_movment = camera_movments[frame_num]
                    new_position = (immeditely_player_position[0] - camera_movment[0] , immeditely_player_position[1] - camera_movment[1])
                    tracks[track_type][frame_num][track_id]['position_with_camera_movment'] = new_position
        return tracks
                




    def calculate_camera_movment(self , frames  , stub = True , stub_path = None):

        if stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path , 'rb') as file:
                camera_movments = pickle.load(file)
                return camera_movments


        old_frame = frames[0]
        old_frame_gray = cv2.cvtColor(old_frame , cv2.COLOR_BGR2GRAY)

        old_features = cv2.goodFeaturesToTrack(old_frame_gray , **self.features_params)
        camera_movments = [[0,0]]*len(frames)

        for frame_num in range(1 , len(frames)):
            max_distance = 0
            new_frame = frames[frame_num]
            new_frame_gray = cv2.cvtColor(new_frame , cv2.COLOR_BGR2GRAY)

            new_features,_,_ = cv2.calcOpticalFlowPyrLK(old_frame_gray , new_frame_gray , old_features , None , **self.lk_params)


            for i  , (new , old) in enumerate(zip(new_features  , old_features)):
                new_point = new.ravel()
                old_point = old.ravel()

                distance = measure_distance(new_point , old_point)

                if distance > max_distance:
                    max_distance = distance
                    x,y = calc_xy_distance(new_point , old_point)

            if max_distance > self.min_distance:
                camera_movments[frame_num]  = (x , y)
                old_features = cv2.goodFeaturesToTrack(new_frame_gray , **self.features_params)
                 
            old_frame_gray = new_frame_gray.copy()


        if stub_path is not None:
            with open(stub_path , 'wb') as file:
                pickle.dump(camera_movments  ,file)

        return camera_movments
    

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            
            # Panel dimensions
            panel_width = 350
            panel_height = 100
            panel_x = 20
            panel_y = 20
            
            # Shadow
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x + 3, panel_y + 3),
                        (panel_x + panel_width + 3, panel_y + panel_height + 3),
                        (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Panel background
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y),
                        (panel_x + panel_width, panel_y + panel_height),
                        (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            
            # Border
            cv2.rectangle(frame, (panel_x, panel_y),
                        (panel_x + panel_width, panel_y + panel_height),
                        (200, 200, 200), 2, cv2.LINE_AA)
            
            # Title
            cv2.putText(frame, "CAMERA TRACKING", (panel_x + 15, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
            
            # X Movement with icon - BIGGER TEXT
            cv2.circle(frame, (panel_x + 25, panel_y + 62), 6, (100, 150, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, f"X: {x_movement:+.2f}", (panel_x + 40, panel_y + 68),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 150, 255), 2, cv2.LINE_AA)
            
            # Y Movement with icon - BIGGER TEXT
            cv2.circle(frame, (panel_x + 200, panel_y + 62), 6, (255, 100, 100), -1, cv2.LINE_AA)
            cv2.putText(frame, f"Y: {y_movement:+.2f}", (panel_x + 215, panel_y + 68),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 100, 100), 2, cv2.LINE_AA)
            
            # Divider line
            cv2.line(frame, (panel_x + 185, panel_y + 45), (panel_x + 185, panel_y + 80),
                    (220, 220, 220), 2, cv2.LINE_AA)
            
            output_frames.append(frame)
        
        return output_frames