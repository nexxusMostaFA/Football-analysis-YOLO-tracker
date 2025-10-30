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
    

    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames

        






        
