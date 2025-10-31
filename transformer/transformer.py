import numpy as np
import cv2

class Transformer:

    def __init__(self):
        court_width = 68
        court_length = 23.32
          
        self.vertices = np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]])
          
        self.standered_veritces = np.array([   [0,court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]])
        
        self.vertices = self.vertices.astype(np.float32)
        self.standered_veritces = self.standered_veritces.astype(np.float32)
        
        self.matrix = cv2.getPerspectiveTransform(self.vertices, self.standered_veritces)


    def calc_transformer(self , point):
        print("point before" , point)
        point = (int(point[0]), int(point[1]))
        print("point after1" , point)

        is_in_court = cv2.pointPolygonTest(self.vertices, point, False)>=0
        if not is_in_court: 
            return None
        
        point = np.array([[point]], dtype=np.float32)
        # print("point after2" , point)

        point = point.reshape(-1,1,2)
        # print("point after3" , point)

        transformed_point = cv2.perspectiveTransform(point, self.matrix)
        # print("point after4" , transformed_point)

        transformed_point = transformed_point.reshape(-1,2)
        # print("point after5" , transformed_point)

        return transformed_point
    

    def add_transformed_points_to_tracks(self , tracks):
        for track_type , track_dict in tracks.items():
            for frame_num , frame_dict in enumerate(track_dict):
                for track_id , track_info in frame_dict.items():
                    immeditely_player_position = track_info['position_with_camera_movment']
                    transformed_position = self.calc_transformer(immeditely_player_position)
                    tracks[track_type][frame_num][track_id]['transformed_position'] = transformed_position
        return tracks

        
        
