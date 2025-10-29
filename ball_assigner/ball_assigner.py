from utils import foot_position_of_bbox , center_of_bbox ,measure_distance

class BallAssigner():
    def __init__(self):
        self.max_distance = 60

    def ball_assigner(self , player_track , ball_bbox):
        min_distance = 9999
        assigned_player = -1

        ball_position = center_of_bbox(ball_bbox)

        for track_id , track_info in player_track.items():
                player_bbox = track_info['bbox']

                left_foot_distance = measure_distance((player_bbox[0] , player_bbox[-1]) , ball_position)
                right_foot_distance = measure_distance((player_bbox[2] , player_bbox[-1]) , ball_position)

                distance = min(left_foot_distance , right_foot_distance)

                if distance < self.max_distance : 
                    if distance < min_distance:
                        min_distance = distance
                        assigned_player = track_id

        return assigned_player


        

