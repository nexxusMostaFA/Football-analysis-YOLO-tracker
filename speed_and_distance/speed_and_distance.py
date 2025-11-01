from utils import measure_distance, foot_position_of_bbox
import cv2
import numpy as np

class SpeedAndDistance:
    def __init__(self):
        self.frames_rate = 24
        self.frames_batch_size = 5

    def calculate_speed_and_distance(self, tracks):
        total_distance = {}

        for track_type, track_dict in tracks.items():
            if track_type != 'players':
                continue
            frames_length = len(track_dict)

            for frame_num in range(0, frames_length, self.frames_batch_size):
                last_frame = min(frame_num + self.frames_batch_size, frames_length - 1)

                for track_id, _ in track_dict[frame_num].items():
                    if track_id not in track_dict[last_frame]:
                        continue

                    start_position = track_dict[frame_num][track_id]['transformed_position']
                    last_position = track_dict[last_frame][track_id]['transformed_position']

                    if start_position is None or last_position is None:
                        continue

                    start_position = tuple(start_position[0])
                    last_position = tuple(last_position[0])

                    dictance = measure_distance(start_position, last_position)
                    time = (last_frame - frame_num) / self.frames_rate
                    speed = dictance / time 
                    speed_kmh = speed * 3.6

                    if track_type not in total_distance:
                        total_distance[track_type] = {}

                    if track_id not in total_distance[track_type]:
                        total_distance[track_type][track_id] = 0

                    total_distance[track_type][track_id] += dictance

                    for frame in range(frame_num, last_frame):
                        if track_id not in tracks[track_type][frame]:
                            continue
                        tracks[track_type][frame][track_id]['speed'] = speed_kmh
                        tracks[track_type][frame][track_id]['distance'] = total_distance[track_type][track_id]

        return tracks
    
    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:
                            continue
                        
                        bbox = track_info['bbox']
                        foot_pos = foot_position_of_bbox(bbox)
                        foot_pos = tuple(map(int, foot_pos))
                        
                        card_width = 80
                        card_height = 40
                        card_x = foot_pos[0] - card_width // 2
                        card_y = foot_pos[1] + 5  # Just below feet
                        
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (card_x, card_y),
                                    (card_x + card_width, card_y + card_height),
                                    (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                        
                        speed_color = self._get_speed_color(speed)
                        cv2.rectangle(frame, (card_x, card_y),
                                    (card_x + card_width, card_y + 3),
                                    speed_color, -1, cv2.LINE_AA)
                        
                        cv2.rectangle(frame, (card_x, card_y),
                                    (card_x + card_width, card_y + card_height),
                                    (180, 180, 180), 1, cv2.LINE_AA)
                        
                        speed_text = f"{speed:.1f} km/h"
                        cv2.putText(frame, speed_text, (card_x + 6, card_y + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, speed_color, 2, cv2.LINE_AA)
                        
                        distance_text = f"{distance:.1f} m"
                        cv2.putText(frame, distance_text, (card_x + 6, card_y + 33),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 2, cv2.LINE_AA)
                    
            output_frames.append(frame)
        
        return output_frames

    def _get_speed_color(self, speed):
        if speed < 10:
            return (0, 0, 0)  
        elif speed < 20:
            return (255, 0, 0)  
        else:
            return (0, 255, 0)  