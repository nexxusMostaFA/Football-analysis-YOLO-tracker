from ultralytics import YOLO
import supervision as sv
import pickle
import os

class Tracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
       
    def yolo_prediction(self, frames):
        batch_size = 5
        predictions = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i+batch_size]
            results = self.model.predict(batch_frames)
            predictions += results
            break
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
            ,'balls':[]
        }

        for frame_num , prediction_frame in enumerate(yolo_predictions):

            class_names = prediction_frame.names
            class_names_inv = { v:k for k , v in class_names.items()}
            print("#"*50)
            print("class_names",class_names)
            print("#"*50)
            print( "class_names inverse", class_names_inv)
            print("#"*50)
            supervision_detiction = sv.Detections.from_ultralytics(prediction_frame)
            print("#"*50)
            print("sv detiction" , supervision_detiction)
            print("#"*50)
            print("class_id " , supervision_detiction.class_id)
            print("#"*50)
            # convert goalkeeper to player
            for index , class_id in enumerate(supervision_detiction.class_id):
                if class_id == class_names_inv['goalkeeper']:
                    supervision_detiction.class_id[index] = class_names_inv['player']

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['balls'].append({})

            detection_with_tracks = self.tracker.update_with_detections(supervision_detiction)
            print("#"*50)
            print( 'detection_with_tracks' , detection_with_tracks)
            print("#"*50)

            for object in detection_with_tracks:
                bbox = object[0].tolist()
                class_id = object[3]
                print("#"*50)
                print( "type of class id", class_id.dtype)
                print("#"*50)
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
                    tracks['balls'][frame_num][track_id] = {'bbox' : bbox}

            if stub_path is not None:
                with open(stub_path , 'wb') as f:
                    pickle.dump(tracks , f)

            return tracks
        

    def find_position_tracks(self , tracks):
        for object , track_info in tracks.items:
            for index , frame_info in enumerate(track_info):
                for track_id , info in frame_info.items():
                    bbox = info['bbox']
                    if object == 'balls':
                        position = center_of_bbox(bbox)
                    else:
                        position = foot_position_of_bbox(bbox)
                    tracks[object][index][track_id]['position'] = position


        


                




          