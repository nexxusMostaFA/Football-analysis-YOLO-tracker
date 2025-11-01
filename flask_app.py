from flask import Flask, request, jsonify, send_file
import os
import uuid
import cv2
import numpy as np
from main import *  

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/")
def index():
    return jsonify({"message": "Football Analysis API is running"})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        video_file = request.files["video"]
        video_id = str(uuid.uuid4())

        input_path = os.path.join(UPLOAD_DIR, f"{video_id}_input.mp4")
        output_path = os.path.join(UPLOAD_DIR, f"{video_id}_output.avi")

        video_file.save(input_path)

        from utils import read_video, save_video
        from tracker import Tracker
        from assign_players_teams import PlayerTeamAssigner
        from ball_assigner import BallAssigner
        from transformer import Transformer
        from speed_and_distance import SpeedAndDistance
        from camera_movment_calc import CameraMovmentCalculator

        frames = read_video(input_path)
        standard_size = (1920, 1080)
        frames = [cv2.resize(f, standard_size) for f in frames]

        model_path = r"models/best.pt"
        tracker = Tracker(model_path)
        tracks = tracker.get_tracks(frames, stub=False)
        tracks = tracker.find_position_tracks(tracks)
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        player_team_assigner = PlayerTeamAssigner()
        player_team_assigner.assign_colors(frames[0], tracks["players"][0])

        for frame_num, player in enumerate(tracks["players"]):
            for track_id, track in player.items():
                bbox = track["bbox"]
                team_id = player_team_assigner.assign_teams(frames[frame_num], bbox, track_id)
                tracks["players"][frame_num][track_id]["team_id"] = team_id
                color = player_team_assigner.teams_dict[team_id]
                tracks["players"][frame_num][track_id]["color"] = color

        ball_assigner = BallAssigner()
        team_ball_control = []

        for frame_num, player_track in enumerate(tracks["players"]):
            ball_bbox = tracks["ball"][frame_num][1]["bbox"]
            assigned_player = ball_assigner.ball_assigner(player_track, ball_bbox)
            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(tracks["players"][frame_num][assigned_player]["team_id"])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

        team_ball_control = np.array(team_ball_control)

        camera_movment_calculator = CameraMovmentCalculator(frames[0])
        camera_movments = camera_movment_calculator.calculate_camera_movment(frames)
        tracks = camera_movment_calculator.add_camera_movments_to_tracks(tracks, camera_movments)
        frames = camera_movment_calculator.draw_camera_movement(frames, camera_movments)

        transformer = Transformer()
        tracks = transformer.add_transformed_points_to_tracks(tracks)

        speed_and_distance_calculator = SpeedAndDistance()
        tracks = speed_and_distance_calculator.calculate_speed_and_distance(tracks)

        frames = tracker.draw_annotations(frames, tracks, team_ball_control)
        frames = speed_and_distance_calculator.draw_speed_and_distance(frames, tracks)

        save_video(frames, output_path)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
