# 🏟️ Football Analysis YOLO Tracker  

An advanced **football (soccer) analysis system** powered by **YOLOv8**, **ByteTrack**, and **OpenCV**.  
This project detects, tracks, and analyzes players, referees, and the ball to extract rich insights from football videos — such as **speed, distance, team control, and camera movement** — and produces a fully **annotated output video**.

---

## 🚀 Features  

- ✅ Player, referee, and ball detection using YOLOv8  
- ✅ Multi-object tracking with ByteTrack  
- ✅ Automatic team color detection (K-Means clustering)  
- ✅ Ball possession and team control visualization  
- ✅ Speed & distance estimation (meters + km/h)  
- ✅ Camera movement compensation  
- ✅ Perspective transformation (bird’s-eye correction)  
- ✅ Output annotated video with all overlays  
- ✅ Optional Flask

---

## 🧠 System Overview  

Input Video
↓
YOLOv8 Detection → Tracker (ByteTrack)
↓
Player & Ball Tracks
↓
Camera Movement Estimator → View Transformer
↓
Speed & Distance Calculator → Team Assigner → Ball Control
↓
Annotated Output Video (.avi or .mp4)

yaml
Copy code

---

## 🛠️ Installation  

### 1️⃣ Clone the Repository 

git clone https://github.com/yourusername/Football-analysis-YOLO-tracker.git
cd Football-analysis-YOLO-tracker


### 1️⃣ Clone the Repository 
python -m venv venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # macOS/Linux

### 3️⃣ Install Dependencies
pip install -r requirements.txt

# 🎬 Usage
▶ Step 1 — Add your input video

Place your football match video in:

data/input_video.mp4

▶ Step 2 — Run the Analysis
python main.py

▶ Step 3 — Output

The annotated video will be automatically saved to:

data/Output-Video.avi

### 📂 Project Structure
FOOTBALL-ANALYSIS-YOLO-TRACKER/
│
├── assign_players_teams/
│   ├── __init__.py
│   ├── analysis_player_assigner.ipynb
│   └── player_team_assigner.py
│
├── ball_assigner/
│   ├── __init__.py
│   └── ball_assigner.py
│
├── camera_movment_calc/
│   ├── __init__.py
│   └── camera_movment.py
│
├── data/
│   ├── camera_movments_stub.pkl
│   ├── cropped_image.jpg
│   ├── input_video.mp4
│   ├── Output-Video.avi
│   └── tracks_stub.pkl
│
├── models/
│   └── best.pt                     # YOLOv8 model weights
│
├── speed_and_distance/
│   ├── __init__.py
│   └── speed_and_distance.py
│
├── tracker/
│   ├── __init__.py
│   └── tracker.py
│
├── transformer/
│   ├── __init__.py
│   └── transformer.py
│
├── uploads/
│   ├── 243248f3-18c7-4cfb-9007-38034d15a59a_input.mp4
│   └── 243248f3-18c7-4cfb-9007-38034d15a59a_output.avi
│
├── utils/
│   ├── __init__.py
│   ├── bbox_utils.py
│   └── video_utils.py
│
├── venv/                           # virtual environment
├── .gitignore
├── flask_app.py                    # Flask web API for uploads & analysis
├── main.py                         # Main analysis pipeline
├── training.ipynb                  # YOLO training / experimentation
├── requirements.txt                # Dependencies list
└── README.md                       # Project documentation

### 🌐 Optional: Run as a Flask API

You can analyze uploaded videos directly via a web endpoint.

Example: flask_app.py
from flask import Flask, request, send_file
from main import main

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_video():
    video = request.files['video']
    input_path = 'uploads/input_video.mp4'
    output_path = 'uploads/output_video.avi'
    video.save(input_path)
    main(input_path, output_path)
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)


Run:

## python flask_app.py


Then test with:

curl -X POST -F "video=@data/input_video.mp4" http://127.0.0.1:5000/analyze --output result.avi

# 📊 Output Visualization

# 🟢 Ball → green triangle

# 🔵Player → ellipse with team color

# 🟠 Referee → yellow ellipse

# ⚡ Speed & Distance displayed on screen

# 🎯 Team Control Panel shows real-time ball possession

# 🎥 Camera Movement Overlay displays X & Y movement