import cv2
import os 

video_path = r"C:\Users\mostafa\Documents\GitHub\Football-analysis-YOLO-tracker\data\Input-Video.mp4"


def read_video(video_path):
    video_frames = cv2.VideoCapture(video_path)
    video = []
    while True:
        ret , frame = video_frames.read()
        if not ret:
            break
        video.append(frame)
    return video


def save_video(video , output_path , fbs = 30):
    codcc = cv2.VideoWriter_fourcc(*'XVID')
    video_frames = cv2.VideoWriter(output_path , codcc , fbs , (video[0].shape[1] , video[0].shape[0]))
    for frame in video:
        video_frames.write(frame)
    video_frames.release()

   
        



    