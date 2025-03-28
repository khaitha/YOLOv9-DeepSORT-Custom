import cv2


def get_video_fps(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the FPS property
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()

    return fps


# Usage
video_path = 'data.mp4'
fps = get_video_fps(video_path)
print(f"The video FPS is: {fps}")
