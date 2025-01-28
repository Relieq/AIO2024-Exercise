import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("samples/highway.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    cap.get(cv2.CAP_PROP_FPS)
)

# Define region points
# region_points = [(20, 400), (1080, 400)] # For line counting
region_points = [
    (430, 700),
    (1600, 700),
    (1600, 1080),
    (430, 1080),
]  # For rectangle region counting: top left, top right, bottom right, bottom left

# Video writer
video_writer = cv2.VideoWriter(
    "./run/highway_counted.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=False,  # Display the output
    region=region_points,  # Pass region points
    model="yolo11x.pt"  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()