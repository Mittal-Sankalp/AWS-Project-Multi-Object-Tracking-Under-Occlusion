import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("store.mp4")

# Get video properties for the writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_store.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking using ByteTrack
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])

    # Plot the results on the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video file
    out.write(annotated_frame)

    # Display the output (optional while recording)
    cv2.imshow("Advanced Multi-Object Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything when finished
cap.release()
out.release()
cv2.destroyAllWindows()
print("Processed video saved successfully as output_store.mp4!")