import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

# Load the model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("store.mp4")

# Dictionary to store tracking logs: {track_id: {"first_seen": frame_num, "last_seen": frame_num}}
active_tracks_log = {}

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or pd.isna(fps):
    fps = 30  # Fallback FPS if video metadata is missing

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time_sec = frame_count / fps

    # Run tracking using ByteTrack
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])

    # Extract IDs present in the current frame
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for tid in track_ids:
            if tid not in active_tracks_log:
                # First time this ID appears (Entry)
                active_tracks_log[tid] = {
                    "Track ID": int(tid),
                    "Entry Frame": frame_count,
                    "Entry Time (s)": round(current_time_sec, 2),
                    "Last Seen Frame": frame_count,
                    "Last Seen Time (s)": round(current_time_sec, 2)
                }
            else:
                # Update last seen frame/time while they are still in frame
                active_tracks_log[tid]["Last Seen Frame"] = frame_count
                active_tracks_log[tid]["Last Seen Time (s)"] = round(current_time_sec, 2)

    # Plot results and display
    annotated_frame = results[0].plot()
    cv2.imshow("Retail Tracking Analytics", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# --- POST-PROCESSING & EXCEL EXPORT ---
# Calculate total duration for each customer/track ID
data_rows = []
for tid, details in active_tracks_log.items():
    duration_sec = details["Last Seen Time (s)"] - details["Entry Time (s)"]
    data_rows.append({
        "ID": details["Track ID"],
        "Entry Time (s)": details["Entry Time (s)"],
        "Last Seen Time (s)": details["Last Seen Time (s)"],
        "Duration Inside Store (s)": round(duration_sec, 2)
    })

# Convert to a Pandas DataFrame
df = pd.DataFrame(data_rows)

# Export to an Excel spreadsheet
excel_filename = "retail_tracking_analytics.xlsx"
df.to_excel(excel_filename, index=False)

print(f"\n[INFO] Tracking session ended. Analytics successfully exported to '{excel_filename}'!")
print(df.head(10)) # Print first few rows to terminal
