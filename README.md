# AWS-Project-Multi-Object-Tracking-Under-Occlusion
Developed an end-to-end computer vision pipeline using Python, OpenCV, YOLOv8, and ByteTrack to count and track retail customers via ceiling cameras under occlusion. The system uses Kalman filters for ID consistency, evaluates performance using MOTA and ID switches, and includes failure analysis for dense retail environments.
# Multi-Object Tracking Under Occlusion (Retail Analytics)

## Scenario & Objective
A retail store requires a computer vision pipeline to count and track customers using ceiling-mounted cameras. Because customers frequently cross paths and block each other from view, standard frame-by-frame object detectors break down. 
**Objective:** Build a tracking pipeline that detects people per frame and maintains consistent identity assignments across frames, even through brief periods of occlusion.

---

## Dataset
* **Benchmark:** MOT17 or MOT20 Multiple Object Tracking Benchmark (or custom retail video feeds).

---

## Technical Approach & Architecture

1. **Per-Frame Detection:** 
   * Uses a pretrained **YOLOv8** model (`yolov8n.pt`) to detect humans (`class=0`) with high accuracy and speed without needing to train a detector from scratch[cite: 1].
2. **Tracking & ID Association:** 
   * Implements **ByteTrack** (`bytetrack.yaml`) combined with Kalman filters for motion prediction and the Hungarian algorithm for data association[cite: 1]. 
   * Unlike basic trackers that discard low-confidence detection boxes, ByteTrack retains them to recover track fragments during heavy occlusion.
3. **Graceful Track Loss Handling:** 
   * Leverages a track buffer (`track_buffer`) to remember lost tracks temporarily so that when a person reappears after being occluded, they retain their original ID rather than spawning a new one[cite: 1].

---

## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Mittal-Sankalp/AWS-Project-Multi-Object-Tracking-Under-Occlusion.git](https://github.com/Mittal-Sankalp/AWS-Project-Multi-Object-Tracking-Under-Occlusion.git)
   cd your-repo-name
