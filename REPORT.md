# Multi-Object Tracking Under Occlusion for Retail Analytics: Technical Report

**Author:** Sankalp Mittal  
**Registration Number:** 26BIT0031  
**Institution:** Vellore Institute of Technology, Vellore  

---

## 1. Introduction & Project Objective
In modern retail analytics, tracking customer movement and counting traffic via ceiling-mounted cameras is crucial for understanding store optimization, foot traffic patterns, and product engagement. However, traditional object detection pipelines suffer from identity switches and track fragmentation when customers cross paths and occlude each other. The objective of this project is to build a robust Multi-Object Tracking (MOT) pipeline that maintains consistent object identities even through temporary heavy occlusions and automatically logs analytics data.

## 2. Approach & Architecture
The system architecture decouples spatial detection from temporal association to ensure high processing speed and tracking stability:
* **Per-Frame Detection (YOLOv8):** A pretrained YOLOv8 model (`yolov8n.pt`) is utilized to detect humans (`class=0`) frame-by-frame. Using a pretrained model avoids the heavy resource overhead of training a custom detector from scratch while providing state-of-the-art detection accuracy.
* **Identity Association & Tracking (ByteTrack):** Instead of relying solely on high-confidence boxes, the pipeline integrates **ByteTrack** (`bytetrack.yaml`). ByteTrack matches both high- and low-score detection bounding boxes using motion prediction (Kalman filters) and the Hungarian matching algorithm. This prevents the loss of tracks when a person's features are partially obscured.
* **Track Buffer & Occlusion Handling:** A configurable `track_buffer` is implemented to keep lost tracks alive for a brief window of frames. If a customer reappears after being hidden behind a pillar or another shopper, they retain their original identification number rather than spawning a new one.

## 3. Key Engineering Decisions
* **Decoupling Detection from Tracking:** By treating detection and association as separate stages, the system can run efficiently at near real-time frame rates.
* **Choosing ByteTrack over Basic Trackers:** Standard trackers drop tracks immediately if detection confidence dips below a threshold during occlusion. Retaining low-confidence detections via ByteTrack is essential for high-density retail store environments.

## 4. Business Intelligence & Automated Data Logging
To translate raw computer vision frames into actionable retail insights, the pipeline features an automated analytics logging layer built with `pandas`:
* **Timestamp & Dwell Tracking:** The system dynamically records each unique customer's **Entry Time**, **Last Seen Time**, and computes their **Total Duration Inside Store**.
* **Automated Excel Export:** Upon session completion, all tracking events are packaged and exported directly into a structured spreadsheet (`retail_tracking_analytics.xlsx`), enabling store managers to analyze peak traffic hours, customer dwell times, and store occupancy metrics.

## 5. Evaluation & Metrics
Unlike standard object detection (which uses mAP), Multi-Object Tracking relies on tracking-specific metrics:
* **MOTA (Multiple Object Tracking Accuracy):** Evaluates overall performance by combining false positives, missed targets, and identity mismatches.
* **ID Switches:** Measures the frequency with which a tracked entity erroneously swaps ID numbers with another individual.
* **Execution Verification:** The pipeline processes the video stream, overlays persistent bounding box IDs, logs analytics records, and exports the annotated trajectory file to `output_store.mp4`.

## 6. Failure Case Analysis & Robustness
While testing under standard clips yields reliable performance, edge cases in production environments include:
1. **Extended Occlusion:** If an occlusion exceeds the tracker's buffer length, the ID expires, creating a new ID upon reappearance. (Mitigation: Increase `track_buffer` or integrate appearance-based re-ID models like BoT-SORT).
2. **Dense Crowding & Similar Appearance:** Spatial motion models can struggle when individuals wearing identical clothing cross paths at matching velocities. (Mitigation: Add appearance embeddings for color/clothing features).
