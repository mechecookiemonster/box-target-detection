# box-target-detection

Box and Target Detection
Real-time computer vision module built for a contest. Detects a colored box via HSV segmentation and an ArUco fiducial marker simultaneously from a live camera feed, computes the pixel-space displacement between them, and annotates the frame with live distance readout.

## Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Built as my contribution to a team project — the pipeline was later integrated and extended by the team for the full robot system.

## Stack: Python, OpenCV

## Intended hardware: Raspberry Pi camera
