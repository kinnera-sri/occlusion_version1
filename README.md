# Occlusion Detection

Experiments for placing a green test ad inside a manually selected video region while keeping foreground objects in front of it.

## Main scripts

- `occlusion.py`: original BGR frame-difference baseline.
- `occlusion_lighting.py` and `occlusion_lighting_v2.py`: LAB chroma difference with different distance calculations and thresholds.
- `occlusion_lighting_v3.py`: normalized BGR colour comparison.
- `occlusion_lighting_optical_flow.py`: lighting-based detection with optical-flow-assisted tracking.
- `occlusion_edge.py` through `occlusion_edge_v5.py`: edge-based experiments, progressing from contour filling to GrabCut and boundary refinement. Version 5 is the latest, not a proven accuracy winner.

## Run

Requires Python 3, NumPy, and OpenCV with desktop window support.

```bash
python3 -m pip install numpy opencv-python
python3 occlusion_edge_v5.py --vid /path/to/video.mp4
```

Click four points clockwise around the ad region in the first frame. During selection, press `r` to reset or `q` to quit. Processing starts after the fourth click. Replace the script name to try another main method.

The example writes `results/<video_name>_edge_v5.mp4`. Re-running the same method on the same input name overwrites its output.

## Results and report

- `results/`: generated output videos.
- `results/final_results/`: collected comparison videos.
- [Occlusion Detection report](Occlusion%20Detection%20-%20lighting%20and%20edge%20report.docx): detailed methods, timestamped screenshots, and conclusions in editable Word format.

## Findings and limitations

Lighting-aware comparisons preserve the background better but can miss similarly coloured objects. Selective edges reduce false detections; aggressive contour filling can instead mask static background details. The screenshot comparisons are qualitative, with no established overall winner.

The main methods assume a fixed camera, a fixed ad region, and the first frame as the background reference. Camera motion, lighting changes, weak object contrast, and frame-to-frame mask flicker remain limitations.
