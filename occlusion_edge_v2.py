import argparse
import os

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def gen_occl_mask(background_frame, current_frame, roi_points=None, debug=False):
    """Return a visibility mask: 255 means the ad is visible, 0 is occluded."""
    if background_frame is None or current_frame is None:
        raise ValueError("Background and current frames must not be None.")
    if background_frame.shape != current_frame.shape:
        raise ValueError("Background and current frames must have the same shape.")

    frame_height, frame_width = background_frame.shape[:2]
    x_min, y_min, x_max, y_max = 0, 0, frame_width, frame_height
    polygon_roi_mask = np.full((frame_height, frame_width), 255, dtype=np.uint8)

    if roi_points is not None:
        if len(roi_points) < 3:
            raise ValueError("At least three ROI points are required.")

        points_array = np.asarray(roi_points, dtype=np.int32)
        x_min = max(int(points_array[:, 0].min()) - 10, 0)
        x_max = min(int(points_array[:, 0].max()) + 11, frame_width)
        y_min = max(int(points_array[:, 1].min()) - 10, 0)
        y_max = min(int(points_array[:, 1].max()) + 11, frame_height)

        polygon_roi_mask.fill(0)
        cv2.fillPoly(polygon_roi_mask, [points_array], 255)

    background_crop = background_frame[y_min:y_max, x_min:x_max]
    current_crop = current_frame[y_min:y_max, x_min:x_max]
    local_roi_mask = polygon_roi_mask[y_min:y_max, x_min:x_max]

    bg_lab = cv2.GaussianBlur(
        cv2.cvtColor(background_crop, cv2.COLOR_BGR2LAB), (3, 3), 0
    )
    cur_lab = cv2.GaussianBlur(
        cv2.cvtColor(current_crop, cv2.COLOR_BGR2LAB), (3, 3), 0
    )
    _, bg_a, bg_b = cv2.split(bg_lab)
    _, cur_a, cur_b = cv2.split(cur_lab)

    # Foreground evidence: pixels whose chroma differs from the background.
    diff_a = cv2.absdiff(bg_a, cur_a)
    diff_b = cv2.absdiff(bg_b, cur_b)
    colour_diff = cv2.addWeighted(diff_a, 0.5, diff_b, 0.5, 0)

    # Colour-change pixels are the initial foreground mask.
    _, colour_mask = cv2.threshold(
        colour_diff, 5, 255, cv2.THRESH_BINARY
    )

    # Use a/b edges from the current frame. Unlike L-channel edges, these are
    # much less responsive to neutral shadows and illumination changes.
    edges_a = cv2.Canny(cur_a, 20, 60)
    edges_b = cv2.Canny(cur_b, 20, 60)
    chroma_edges = cv2.bitwise_or(edges_a, edges_b)

    # Edges refine the colour mask; unrelated edges are discarded.
    seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    seed_neighbourhood = cv2.dilate(colour_mask, seed_kernel, iterations=1)
    relevant_edges = cv2.bitwise_and(chroma_edges, seed_neighbourhood)

    # Keep colour evidence and add a narrow band around supporting edges.
    edge_band = cv2.dilate(
        relevant_edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    occlusion_mask = cv2.bitwise_or(colour_mask, edge_band)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    occlusion_mask = cv2.morphologyEx(
        occlusion_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1
    )

    # The padded crop is analysis context. Clip to the ad polygon only after
    # segmentation, so a boundary just outside the ad can still support it.
    occlusion_mask = cv2.bitwise_and(occlusion_mask, local_roi_mask)

    if debug:
        cv2.imshow("01 colour difference", colour_diff)
        cv2.imshow("02 colour mask", colour_mask)
        cv2.imshow("03 relevant edges", relevant_edges)
        cv2.imshow("04 occlusion mask", occlusion_mask)
        cv2.waitKey(1)

    full_occlusion_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    full_occlusion_mask[y_min:y_max, x_min:x_max] = occlusion_mask
    return cv2.bitwise_not(full_occlusion_mask)


points = []
display = None


def mouse_callback(event, x, y, flags, param):
    global display
    if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
        return

    points.append([x, y])
    cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(
        display,
        str(len(points)),
        (x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Select 4 Points", display)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma-edge ad occlusion test")
    parser.add_argument("--vid", required=True, help="Path to video file")
    parser.add_argument("--debug", action="store_true", help="Show live mask diagnostics")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.vid)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.vid}")

    ret, background = cap.read()
    if not ret or background is None:
        cap.release()
        raise RuntimeError(f"Could not read the first frame from: {args.vid}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    height, width = background.shape[:2]
    output_path = os.path.join(OUTPUT_DIR, args.vid.split(os.sep)[-1].split(".")[0] + "_edge_v2.mp4")
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    display = background.copy()
    cv2.namedWindow("Select 4 Points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select 4 Points", mouse_callback)
    print("\nClick 4 points CLOCKWISE: top-left -> top-right -> bottom-right -> bottom-left")
    print("Press r to reset, or q to quit.\n")

    while len(points) < 4:
        cv2.imshow("Select 4 Points", display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
            display = background.copy()
        elif key == ord("q"):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)

    cv2.destroyWindow("Select 4 Points")
    polygon = np.asarray(points, dtype=np.int32)
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [polygon], 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        visible_mask = gen_occl_mask(
            background, frame, roi_points=points, debug=args.debug
        )
        visible_mask = cv2.bitwise_and(visible_mask, roi_mask)

        green_ad = np.full_like(frame, (0, 255, 0))
        alpha = 0.4 * (visible_mask.astype(np.float32) / 255.0)
        composited = (
            green_ad * alpha[..., np.newaxis]
            + frame * (1.0 - alpha[..., np.newaxis])
        ).astype(np.uint8)
        out.write(composited)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
