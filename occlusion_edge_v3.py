import argparse
import os

import cv2 # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def gen_occl_mask(background_frame, current_frame, roi_points=None, debug=False):
    """Return a soft visibility mask: 255 means ad visible, 0 occluded."""
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
        padding = 5
        x_min = max(int(points_array[:, 0].min()) - padding, 0)
        x_max = min(int(points_array[:, 0].max()) + padding + 1, frame_width)
        y_min = max(int(points_array[:, 1].min()) - padding, 0)
        y_max = min(int(points_array[:, 1].max()) + padding + 1, frame_height)

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

    # Strong Lab chroma changes are reliable foreground seeds.
    diff_a = cv2.absdiff(bg_a, cur_a)
    diff_b = cv2.absdiff(bg_b, cur_b)
    colour_diff = cv2.addWeighted(diff_a, 0.5, diff_b, 0.5, 0)
    _, colour_mask = cv2.threshold(colour_diff, 5, 255, cv2.THRESH_BINARY)

    # Current-frame intensity edges provide only a rough object location. They
    # may include shadows/reflections; GrabCut decides which regions survive.
    current_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)
    edge_mask = cv2.Canny(current_gray, 50, 150)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_edges = cv2.morphologyEx(
        edge_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2
    )
    # cv2.imshow("closed_edges", closed_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    contours, _ = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rough_silhouette = np.zeros_like(edge_mask)
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            cv2.drawContours(
                rough_silhouette, [contour], -1, 255, cv2.FILLED
            )
    # cv2.imshow("rough_silhouette", rough_silhouette)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # GrabCut labels:
    #   rough edge silhouette -> probable foreground
    #   Lab chroma difference -> definite foreground
    #   everything else       -> probable background
    grabcut_mask = np.full(
        current_crop.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8
    )
    grabcut_mask[rough_silhouette > 0] = cv2.GC_PR_FGD
    grabcut_mask[colour_mask > 0] = cv2.GC_FGD

    # The padded crop border is known background and guarantees that GrabCut
    # always has background samples for its colour model.
    grabcut_mask[[0, -1], :] = cv2.GC_BGD
    grabcut_mask[:, [0, -1]] = cv2.GC_BGD

    # Check the final GrabCut labels after the border assignment. Detections
    # located only on the crop border have just been overwritten as background.
    foreground_pixels = (grabcut_mask == cv2.GC_FGD) | (
        grabcut_mask == cv2.GC_PR_FGD
    )
    background_pixels = (grabcut_mask == cv2.GC_BGD) | (
        grabcut_mask == cv2.GC_PR_BGD
    )
    can_run_grabcut = np.any(foreground_pixels) and np.any(background_pixels)

    if can_run_grabcut:
        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            current_crop,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            3,
            cv2.GC_INIT_WITH_MASK,
        )

        occlusion_mask = np.where(
            (grabcut_mask == cv2.GC_FGD)
            | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        
    else:
        occlusion_mask = np.zeros_like(edge_mask)

    # Segment with padded context, then limit the completed mask to the ad.
    occlusion_mask = cv2.bitwise_and(occlusion_mask, local_roi_mask)

    # cv2.imshow("GrabCut occlusion", occlusion_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # A real motion-blurred boundary is partially foreground, not binary.
    occlusion_mask = cv2.GaussianBlur(occlusion_mask, (5, 5), 0)

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
    parser = argparse.ArgumentParser(description="GrabCut ad occlusion test")
    parser.add_argument("--vid", required=True, help="Path to video file")
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
    input_name = os.path.splitext(os.path.basename(args.vid))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{input_name}_edge_v3.mp4")
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    display = background.copy()
    cv2.namedWindow("Select 4 Points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select 4 Points", mouse_callback)
    print("\nClick 4 points CLOCKWISE around the ad plane.")
    print("Press r to reset or q to quit.\n")

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
            background, frame, roi_points=points)
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
