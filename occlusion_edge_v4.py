import argparse
import os

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def gen_occl_mask(background_frame, current_frame, roi_points=None):
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
        padding = 20
        x_min = max(int(points_array[:, 0].min()) - padding, 0)
        x_max = min(int(points_array[:, 0].max()) + padding + 1, frame_width)
        y_min = max(int(points_array[:, 1].min()) - padding, 0)
        y_max = min(int(points_array[:, 1].max()) + padding + 1, frame_height)

        polygon_roi_mask.fill(0)
        cv2.fillPoly(polygon_roi_mask, [points_array], 255)

    background_crop = background_frame[y_min:y_max, x_min:x_max]
    current_crop = current_frame[y_min:y_max, x_min:x_max]
    local_roi_mask = polygon_roi_mask[y_min:y_max, x_min:x_max]

    # cv2.imshow("local_roi_mask", local_roi_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(np.shape(local_roi_mask))

    bg_lab = cv2.GaussianBlur(
        cv2.cvtColor(background_crop, cv2.COLOR_BGR2LAB), (3, 3), 0
    )
    cur_lab = cv2.GaussianBlur(
        cv2.cvtColor(current_crop, cv2.COLOR_BGR2LAB), (3, 3), 0
    )
    _, bg_a, bg_b = cv2.split(bg_lab)
    _, cur_a, cur_b = cv2.split(cur_lab)

    # Lab chroma changes provide filled foreground evidence.
    diff_a = cv2.absdiff(bg_a, cur_a)
    diff_b = cv2.absdiff(bg_b, cur_b)
    colour_diff = cv2.addWeighted(diff_a, 0.5, diff_b, 0.5, 0)
    _, colour_mask = cv2.threshold(colour_diff, 5, 255, cv2.THRESH_BINARY)
    # cv2.imshow("colour_mask", colour_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # GrabCut is used only to obtain a coarse foreground/background split from
    # the Lab change mask. Edge detection is deliberately applied afterwards.
    seed_region = colour_mask
    allowed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    allowed_region = cv2.dilate(seed_region, allowed_kernel, iterations=1)
    
    # GrabCut labels:
    #   outside allowed region -> definite background
    #   allowed margin         -> probable background
    #   Lab colour difference  -> definite foreground
    grabcut_mask = np.full(
        current_crop.shape[:2], cv2.GC_BGD, dtype=np.uint8
    )
    grabcut_mask[allowed_region > 0] = cv2.GC_PR_BGD
    grabcut_mask[colour_mask > 0] = cv2.GC_FGD

    # The padded crop border supplies definite background samples.
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

        grabcut_foreground = np.where(
            (grabcut_mask == cv2.GC_FGD)
            | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
    else:
        grabcut_foreground = np.zeros_like(colour_mask)

    # GrabCut supplies only the coarse foreground/background separation. Refine
    # its boundary using current-frame edges inside a narrow search band.
    current_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)
    refinement_gray = cv2.GaussianBlur(current_gray, (3, 3), 0)
    edge_mask = cv2.Canny(refinement_gray, 50, 150)
    # cv2.imshow("edge mask", edge_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    refinement_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_foreground = cv2.erode(
        grabcut_foreground, refinement_kernel, iterations=1
    )
    outer_foreground = cv2.dilate(
        grabcut_foreground, refinement_kernel, iterations=1
    )
    boundary_band = cv2.subtract(outer_foreground, sure_foreground)
    candidate_edges = cv2.bitwise_and(edge_mask, boundary_band)
    # cv2.imshow("candidate_edges", candidate_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    has_coarse_foreground = cv2.countNonZero(grabcut_foreground) > 0
    has_sure_foreground = cv2.countNonZero(sure_foreground) > 0
    has_candidate_edges = cv2.countNonZero(candidate_edges) > 0

    if has_coarse_foreground and has_sure_foreground and has_candidate_edges:
        # Build a gradient surface and reinforce only the Canny edges close to
        # the coarse GrabCut boundary. Watershed treats these as likely barriers.
        gradient_x = cv2.convertScaleAbs(
            cv2.Sobel(refinement_gray, cv2.CV_16S, 1, 0, ksize=3)
        )
        gradient_y = cv2.convertScaleAbs(
            cv2.Sobel(refinement_gray, cv2.CV_16S, 0, 1, ksize=3)
        )
        gradient = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
        gradient = cv2.max(gradient, candidate_edges)
        watershed_image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

        # Marker 1 is known background, marker 2 is known foreground, and zero
        # is the boundary band that watershed must divide using the edge surface.
        markers = np.zeros(current_crop.shape[:2], dtype=np.int32)
        markers[outer_foreground == 0] = 1
        markers[sure_foreground > 0] = 2
        markers[[0, -1], :] = 1
        markers[:, [0, -1]] = 1

        cv2.watershed(watershed_image, markers)
        refined_foreground = markers == 2
        inner_ridge = (markers == -1) & (grabcut_foreground > 0)
        occlusion_mask = np.where(
            refined_foreground | inner_ridge, 255, 0
        ).astype(np.uint8)
        occlusion_mask = cv2.bitwise_and(occlusion_mask, outer_foreground)
    else:
        # Very thin or tiny components may disappear during erosion. Preserve
        # GrabCut's coarse result instead of dropping those components entirely.
        occlusion_mask = grabcut_foreground

    # cv2.imshow("occlusion_mask_before", occlusion_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # The edge-refined result is finally limited to the exact ad polygon.
    occlusion_mask = cv2.bitwise_and(occlusion_mask, local_roi_mask)
    # cv2.imshow("occlusion_mask_after", occlusion_mask)
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
    parser = argparse.ArgumentParser(
        description="GrabCut foreground extraction with edge-guided refinement"
    )
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
    output_path = os.path.join(OUTPUT_DIR, f"{input_name}_edge_v4.mp4")
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
