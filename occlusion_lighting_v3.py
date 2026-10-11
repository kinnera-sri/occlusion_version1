import cv2
import argparse
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def debug_stats(name, arr):
    print(f"{name}: shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()} mean={arr.mean():.3f}")


def gen_occl_mask(background_frame, current_frame, roi_points=None):
    # """
    # Generate a fast, high-quality occlusion mask using color subtraction before grayscale.
    # """ 
    # if roi_points is not None:
    #     # Store original shape
    #     original_height, original_width = background_frame.shape[:2]

    #     # get square bounding box of roi points
    #     x_coords = [p[0] for p in roi_points]
    #     y_coords = [p[1] for p in roi_points]

    #     x_min, x_max = int(max(min(x_coords) - 10, 0)), int(min(max(x_coords) + 10, original_width))
    #     y_min, y_max = int(max(min(y_coords) - 10, 0)), int(min(max(y_coords) + 10, original_height))

    #     background_frame = background_frame[y_min:y_max, x_min:x_max]

    #     current_frame = current_frame[y_min:y_max, x_min:x_max]

    # bg_lab = cv2.cvtColor(background_frame, cv2.COLOR_BGR2LAB)
    # cur_lab = cv2.cvtColor(current_frame, cv2.COLOR_BGR2LAB)

    # # Light blur to reduce noise without much latency
    # bg_lab = cv2.blur(bg_lab, (3, 3))
    # cur_lab = cv2.blur(cur_lab, (3, 3))

    # bg_l, bg_a, bg_b = cv2.split(bg_lab)
    # cur_l, cur_a, cur_b = cv2.split(cur_lab)

    # # cv2.imshow("Background L Channel", bg_l)
    # # cv2.imshow("Current L Channel", cur_l)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # diff_l = cv2.absdiff(bg_l, cur_l)
    # diff_a = cv2.absdiff(bg_a, cur_a)
    # diff_b = cv2.absdiff(bg_b, cur_b)

    # weights = [0.5, 0.25, 0.25]  # Adjust weights for L, A, B channels
    # weighted_diff = weights[0] * diff_l + weights[1] * diff_a + weights[2] * diff_b
    # weighted_diff = np.clip(weighted_diff, 0, 255).astype(np.uint8)
    # # cv2.imshow("Weighted Difference", weighted_diff)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
 
    # # Color subtraction (captures hue differences)
    # diff_mask = cv2.absdiff(bg_lab, cur_lab)

    # # Convert to grayscale after subtraction (aggregate color differences)
    # diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY)

    # # Threshold (tune threshold value for sensitivity)
    # _, diff_mask = cv2.threshold(weighted_diff, 7, 255, cv2.THRESH_BINARY)
    # # debug_stats("diff_mask (after threshold)", diff_mask)

    # # Morphological cleanup (fast & effective)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # # debug_stats("diff_mask (after open)", diff_mask)

    # diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # # debug_stats("diff_mask (after close)", diff_mask)
    # # print(f"diff_mask: shape={diff_mask.shape} dtype={diff_mask.dtype} min={diff_mask.min()} max={diff_mask.max()} mean={diff_mask.mean():.3f}")

    # if roi_points is not None:
    #     # Resize mask back to original frame size
    #     full_mask = np.zeros((original_height, original_width), dtype=np.uint8)
    #     full_mask[y_min:y_max, x_min:x_max] = diff_mask
    #     diff_mask = full_mask

    # # cv2.imshow("Occlusion Mask", diff_mask)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # # debug_stats("background_frame (cropped)", background_frame)
    # # debug_stats("current_frame (cropped)", current_frame)
    # # debug_stats("bg_l", bg_l)
    # # debug_stats("cur_l", cur_l)
    # # debug_stats("diff_l", diff_l)
    # # debug_stats("diff_a", diff_a)
    # # debug_stats("diff_b", diff_b)
    # # debug_stats("weighted_diff", weighted_diff)

    # return ~diff_mask

# -------------------
# original code below
# -------------------

    # """
    # Generate a fast, high-quality occlusion mask using color subtraction before grayscale.
    # """

    if roi_points is not None:
        # Store original shape
        original_height, original_width = background_frame.shape[:2]
        # get square bounding box of roi points
        x_coords = [p[0] for p in roi_points]
        y_coords = [p[1] for p in roi_points]
        x_min, x_max = int(max(min(x_coords) - 10, 0)), int(min(max(x_coords) + 10, original_width))
        y_min, y_max = int(max(min(y_coords) - 10, 0)), int(min(max(y_coords) + 10, original_height))
        background_frame = background_frame[y_min:y_max, x_min:x_max]
        current_frame = current_frame[y_min:y_max, x_min:x_max]

    # Light blur to reduce noise without much latency
    background_frame = cv2.blur(background_frame, (3, 3))
    current_frame = cv2.blur(current_frame, (3, 3))

    # Work in float32 so the channel sums do not saturate at 255 and all
    # arrays passed to absdiff have the same type.
    bg_b, bg_g, bg_r = (
        channel.astype(np.float32) for channel in cv2.split(background_frame)
    )
    cur_b, cur_g, cur_r = (
        channel.astype(np.float32) for channel in cv2.split(current_frame)
    )

    # Normalize each frame by its own brightness. Clamping the sums prevents
    # division by zero for black pixels.
    sum_bg = np.maximum(bg_b + bg_g + bg_r, 1.0)
    sum_cur = np.maximum(cur_b + cur_g + cur_r, 1.0)

    diff_b = cv2.absdiff(cur_b / sum_cur, bg_b / sum_bg)
    diff_g = cv2.absdiff(cur_g / sum_cur, bg_g / sum_bg)
    diff_r = cv2.absdiff(cur_r / sum_cur, bg_r / sum_bg)

    # Average the normalized colour differences, then return to an 8-bit
    # mask so the existing threshold and morphology can be kept unchanged.
    diff_mask = (diff_b + diff_g + diff_r) / 3.0
    diff_mask = np.clip(diff_mask * 255.0, 0, 255).astype(np.uint8)

    # Convert to grayscale after subtraction (aggregate color differences)
    # diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY)

    # Threshold (tune threshold value for sensitivity)
    _, diff_mask = cv2.threshold(diff_mask, 3, 255, cv2.THRESH_BINARY)

    # Morphological cleanup (fast & effective)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if roi_points is not None:
        # Resize mask back to original frame size
        full_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        full_mask[y_min:y_max, x_min:x_max] = diff_mask
        diff_mask = full_mask

    return ~diff_mask

# ============================================================
# SELECT 4 POINTS
# ============================================================

points = []
# display = dst.copy()

def mouse_callback(event, x, y, flags, param):
    global points, display

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            display, str(len(points)), (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.namedWindow("Select 4 Points", cv2.WINDOW_NORMAL)
        cv2.imshow("Select 4 Points", display)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracker test")
    parser.add_argument("--vid", required=True, help="Path to video file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.vid)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.vid}")

    ret, img0 = cap.read()

    # cap = cv2.VideoCapture(args.vid)
    # ret, img0 = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    height, width = img0.shape[:2]

    out = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, args.vid.split(os.sep)[-1].split(".")[0] + "_lighting_v3.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    display = img0.copy()

    cv2.namedWindow("Select 4 Points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select 4 Points", mouse_callback)

    print("\nClick 4 points CLOCKWISE:")
    print("Top-left -> Top-right -> Bottom-right -> Bottom-left\n")

    while True:
        cv2.imshow("Select 4 Points", display)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            points = []
            display = img0.copy()

        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        elif len(points) == 4:
            break

    cv2.destroyAllWindows()

    roi_vis = img0.copy()

    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(roi_vis, [pts], (0, 255, 0))   # solid green

    # cv2.imwrite(os.path.join(OUTPUT_DIR, "background_roi.png"), roi_vis)

    i = 0
    while True:
        ret, img1 = cap.read()

        if not ret:
            break

        i += 1

        occl_mask = gen_occl_mask(img0, img1, roi_points=points)

        # Visualization image
        vis = img1.copy()

        # Polygon ROI
        roi_mask = np.zeros(img1.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [pts], 255)

        # Only keep ROI pixels that are background (255 in occl_mask)
        visible_mask = cv2.bitwise_and(roi_mask, occl_mask)

        # Green overlay
        overlay = vis.copy()
        overlay[visible_mask == 255] = (0, 255, 0)

        # Blend with original frame
        alpha = 0.4
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        # cv2.imshow("Occlusion Mask", vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        out.write(vis)   
        # Optional: save the binary mask too
        # cv2.imwrite(
        #     os.path.join(OUTPUT_DIR, f"mask_{i:04d}.png"),
        #     occl_mask
        # )

    # cv2.imshow("Occlusion Mask", occl_mask)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
