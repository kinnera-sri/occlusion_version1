import cv2
import argparse
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def debug_stats(name, arr):
    print(f"{name}: shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()} mean={arr.mean():.3f}")

def srgb_to_linear(c):
    """sRGB [0,1] -> linear light [0,1]."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def gen_occl_mask(background_frame, current_frame, roi_points=None):

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

    bg_lab = cv2.cvtColor(background_frame, cv2.COLOR_BGR2LAB)
    cur_lab = cv2.cvtColor(current_frame, cv2.COLOR_BGR2LAB)

    # Light blur to reduce noise without much latency
    background_frame = cv2.blur(bg_lab, (3, 3))
    current_frame = cv2.blur(cur_lab, (3, 3))

    cur_l, cur_a, cur_b = cv2.split(current_frame)
    bg_l, bg_a, bg_b = cv2.split(background_frame)

    # cv2.imshow("Background L Channel", bg_l)
    # cv2.waitKey(0)
    # cv2.imshow("Current L Channel", cur_l)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("Background A Channel", bg_a)
    # cv2.waitKey(0)
    # cv2.imshow("Current A Channel", cur_a)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("Background B Channel", bg_b)
    # cv2.waitKey(0)
    # cv2.imshow("Current B Channel", cur_b)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    diff_a = cv2.absdiff(bg_a, cur_a)
    diff_b = cv2.absdiff(bg_b, cur_b)

    # if diff_a.max() > 0:
    #     diff_a = (diff_a / diff_a.max() * 255).astype(np.uint8)
    # if diff_b.max() > 0:
    #     diff_b = (diff_b / diff_b.max() * 255).astype(np.uint8)

    # cv2.imshow("diff a", diff_a)
    # cv2.waitKey(0)
    # cv2.imshow("diff b", diff_b)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # take average of diff_a and diff_b to get a single channel mask
    diff_mask = cv2.addWeighted(diff_a, 0.5, diff_b, 0.5, 0)

    # binarize the color diff so it's an actual "changed" mask, not raw grayscale
    _, diff_bin = cv2.threshold(diff_mask, 0, 255, cv2.THRESH_BINARY)

    edge_mask = cv2.Canny(current_frame, 50, 150)

    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)  # bridge small gaps

    combined = cv2.bitwise_or(diff_bin, edge_mask)
    # cv2.imshow("combined", combined)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros_like(combined)

    if contours:
        min_area = 0.01 * combined.shape[0] * combined.shape[1]  # tune to your object size
        good = [c for c in contours if cv2.contourArea(c) > min_area]
        cv2.drawContours(final_mask, good, -1, 255, thickness=cv2.FILLED)

    # cv2.imshow("final_mask", final_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

    if roi_points is not None:
        # Resize mask back to original frame size
        full_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        full_mask[y_min:y_max, x_min:x_max] = final_mask
        final_mask = full_mask

    return ~final_mask

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
        os.path.join(OUTPUT_DIR, args.vid.split(os.sep)[-1].split(".")[0] + "_edge.mp4"),
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
        # cv2.imshow("img1", img1)
        # cv2.waitKey(0)

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