import numpy as np
import cv2
import os
import csv

def read_info_file(filename):
    """
    Parse the DeepSORT info.txt, returning a list of
    (frame_number, obj_id, x1, y1, x2, y2) tuples.
    """
    data = []
    current_frame = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Frame:"):
                parts = line.split()
                current_frame = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else None
            else:
                vals = list(map(int, line.replace('[','').replace(']','').split()))
                if len(vals) == 5 and current_frame is not None:
                    obj_id, x1, y1, x2, y2 = vals
                    data.append((current_frame, obj_id, x1, y1, x2, y2))
    return data

def extract_info(data):
    """
    Build:
      - positions[obj_id]    = list of (cx, cy) across all frames
      - bboxes[obj_id]       = last-seen (x1, y1, x2, y2)
      - first_frames[obj_id] = frame index of first detection
    """
    positions    = {}
    bboxes       = {}
    first_frames = {}
    for frame, obj_id, x1, y1, x2, y2 in data:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        positions.setdefault(obj_id, []).append((cx, cy))
        bboxes[obj_id] = (x1, y1, x2, y2)
        first_frames.setdefault(obj_id, frame)
    return positions, bboxes, first_frames

def extract_frame(video_path, frame_number):
    """
    Capture exactly frame_number from video.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Couldn’t read frame {frame_number} from {video_path}")
    return frame

def plot_trail_with_image(
    obj_id,
    positions,
    bboxes,
    first_frame_idx,
    video_path,
    draw_trails=True,
    csv_writer=None
):
    """
    Left: original frame at first_frame_idx (uncropped).
    Right: white canvas same size showing object’s trail.
    """
    if obj_id not in positions:
        print(f"No data for Object ID {obj_id}")
        return

    # 1) load the specified frame
    first_frame = extract_frame(video_path, first_frame_idx)
    h, w = first_frame.shape[:2]

    # 2) draw bounding box on it
    if obj_id in bboxes:
        x1, y1, x2, y2 = bboxes[obj_id]
        cv2.rectangle(first_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(first_frame, f'ID: {obj_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 3) prepare trail canvas
    trail_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    trails     = positions[obj_id]
    total_dist = up = down = 0
    line_data  = []

    if draw_trails:
        for (px, py), (cx, cy) in zip(trails, trails[1:]):
            if abs(px - cx) < 50 and abs(py - cy) < 50:
                p = (int(px), int(py))
                c = (int(cx), int(cy))
                cv2.line(trail_img, p, c, (255, 0, 0), 1)
                line_data.append((p, c))
                d = np.hypot(cx - px, cy - py)
                total_dist += d
                up   += (cy < py)
                down += (cy > py)

    # 4) compute stats
    displacement = np.hypot(trails[-1][0] - trails[0][0],
                            trails[-1][1] - trails[0][1]) if trails else 0
    speed  = total_dist / 60.0
    status = "Motile" if total_dist > 3 else "Not Moving"

    # 5) side-by-side combine
    combined = np.hstack((first_frame, trail_img))

    # 6) overlay stats on the right half
    sx = w + 50
    sy = h - 50
    lh = 30
    stats = [
        f"ID: {obj_id}",
        f"Distance: {int(total_dist)}",
        f"Displacement: {int(displacement)}",
        f"Speed: {speed:.2f}",
        f"Up: {up} | Down: {down}",
        f"Status: {status}"
    ]
    for i, txt in enumerate(stats):
        y = sy - (len(stats) - i - 1) * lh
        cv2.putText(combined, txt, (sx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # 7) optional CSV
    if csv_writer:
        csv_writer.writerow([
            obj_id,
            int(total_dist),
            int(displacement),
            speed,
            up,
            down,
            status,
            line_data
        ])

    # 8) save
    base    = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = f"trail_result/{base}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/trail_{obj_id}.png"
    cv2.imwrite(out_path, combined)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    INFO_FILE  = 'prediction/axon_ICC_chan01_info.txt'
    VIDEO_PATH = 'Videos/axon_ICC_chan01.mp4'  # ← your video
    CSV_OUT    = 'tracking_data.csv'

    raw           = read_info_file(INFO_FILE)
    positions, bboxes, first_frames = extract_info(raw)

    with open(CSV_OUT, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Object ID", "Total Distance", "Displacement",
            "Speed", "Up Movements", "Down Movements",
            "Status", "Line Data"
        ])
        for obj_id, frame_idx in first_frames.items():
            plot_trail_with_image(
                obj_id,
                positions,
                bboxes,
                first_frame_idx=frame_idx,
                video_path=VIDEO_PATH,
                csv_writer=writer
            )
