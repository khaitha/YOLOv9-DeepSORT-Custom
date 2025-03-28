import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import csv

def read_data_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        if line.strip().startswith("Frame:"):
            continue
        cleaned_line = line.strip().replace('[', '').replace(']', '')
        data.append(list(map(int, cleaned_line.split())))
    return np.array(data)

def extract_positions(data):
    positions = {}
    bboxes = {}  # Dictionary to store bounding boxes

    for row in data:
        obj_id, x1, y1, x2, y2 = row
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if obj_id not in positions:
            positions[obj_id] = []
        positions[obj_id].append((cx, cy))

        # Store the latest bounding box for each obj_id
        bboxes[obj_id] = (x1, y1, x2, y2)

    return positions, bboxes

def plot_trail_with_image(obj_id, positions, bboxes, img_size=(2048, 2048), crop_size=600, draw_trails=True, csv_writer=None):
    if obj_id not in positions:
        print(f"No data found for Object ID {obj_id}.")
        return
    os.makedirs('trail_result', exist_ok=True)

    trail_img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
    trails = positions[obj_id]

    total_distance = 0
    up_movements = 0
    down_movements = 0
    line_data = []

    if draw_trails:
        for i in range(1, len(trails)):
            if trails[i - 1] is None or trails[i] is None:
                continue
            prev_x, prev_y = trails[i - 1]
            curr_x, curr_y = trails[i]

            if abs(prev_x - curr_x) < 50 and abs(prev_y - curr_y) < 50:
                start_point = (int(prev_x), int(prev_y))
                end_point = (int(curr_x), int(curr_y))
                cv2.line(trail_img, start_point, end_point, (255, 0, 0), 1)
                line_data.append((start_point, end_point))

                total_distance += np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)

                if curr_y < prev_y:
                    up_movements += 1
                elif curr_y > prev_y:
                    down_movements += 1

    displacement = np.sqrt((trails[-1][0] - trails[0][0]) ** 2 + (trails[-1][1] - trails[0][1]) ** 2) if trails else 0
    speed = total_distance / 60.0
    status = "Motile" if total_distance > 3 else "Not Moving"

    img_path = 'first_frames/reference_img.jpg'
    if os.path.exists(img_path):
        first_frame = cv2.imread(img_path)
        first_frame = cv2.resize(first_frame, (img_size[0], img_size[1]))
        if obj_id in bboxes:
            x1, y1, x2, y2 = bboxes[obj_id]
            cv2.rectangle(first_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(first_frame, f'ID: {obj_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        first_frame = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 200
        cv2.putText(first_frame, 'Image Not Found', (50, img_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    first_frame_cropped = first_frame[:, :-crop_size]
    trail_img_cropped = trail_img[:, crop_size + 150:]
    combined = np.hstack((first_frame_cropped, trail_img_cropped))

    start_x = combined.shape[1] - 500
    start_y = combined.shape[0] - 100
    line_height = 40

    cv2.putText(combined, f"ID: {obj_id}", (start_x, start_y - 5 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, f"Distance: {int(total_distance)}", (start_x, start_y - 4 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, f"Displacement: {int(displacement)}", (start_x, start_y - 3 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, f"Speed: {speed:.2f}", (start_x, start_y - 2 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, f"Up: {up_movements} | Down: {down_movements}", (start_x, start_y - line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, f"Status: {status}", (start_x, start_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    if csv_writer:
        csv_writer.writerow([obj_id, int(total_distance), int(displacement), speed, up_movements, down_movements, status, line_data])

    output_path = f'trail_result/trail_{obj_id}.png'
    cv2.imwrite(output_path, combined)
    print(f"Saved: {output_path}")

frames = read_data_from_file('info.txt')
positions, bboxes = extract_positions(frames)

with open('tracking_data.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["Object ID", "Total Distance", "Displacement", "Speed", "Up Movements", "Down Movements", "Status", "Line Data"])

    for obj_id in positions.keys():
        plot_trail_with_image(obj_id, positions, bboxes, csv_writer=csv_writer)
