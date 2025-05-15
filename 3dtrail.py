import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def read_data_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = []
    frame_idx = -1
    for line in lines:
        if line.strip().startswith("Frame:"):
            frame_idx += 1  # Increment frame index when a new frame starts
            continue
        cleaned_line = line.strip().replace('[', '').replace(']', '')
        values = list(map(int, cleaned_line.split()))
        if values:
            data.append([frame_idx] + values)  # Prepend frame index
    return np.array(data)

def extract_positions(data):
    positions = {}

    for row in data:
        frame, obj_id, x1, y1, x2, y2 = row
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if obj_id not in positions:
            positions[obj_id] = []
        positions[obj_id].append((cx,frame,cy))  # Store X, Y, Time

    return positions

def plot_3d_trails_fixed(positions):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Assign random colors for different objects
    colors = {obj_id: (random.random(), random.random(), random.random()) for obj_id in positions.keys()}

    # Create trails for each object
    for obj_id, trail in positions.items():
        trail = np.array(trail)
        x, y, z = trail[:, 0], trail[:, 1], trail[:, 2]

        # Make each trail a horizontal line (use the same Z axis for all points of one object)
        ax.scatter(x, y, z, color=colors[obj_id], label=f'Object {obj_id}', alpha=0.7)

        # Connect the points with lines to show the trajectory
        ax.plot(x, y, z, linestyle='-', color=colors[obj_id], alpha=0.6)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Object ID (Time Layers)')
    ax.set_title('3D Trail Visualization with Fixed Spatial Coordinates')
    ax.legend()

    plt.show()

# Read data and extract positions
frames = read_data_from_file('info.txt')
positions = extract_positions(frames)

# Generate 3D trail visualization with fixed positions
plot_3d_trails_fixed(positions)
