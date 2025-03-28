import numpy as np
import plotly.graph_objects as go


# Read data from file
def read_data_from_file(filename):
    frame = 0
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Process each line to remove unwanted characters and convert to integers
        data = []
        for line in lines:
            if line.strip().startswith("Frame:"):
                continue
            # Remove square brackets and extra spaces
            cleaned_line = line.strip().replace('[', '').replace(']', '')
            # Split the line into elements and convert them to integers
            data.append(list(map(int, cleaned_line.split())))
            #print('Frame: ', frame)
            frame += 1
    return np.array(data)


# Extract positions from frames
def extract_positions(data):
    positions = {}
    for row in data:
        obj_id, x1, y1, x2, y2 = row
        print("Object Id:", obj_id, x1, y1, x2, y2)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # Compute center

        if obj_id not in positions:  # Ensure obj_id exists in the dictionary
            positions[obj_id] = []
        positions[obj_id].append((cx, cy))
    print("Length of Positions:", len(positions))
    return positions


# Read all frames from info.txt
frames = read_data_from_file('info.txt')
positions = extract_positions(frames)

# Create an interactive animation
fig = go.Figure()

# Initialize traces for all objects
traces = {}
for obj_id, coords in positions.items():
    x_vals, y_vals = zip(*coords)
    trace = go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name=f'ID {obj_id}')
    traces[obj_id] = trace
    fig.add_trace(trace)

# Prepare frames for animation
print("Positions: ", len(positions))
animation_frames = []

# Find the maximum number of frames across all objects
num_frames = max(len(coords) for coords in positions.values())
print("The number of Frames:", num_frames)

# Store last known positions for objects without updates
last_positions = {obj_id: None for obj_id in positions}

for frame_idx in range(num_frames):
    frame_data = []

    for obj_id, coords in positions.items():
        if frame_idx < len(coords):  # If the object has a position update
            last_positions[obj_id] = coords[frame_idx]  # Update last known position
        elif last_positions[obj_id] is None:  # If no updates have ever been received
            continue  # Skip this object

        # Extract x and y values up to the current frame
        x_vals, y_vals = zip(*coords[:frame_idx + 1]) if frame_idx < len(coords) else zip(*coords)
        y_vals =[-y for y in y_vals]
        frame_data.append(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name=f'ID {obj_id}'))

    animation_frames.append(go.Frame(
        data=frame_data,
        name=str(frame_idx)
    ))

# Update layout for the animation
fig.update_layout(
    title='Trajectory of Objects (Time Lapse)',
    xaxis_title='X Coordinate',
    yaxis_title='Y Coordinate',
    showlegend=True,
    hovermode='closest',
    updatemenus=[{
        'buttons': [
            {
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 100, 'redraw': True},
                    'fromcurrent': True,
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }],
    sliders=[{
        'active': 0,
        'currentvalue': {'font': {'size': 20}, 'prefix': 'Frame: ', 'visible': True},
        'pad': {'b': 10},
        'steps': [{
            'args': [
                [str(i)],
                {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate'}
            ],
            'label': str(i),
            'method': 'animate'
        } for i in range(num_frames)]
    }]
)

# Set frames
fig.frames = animation_frames

# Save the animation as an interactive HTML file
fig.write_html("trajectory_with_timelapse.html")
print(
    "Time-lapse interactive trajectory saved as 'trajectory_with_timelapse.html'. Open this file in a browser to explore.")
