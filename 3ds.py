import streamlit as st
import numpy as np
import plotly.graph_objects as go

def read_data_from_file(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.splitlines()
    data = []
    frame_idx = -1
    for line in lines:
        if line.strip().startswith("Frame:"):
            try:
                # Try to parse the frame number from the line
                frame_idx = int(line.split()[1])
            except Exception:
                frame_idx += 1  # fallback if parsing fails
            continue
        cleaned_line = line.strip().replace('[', '').replace(']', '')
        if cleaned_line:
            try:
                values = list(map(int, cleaned_line.split()))
                # Prepend the frame number: [frame, id, x1, y1, x2, y2]
                data.append([frame_idx] + values)
            except Exception as e:
                st.write("Failed to parse line:", line, e)
    return np.array(data)

def extract_positions(data):
    positions = {}
    for row in data:
        frame, obj_id, x1, y1, x2, y2 = row
        # Compute center of bounding box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if obj_id not in positions:
            positions[obj_id] = []
        # Store as (frame, cx, cy) so time is along the x-axis
        positions[obj_id].append((frame, cx, cy))
    return positions

def plot_3d_trails_plotly(positions, gap_threshold=3):
    fig = go.Figure()

    for obj_id, points in positions.items():
        # Sort points by frame number
        sorted_points = sorted(points, key=lambda x: x[0])
        frames = []
        x_centers = []
        y_centers = []
        prev_frame = None
        for point in sorted_points:
            frame, cx, cy = point
            if prev_frame is not None:
                # If the gap between this point and the previous exceeds the threshold,
                # insert a None to break the line.
                if frame - prev_frame > gap_threshold:
                    frames.append(None)
                    x_centers.append(None)
                    y_centers.append(None)
            frames.append(frame)
            x_centers.append(cx)
            y_centers.append(cy)
            prev_frame = frame

        # Plot using a single trace per object, which now has gaps (None values)
        fig.add_trace(go.Scatter3d(
            x=frames,          # x-axis: Time (Frame)
            y=x_centers,       # y-axis: X Center (spatial)
            z=y_centers,       # z-axis: Y Center (spatial)
            mode='lines+markers',
            name=f'Object {obj_id}',
            line=dict(width=4),
            marker=dict(size=1)
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Time (Frame)',
            yaxis_title='X Center',
            zaxis_title='Y Center',
            aspectmode='cube'  # enforce uniform scaling on all axes
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title='3D Organelle Trails'
    )
    return fig

# Streamlit UI
st.title("3D Cell Organelle Trail Visualization")

uploaded_file = st.file_uploader("Upload your organelle trail file (info.txt)", type=["txt"])
if uploaded_file is not None:
    data = read_data_from_file(uploaded_file)
    positions = extract_positions(data)
    st.write("Sample computed positions:", {k: positions[k][:5] for k in positions})
    fig = plot_3d_trails_plotly(positions)
    st.plotly_chart(fig)
