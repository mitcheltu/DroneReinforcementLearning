"""Generate an offline 3D trajectory player from a notebook diagnostic trace."""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from training.physics.quaternion import rotation


def visualize(path, output):
    with np.load(path, allow_pickle=False) as archive:
        states = archive["states"].copy()
        metadata = json.loads(str(archive["metadata_json"]))
    data = []
    for gate in metadata["course"]["gates"]:
        yaw = gate["yaw_rad"]
        horizontal = np.array([-np.sin(yaw), np.cos(yaw), 0]) * gate["width_m"] / 2
        vertical = np.array([0, 0, gate["height_m"] / 2])
        center = np.array(gate["center_m"])
        corners = np.array(
            [
                center - horizontal - vertical,
                center + horizontal - vertical,
                center + horizontal + vertical,
                center - horizontal + vertical,
                center - horizontal - vertical,
            ]
        )
        data.append(
            go.Scatter3d(
                x=corners[:, 0],
                y=corners[:, 1],
                z=corners[:, 2],
                mode="lines",
                name=f"Gate {gate['label']}",
                line={"width": 6},
            )
        )
    data.append(
        go.Scatter3d(
            x=states[:, 1],
            y=states[:, 2],
            z=states[:, 3],
            mode="lines",
            name="Recorded flight",
            line={"color": "#8295b5", "width": 3},
        )
    )
    drone_index = len(data)

    def drone(row):
        p, matrix = row[1:4], rotation(row[4:8])
        b = 0.2 / np.sqrt(2)
        vertices = (
            np.array([[b, b, 0], [-b, -b, 0], [np.nan] * 3, [-b, b, 0], [b, -b, 0]]) @ matrix.T + p
        )
        return go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="lines+markers",
            name="Drone",
            line={"width": 9, "color": "#e85c41"},
        )

    data.append(drone(states[0]))
    stride = max(1, int(np.ceil(len(states) / 600)))
    sampled = states[::stride]
    if sampled[-1, 0] != states[-1, 0]:
        sampled = np.vstack([sampled, states[-1]])
    frames = [
        go.Frame(name=str(i), data=[drone(row)], traces=[drone_index])
        for i, row in enumerate(sampled)
    ]
    figure = go.Figure(data=data, frames=frames)
    figure.update_layout(
        title=(
            f"{metadata['metrics']['outcome']} · stage {metadata['metrics']['stage']}"
            f" · {states[-1, 0]:.2f}s"
        ),
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "aspectmode": "data",
        },
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": stride * 1000 / 120, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "label": f"{row[0]:.2f}s",
                        "method": "animate",
                        "args": [
                            [str(i)],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": True}},
                        ],
                    }
                    for i, row in enumerate(sampled)
                ]
            }
        ],
    )
    figure.write_html(output, include_plotlyjs=True, auto_play=False)
    return Path(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("output")
    args = parser.parse_args()
    print(visualize(args.trace, args.output))
