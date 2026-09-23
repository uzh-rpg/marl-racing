import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def plot_lineup(folder, lineup):
    folder = Path(folder)
    race_folder = folder / "races" if (folder / "races").is_dir() else folder
    races = sorted(race_folder.glob(f"race_{lineup['permutation']:04d}_*/states.npz"))
    if not races:
        raise ValueError("No saved trajectories for this lineup")
    agents = len(lineup["slots"])
    paths = [[] for _ in range(agents)]
    finished = np.zeros(agents, dtype=int)
    for race in races:
        with np.load(race) as data:
            times, states = data["t"], data["states"]
        results = json.loads((race.parent / "results.json").read_text())
        if results["infrastructure_error"]:
            raise ValueError(f"Failed race: {race.parent}")
        for slot, result in enumerate(results["agents"]):
            end = result["terminal_time"]
            positions = states[(times > 0) & (times <= end + 1e-9), slot, :3]
            paths[slot].append(positions)
            finished[slot] += result["finished"]
    manifest = json.loads((races[0].parent / "manifest.json").read_text())
    track = yaml.safe_load(
        Path(manifest["simulator_config"]["environment"]["tracks"]).read_text()
    )
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.get_cmap("tab10").colors
    handles = []
    for slot in range(agents):
        color = colors[slot % len(colors)]
        ax.add_collection3d(
            Line3DCollection(paths[slot], colors=[color], linewidths=0.65, alpha=0.18)
        )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                label=f"Start slot {slot}: {finished[slot]}/{len(races)} finished",
            )
        )
    for index in range(track["gates"]["N"]):
        gate = track["gates"][f"Gate{index + 1}"]
        width, height = np.asarray(gate["size"])[1:] / 2
        corners = np.array(
            [
                [0, -width, -height],
                [0, width, -height],
                [0, width, height],
                [0, -width, height],
                [0, -width, -height],
            ]
        )
        quat = np.asarray(gate["rotation"])
        corners = (
            Rotation.from_quat(quat[[1, 2, 3, 0]]).apply(corners) + gate["position"]
        )
        ax.plot(*corners.T, color="0.2", linewidth=1.1)
    points = np.concatenate([path for group in paths for path in group])
    low, high = points.min(axis=0) - 0.4, points.max(axis=0) + 0.4
    ax.set(
        xlim=(low[0], high[0]),
        ylim=(low[1], high[1]),
        zlim=(low[2], high[2]),
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
    )
    ax.set_box_aspect(high - low)
    ax.view_init(elev=45, azim=-160)
    labels = list(dict.fromkeys(s["checkpoint"] for s in lineup["slots"]))
    ax.set_title(f"{', '.join(labels)}\n{len(races)} races · {agents} agents", pad=18)
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    output = folder / f"trajectories_{lineup['permutation']:04d}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Benchmark output folder")
    args = parser.parse_args()
    plan = json.loads((args.folder / "plan.json").read_text())
    for lineup in plan["lineups"]:
        print(plot_lineup(args.folder, lineup))


if __name__ == "__main__":
    main()
