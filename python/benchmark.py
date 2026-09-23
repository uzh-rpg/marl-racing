import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["lp", "lp_mlp", "sa", "ippo", "sp"]


def entries():
    manifest = json.loads((ROOT / "checkpoints/manifest.json").read_text())
    result = []
    for method in METHODS:
        for path in manifest["methods"][method]:
            result.append(
                dict(controller=method, checkpoint=path, sampling_alias=len(result))
            )
    return result


def race(job):
    lineup, grid, root = job
    (root / "races").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    folder = root / "races" / f"race_{lineup['permutation']:04d}_{grid:02d}"
    command = [
        sys.executable,
        str(ROOT / "python/evaluate.py"),
        "--pool-lineup",
        str(root / f"lineup_{lineup['permutation']:04d}.json"),
        "--grid-index",
        str(grid),
        "--seed",
        str(grid + lineup.get("seed_offset", 0)),
        "--output",
        str(folder),
    ]
    with (root / "logs" / f"{folder.name}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((folder / "results.json").read_text())
    if result["infrastructure_error"]:
        raise RuntimeError(result["infrastructure_error"])
    return result["agents"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["self", "pool"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--lineups", type=int, default=1000)
    parser.add_argument("--starts", type=int, default=64, choices=range(1, 65))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--plot", action="store_true", help="Plot trajectories by agent start slot"
    )
    parser.add_argument(
        "--agents",
        type=int,
        nargs="+",
        choices=range(1, 9),
        default=list(range(1, 9)),
        help="Self races only; pooled races always use four agents",
    )
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    pool = entries()
    if args.mode == "pool":
        rng = np.random.default_rng(0)
        lineups = [
            dict(
                permutation=i + 1,
                slots=[pool[int(j)] for j in rng.choice(len(pool), 4, replace=True)],
            )
            for i in range(args.lineups)
        ]
    else:
        lineups = []
        for method_index in range(len(METHODS)):
            snapshots = pool[4 * method_index : 4 * method_index + 4]
            for snapshot in snapshots:
                for agents in args.agents:
                    lineups.append(
                        dict(
                            permutation=len(lineups) + 1,
                            slots=[snapshot] * agents,
                            seed_offset=42,
                            protocol="self",
                        )
                    )
    (root / "plan.json").write_text(
        json.dumps(dict(arguments=vars(args), lineups=lineups), indent=2)
    )
    for lineup in lineups:
        (root / f"lineup_{lineup['permutation']:04d}.json").write_text(
            json.dumps(lineup)
        )
    jobs = [(lineup, grid, root) for lineup in lineups for grid in range(args.starts)]
    rows = []
    with ProcessPoolExecutor(args.workers) as executor:
        for completed, agents in enumerate(executor.map(race, jobs), 1):
            rows.extend([dict(row, num_agents=len(agents)) for row in agents])
            print(f"{completed}/{len(jobs)} races", flush=True)
    if args.plot:
        from plot import plot_lineup

        for lineup in lineups:
            plot_lineup(root, lineup)
    with (root / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "agents",
                "trials",
                "finish_fraction",
                "mean_gate_fraction",
                "mean_rank",
                "mean_best_simulator_lap_s",
                "gate_collision_fraction",
                "world_collision_fraction",
                "drone_collision_fraction",
            ],
        )
        writer.writeheader()
        for method, num_agents in [
            (m, n) for m in METHODS for n in sorted({r["num_agents"] for r in rows})
        ]:
            selected = [
                r
                for r in rows
                if r["controller"] == method and r["num_agents"] == num_agents
            ]
            if not selected:
                continue
            laps = [
                min(r["simulator_lap_times"])
                for r in selected
                if r["simulator_lap_times"]
            ]
            writer.writerow(
                dict(
                    method=method,
                    agents=num_agents,
                    trials=len(selected),
                    finish_fraction=np.mean([r["finished"] for r in selected]),
                    mean_gate_fraction=np.mean([r["gate_fraction"] for r in selected]),
                    mean_rank=np.mean([r["rank"] for r in selected]),
                    mean_best_simulator_lap_s=np.mean(laps) if laps else "",
                    **{
                        f"{cause}_collision_fraction": np.mean(
                            [r["terminal_cause"] == cause for r in selected]
                        )
                        for cause in ["gate", "world", "drone"]
                    },
                )
            )


if __name__ == "__main__":
    main()
