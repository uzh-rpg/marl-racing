from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

from evaluation_utils import TrackGeometry, gate_overtakes, assign_ranks

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import flightgym
    from flightgym import MarlNoCameraVecEnv_v0
    from env import MarlFlightEnvVec
    from src.utils.train_utils import configure_seed, load_teacher, set_up_cfg_yaml
    from src.utils.marl_eval_utils import get_pos_grid

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pool_lineup = (
        json.loads(Path(args.pool_lineup).read_text())
        if getattr(args, "pool_lineup", None)
        else None
    )
    kinds = (
        [s["controller"] for s in pool_lineup["slots"]]
        if pool_lineup
        else args.controllers.split(",")
    )
    allowed = ("lp", "lp_mlp", "sa", "ippo", "sp")
    if not 1 <= len(kinds) <= 8 or any((k not in allowed for k in kinds)):
        raise ValueError("Invalid controller composition")
    catalog = json.loads((CHECKPOINTS / "manifest.json").read_text())
    if pool_lineup is None:
        slots = []
        for slot, kind in enumerate(kinds):
            path = catalog["methods"][kind][(slot + args.rotation) % 4]
            slots.append(dict(controller=kind, checkpoint=path))
        pool_lineup = dict(slots=slots)
    args.controllers = ",".join(kinds)
    n = len(kinds)
    with open(ROOT / "track.yaml") as f:
        track = yaml.safe_load(f)
    geometry = TrackGeometry(track)
    config_paths = {
        slot: CHECKPOINTS / catalog["configs"][spec["checkpoint"]]
        for slot, spec in enumerate(pool_lineup["slots"])
    }
    policy_configs = {
        slot: yaml.safe_load(path.read_text()) for slot, path in config_paths.items()
    }
    cfg = deepcopy(policy_configs[0])
    cfg["main"].update(num_envs=1, num_threads=1, test_env="yes")
    cfg["environment"].update(
        num_agents=n,
        num_selfplay_agents=0,
        tracks=str(ROOT / "track.yaml"),
        max_laps=3,
        max_t=30.0,
        simulation_files=["kolibri_simulation.yaml"] * n,
        derated_thrust_max=["3.5"] * n,
        agent_collision_radius=0.1,
        agent_invisible_prob=0.0,
        air_disturbance_modeling="true",
    )
    cfg["extra_infos"]["components"] = list(
        dict.fromkeys(
            cfg["extra_infos"]["components"] + ["cumulative_laptime", "finished_laps"]
        )
    )
    cfg["curriculum"].update(
        relative_gate_size_points=1,
        relative_gate_size=[1.0],
        relative_gate_size_counter=[0],
    )
    grid = get_pos_grid(
        *[float(v) / 1.5 for v in cfg["randomization"]["rand_init_pos"]], n=4
    )
    nominal = geometry.start - geometry.centers[0]
    nominal[2] = 0.0
    radius = np.linalg.norm(nominal)
    from scipy.spatial.transform import Rotation

    offsets = []
    for slot in range(n):
        base_offset = -max(0, n - 4) / 2
        direction = Rotation.from_euler("z", (slot + base_offset) / radius).apply(
            nominal
        )
        direction /= np.linalg.norm(direction)
        axis = np.cross([1.0, 0, 0], direction)
        if np.linalg.norm(axis) > 1e-06:
            axis /= np.linalg.norm(axis)
            agent_rot = Rotation.from_rotvec(
                axis * np.arccos(np.clip(direction[0], -1, 1))
            )
            rotate = Rotation.from_matrix(geometry.rotations[0]) * agent_rot
        else:
            rotate = Rotation.from_matrix(geometry.rotations[0])
        offsets.append(rotate.apply(grid[args.grid_index])[None, :])
    env = MarlFlightEnvVec(
        MarlNoCameraVecEnv_v0(set_up_cfg_yaml(cfg)), num_agents=n, num_selfplay_agents=0
    )
    if pool_lineup.get("protocol") == "self":
        env.curriculum_update(1600)
    configure_seed(args.seed, env=env)
    policies = {}
    checkpoints = {}
    for (slot, kind) in enumerate(kinds):
        checkpoint = CHECKPOINTS / pool_lineup["slots"][slot]["checkpoint"]
        policies[slot] = load_teacher(
            checkpoint, device="cpu", config=policy_configs[slot], env=env
        )
        policies[slot].eval()
        checkpoints[slot] = {
            "path": str(checkpoint),
            "sha256": fingerprint(checkpoint),
            "config_path": str(config_paths[slot]),
            "config_sha256": fingerprint(config_paths[slot]),
        }
    obs = env.reset(random_mode=False, init_pos_offset=np.asarray(offsets))
    env.curriculum_update(10000)
    initial = env.get_quad_state().reshape(n, -1).copy()
    manifest = dict(
        arguments=vars(args),
        checkpoints=checkpoints,
        track_sha256=fingerprint(ROOT / "track.yaml"),
        simulator_binary_sha256=fingerprint(flightgym.__file__),
        simulator_config=cfg,
        initial_states=initial.tolist(),
        runtime_mode="synchronous offline",
        agilicious_base="02696bc1d5a97d8aed5a3e9fb7bdaf381c6b6041",
    )
    manifest["pool_lineup"] = pool_lineup
    manifest["pool_lineup_sha256"] = hashlib.sha256(
        json.dumps(pool_lineup, sort_keys=True).encode()
    ).hexdigest()
    manifest["policy_visible_opponents"] = {
        str(i): n - 1
        if policy_configs[i]["environment"]["num_agents"] - 1 > 1
        else policy_configs[i]["environment"]["num_agents"] - 1
        for i in policies
    }
    manifest["source_sha256"] = {
        str(p.relative_to(ROOT)): fingerprint(p)
        for p in ROOT.joinpath("python").rglob("*.py")
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lstm = {i: None for i in policies}
    active = np.ones(n, bool)
    states = [initial]
    times = [0.0]
    results = [
        dict(
            agent=i,
            controller=k,
            finished=False,
            gate_passages=0,
            finish_time=None,
            terminal_cause=None,
            terminal_time=None,
            lap_times=[],
            simulator_lap_times=[],
            gate_times=[],
            passed_traversal_errors=[],
            failed_traversal_error=None,
        )
        for (i, k) in enumerate(kinds)
    ]
    brackets = {i: [] for i in range(n)}
    infrastructure_error = None
    try:
        for step in range(args.steps):
            action = np.zeros((1, n * 4))
            for (slot, policy) in policies.items():
                if active[slot]:
                    policy_obs = obs[:, slot]
                    policy_obs = policy_obs.copy()
                    visible = manifest["policy_visible_opponents"][str(slot)]
                    policy_obs[:, 36 + 6 * visible :] = 0.0
                    (value, lstm[slot]) = policy.predict(
                        policy_obs, lstm[slot], deterministic=True
                    )
                    action[0, slot * 4 : slot * 4 + 4] = value[0]
            (obs, _, done, infos) = env.step(action)
            state = env.get_quad_state().reshape(n, -1).copy()
            times.append((step + 1) * 0.02)
            states.append(state)
            for slot in range(n):
                info = infos[0]["extra_info"][slot]
                mode = int(info[f"flightmode_{slot}"])
                if not active[slot]:
                    continue
                row = results[slot]
                if mode == 1:
                    gate = row["gate_passages"] % len(geometry.centers)
                    row["gate_passages"] += 1
                    row["gate_times"].append(times[-1])
                    row["passed_traversal_errors"].append(
                        float(info[f"traversal_err_{slot}"])
                    )
                    if row["gate_passages"] > 1 and (row["gate_passages"] - 1) % 7 == 0:
                        row["simulator_lap_times"].append(
                            float(info[f"laptime_{slot}"])
                        )
                        row["lap_times"].append(
                            row["gate_times"][-1] - row["gate_times"][-8]
                        )
                    bracket = (None, None)
                    for j in range(max(0, len(states) - 4), len(states) - 1):
                        a = geometry.rotations[gate].T @ (
                            states[j][slot, :3] - geometry.centers[gate]
                        )
                        b = geometry.rotations[gate].T @ (
                            states[j + 1][slot, :3] - geometry.centers[gate]
                        )
                        if a[0] <= 0 < b[0]:
                            bracket = (times[j], times[j + 1])
                    brackets[slot].append(bracket)
                finished = mode == 5 or info.get(f"finished_laps_{slot}", 0) >= 3
                cause = {2: "world", 3: "gate", 4: "drone"}.get(mode)
                if cause == "gate":
                    row["failed_traversal_error"] = float(info[f"traversal_err_{slot}"])
                if finished or cause or done[0, slot]:
                    row["finished"] = bool(finished)
                    row["terminal_cause"] = (
                        "finish" if finished else cause or "environment_done"
                    )
                    row["terminal_time"] = times[-1]
                    if finished:
                        row["finish_time"] = float(
                            info.get(f"cumulative_laptime_{slot}", times[-1])
                        )
                    active[slot] = False
            if (step + 1) % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            step=step + 1,
                            active=int(active.sum()),
                            gates=[r["gate_passages"] for r in results],
                        )
                    ),
                    flush=True,
                )
            if not active.any():
                break
    except Exception as exc:
        infrastructure_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for (slot, row) in enumerate(results):
            if row["terminal_cause"] is None:
                row["terminal_cause"] = (
                    "evaluation_failure" if infrastructure_error else "timeout"
                )
                row["terminal_time"] = times[-1]
            row["gate_fraction"] = min(row["gate_passages"] / 22.0, 1.0)
        (overtakes, audit) = gate_overtakes(brackets)
        assign_ranks(results, states[-1][:, :3], geometry.centers)
        np.savez_compressed(output / "states.npz", t=times, states=states)
        (output / "results.json").write_text(
            json.dumps(
                dict(
                    agents=results,
                    brackets=brackets,
                    overtakes=overtakes,
                    ranking_audit=audit,
                    infrastructure_error=infrastructure_error,
                ),
                indent=2,
            )
        )
    print(json.dumps(results, indent=2))
    if infrastructure_error:
        raise RuntimeError(infrastructure_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", default="lp,lp,lp,lp")
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--grid-index", type=int, default=21, choices=range(64))
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pool-lineup", help="Frozen pool slots and checkpoint identities"
    )
    run(parser.parse_args())
