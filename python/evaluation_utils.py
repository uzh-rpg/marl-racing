from itertools import combinations

import numpy as np
from scipy.spatial.transform import Rotation


class TrackGeometry:
    def __init__(self, track):
        gates = track["gates"]
        self.centers = np.array(
            [gates[f"Gate{i+1}"]["position"] for i in range(gates["N"])]
        )
        quats = np.array([gates[f"Gate{i+1}"]["rotation"] for i in range(gates["N"])])
        self.rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        self.start = np.asarray(track["start_pos"], float)


def gate_overtakes(crossings, finish_passages=22, max_bracket=0.050):

    ranking = None
    events = []
    audit = []
    for k in range(finish_passages):
        reaching = [a for a, v in crossings.items() if len(v) > k]
        previous = None if ranking is None else [a for a in ranking if a in reaching]
        brackets = {a: crossings[a][k] for a in reaching}
        clear = bool(reaching) and all(
            lo is not None and hi is not None and 0 <= hi - lo <= max_bracket + 1e-12
            for lo, hi in brackets.values()
        )
        if clear:
            clear = all(
                brackets[a][1] < brackets[b][0] or brackets[b][1] < brackets[a][0]
                for a, b in combinations(reaching, 2)
            )
        if clear:
            ranking = sorted(reaching, key=lambda a: brackets[a][0])
            if previous is not None:
                for a, b in combinations(previous, 2):
                    if ranking.index(a) > ranking.index(b):
                        events.append(dict(passage=k + 1, passer=b, passed=a))
        else:
            ranking = previous
        audit.append(
            dict(
                passage=k + 1,
                reaching=reaching,
                clear=clear,
                ranking=None if ranking is None else ranking.copy(),
            )
        )
    return events, audit


def assign_ranks(rows, final_positions, gate_centers):
    positions = np.asarray(final_positions, dtype=float)
    centers = np.asarray(gate_centers, dtype=float)
    assert len(positions) == len(rows) and np.isfinite(positions).all()
    assert np.isfinite(centers).all()
    for slot, row in enumerate(rows):
        assert row["agent"] == slot
        row["next_gate_distance_m"] = float(
            np.linalg.norm(
                positions[slot, :3] - centers[row["gate_passages"] % len(centers)]
            )
        )
        if row["finished"]:
            assert np.isfinite(row["finish_time"]) and row["gate_passages"] == 22

    def key(slot):
        row = rows[slot]
        if row["finished"]:
            return (0, row["finish_time"], 0, slot)
        return (1, -row["gate_passages"], row["next_gate_distance_m"], slot)

    for rank, slot in enumerate(sorted(range(len(rows)), key=key), 1):
        rows[slot]["rank"] = rank
        rows[slot]["rank_source"] = "time_gates_final_distance_slot"
