"""Convert anatomy, electrodes and detector output to a browser scene."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import nibabel as nib
import numpy as np


def read_electrodes(path: Path) -> list[dict]:
    """Read electrode positions in the patient's T1 scanner RAS, in millimetres."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"channel_number", "label", "x", "y", "z"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Electrode CSV needs {', '.join(sorted(required))}")
    result = []
    seen = set()
    for row in rows:
        number = int(row["channel_number"])
        if number in seen:
            raise ValueError(f"Duplicate electrode channel {number}")
        seen.add(number)
        xyz = [float(row[key]) for key in ("x", "y", "z")]
        if not np.all(np.isfinite(xyz)):
            raise ValueError(f"Non-finite electrode position: {number}")
        result.append({"channel_number": number, "label": row["label"], "xyz": xyz})
    return result


def read_events(path: Path | None, kind: str, known_channels: set[int]) -> list[dict]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    events = []
    for index, row in enumerate(rows):
        try:
            channel = int(row["channel_number"])
            peak = float(row["peak_s"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid {kind} event at row {index + 2}") from exc
        if channel not in known_channels or not np.isfinite(peak):
            continue
        events.append({"id": f"{kind}-{index}", "kind": kind, "channel": channel,
                       "peak_s": peak, "score": float(row.get("score") or 0)})
    return events


def read_surface(subject_dir: Path, hemi: str, target_faces: int = 18000) -> dict:
    path = subject_dir / "surf" / f"{hemi}.pial"
    vertices, faces = nib.freesurfer.read_geometry(str(path))
    orig = nib.load(str(subject_dir / "mri" / "orig.mgz"))
    # FreeSurfer surfaces are in tkRAS; put all anatomy in scanner RAS.
    transform = orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    vertices = nib.affines.apply_affine(transform, vertices)
    stride = max(1, int(np.ceil(len(faces) / target_faces)))
    faces = faces[::stride]
    used, compact_faces = np.unique(faces.reshape(-1), return_inverse=True)
    return {"vertices": np.round(vertices[used], 2).tolist(),
            "faces": compact_faces.reshape(-1, 3).tolist(), "hemi": hemi}


def read_tracts(path: Path, transform_path: Path | None = None,
                max_tracks: int = 1800, max_points: int = 70) -> list[list[list[float]]]:
    if not path.name.endswith((".trk", ".trk.gz", ".tck")):
        raise ValueError("Use DSI Studio exported .trk or .trk.gz tracts")
    matrix = np.eye(4)
    if transform_path:
        matrix = np.loadtxt(transform_path)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("Tract transform must be a finite 4 x 4 matrix")
    def collect(tractogram) -> list[list[list[float]]]:
        result = []
        for index, streamline in enumerate(tractogram.streamlines):
            if index >= max_tracks:
                break
            points = np.asarray(streamline)
            if len(points) < 2:
                continue
            points = points[np.linspace(0, len(points) - 1, min(len(points), max_points), dtype=int)]
            points = nib.affines.apply_affine(matrix, points)
            result.append(np.round(points, 2).tolist())
        return result

    if path.name.endswith(".trk.gz"):
        with gzip.open(path, "rb") as stream:
            return collect(nib.streamlines.trk.TrkFile.load(stream, lazy_load=True))
    return collect(nib.streamlines.load(str(path), lazy_load=True))


def route_for_pair(a: np.ndarray, b: np.ndarray, tracts: list[np.ndarray], radius_mm: float):
    """Choose a streamline passing near both contacts, with order from a to b."""
    best = None
    for track_id, points in enumerate(tracts):
        da = np.linalg.norm(points - a, axis=1)
        db = np.linalg.norm(points - b, axis=1)
        ia, ib = int(np.argmin(da)), int(np.argmin(db))
        if ia == ib or da[ia] > radius_mm or db[ib] > radius_mm:
            continue
        part = points[min(ia, ib):max(ia, ib) + 1]
        length = float(np.linalg.norm(np.diff(part, axis=0), axis=1).sum())
        if ia > ib:
            part = part[::-1]
        rank = (float(da[ia] + db[ib]), length)
        if best is None or rank < best[0]:
            best = (rank, track_id, part.tolist(), length)
    return None if best is None else {"track_id": best[1], "path": best[2], "length_mm": round(best[3], 1)}


def find_propagations(events: list[dict], electrodes: list[dict], tracts: list[list],
                      min_delay_ms: float, max_delay_ms: float, radius_mm: float,
                      cross_type: bool = False) -> list[dict]:
    if not (0 <= min_delay_ms <= max_delay_ms) or radius_mm <= 0:
        raise ValueError("Invalid propagation parameters")
    positions = {e["channel_number"]: np.asarray(e["xyz"]) for e in electrodes}
    arrays = [np.asarray(t) for t in tracts]
    route_cache = {}
    result = []
    ordered = sorted(events, key=lambda e: e["peak_s"])
    consumed = set()
    for i, source in enumerate(ordered):
        if source["id"] in consumed:
            continue
        consumed.add(source["id"])
        for target in ordered[i + 1:]:
            if target["id"] in consumed:
                continue
            delay_ms = (target["peak_s"] - source["peak_s"]) * 1000
            if delay_ms > max_delay_ms:
                break
            if delay_ms < min_delay_ms or source["channel"] == target["channel"]:
                continue
            if source["kind"] != target["kind"] and not cross_type:
                continue
            pair = (source["channel"], target["channel"])
            if pair not in route_cache:
                route_cache[pair] = route_for_pair(positions[pair[0]], positions[pair[1]], arrays, radius_mm)
            route = route_cache[pair]
            if route:
                consumed.add(target["id"])
                result.append({"source_id": source["id"], "target_id": target["id"],
                               "cluster_id": f"cluster-{source['id']}",
                               "kind": source["kind"] if source["kind"] == target["kind"] else "mixed",
                               "start_s": source["peak_s"], "end_s": target["peak_s"],
                               "delay_ms": round(delay_ms, 2), "source_channel": pair[0],
                               "target_channel": pair[1], **route})
    return result


def build_scene(subject_dir: Path | None, electrodes_path: Path, spike_path: Path | None,
                hfo_path: Path | None, tracts_path: Path | None, transform_path: Path | None,
                min_delay_ms: float, max_delay_ms: float, radius_mm: float,
                cross_type: bool, source_label: str) -> dict:
    electrodes = read_electrodes(electrodes_path)
    known = {e["channel_number"] for e in electrodes}
    events = read_events(spike_path, "Spike", known) + read_events(hfo_path, "HFO", known)
    events.sort(key=lambda e: e["peak_s"])
    surfaces = [read_surface(subject_dir, h) for h in ("lh", "rh")] if subject_dir else []
    tracts = read_tracts(tracts_path, transform_path) if tracts_path else []
    propagations = find_propagations(events, electrodes, tracts, min_delay_ms,
                                     max_delay_ms, radius_mm, cross_type)
    return {"title": "DYNAMIC TRACTOGRAPHY", "source_label": source_label,
            "surfaces": surfaces, "electrodes": electrodes, "tracts": tracts,
            "events": events, "propagations": propagations,
            "settings": {"min_delay_ms": min_delay_ms, "max_delay_ms": max_delay_ms,
                         "radius_mm": radius_mm, "cross_type": cross_type}}



