"""Build the demonstration from the published HCP1065 TRK tract atlas."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import re
from urllib.request import urlretrieve
from zipfile import ZipFile

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .demo import CONTACT_DIAMETER_MM, CONTACT_GAP_MM, CONTACT_PITCH_MM, CONTACT_DEPTH_MM, CONTACTS_PER_LEAD
from .scene import read_tracts

ATLAS_URL = "https://github.com/data-others/atlas/releases/download/hcp1065/hcp1065_avg_tracts_trk.zip"
BUNDLE_PATTERNS = {
    "TEMPORAL": (r"(?:^|[_\W])ilf(?:[_\W]|$)", r"inferior.longitudinal"),
    "IFOF": (r"ifof", r"inferior.fronto.occipital"),
    "ARCUATE": (r"(?:^|[_\W])af(?:[_\W]|$)", r"arcuate"),
    "UNCINATE": (r"(?:^|[_\W])uf(?:[_\W]|$)", r"uncinate"),
}


def _atlas_path(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"HCP1065 atlas not found: {path}")
        return path
    cache = Path.home() / ".cache" / "dynamic_tractography"
    cache.mkdir(parents=True, exist_ok=True)
    selected = cache / "hcp1065_left"
    if all((selected / f"{name}_L.trk.gz").is_file()
           for name in ("ILF", "IFOF", "AF", "UF")):
        return selected
    archive = cache / "hcp1065_avg_tracts_trk.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        print("Downloading HCP1065 TRK atlas (588 MB); this is needed only once...")
        try:
            urlretrieve(ATLAS_URL, temporary)
            with ZipFile(temporary) as source:
                if not source.namelist():
                    raise ValueError("HCP1065 atlas download is empty")
            temporary.replace(archive)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return archive


def _bundle_for_name(name: str) -> str | None:
    stem = Path(name).name.lower()
    if not stem.endswith((".trk", ".trk.gz")):
        return None
    # The atlas contains separate hemispheric files. Never mirror right tracks.
    if not (re.search(r"(?:^|[_\W])(?:l|lh|left)(?:[_\W]|$)", stem) or "left" in name.lower()):
        return None
    for bundle, patterns in BUNDLE_PATTERNS.items():
        if any(re.search(pattern, stem) for pattern in patterns):
            return bundle
    return None


def _load_bundles(atlas_path: Path, cache_dir: Path) -> dict[str, list[np.ndarray]]:
    files: dict[str, Path] = {}
    if atlas_path.is_dir():
        candidates = atlas_path.rglob("*.trk*")
        for path in candidates:
            bundle = _bundle_for_name(path.name)
            if bundle is not None and bundle not in files:
                files[bundle] = path
    else:
        with ZipFile(atlas_path) as archive:
            for name in archive.namelist():
                bundle = _bundle_for_name(name)
                if bundle is None or bundle in files:
                    continue
                suffix = ".trk.gz" if name.lower().endswith(".trk.gz") else ".trk"
                target = cache_dir / (bundle.lower() + suffix)
                if not target.exists():
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as source, target.open("wb") as output:
                        import shutil
                        shutil.copyfileobj(source, output)
                files[bundle] = target
    if not files:
        raise ValueError("No left IFOF, arcuate, uncinate, or ILF TRK files found in the HCP1065 atlas")
    bundles = {}
    for bundle, path in files.items():
        tracks = read_tracts(path, max_tracks=1200, max_points=90)
        bundles[bundle] = [np.asarray(track, dtype=float) for track in tracks if len(track) >= 2]
    return bundles


def _standard_brain():
    from nilearn.datasets import load_mni152_brain_mask
    from skimage.measure import marching_cubes

    image = load_mni152_brain_mask(resolution=2, threshold=0.2)
    mask = np.asarray(image.dataobj, dtype=bool)
    smooth = gaussian_filter(mask.astype(np.float32), sigma=0.7)
    surfaces = []
    for hemi, half in (("lh", -1), ("rh", 1)):
        field = smooth.copy()
        x_world = nib.affines.apply_affine(image.affine, np.column_stack((np.arange(field.shape[0]),
                     np.zeros(field.shape[0]), np.zeros(field.shape[0]))))[:, 0]
        field[(x_world >= 0) if half == -1 else (x_world < 0), :, :] = 0
        vertices, faces, _, _ = marching_cubes(field, level=0.45, step_size=2)
        vertices = nib.affines.apply_affine(image.affine, vertices)
        surfaces.append({"hemi": hemi, "vertices": np.round(vertices, 2).tolist(),
                         "faces": faces.astype(int).tolist()})
    return surfaces, mask, image.affine


def _lateral_entry(mask: np.ndarray, inverse: np.ndarray, point: np.ndarray) -> float | None:
    voxel = np.rint(nib.affines.apply_affine(inverse, point)).astype(int)
    y, z = voxel[1], voxel[2]
    if not (0 <= y < mask.shape[1] and 0 <= z < mask.shape[2]):
        return None
    xs = np.flatnonzero(mask[:, y, z])
    if not len(xs):
        return None
    # Standard MNI affine may have either x orientation. Work in world coordinates.
    world_x = nib.affines.apply_affine(np.linalg.inv(inverse),
                                      np.column_stack((xs, np.full(len(xs), y), np.full(len(xs), z))))[:, 0]
    left = np.sort(world_x[world_x < -10])
    if not len(left):
        return None
    gaps = np.flatnonzero(np.diff(left) > 2.5)
    runs = np.split(left, gaps + 1)
    runs = [run for run in runs if run[-1] - run[0] >= CONTACT_DEPTH_MM + 1]
    if not runs:
        return None
    return float(min(run[0] for run in runs))


def _lead_at(name: str, region: str, point: np.ndarray, mask: np.ndarray,
             inverse: np.ndarray, number: int):
    entry_x = _lateral_entry(mask, inverse, point)
    if entry_x is None:
        return None
    entry = np.array([entry_x, point[1], point[2]])
    centers = [entry + np.array([4 + i * CONTACT_PITCH_MM, 0, 0])
               for i in range(CONTACTS_PER_LEAD)]
    distances = [np.linalg.norm(point - center) for center in centers]
    contact_index = int(np.argmin(distances))
    if distances[contact_index] > 7 or entry_x + CONTACT_DEPTH_MM >= -10:
        return None
    contacts = [{"channel_number": number * CONTACTS_PER_LEAD + i + 1,
                 "label": f"{name}{i + 1}", "xyz": np.round(center, 2).tolist(),
                 "lead": name, "contact_diameter_mm": CONTACT_DIAMETER_MM}
                for i, center in enumerate(centers)]
    lead = {"name": name, "region": region, "hemi": "lh",
            "path": [np.round(entry - [3, 0, 0], 2).tolist(),
                     np.round(entry + [CONTACT_DEPTH_MM, 0, 0], 2).tolist()],
            "contacts": [c["channel_number"] for c in contacts],
            "contact_diameter_mm": CONTACT_DIAMETER_MM,
            "contact_gap_mm": CONTACT_GAP_MM}
    return contacts, lead, contacts[contact_index]


def _sources(bundles: dict, mask: np.ndarray, inverse: np.ndarray):
    clouds = {}
    trees = {}
    for bundle, tracks in bundles.items():
        points = np.concatenate([track[::max(1, len(track)//20)] for track in tracks[:600]])
        clouds[bundle] = points
        trees[bundle] = cKDTree(points)
    options = []
    for origin_bundle, points in clouds.items():
        for point in points[::max(1, len(points)//450)]:
            if not (-55 < point[0] < -15 and -55 < point[1] < 12 and -35 < point[2] < 5):
                continue
            lead = _lead_at("A", "Temporal source", point, mask, inverse, 0)
            if lead is None:
                continue
            contact = np.asarray(lead[2]["xyz"])
            near = {bundle: trees[bundle].query(contact)[0] for bundle in trees}
            score = (sum(distance < 8 for distance in near.values()),
                     sum(distance < 12 for distance in near.values()),
                     -sum(min(distance, 30) for distance in near.values()))
            options.append((score, contact, lead, origin_bundle))
    if not options:
        raise ValueError("No HCP1065 tract point can be reached by a left orthogonal depth electrode")
    options.sort(key=lambda item: item[0], reverse=True)
    chosen = []
    for bundle in clouds:
        count = 0
        for option in options:
            if option[3] != bundle or any(np.linalg.norm(option[1][1:] - other[1][1:]) < 7
                                          for other in chosen):
                continue
            chosen.append(option)
            count += 1
            if count == 8:
                break
    return [(contact, lead) for _, contact, lead, _ in chosen]


def _select_routes(bundles: dict, source_xyz: np.ndarray, mask: np.ndarray,
                   inverse: np.ndarray, only_bundle: str | None = None):
    selected = []
    desired = [only_bundle] if only_bundle else ["TEMPORAL", "IFOF", "ARCUATE", "UNCINATE"]
    paths_by_bundle = defaultdict(list)
    for bundle, tracks in bundles.items():
        if only_bundle and bundle != only_bundle:
            continue
        for track in tracks:
            distance = np.linalg.norm(track - source_xyz, axis=1)
            near = int(np.argmin(distance))
            if distance[near] > 8:
                continue
            for end in (0, len(track) - 1):
                segment = track[min(near, end):max(near, end) + 1]
                if end < near:
                    segment = segment[::-1]
                if len(segment) < 8 or np.linalg.norm(segment[-1] - source_xyz) < 30:
                    continue
                paths_by_bundle[bundle].append((float(distance[near]), segment))
        paths_by_bundle[bundle].sort(key=lambda item: item[0])
    for bundle in desired * 9:
        if len(selected) >= 9:
            break
        for _, segment in paths_by_bundle[bundle]:
            name = chr(ord("B") + len(selected))
            for target_point in segment[::-1][::3]:
                if np.linalg.norm(target_point - source_xyz) < 25:
                    continue
                if any(np.linalg.norm(target_point[1:] - item[3][1:]) < 8 for item in selected):
                    continue
                target_lead = _lead_at(name, bundle.title(), target_point, mask, inverse, len(selected) + 1)
                if target_lead is not None:
                    selected.append((bundle, segment, target_lead, target_point))
                    break
            else:
                continue
            break
    return selected


def atlas_demo_scene(atlas_path: Path | None = None) -> dict:
    atlas_path = _atlas_path(atlas_path)
    stat = atlas_path.stat()
    cache_key = sha256(f"{atlas_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:12]
    cache_dir = Path.home() / ".cache" / "dynamic_tractography" / cache_key
    bundles = _load_bundles(atlas_path, cache_dir)
    surfaces, mask, affine = _standard_brain()
    inverse = np.linalg.inv(affine)
    candidates = _sources(bundles, mask, inverse)
    cluster_scenes, clusters = {}, []
    specs = (("spike-temporal", "SPIKE · TEMPORAL", "Spike", 2.0, "TEMPORAL"),
             ("hfo-occipital", "HFO · IFOF", "HFO", 5.0, "IFOF"),
             ("spike-frontal", "SPIKE · ARCUATE", "Spike", 8.0, "ARCUATE"),
             ("hfo-network", "HFO · UNCINATE", "HFO", 11.0, "UNCINATE"))
    filler_points = np.concatenate([track[::20] for tracks in bundles.values() for track in tracks[:30]])
    for cluster_id, title, kind, start, bundle in specs:
        if bundle not in bundles:
            raise ValueError(f"HCP1065 atlas is missing left {bundle} TRK bundle")
        best = None
        for source_xyz, source_lead in candidates:
            selected = _select_routes(bundles, source_xyz, mask, inverse, bundle)
            if best is None or len(selected) > len(best[2]):
                best = (source_xyz, source_lead, selected)
            if len(selected) >= 9:
                break
        if best is None or len(best[2]) < 5:
            raise ValueError(f"Cannot place five left depth contacts along {bundle} atlas streamlines")
        source_xyz, source_lead, selected = best
        selected = selected[:5]
        electrodes, depth_electrodes = list(source_lead[0]), [source_lead[1]]
        source_contact = source_lead[2]
        for _, _, lead, _ in selected:
            electrodes.extend(lead[0])
            depth_electrodes.append(lead[1])
        while len(depth_electrodes) < 10:
            name = chr(ord("A") + len(depth_electrodes))
            for point in filler_points:
                if any(np.linalg.norm(point[1:] - np.asarray(lead["path"][-1])[1:]) < 12
                       for lead in depth_electrodes):
                    continue
                lead = _lead_at(name, "Atlas tract", point, mask, inverse, len(depth_electrodes))
                if lead:
                    electrodes.extend(lead[0])
                    depth_electrodes.append(lead[1])
                    break
            else:
                raise ValueError("Could not place ten independent orthogonal depth leads")
        tracts, events, propagations = [], [], []
        source_id = cluster_id + "-source"
        events.append({"id": source_id, "kind": kind, "channel": source_contact["channel_number"],
                       "peak_s": start, "score": 8.5, "cluster_id": cluster_id})
        event_ids = [source_id]
        for j, (_, segment, lead, _) in enumerate(selected):
            contact = lead[2]
            # The highlighted path is an unmodified segment of a real atlas streamline.
            start_index = int(np.argmin(np.linalg.norm(segment - source_xyz, axis=1)))
            end_index = int(np.argmin(np.linalg.norm(segment - np.asarray(contact["xyz"]), axis=1)))
            points = segment[min(start_index, end_index):max(start_index, end_index) + 1]
            if start_index > end_index:
                points = points[::-1]
            if len(points) < 2:
                continue
            route = np.round(points, 2).tolist()
            track_id = len(tracts)
            tracts.append(route)
            delay = (15, 22, 30, 38, 46)[j]
            target_id = f"{cluster_id}-target-{j+1}"
            event_ids.append(target_id)
            events.append({"id": target_id, "kind": kind, "channel": contact["channel_number"],
                           "peak_s": round(start + delay / 1000, 3), "score": 8.5,
                           "cluster_id": cluster_id})
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            propagations.append({"cluster_id": cluster_id, "source_id": source_id,
                                 "target_id": target_id, "kind": kind, "start_s": start,
                                 "end_s": round(start + delay / 1000, 3), "delay_ms": delay,
                                 "source_channel": source_contact["channel_number"],
                                 "target_channel": contact["channel_number"], "track_id": track_id,
                                 "path": route, "length_mm": round(length, 1),
                                 "route_label": bundle, "bundle": bundle})
        clusters.append({"id": cluster_id, "title": title, "kind": kind,
                         "source_label": source_contact["label"], "event_ids": event_ids,
                         "start_s": start, "end_s": start + .046})
        cluster_scenes[cluster_id] = {"electrodes": electrodes,
                                      "depth_electrodes": depth_electrodes,
                                      "tracts": tracts, "events": events,
                                      "propagations": propagations}
    first = cluster_scenes[specs[0][0]]
    return {"title": "DYNAMIC TRACTOGRAPHY",
            "source_label": "HCP1065 · DSI STUDIO TRK ATLAS · ICBM152 2009a",
            "surfaces": surfaces, **first, "clusters": clusters,
            "cluster_scenes": cluster_scenes,
            "settings": {"min_delay_ms": 10, "max_delay_ms": 50,
                         "radius_mm": 8, "cross_type": False, "step_limit": 1},
            "demo": {"template": "ICBM152 2009a", "atlas": "HCP1065 TRK",
                     "leads": 10, "contacts_per_lead": CONTACTS_PER_LEAD,
                     "contact_diameter_mm": CONTACT_DIAMETER_MM,
                     "contact_gap_mm": CONTACT_GAP_MM}}

