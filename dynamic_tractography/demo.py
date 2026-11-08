"""Clustered one-step propagation demo on the FreeSurfer fsaverage5 brain."""

from __future__ import annotations

import nibabel as nib
import numpy as np

CONTACT_DIAMETER_MM = 2.0
CONTACT_GAP_MM = 3.5
CONTACT_PITCH_MM = CONTACT_DIAMETER_MM + CONTACT_GAP_MM
CONTACTS_PER_LEAD = 8
CONTACT_DEPTH_MM = 5 + (CONTACTS_PER_LEAD - 1) * CONTACT_PITCH_MM

# Approximate entry and deep target coordinates on the left fsaverage5 surface.
LEADS = (
    ("A", "Medial temporal", (-75, -25, -18), (-20, -22, -18)),
    ("B", "Lateral temporal", (-82, -5, -8), (-48, -7, -10)),
    ("C", "Posterior temporal", (-80, -50, 5), (-48, -52, 6)),
    ("D", "Occipital", (-66, -82, 12), (-35, -78, 13)),
    ("E", "Dorsal frontal", (-72, 27, 37), (-43, 26, 35)),
    ("F", "Orbitofrontal", (-62, 46, -18), (-30, 43, -17)),
    ("G", "Temporal pole", (-64, 55, -15), (-41, 50, -12)),
    ("H", "Inferior frontal", (-74, 17, 8), (-42, 20, 10)),
    ("I", "Posterior occipital", (-55, -93, 28), (-30, -86, 24)),
    ("J", "Superior frontal", (-55, 48, 48), (-30, 47, 42)),
)

ROUTES = {
    "B": ("Medial → lateral temporal", "TEMPORAL"),
    "C": ("Medial → posterior temporal", "TEMPORAL"),
    "D": ("Occipital · IFOF", "IFOF"),
    "E": ("Frontal · arcuate", "ARCUATE"),
    "F": ("Orbitofrontal · uncinate", "UNCINATE"),
    "G": ("Temporal pole · uncinate", "UNCINATE"),
    "H": ("Inferior frontal · arcuate", "ARCUATE"),
    "I": ("Posterior occipital · IFOF", "IFOF"),
    "J": ("Superior frontal · arcuate", "ARCUATE"),
}

CLUSTER_SPECS = (
    ("spike-temporal", "SPIKE · TEMPORAL FAN", "Spike", 2.0, ("B", "C", "D", "E", "F")),
    ("hfo-occipital", "HFO · OCCIPITAL FAN", "HFO", 5.0, ("B", "G", "H", "I", "J")),
    ("spike-frontal", "SPIKE · FRONTAL FAN", "Spike", 8.0, ("C", "D", "E", "F", "G")),
    ("hfo-network", "HFO · FRONTOTEMPORAL FAN", "HFO", 11.0, ("D", "E", "F", "H", "I")),
)


def _standard_surfaces() -> tuple[list[dict], np.ndarray]:
    from nilearn.datasets import fetch_surf_fsaverage

    atlas = fetch_surf_fsaverage("fsaverage5")
    surfaces = []
    left_vertices = None
    for hemi, key in (("lh", "pial_left"), ("rh", "pial_right")):
        vertices, faces = nib.load(atlas[key]).agg_data()
        if hemi == "lh":
            left_vertices = np.asarray(vertices, dtype=float)
        surfaces.append({"hemi": hemi, "vertices": np.round(vertices, 2).tolist(),
                         "faces": np.asarray(faces, dtype=int).tolist()})
    return surfaces, left_vertices


def _electrode_array(left_vertices: np.ndarray) -> tuple[list[dict], list[dict]]:
    electrodes, depth_electrodes = [], []
    eligible = left_vertices[left_vertices[:, 0] < -35]
    if not len(eligible):
        raise ValueError("fsaverage5 left hemisphere has no lateral entry vertices")
    for lead_index, (name, region, wanted_entry, target) in enumerate(LEADS):
        wanted_entry = np.asarray(wanted_entry, dtype=float)
        target = np.asarray(target, dtype=float)
        entry = eligible[np.argmin(np.linalg.norm(eligible - wanted_entry, axis=1))].copy()
        direction = target - entry
        direction /= np.linalg.norm(direction)
        max_inward_x = (-12 - entry[0]) / CONTACT_DEPTH_MM
        if direction[0] > max_inward_x:
            yz = direction[1:]
            if np.linalg.norm(yz) < 1e-6:
                yz = np.array([.8, .6])
            yz /= np.linalg.norm(yz)
            direction = np.r_[max_inward_x, yz * np.sqrt(1 - max_inward_x**2)]
        contact_numbers = []
        for contact_index in range(CONTACTS_PER_LEAD):
            number = lead_index * CONTACTS_PER_LEAD + contact_index + 1
            center = entry + direction * (4 + contact_index * CONTACT_PITCH_MM)
            electrodes.append({"channel_number": number,
                               "label": f"{name}{contact_index + 1}",
                               "xyz": np.round(center, 2).tolist(), "lead": name,
                               "contact_diameter_mm": CONTACT_DIAMETER_MM})
            contact_numbers.append(number)
        tip = entry + direction * CONTACT_DEPTH_MM
        if tip[0] >= -10:
            raise ValueError(f"Depth electrode {name} crosses the left midline")
        depth_electrodes.append({"name": name, "region": region, "hemi": "lh",
                                 "path": [np.round(entry - direction * 3, 2).tolist(),
                                          np.round(tip, 2).tolist()],
                                 "contacts": contact_numbers,
                                 "contact_diameter_mm": CONTACT_DIAMETER_MM,
                                 "contact_gap_mm": CONTACT_GAP_MM})
    return electrodes, depth_electrodes


def _route(a: np.ndarray, b: np.ndarray, bundle: str) -> list[list[float]]:
    """Illustrative curve with a distinct shape for each named route."""
    if bundle == "IFOF":
        c1, c2 = np.array([-37, -38, -17]), np.array([-42, -68, 0])
    elif bundle == "ARCUATE":
        c1, c2 = np.array([-52, -42, 25]), np.array([-48, 8, 44])
    elif bundle == "UNCINATE":
        c1, c2 = np.array([-34, -7, -30]), np.array([-40, 25, -26])
    else:
        c1, c2 = a + [-19, -8, 3], b + [8, -8, 2]
    return [np.round((1-t)**3*a + 3*(1-t)**2*t*c1 +
                     3*(1-t)*t**2*c2 + t**3*b, 2).tolist()
            for t in np.linspace(0, 1, 42)]


def demo_scene() -> dict:
    surfaces, left_vertices = _standard_surfaces()
    electrodes, depth_electrodes = _electrode_array(left_vertices)
    contacts = {e["label"]: e for e in electrodes}
    tracts, events, propagations, clusters = [], [], [], []
    delays_ms = (15, 22, 30, 38, 46)
    for cluster_id, title, kind, start_s, targets in CLUSTER_SPECS:
        origin = contacts["A6"]
        source_id = f"{cluster_id}-source"
        event_ids = [source_id]
        events.append({"id": source_id, "kind": kind,
                       "channel": origin["channel_number"], "peak_s": start_s,
                       "score": 8.5, "cluster_id": cluster_id})
        for index, (lead, delay) in enumerate(zip(targets, delays_ms)):
            target = contacts[f"{lead}6"]
            route_label, bundle = ROUTES[lead]
            points = _route(np.asarray(origin["xyz"]), np.asarray(target["xyz"]), bundle)
            track_id = len(tracts)
            tracts.append(points)
            target_id = f"{cluster_id}-target-{index + 1}"
            event_ids.append(target_id)
            end_s = round(start_s + delay / 1000, 3)
            events.append({"id": target_id, "kind": kind,
                           "channel": target["channel_number"], "peak_s": end_s,
                           "score": 8.5, "cluster_id": cluster_id})
            length = np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum()
            propagations.append({"cluster_id": cluster_id, "source_id": source_id,
                                 "target_id": target_id, "kind": kind,
                                 "start_s": start_s, "end_s": end_s,
                                 "delay_ms": delay, "source_channel": origin["channel_number"],
                                 "target_channel": target["channel_number"],
                                 "track_id": track_id, "path": points,
                                 "length_mm": round(float(length), 1),
                                 "route_label": route_label, "bundle": bundle})
        clusters.append({"id": cluster_id, "title": title, "kind": kind,
                         "source_label": origin["label"], "event_ids": event_ids,
                         "start_s": start_s, "end_s": round(start_s + .046, 3)})
    return {"title": "DYNAMIC TRACTOGRAPHY",
            "source_label": "FSAVERAGE5 · SCHEMATIC ROUTES",
            "surfaces": surfaces, "electrodes": electrodes,
            "depth_electrodes": depth_electrodes, "tracts": tracts,
            "events": events, "propagations": propagations, "clusters": clusters,
            "settings": {"min_delay_ms": 10, "max_delay_ms": 50,
                         "radius_mm": 8, "cross_type": False, "step_limit": 1},
            "demo": {"template": "FreeSurfer fsaverage5", "leads": 10,
                     "contacts_per_lead": CONTACTS_PER_LEAD,
                     "contact_diameter_mm": CONTACT_DIAMETER_MM,
                     "contact_gap_mm": CONTACT_GAP_MM}}

