"""Clustered one-step propagation demo on the FreeSurfer fsaverage5 brain."""

from __future__ import annotations

import nibabel as nib
import numpy as np

CONTACT_DIAMETER_MM = 2.0
CONTACT_GAP_MM = 3.5
CONTACT_PITCH_MM = CONTACT_DIAMETER_MM + CONTACT_GAP_MM
CONTACTS_PER_LEAD = 8
CONTACT_DEPTH_MM = 5 + (CONTACTS_PER_LEAD - 1) * CONTACT_PITCH_MM

# Entry and tip were selected on the fsaverage5 left hemisphere. Each 43.5 mm
# shaft remains inside its pial surface after entering and ends left of x=-12.
PLACEMENTS = {
    "A": ((-65.77, -23.72, -19.69), (-22.31, -21.93, -19.08)),
    "B": ((-68.02, -22.90, 0.41), (-35.68, -2.98, -20.78)),
    "C": ((-68.50, -31.55, 6.78), (-33.76, -57.42, 2.82)),
    "D": ((-62.87, -59.78, 3.84), (-23.03, -72.33, 15.97)),
    "E": ((-57.30, 15.35, 15.99), (-18.89, 31.92, 27.92)),
    "F": ((-53.64, 31.07, 0.99), (-16.80, 45.97, -16.70)),
    "G": ((-60.52, 4.04, -6.42), (-32.71, 19.38, -36.14)),
    "H": ((-59.97, 1.96, 4.90), (-23.68, 24.83, 12.17)),
    "I": ((-46.74, -74.84, 31.45), (-17.83, -86.52, 1.11)),
    "J": ((-45.57, 17.87, 42.34), (-16.55, 47.98, 30.37)),
}

LEADS = (
    ("A", "Medial temporal"),
    ("B", "Lateral temporal"),
    ("C", "Posterior temporal"),
    ("D", "Occipital"),
    ("E", "Dorsal frontal"),
    ("F", "Orbitofrontal"),
    ("G", "Temporal pole"),
    ("H", "Inferior frontal"),
    ("I", "Posterior occipital"),
    ("J", "Superior frontal"),
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
    for lead_index, (name, region) in enumerate(LEADS):
        entry, intended_tip = (np.asarray(p, dtype=float) for p in PLACEMENTS[name])
        if np.min(np.linalg.norm(left_vertices - entry, axis=1)) > 1.0:
            raise ValueError(f"Depth electrode {name} entry does not match fsaverage5")
        direction = intended_tip - entry
        direction /= np.linalg.norm(direction)
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



