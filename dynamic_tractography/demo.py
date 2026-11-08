"""Demo on the FreeSurfer fsaverage5 standard brain."""

from __future__ import annotations

import numpy as np
import nibabel as nib

from .scene import find_propagations


CONTACT_DIAMETER_MM = 2.0
CONTACT_GAP_MM = 3.5
CONTACT_PITCH_MM = CONTACT_DIAMETER_MM + CONTACT_GAP_MM
CONTACTS_PER_LEAD = 8


def _standard_surfaces() -> tuple[list[dict], dict[str, np.ndarray]]:
    from nilearn.datasets import fetch_surf_fsaverage

    atlas = fetch_surf_fsaverage("fsaverage5")
    surfaces = []
    coordinates = {}
    for hemi, key in (("lh", "pial_left"), ("rh", "pial_right")):
        vertices, faces = nib.load(atlas[key]).agg_data()
        coordinates[hemi] = np.asarray(vertices, dtype=float)
        surfaces.append({"hemi": hemi,
                         "vertices": np.round(vertices, 2).tolist(),
                         "faces": np.asarray(faces, dtype=int).tolist()})
    return surfaces, coordinates


def _electrode_array(coordinates: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    electrodes = []
    depth_electrodes = []
    targets = [(-68, 20), (-36, 31), (-5, 38), (29, 31), (59, 17)]
    for lead_index in range(10):
        side = -1 if lead_index < 5 else 1
        hemi = "lh" if side < 0 else "rh"
        target_y, target_z = targets[lead_index % 5]
        vertices = coordinates[hemi]
        eligible = vertices[vertices[:, 0] * side > 30]
        score = (eligible[:, 1] - target_y) ** 2 + 1.4 * (eligible[:, 2] - target_z) ** 2
        score -= 9 * eligible[:, 0] * side
        entry = eligible[int(np.argmin(score))].copy()
        direction = np.array([-side, -entry[1] * .002, -entry[2] * .002], dtype=float)
        direction /= np.linalg.norm(direction)
        name = chr(ord("A") + lead_index)
        contact_numbers = []
        for contact_index in range(CONTACTS_PER_LEAD):
            number = lead_index * CONTACTS_PER_LEAD + contact_index + 1
            center = entry + direction * (4 + contact_index * CONTACT_PITCH_MM)
            electrodes.append({"channel_number": number,
                               "label": f"{name}{contact_index + 1}",
                               "xyz": np.round(center, 2).tolist(),
                               "lead": name, "contact_diameter_mm": CONTACT_DIAMETER_MM})
            contact_numbers.append(number)
        tip = entry + direction * (5 + (CONTACTS_PER_LEAD - 1) * CONTACT_PITCH_MM)
        depth_electrodes.append({"name": name, "hemi": hemi,
                                 "path": [np.round(entry - direction * 3, 2).tolist(),
                                          np.round(tip, 2).tolist()],
                                 "contacts": contact_numbers,
                                 "contact_diameter_mm": CONTACT_DIAMETER_MM,
                                 "contact_gap_mm": CONTACT_GAP_MM})
    return electrodes, depth_electrodes


def _fiber(a: np.ndarray, b: np.ndarray, side: int, bend: float = 14) -> list[list[float]]:
    # An illustrative curve in the standard brain. This is not measured tractography.
    points = []
    for t in np.linspace(0, 1, 34):
        p = (1 - t) * a + t * b
        p = p + np.array([-side * bend, 0, -8]) * np.sin(np.pi * t)
        points.append(np.round(p, 2).tolist())
    return points


def demo_scene() -> dict:
    surfaces, coordinates = _standard_surfaces()
    electrodes, depth_electrodes = _electrode_array(coordinates)
    by_number = {e["channel_number"]: np.asarray(e["xyz"]) for e in electrodes}
    tracts = []
    # Three nearby curves per inter-lead connection, all touching the two event contacts.
    for first in (0, 5):
        side = -1 if first == 0 else 1
        for lead_index in range(first, first + 4):
            a = by_number[lead_index * 8 + 5]
            b = by_number[(lead_index + 1) * 8 + 5]
            for bend in (11, 14, 17):
                tracts.append(_fiber(a, b, side, bend))
    # Add an along-lead path for each shaft, linking its contact centers.
    for lead in depth_electrodes:
        tracts.append([by_number[number].tolist() for number in lead["contacts"]])

    events = []
    patterns = [
        (2.0, "Spike", range(0, 5), 5, .032),
        (5.0, "HFO", range(5, 10), 5, .029),
        (8.0, "Spike", range(4, -1, -1), 5, .035),
        (11.0, "HFO", range(9, 4, -1), 5, .031),
    ]
    for base, kind, leads, contact, step in patterns:
        for index, lead_index in enumerate(leads):
            number = lead_index * 8 + contact
            events.append({"id": f"{kind}-{base}-{index}", "kind": kind,
                           "channel": number, "peak_s": round(base + index * step, 3),
                           "score": 8.5})
    for base, kind, lead_index in ((14.0, "Spike", 0), (16.0, "HFO", 9)):
        for index, contact in enumerate((2, 4, 6)):
            events.append({"id": f"{kind}-{base}-{index}", "kind": kind,
                           "channel": lead_index * 8 + contact,
                           "peak_s": round(base + index * .028, 3), "score": 8.5})

    propagations = find_propagations(events, electrodes, tracts, 10, 50, 8)
    return {"title": "DYNAMIC TRACTOGRAPHY",
            "source_label": "FSAVERAGE5 · DEMO FIBERS",
            "surfaces": surfaces, "electrodes": electrodes,
            "depth_electrodes": depth_electrodes, "tracts": tracts,
            "events": events, "propagations": propagations,
            "settings": {"min_delay_ms": 10, "max_delay_ms": 50,
                         "radius_mm": 8, "cross_type": False},
            "demo": {"template": "FreeSurfer fsaverage5",
                     "leads": 10, "contacts_per_lead": CONTACTS_PER_LEAD,
                     "contact_diameter_mm": CONTACT_DIAMETER_MM,
                     "contact_gap_mm": CONTACT_GAP_MM}}

