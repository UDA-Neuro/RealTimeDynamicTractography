"""Synthetic anatomy and events for previewing the viewer without patient data."""

from __future__ import annotations

import math
import numpy as np

from .scene import find_propagations


def demo_scene() -> dict:
    surfaces = []
    n_lat, n_lon = 36, 54
    for hemi, sign in (("lh", -1), ("rh", 1)):
        vertices, faces = [], []
        for i in range(n_lat + 1):
            theta = math.pi * i / n_lat
            for j in range(n_lon):
                phi = 2 * math.pi * j / n_lon
                fold = 1 + .045 * math.sin(12 * theta + 5 * phi) * math.sin(9 * phi - 3 * theta)
                x = sign * (38 + 37 * math.sin(theta) * math.cos(phi) * fold)
                y = 68 * math.sin(theta) * math.sin(phi) * fold
                z = 60 * math.cos(theta) * fold
                vertices.append([x, y, z])
        for i in range(n_lat):
            for j in range(n_lon):
                a = i * n_lon + j
                b = i * n_lon + (j + 1) % n_lon
                c = (i + 1) * n_lon + j
                d = (i + 1) * n_lon + (j + 1) % n_lon
                faces.extend(([a, b, c], [b, d, c]))
        surfaces.append({"vertices": vertices, "faces": faces, "hemi": hemi})
    electrodes = [
        {"channel_number": 1, "label": "A1", "xyz": [-77, -25, 27]},
        {"channel_number": 2, "label": "A2", "xyz": [-75, 4, 30]},
        {"channel_number": 3, "label": "A3", "xyz": [-69, 32, 28]},
        {"channel_number": 4, "label": "B1", "xyz": [78, -25, 24]},
        {"channel_number": 5, "label": "B2", "xyz": [76, 7, 27]},
        {"channel_number": 6, "label": "B3", "xyz": [69, 34, 25]},
    ]
    tracts = []
    rng = np.random.default_rng(7)
    for side in (-1, 1):
        group = electrodes[:3] if side < 0 else electrodes[3:]
        for k in range(90):
            offset = rng.normal(0, 5, 3)
            points = []
            for t in np.linspace(0, 1, 30):
                a = np.array(group[0]["xyz"])
                b = np.array(group[2]["xyz"])
                p = a * (1 - t) + b * t + offset * math.sin(math.pi * t)
                p[0] -= side * (2 + k % 6) * math.sin(math.pi * t)
                p[2] -= 3 * math.sin(math.pi * t)
                points.append(np.round(p, 2).tolist())
            tracts.append(points)
    events = []
    for base, kind, channels in ((2.5, "Spike", (1, 2, 3)), (5.1, "HFO", (4, 5, 6)),
                                 (7.3, "Spike", (3, 2, 1))):
        for i, channel in enumerate(channels):
            events.append({"id": f"{kind}-{base}-{i}", "kind": kind,
                           "channel": channel, "peak_s": round(base + i * .029, 3), "score": 8.5})
    paths = find_propagations(events, electrodes, tracts, 10, 50, 8)
    return {"title": "DYNAMIC TRACTOGRAPHY", "source_label": "SYNTHETIC DEMO",
            "surfaces": surfaces, "electrodes": electrodes, "tracts": tracts,
            "events": events, "propagations": paths,
            "settings": {"min_delay_ms": 10, "max_delay_ms": 50, "radius_mm": 8, "cross_type": False}}

