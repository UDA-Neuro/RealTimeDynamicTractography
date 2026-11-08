import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from dynamic_tractography.demo import PLACEMENTS, _electrode_array, demo_scene
from dynamic_tractography.scene import find_propagations, read_tracts


class SceneTests(unittest.TestCase):
    def test_demo_has_ten_depth_leads_and_default_window_paths(self):
        # Geometry stub keeps this test independent of the packaged fsaverage5 mesh.
        coordinates = np.array([entry for entry, _ in PLACEMENTS.values()], dtype=float)
        with patch("dynamic_tractography.demo._standard_surfaces",
                   return_value=([], coordinates)):
            scene = demo_scene()
        self.assertEqual(len(scene["depth_electrodes"]), 10)
        self.assertEqual(len(scene["electrodes"]), 60)
        self.assertEqual(len(scene["clusters"]), 4)
        self.assertEqual(len(scene["propagations"]), 20)
        for cluster in scene["clusters"]:
            paths = [p for p in scene["propagations"] if p["cluster_id"] == cluster["id"]]
            self.assertEqual(len(paths), 5)
            self.assertEqual({p["source_channel"] for p in paths}, {6})
            self.assertEqual(len({p["target_channel"] for p in paths}), 5)
        self.assertEqual({e["kind"] for e in scene["events"]}, {"Spike", "HFO"})
        self.assertEqual({p["bundle"] for p in scene["propagations"]},
                         {"TEMPORAL", "IFOF", "ARCUATE", "UNCINATE"})
        self.assertTrue(all(lead["hemi"] == "lh" and lead["path"][-1][0] < -10
                            for lead in scene["depth_electrodes"]))
        for lead in scene["depth_electrodes"]:
            entry, tip = np.asarray(lead["path"])
            self.assertGreater(tip[0], entry[0])
            np.testing.assert_allclose(tip[1:], entry[1:])
            self.assertEqual(len(lead["contacts"]), 6)
        self.assertTrue(all(e["xyz"][0] < -10 for e in scene["electrodes"]))
        first = [e for e in scene["electrodes"] if e["lead"] == "A"]
        pitch = np.linalg.norm(np.asarray(first[1]["xyz"]) - first[0]["xyz"])
        self.assertAlmostEqual(pitch, 5.5, delta=.02)
        self.assertTrue(all(10 <= p["delay_ms"] <= 50 for p in scene["propagations"]))

    def test_time_and_tract_both_required(self):
        contacts = [{"channel_number": 1, "xyz": [0, 0, 0]},
                    {"channel_number": 2, "xyz": [20, 0, 0]}]
        events = [{"id": "a", "kind": "Spike", "channel": 1, "peak_s": 1},
                  {"id": "b", "kind": "Spike", "channel": 2, "peak_s": 1.03}]
        self.assertEqual(len(find_propagations(events, contacts, [[[0, 0, 0], [20, 0, 0]]], 10, 50, 3)), 1)
        self.assertEqual(find_propagations(events, contacts, [[[0, 20, 0], [20, 20, 0]]], 10, 50, 3), [])
        self.assertEqual(find_propagations(events, contacts, [[[0, 0, 0], [20, 0, 0]]], 40, 50, 3), [])

    def test_one_step_rule_does_not_chain_targets(self):
        contacts = [{"channel_number": i + 1, "xyz": [i * 10, 0, 0]} for i in range(3)]
        events = [{"id": str(i), "kind": "Spike", "channel": i + 1,
                   "peak_s": 1 + i * .02} for i in range(3)]
        tracks = [[[0, 0, 0], [10, 0, 0]], [[0, 0, 0], [20, 0, 0]]]
        paths = find_propagations(events, contacts, tracks, 10, 50, 2)
        self.assertEqual({p["source_id"] for p in paths}, {"0"})
        self.assertEqual({p["target_id"] for p in paths}, {"1", "2"})

    @unittest.skipUnless(importlib.util.find_spec("nilearn"), "Nilearn is not installed")
    def test_real_fsaverage5_contains_every_demo_contact(self):
        from nilearn.datasets import fetch_surf_fsaverage

        atlas = fetch_surf_fsaverage("fsaverage5")
        vertices, faces = nib.load(atlas["pial_left"]).agg_data()
        vertices = np.asarray(vertices, float)
        triangles = vertices[faces]
        electrodes, leads = _electrode_array(vertices)
        self.assertEqual(len(electrodes), 60)
        for lead in leads:
            contacts = [e["xyz"] for e in electrodes if e["lead"] == lead["name"]]
            points = contacts + [lead["path"][-1]]
            for x, y, z in points:
                mask = ((triangles[:, :, 1].min(axis=1) <= y) &
                        (triangles[:, :, 1].max(axis=1) >= y) &
                        (triangles[:, :, 2].min(axis=1) <= z) &
                        (triangles[:, :, 2].max(axis=1) >= z))
                a, b, c = triangles[mask].transpose(1, 0, 2)
                u, v = b - a, c - a
                denom = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
                valid = np.abs(denom) > 1e-8
                s = ((y-a[:, 1])*v[:, 2] - (z-a[:, 2])*v[:, 1]) / np.where(valid, denom, 1)
                t = (u[:, 1]*(z-a[:, 2]) - u[:, 2]*(y-a[:, 1])) / np.where(valid, denom, 1)
                valid &= (s >= -1e-6) & (t >= -1e-6) & (s+t <= 1+1e-6)
                intersections = np.unique(np.round(a[valid, 0] + s[valid]*u[valid, 0] + t[valid]*v[valid, 0], 2))
                self.assertTrue(any(left < x < right for left, right in zip(intersections[::2], intersections[1::2])),
                                f"{lead['name']} has a contact outside the pial surface")

    def test_gzipped_trk(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder)
            points = np.array([[0., 0., 0.], [1., 2., 3.], [2., 3., 4.]])
            nib.streamlines.save(nib.streamlines.Tractogram([points], affine_to_rasmm=np.eye(4)),
                                 str(path / "sample.trk"))
            with (path / "sample.trk").open("rb") as source, gzip.open(path / "sample.trk.gz", "wb") as target:
                target.write(source.read())
            tracks = read_tracts(path / "sample.trk.gz")
            self.assertEqual(len(tracks), 1)
            np.testing.assert_allclose(tracks[0][-1], [2, 3, 4])


if __name__ == "__main__":
    unittest.main()









