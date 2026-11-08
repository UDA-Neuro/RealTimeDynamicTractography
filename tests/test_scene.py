import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from dynamic_tractography.demo import demo_scene
from dynamic_tractography.scene import find_propagations, read_tracts


class SceneTests(unittest.TestCase):
    def test_demo_has_ten_depth_leads_and_default_window_paths(self):
        # Geometry stub keeps this test independent of the packaged fsaverage5 mesh.
        targets = [(-68, 20), (-36, 31), (-5, 38), (29, 31), (59, 17)]
        coordinates = {hemi: np.array([[side * 70, y, z] for y, z in targets])
                       for hemi, side in (("lh", -1), ("rh", 1))}
        with patch("dynamic_tractography.demo._standard_surfaces",
                   return_value=([], coordinates)):
            scene = demo_scene()
        self.assertEqual(len(scene["depth_electrodes"]), 10)
        self.assertEqual(len(scene["electrodes"]), 80)
        self.assertGreaterEqual(len(scene["propagations"]), 16)
        self.assertEqual({e["kind"] for e in scene["events"]}, {"Spike", "HFO"})
        first = [e for e in scene["electrodes"] if e["lead"] == "A"]
        pitch = np.linalg.norm(np.asarray(first[1]["xyz"]) - first[0]["xyz"])
        self.assertAlmostEqual(pitch, 5.5, places=2)
        self.assertTrue(all(10 <= p["delay_ms"] <= 50 for p in scene["propagations"]))

    def test_time_and_tract_both_required(self):
        contacts = [{"channel_number": 1, "xyz": [0, 0, 0]},
                    {"channel_number": 2, "xyz": [20, 0, 0]}]
        events = [{"id": "a", "kind": "Spike", "channel": 1, "peak_s": 1},
                  {"id": "b", "kind": "Spike", "channel": 2, "peak_s": 1.03}]
        self.assertEqual(len(find_propagations(events, contacts, [[[0, 0, 0], [20, 0, 0]]], 10, 50, 3)), 1)
        self.assertEqual(find_propagations(events, contacts, [[[0, 20, 0], [20, 20, 0]]], 10, 50, 3), [])
        self.assertEqual(find_propagations(events, contacts, [[[0, 0, 0], [20, 0, 0]]], 40, 50, 3), [])

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



