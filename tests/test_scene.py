import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import nibabel as nib
import numpy as np

from dynamic_tractography.atlas_demo import atlas_demo_scene, _bundle_for_name
from dynamic_tractography.scene import find_propagations, read_tracts


class SceneTests(unittest.TestCase):
    def test_atlas_name_selection_excludes_right_hemisphere(self):
        self.assertEqual(_bundle_for_name("IFOF_L.trk"), "IFOF")
        self.assertEqual(_bundle_for_name("left_uncinate.trk.gz"), "UNCINATE")
        self.assertIsNone(_bundle_for_name("IFOF_R.trk"))

    def test_hcp_atlas_demo_uses_unmodified_trk_streamline_segments(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            folder = Path(folder)
            archive_path = folder / "atlas.zip"
            all_lines = []
            with ZipFile(archive_path, "w") as archive:
                for bundle in ("ILF", "IFOF", "AF", "UF"):
                    lines = []
                    for index in range(12):
                        start = np.array([-40., -20., -20.])
                        end = np.array([-40., -48. + index * 8, -36. + (index % 4) * 12])
                        line = np.linspace(start, end, 45)
                        lines.append(line)
                        all_lines.append(line)
                    path = folder / f"{bundle}_L.trk"
                    nib.streamlines.save(nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)), str(path))
                    archive.write(path, arcname=path.name)
            affine = np.diag([2., 2., 2., 1.])
            affine[:3, 3] = -100
            mask = np.zeros((70, 70, 70), dtype=bool)
            mask[15:45, 15:65, 15:65] = True
            with patch("dynamic_tractography.atlas_demo._standard_brain",
                       return_value=([], mask, affine)), patch.object(Path, "home", return_value=folder):
                scene = atlas_demo_scene(archive_path)
            self.assertEqual(len(scene["depth_electrodes"]), 10)
            self.assertEqual(len(scene["electrodes"]), 60)
            self.assertEqual(len(scene["clusters"]), 4)
            self.assertEqual(len(scene["propagations"]), 20)
            self.assertTrue(all(lead["hemi"] == "lh" and lead["path"][-1][0] < -10
                                for lead in scene["depth_electrodes"]))
            for lead in scene["depth_electrodes"]:
                entry, tip = np.asarray(lead["path"])
                self.assertGreater(tip[0], entry[0])
                np.testing.assert_allclose(tip[1:], entry[1:])
            for route in scene["tracts"]:
                self.assertTrue(any(np.min(np.linalg.norm(line[:, None, :] - np.asarray(route)[None, :, :], axis=2), axis=0).max() < .03
                                    for line in all_lines))

    def test_time_and_tract_both_required(self):
        contacts = [{"channel_number": 1, "xyz": [0, 0, 0]},
                    {"channel_number": 2, "xyz": [20, 0, 0]}]
        events = [{"id": "a", "kind": "Spike", "channel": 1, "peak_s": 1},
                  {"id": "b", "kind": "Spike", "channel": 2, "peak_s": 1.03}]
        self.assertEqual(len(find_propagations(events, contacts, [[[0, 0, 0], [20, 0, 0]]], 10, 50, 3)), 1)
        self.assertEqual(find_propagations(events, contacts, [[[0, 20, 0], [20, 20, 0]]], 10, 50, 3), [])

    def test_one_step_rule_does_not_chain_targets(self):
        contacts = [{"channel_number": i + 1, "xyz": [i * 10, 0, 0]} for i in range(3)]
        events = [{"id": str(i), "kind": "Spike", "channel": i + 1,
                   "peak_s": 1 + i * .02} for i in range(3)]
        tracks = [[[0, 0, 0], [10, 0, 0]], [[0, 0, 0], [20, 0, 0]]]
        paths = find_propagations(events, contacts, tracks, 10, 50, 2)
        self.assertEqual({p["source_id"] for p in paths}, {"0"})

    def test_gzipped_trk(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder)
            points = np.array([[0., 0., 0.], [1., 2., 3.], [2., 3., 4.]])
            nib.streamlines.save(nib.streamlines.Tractogram([points], affine_to_rasmm=np.eye(4)),
                                 str(path / "sample.trk"))
            with (path / "sample.trk").open("rb") as source, gzip.open(path / "sample.trk.gz", "wb") as target:
                target.write(source.read())
            tracks = read_tracts(path / "sample.trk.gz")
            np.testing.assert_allclose(tracks[0][-1], [2, 3, 4])


if __name__ == "__main__":
    unittest.main()

