import unittest

import numpy as np

from apps.methods.hdr_texture_data import copy_hdr_fields, mark_hdr_preview_modified


class HdrTextureDataCopyTests(unittest.TestCase):
    def test_pixel_array_is_copied_independently(self):
        source = {
            "hdr_data": np.array([[[1.5, -0.25, 2.0]]], dtype=np.float32),
            "hdr_metadata": {"channels": ["R", "G", "B"]},
            "hdr_source": {"format": "EXR"},
        }
        target = copy_hdr_fields(source, {})

        self.assertIsNot(target["hdr_data"], source["hdr_data"])
        np.testing.assert_array_equal(target["hdr_data"], source["hdr_data"])
        target["hdr_data"][0, 0, 0] = 9.0
        self.assertEqual(source["hdr_data"][0, 0, 0], 1.5)

    def test_metadata_is_copied_independently(self):
        source = {"hdr_metadata": {"channels": ["R", "G", "B"]}}
        target = copy_hdr_fields(source, {})
        target["hdr_metadata"]["channels"].append("A")
        self.assertEqual(source["hdr_metadata"]["channels"], ["R", "G", "B"])

    def test_preview_stale_flag_is_copied(self):
        source = {"hdr_data": np.ones((1, 1, 3)), "hdr_preview_stale": True}
        target = copy_hdr_fields(source, {})
        self.assertTrue(target["hdr_preview_stale"])

    def test_edit_marks_existing_hdr_source_stale(self):
        texture = {"hdr_data": np.ones((1, 1, 3)), "hdr_preview_stale": False}
        self.assertIs(mark_hdr_preview_modified(texture), texture)
        self.assertTrue(texture["hdr_preview_stale"])

    def test_edit_without_hdr_source_does_not_add_flag(self):
        texture = {"rgba_data": b"\\x00"}
        mark_hdr_preview_modified(texture)
        self.assertNotIn("hdr_preview_stale", texture)

    def test_missing_hdr_fields_are_ignored(self):
        target = copy_hdr_fields({"name": "ordinary"}, {"name": "copy"})
        self.assertEqual(target, {"name": "copy"})


if __name__ == "__main__":
    unittest.main()
