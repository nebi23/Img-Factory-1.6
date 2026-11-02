import unittest

import numpy as np

from apps.methods.hdr_texture_data import copy_hdr_fields


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

    def test_missing_hdr_fields_are_ignored(self):
        target = copy_hdr_fields({"name": "ordinary"}, {"name": "copy"})
        self.assertEqual(target, {"name": "copy"})


if __name__ == "__main__":
    unittest.main()
