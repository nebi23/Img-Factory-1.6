import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

try:
    import OpenEXR
except ImportError:
    OpenEXR = None


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "methods"
    / "float_image_loader.py"
)
SPEC = importlib.util.spec_from_file_location("float_image_loader", MODULE_PATH)
LOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOADER)


class FloatImageLoaderTests(unittest.TestCase):
    def test_float32_rgba_tiff_preserves_values_above_one(self):
        pixels = np.array(
            [
                [[0.25, 0.5, 1.0, 1.0], [1.5, 2.0, 4.0, 0.75]],
                [[-0.5, 0.0, 0.1, 1.0], [3.0, 2.5, 1.25, 0.0]],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgba_float.tif"
            tifffile.imwrite(path, pixels, photometric="rgb")
            result = LOADER.load_float_image(path)

        np.testing.assert_array_equal(result["pixel_data"], pixels)
        self.assertEqual(result["dtype"], "float32")
        self.assertEqual((result["width"], result["height"]), (2, 2))
        self.assertEqual(result["channels"], 4)
        self.assertTrue(result["has_alpha"])
        self.assertEqual(result["value_range"], (-0.5, 4.0))

    def test_float32_grayscale_tiff_adds_single_channel_axis(self):
        pixels = np.array([[0.0, 1.25], [2.5, -1.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gray_float.tif"
            tifffile.imwrite(path, pixels)
            result = LOADER.load_float_image(path)

        np.testing.assert_array_equal(result["pixel_data"][..., 0], pixels)
        self.assertEqual(result["channel_names"], ["Y"])
        self.assertFalse(result["has_alpha"])

    def test_float32_tiff_lzw_compression(self):
        pixels = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 3

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb_lzw.tif"
            tifffile.imwrite(
                path, pixels, photometric="rgb", compression="lzw"
            )
            result = LOADER.load_float_image(path)

        np.testing.assert_array_equal(result["pixel_data"], pixels)
        self.assertEqual(result["dtype"], "float32")
        self.assertEqual(result["channel_names"], ["R", "G", "B"])
        self.assertFalse(result["has_alpha"])

    def test_tiff_value_range_ignores_nan_and_infinity(self):
        pixels = np.array(
            [[np.nan, -2.0], [np.inf, 3.0]], dtype=np.float32
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "non_finite.tif"
            tifffile.imwrite(path, pixels)
            result = LOADER.load_float_image(path)

        self.assertTrue(np.isnan(result["pixel_data"][0, 0, 0]))
        self.assertTrue(np.isinf(result["pixel_data"][1, 0, 0]))
        self.assertEqual(result["value_range"], (-2.0, 3.0))

    def test_integer_tiff_is_rejected(self):
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "integer.tif"
            tifffile.imwrite(path, pixels, photometric="rgb")
            with self.assertRaises(LOADER.FloatImageLoadError):
                LOADER.load_float_image(path)

    @unittest.skipIf(OpenEXR is None, "OpenEXR is not installed")
    def test_float16_rgba_exr_preserves_values_above_one(self):
        pixels = np.array(
            [
                [[0.25, 0.5, 1.0, 1.0], [1.5, 2.0, 4.0, 0.75]],
                [[-0.5, 0.0, 0.1, 1.0], [3.0, 2.5, 1.25, 0.0]],
            ],
            dtype=np.float16,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgba_float.exr"
            header = {
                "compression": OpenEXR.ZIP_COMPRESSION,
                "type": OpenEXR.scanlineimage,
            }
            with OpenEXR.File(header, {"RGBA": pixels}) as exr_file:
                exr_file.write(str(path))
            result = LOADER.load_float_image(path)

        np.testing.assert_array_equal(result["pixel_data"], pixels)
        self.assertEqual(result["dtype"], "float16")
        self.assertEqual((result["width"], result["height"]), (2, 2))
        self.assertEqual(result["channels"], 4)
        self.assertTrue(result["has_alpha"])
        self.assertEqual(result["value_range"], (-0.5, 4.0))

    @unittest.skipIf(OpenEXR is None, "OpenEXR is not installed")
    def test_float32_rgb_exr_without_alpha(self):
        pixels = np.array(
            [
                [[0.1, 0.2, 0.3], [1.5, 2.0, 3.0]],
                [[-1.0, 0.0, 0.5], [4.0, 3.0, 2.0]],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb_float.exr"
            header = {
                "compression": OpenEXR.ZIP_COMPRESSION,
                "type": OpenEXR.scanlineimage,
            }
            with OpenEXR.File(header, {"RGB": pixels}) as exr_file:
                exr_file.write(str(path))
            result = LOADER.load_float_image(path)

        np.testing.assert_array_equal(result["pixel_data"], pixels)
        self.assertEqual(result["dtype"], "float32")
        self.assertEqual(result["channels"], 3)
        self.assertFalse(result["has_alpha"])
        self.assertEqual(result["channel_names"], ["R", "G", "B"])

    def test_corrupt_tiff_is_reported_as_loader_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.tif"
            path.write_bytes(b"not a valid TIFF file")
            with self.assertRaises(LOADER.FloatImageLoadError):
                LOADER.load_float_image(path)

    def test_unsupported_extension_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"not an image")
            with self.assertRaises(LOADER.FloatImageLoadError):
                LOADER.load_float_image(path)


if __name__ == "__main__":
    unittest.main()
