from copy import deepcopy


def copy_hdr_fields(source, target):
    """Copy auxiliary HDR fields without sharing mutable data."""
    if "hdr_data" in source:
        data = source["hdr_data"]
        target["hdr_data"] = data.copy() if hasattr(data, "copy") else deepcopy(data)
    for key in ("hdr_metadata", "hdr_source"):
        if key in source:
            target[key] = deepcopy(source[key])
    return target


def make_hdr_preview_rgba(hdr_result):
    """Create a clipped 8-bit UI preview without modifying the HDR source.

    This is deliberately not a tone mapper. Values outside [0, 1] are clipped
    only in this disposable preview; hdr_result['pixel_data'] stays untouched.
    """
    import numpy as np

    pixels = np.asarray(hdr_result["pixel_data"])
    names = hdr_result["channel_names"]
    if pixels.ndim != 3:
        raise ValueError("HDR preview expects HxWxC pixel data.")

    if names == ["Y"]:
        rgb = np.repeat(pixels, 3, axis=2)
    else:
        rgb = pixels[:, :, :3]

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    if "A" in names:
        alpha = pixels[:, :, names.index("A"):names.index("A") + 1]
        alpha8 = np.rint(
            np.clip(np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            * 255.0
        ).astype(np.uint8)
    else:
        alpha8 = np.full((*pixels.shape[:2], 1), 255, dtype=np.uint8)

    return np.concatenate((rgb8, alpha8), axis=2).tobytes()
