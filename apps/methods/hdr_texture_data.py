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
