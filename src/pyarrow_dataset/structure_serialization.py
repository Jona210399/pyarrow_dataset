def pyarrow_serialize_structure_dict(sdict: dict):
    """Remove empty properties from a pymatgen Structure dictionary."""
    if sdict.get("properties") == {}:
        sdict.pop("properties")

    sites: list[dict] = sdict.get("sites", [])
    for site in sites:
        if site.get("properties") == {}:
            site.pop("properties")

    return sdict


def pyarrow_deserialize_structure_dict(sdict: dict):
    """Reformat a pymatgen Structure dictionary that was serialized by pyarrow to make it compatible with pymatgen.
    Only needs to be used if to_pandas() was called on the pyarrow Table."""
    matrix = sdict["lattice"]["matrix"]
    sdict["lattice"]["matrix"] = [row.tolist() for row in matrix]
    return sdict
