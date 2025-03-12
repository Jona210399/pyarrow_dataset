from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from pymatgen.core import Structure


def pyarrow_serialize_structure_dict(sdict: dict):
    """Remove empty properties from a pymatgen Structure dictionary."""
    if sdict["properties"] == {}:
        sdict.pop("properties")

    sites = sdict["sites"]
    for site in sites:
        if site["properties"] == {}:
            site.pop("properties")

    return sdict


def pyarrow_deserialize_structure_dict(sdict: dict):
    """Reformat a pymatgen Structure dictionary that was serialized by pyarrow to make it compatible with pymatgen. Only needs to be used if to_pandas() was called on the pyarrow Table."""
    matrix = sdict["lattice"]["matrix"]
    sdict["lattice"]["matrix"] = [row.tolist() for row in matrix]
    return sdict


def load_structures():
    path = Path("data/cifs")
    return [Structure.from_file(file) for file in path.glob("*.cif")]


def create_dataset():
    structures = load_structures()
    # Add a property to the first structure to test if it is preserved after serialization and deserialization
    structures[0].properties = {"test": "test"}

    structures = [
        pyarrow_serialize_structure_dict(structure.as_dict())
        for structure in structures
    ]

    table = pa.Table.from_pydict(
        {
            "structure": structures,
        }
    )

    (
        ds.write_dataset(
            table,
            "data/structures",
            format="parquet",
            max_rows_per_file=1,
            max_rows_per_group=1,
        ),
    )


def read():
    data = pl.scan_parquet("data/structures").collect()
    first = data["structure"][0]

    structure = Structure.from_dict(first)
    print(structure)


if __name__ == "__main__":
    create_dataset()
    read()
