from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from pymatgen.core import Structure

from pyarrow_dataset.structure_serialization import pyarrow_serialize_structure_dict


def load_structures():
    path = Path("data/cifs")
    return [Structure.from_file(file) for file in path.glob("*.cif")]


def repeat_list_until_length(list_: list[Any], length: int):
    return (list_ * ((length // len(list_)) + 1))[:length]


def create_table(num_rows: int = 1000, idx_column: str = "idx"):
    structures = load_structures()
    structures = [
        pyarrow_serialize_structure_dict(structure.as_dict())
        for structure in structures
    ]
    structures = repeat_list_until_length(structures, num_rows)

    table = pa.Table.from_pydict(
        {
            "idx": list(range(num_rows)),
            "structure": structures,
            "a": [list(range(5)) for _ in range(num_rows)],
            "b": np.random.rand(num_rows),
        }
    )

    return table


def create_dataset(
    num_rows: int = 1000, idx_column: str = "idx", path: str = "dataset"
):
    table = create_table(num_rows, idx_column)

    ds.write_dataset(
        table, path, format="parquet", max_rows_per_file=1, max_rows_per_group=1
    )

    return table
