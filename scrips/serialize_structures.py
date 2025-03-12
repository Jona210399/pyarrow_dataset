import polars as pl
from pymatgen.core import Structure

from pyarrow_dataset.create_dataset import create_dataset


def main():
    NUM_ROWS = 100
    IDX_COLUMN = "idx"
    PATH = "data/structures"

    table = create_dataset(NUM_ROWS, IDX_COLUMN, PATH)

    data = pl.scan_parquet("data/structures").collect()
    first = data["structure"][0]
    print(data.head())

    structure = Structure.from_dict(first)
    print(structure)


if __name__ == "__main__":
    main()
