import numpy as np
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from pymatgen.core import Structure
from torch.utils.data import DataLoader, Dataset

process = psutil.Process()


def get_structure():
    return Structure.from_file("data/Cu3Ge.cif")


def bytes_to_gb(bytes):
    return bytes / 1024 / 1024 / 1024


def memory_usage():
    print("Memory:", bytes_to_gb(process.memory_info().rss), "GB")


def get_indices(length: int, frac_train: float, shuffle: bool, seed: int):
    indices = list(range(0, length))
    indices_to_shuffle = indices.copy()
    if shuffle:
        random_generator = np.random.default_rng(seed=seed)
        random_generator.shuffle(indices_to_shuffle)

    split_idx = int(length * frac_train)
    train_indices = indices_to_shuffle[:split_idx]
    test_indices = indices_to_shuffle[split_idx:]

    return train_indices, test_indices


def train_test_split(
    data: pl.LazyFrame,
    frac_train: float,
    shuffle: bool,
    seed: int,
):
    num_rows = data.select(pl.len()).collect().item()
    train_indices, test_indices = get_indices(
        num_rows,
        frac_train=frac_train,
        shuffle=shuffle,
        seed=seed,
    )
    train_data = data.filter(pl.col("idx").is_in(train_indices))
    test_data = data.filter(pl.col("idx").is_in(test_indices))
    return train_data, test_data


class LazyDataset(Dataset):
    def __init__(self, data: pl.LazyFrame, column_names: list[str] | str):
        self.data = data.select(pl.col(column_names))

    def __len__(self):
        return self.data.select(pl.len()).collect().item()

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            offset = idx.start
            length = idx.stop - idx.start

        else:
            offset = idx
            length = 1

        item = self.data.slice(offset, length).collect()

        # Convert list columns to torch tensors to make them compatible with the default collate_fn of pytorch DataLoader
        return item.with_columns(
            [
                pl.col(col).map_elements(lambda x: torch.tensor(x), return_dtype=object)
                for col in item.columns
                if item[col].dtype == pl.List
            ]
        ).to_dicts()


def create_dataset():
    import pandas as pd

    num_rows = 1000

    df = pd.DataFrame(
        {
            "idx": np.arange(num_rows),
            "a": [np.arange(2) for _ in range(num_rows)],
            "b": np.random.rand(num_rows),
            "c": np.random.rand(num_rows),
            "d": np.random.rand(num_rows),
        }
    )

    schema = pa.schema(
        [
            pa.field("idx", pa.int64()),
            pa.field("a", pa.list_(pa.float64())),
            pa.field("b", pa.float64()),
            pa.field("c", pa.float64()),
            pa.field("d", pa.float64()),
        ]
    )
    table = pa.Table.from_pandas(df, schema=schema)

    path = "dataset"

    ds.write_dataset(
        table, path, format="parquet", max_rows_per_file=1, max_rows_per_group=1
    )


def test_lazy_loaded_dataset():
    memory_usage()
    data = pl.scan_parquet("dataset")
    memory_usage()

    train_data, test_data = train_test_split(
        data, frac_train=0.8, shuffle=True, seed=42
    )

    train_dataset = LazyDataset(train_data, ["a", "b"])

    memory_usage()
    print(train_data.head().collect())
    memory_usage()

    loader = DataLoader(train_dataset, batch_size=5)

    for i, batch in enumerate(loader):
        memory_usage()
        print(i, batch)
        print(len(batch))

        if i == 2:
            break


if __name__ == "__main__":
    # create_dataset()

    test_lazy_loaded_dataset()
