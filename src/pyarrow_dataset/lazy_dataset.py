import polars as pl
import torch
from torch.utils.data import Dataset


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
