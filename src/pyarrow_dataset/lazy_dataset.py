import polars as pl
import torch
from torch.utils.data import Dataset


class LazyDataset(Dataset):
    def __init__(self, data: pl.LazyFrame, column_names: list[str] | str):
        self.data = data.select(pl.col(column_names))
        self._len = self.data.select(pl.len()).collect().item()

    def __len__(self):
        return self._len

    def __validate_index(self, idx: int | slice) -> tuple[int, int]:
        if isinstance(idx, slice):
            if idx.step is not None and idx.step != 1:
                raise ValueError("Slicing with a step other than 1 is not allowed.")
            if idx.start is not None and idx.start < 0:
                raise ValueError("Slicing with negative start index is not allowed.")
            if idx.stop is not None and idx.stop < 0:
                raise ValueError("Slicing with negative stop index is not allowed.")

            offset = idx.start if idx.start is not None else 0
            length = (
                (idx.stop - offset) if idx.stop is not None else (self._len - offset)
            )
            if offset + length > self._len:
                raise IndexError("Index out of range.")

            return offset, length

        if isinstance(idx, int):
            if idx >= self._len:
                raise IndexError("Index out of range.")
            if idx < 0:
                raise ValueError("Negative indexing is not allowed.")

            return idx, 1

        raise ValueError("Invalid index type.")

    def __getitem__(self, idx: int | slice):
        offset, length = self.__validate_index(idx)
        item = self.data.slice(
            offset, length
        ).collect(
            streaming=False
        )  # Streaming=False is necessary to avoid that all requested columns are loaded. When it was set to False some columns just didnt appear.

        # Convert list columns to torch tensors to make them compatible with the default collate_fn of pytorch DataLoader
        return item.with_columns(
            [
                pl.col(col).map_elements(lambda x: torch.tensor(x), return_dtype=object)
                for col in item.columns
                if item[col].dtype == pl.List
            ]
        ).to_dicts()
