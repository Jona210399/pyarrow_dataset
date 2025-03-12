import numpy as np
import polars as pl


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
    idx_column: str,
    frac_train: float,
    shuffle: bool,
    seed: int,
):
    if idx_column not in data.columns:
        raise ValueError(
            f"Column {idx_column} not found in the DataFrame. In order to split the data, the DataFrame must contain a column with unique indices."
        )

    num_rows = data.select(pl.len()).collect().item()
    train_indices, test_indices = get_indices(
        num_rows,
        frac_train=frac_train,
        shuffle=shuffle,
        seed=seed,
    )
    train_data = data.filter(pl.col(idx_column).is_in(train_indices))
    test_data = data.filter(pl.col(idx_column).is_in(test_indices))
    return train_data, test_data
