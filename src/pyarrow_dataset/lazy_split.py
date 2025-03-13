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
    frac_train: float,
    shuffle: bool,
    seed: int,
):
    """
    Lazily split a LazyFrame into a training and a test set. The split is done based on the row indices of the DataFrame."""
    num_rows = data.select(pl.len()).collect().item()
    train_indices, test_indices = get_indices(
        num_rows,
        frac_train=frac_train,
        shuffle=shuffle,
        seed=seed,
    )
    data = data.with_row_index()
    train_data = data.filter(pl.col("index").is_in(train_indices)).drop("index")
    test_data = data.filter(pl.col("index").is_in(test_indices)).drop("index")
    return train_data, test_data
