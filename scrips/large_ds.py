import polars as pl
from torch.utils.data import DataLoader

from pyarrow_dataset.lazy_dataset import LazyDataset
from pyarrow_dataset.lazy_split import train_test_split
from pyarrow_dataset.utils import memory_usage


def test_lazy_loaded_dataset(path: str):
    memory_usage()
    data = pl.scan_parquet(path)
    memory_usage()

    train_data, test_data = train_test_split(
        data,
        frac_train=0.8,
        shuffle=True,
        seed=42,
    )

    print("Train Data Type: ", type(train_data))
    train_dataset = LazyDataset(train_data, ["problem", "solution"])

    memory_usage()

    loader = DataLoader(train_dataset, batch_size=5)

    for i, batch in enumerate(loader):
        memory_usage()
        print(i, batch)
        print(len(batch))

        if i == 2:
            break


def get_dataset_size():
    df = pl.read_parquet("data/large_ds")
    print(f"Estimated size of the dataset: {df.estimated_size()}")
    memory_usage()


def main():
    get_dataset_size()  # Comment out if you dont want to check the memory footprint of the not lazily loaded dataset
    test_lazy_loaded_dataset(path="data/large_ds")


if __name__ == "__main__":
    main()
