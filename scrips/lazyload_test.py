import polars as pl
import psutil
from torch.utils.data import DataLoader

from pyarrow_dataset.create_dataset import create_dataset
from pyarrow_dataset.lazy_dataset import LazyDataset
from pyarrow_dataset.lazy_split import train_test_split

process = psutil.Process()


def bytes_to_gb(bytes):
    return bytes / 1024 / 1024 / 1024


def memory_usage():
    print("Memory:", bytes_to_gb(process.memory_info().rss), "GB")


def test_lazy_loaded_dataset(path: str = "data/structures"):
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

    # If we put in "structure" here as well, the collate_fn of the DataLoader throws an error because it cant process the list of structures as dicts. This can be easily fixed if we handle the structures correctly in the dataset class. In our use case we use the graph data so the use cas of dataloading structure dicts is not needed.
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


def main():
    NUM_ROWS = 100
    PATH = "data/structures"

    table = create_dataset(NUM_ROWS, PATH)
    test_lazy_loaded_dataset(PATH)


if __name__ == "__main__":
    main()
