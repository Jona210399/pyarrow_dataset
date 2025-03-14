from psutil import Process


def bytes_to_gb(bytes: int):
    return bytes / 1024 / 1024 / 1024


def memory_usage():
    process = Process()
    print("Memory:", bytes_to_gb(process.memory_info().rss), "GB")
