from pathlib import Path


class paths:
    # Directories
    repo = Path(__file__).parents[1].resolve()
    data = repo / "data"
    output = repo / "output"

    # Files
    dataset_csv = data / "dataset_vasospasm.csv"
