"""Download robot data from Google Drive."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import gdown

DATA_INDICES = {
    "pinpad_observation": "14z2nMHXwmxVtkkHFiGKBeSHUIdIf28NQ",
}


def main(data_name: str) -> None:
    """Download the data specified in `data_names`."""
    url = f"https://drive.google.com/uc?id={DATA_INDICES[data_name]}"
    tar_path = f"data/{data_name}.tar.gz"
    gdown.download(url, tar_path, quiet=False)
    tarfile.open(tar_path, "r:gz").extractall()
    Path(tar_path).unlink(missing_ok=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    args = parser.parse_args()
    main(data_name=args.data)
