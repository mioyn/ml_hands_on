import tarfile
import urllib.request
from pathlib import Path

import pandas as pd


def load_housingdata():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.exists():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        print("Downloaded housing dataset.")
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
            print("Extracted housing dataset.")
    housing_csv_path = Path("datasets/housing/housing.csv")
    return pd.read_csv(housing_csv_path)


if __name__ == "__main__":
    data = load_housingdata()
    print(data.head())
