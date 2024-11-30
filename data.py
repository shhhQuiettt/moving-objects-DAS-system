from pathlib import Path
import numpy.typing as npt
import matplotlib.pyplot as plt
import sys
import datetime
import pandas as pd
import numpy as np
import glob


DATA_FOLDER = "20240507/"

DX = 5.106500953873407
DT = 0.0016


def _get_test_files():
    file_start = "090322"
    file_end = "090512"

    h = file_start[:2]
    m = file_start[2:4]
    s = file_start[4:]
    files = []

    while True:
        files.append(f"{h}{m}{s}.npy")
        print(f"{h}{m}{s}")
        if f"{h}{m}{s}" == file_end:
            break

        s = str(int(s) + 10).zfill(2)
        if int(s) >= 60:
            s = str(int(s) % 60).zfill(2)
            m = str(int(m) + 1).zfill(2)
        if int(m) >= 60:
            m = str(int(m) % 60).zfill(2)
            h = str(int(h) + 1).zfill(2)

    return files


TEST_FILES = _get_test_files()


def velocity_from_slope(slope: float) -> float:
    """
    Velocity in m/s

    Because on the x-axis we have space not time, we need to use the inverse of the slope.
    """
    return round(abs(1 / slope) * DX / DT, 2)


def mps_to_kmph(velocity: float) -> float:
    return round(velocity * (3.6), 2)


def load_from_file(filename: str) -> pd.DataFrame:
    file_path = Path.joinpath(Path.cwd(), DATA_FOLDER, filename)
    data = np.load(file_path)

    dx = 5.106500953873407
    dt = 0.0016

    try:
        time_start = datetime.datetime.strptime(
            f"2024-05-07 {filename.split('.')[0]}", "%Y-%m-%d %H%M%S"
        )
    except ValueError:
        print(f"Could not parse filename {filename}, expected format: %H%M%S")
        sys.exit(1)

    index = pd.date_range(start=time_start, periods=len(data), freq=f"{dt}s")

    columns = np.arange(len(data[0])) * dx

    df = pd.DataFrame(data=data, index=index, columns=columns)
    return df


def load_all_files() -> list[pd.DataFrame]:
    file_pahts = [Path.joinpath(Path.cwd(), DATA_FOLDER, file) for file in TEST_FILES]

    return [load_from_file(file) for file in TEST_FILES]


def prepocess(data: npt.NDArray) -> npt.NDArray:
    data = np.abs(data)
    low, high = np.percentile(data, [1, 99])
    data = np.clip(data, low, high)

    data = np.around(
        255 * (data - np.min(data)) / (np.max(data) - np.min(data))
    ).astype(np.uint8)
    return data
    # print(high, np.max(data))
