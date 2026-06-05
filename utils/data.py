import numpy as np
import pandas as pd


def _resolve_column(df, column_ref, default_index):
    """컬럼 이름 또는 인덱스로 Series를 가져온다.

    기존 코드처럼 ``column_ref=None``이면 고정 인덱스를 사용한다.
    차량 주행 노이즈가 합성된 CSV를 EKF에 넣을 때는
    ``voltage_col="discharge_voltage_vehicle_aug_v"``처럼 컬럼명을 넘기면 된다.
    """

    if column_ref is None:
        return df.iloc[:, default_index]

    if isinstance(column_ref, int):
        return df.iloc[:, column_ref]

    if column_ref in df.columns:
        return df[column_ref]

    raise KeyError(f"Column {column_ref!r} not found. Available columns: {list(df.columns)}")


def load_left_block(path, voltage_col=None):
    """충전 블록(왼쪽)을 읽어옴.

    CV 말기 데이터로 OCV(SOC=1.0) 앵커 추출에 사용한다.
    ``voltage_col``을 지정하면 원본 전압 대신 증강 전압 컬럼을 읽을 수 있다.
    """

    df = pd.read_csv(path)
    data = pd.DataFrame({
        "time":        df.iloc[:, 0],
        "current_raw": df.iloc[:, 1],
        "soc_percent": df.iloc[:, 3],
        "voltage":     _resolve_column(df, voltage_col, 4),
    }).dropna()
    data["time"] = data["time"] - data["time"].iloc[0]
    return data.reset_index(drop=True)


## 데이터셋에서 충전부가 아닌, 방전부를 읽어옴
def load_right_block(path, voltage_col=None):
    """방전 블록(오른쪽)을 읽어옴.

    기본값은 원본 방전 전압 컬럼(index 10)이다.
    ``voltage_col``을 지정하면 ``discharge_voltage_vehicle_aug_v`` 같은
    증강 전압 컬럼을 EKF 입력 전압으로 사용할 수 있다.
    """

    df = pd.read_csv(path)

    data = pd.DataFrame(
        {
            "time": df.iloc[:, 6],
            "current_raw": df.iloc[:, 7],
            "capacity_ah": df.iloc[:, 8],
            "soc_dod_percent": df.iloc[:, 9],
            "voltage": _resolve_column(df, voltage_col, 10),
        }
    ).dropna()

    data["time"] = data["time"] - data["time"].iloc[0]

    # 데이터셋과 EKF에 들어가는 전류 방향이 달라서 -부호를 붙임
    data["current"] = -data["current_raw"]

    return data.reset_index(drop=True)


def add_coulomb_counted_soc(data, capacity_coulomb):
    soc = [1.0]
    times = data["time"].to_numpy()
    currents = data["current"].to_numpy()

    for i in range(1, len(data)):
        dt = times[i] - times[i - 1]
        next_soc = soc[-1] - currents[i] * dt / capacity_coulomb
        soc.append(np.clip(next_soc, 0.0, 1.0))

    data = data.copy()
    data["soc_ref"] = soc
    return data


def add_dataset_dod_soc(data):
    data = data.copy()
    data["soc_ref"] = 1.0 - data["soc_dod_percent"] / 100.0
    data["soc_ref"] = data["soc_ref"].clip(0.0, 1.0)
    return data
