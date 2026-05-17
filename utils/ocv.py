import numpy as np


def build_ocv_table_from_cc(cc_data, points=300, degree=7,
                             R0=0.015, R1=0.01, skip_initial_sec=72.0):
    """
    CC 방전 데이터로 OCV(SOC) 룩업테이블 생성.

    IR 보정:
        CC 방전 중 단말전압에는 ohmic drop이 포함되어 있음.
        OCV_corrected = V_terminal + (R0 + R1) * I
        초기 72초(= 3 * RC시상수)는 Vrc 미수렴 구간이므로 제외.

    Polynomial:
        보정된 (SOC, OCV) 데이터를 degree차 다항식으로 피팅.
        여러 파일을 합쳐도 최소제곱법이 자동으로 평균화 처리.
    """
    df = cc_data.copy()

    # SOC (fraction 0~1) 계산
    df["soc"] = 1.0 - df["soc_dod_percent"] / 100.0

    # 초기 과도구간 제외 (Vrc 미수렴)
    if "time" in df.columns:
        t0 = df["time"].min()
        df = df[df["time"] - t0 >= skip_initial_sec]

    # IR 보정: OCV ≈ V_terminal + (R0 + R1) * I
    df["ocv_corrected"] = df["voltage"] + (R0 + R1) * df["current"]

    soc = df["soc"].to_numpy()
    ocv = df["ocv_corrected"].to_numpy()

    # Polynomial 피팅 (최소제곱)
    coeffs = np.polyfit(soc, ocv, deg=degree)

    # 룩업테이블로 변환 (EKF는 np.interp 사용)
    soc_table = np.linspace(soc.min(), soc.max(), points)
    ocv_table = np.polyval(coeffs, soc_table)

    return soc_table, ocv_table
