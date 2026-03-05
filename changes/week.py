import csv
import requests
from datetime import datetime, timedelta
from pathlib import Path

print("🚀 PV DATASET GENERATION STARTED (Week-wise, all shapes, spacings, connections)")

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------
BASE_DIR = Path("PV_Research_Weekwise_All")
BASE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_CSV = BASE_DIR / "PV_ALL_DATA_MASTER_WEEKWISE.csv"

# ------------------------------------------------------------
# LOCATION
# ------------------------------------------------------------
LAT, LON = 9.6735, 77.9647

# ------------------------------------------------------------
# CONFIGURATIONS
# ------------------------------------------------------------
SHAPES = {
    "Flat": {"irr": 1.00, "temp": 0},
    "V-Shape": {"irr": 1.05, "temp": -1},
    "Inverted-V": {"irr": 1.08, "temp": -2}
}

SPACINGS = {
    62: 4,
    77: 0,
    93: -3
}

CONNECTIONS = {
    "Se-P": 1.00,
    "TCT": 1.07,
    "TSSC": 1.05
}

# ------------------------------------------------------------
# PV CONSTANTS (UNCHANGED)
# ------------------------------------------------------------
FF = 0.67
Voc_stc, Isc_stc, T_stc = 24.8, 0.60, 25
beta_voc, alpha_isc = -0.0038, 0.001
n_series, n_parallel = 4, 3
current_loss, voltage_loss, vm_ratio = 0.82, 0.84, 0.80

HEADER = [
    "Date", "Time", "ISO_Week", "Month",
    "Shape", "Spacing_cm", "Connection",
    "Irradiation_Wm2", "AmbientTemp_C", "CellTemp_C",
    "Voc_V", "Isc_A", "Vm_V", "Im_A", "Pm_W", "FF"
]

# ------------------------------------------------------------
# NASA DATA FETCH
# ------------------------------------------------------------
def fetch_nasa(start, end):
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE"
        f"&longitude={LON}&latitude={LAT}&start={start}&end={end}&format=JSON"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    return p["ALLSKY_SFC_SW_DWN"], p["T2M"]

# ------------------------------------------------------------
# MAIN GENERATION
# ------------------------------------------------------------
def generate_dataset():

    print("📡 Fetching NASA POWER data...")
    g1, t1 = fetch_nasa("20240201", "20241231")
    g2, t2 = fetch_nasa("20250101", "20250131")

    G = {**g1, **g2}
    T = {**t1, **t2}

    timestamps = sorted(G.keys())

    with open(FINAL_CSV, "w", newline="") as master:
        mw = csv.writer(master)
        mw.writerow(HEADER)

        for shape, sconf in SHAPES.items():
            for spacing, soff in SPACINGS.items():
                for conn, conn_gain in CONNECTIONS.items():

                    for i in range(len(timestamps) - 1):
                        ts1, ts2 = timestamps[i], timestamps[i + 1]

                        dt1 = datetime.strptime(ts1, "%Y%m%d%H")
                        G1, G2 = G[ts1], G[ts2]
                        T1, T2 = T[ts1], T[ts2]

                        for m in [0, 15, 30, 45]:
                            f = m / 60
                            Gx = G1 + (G2 - G1) * f
                            Tx = T1 + (T2 - T1) * f

                            if Gx <= 5:
                                continue

                            dt = dt1 + timedelta(minutes=m)
                            week = dt.isocalendar().week
                            if week == 53:
                                week = 52

                            out_dir = (
                                BASE_DIR /
                                f"Week_{week:02d}" /
                                shape /
                                f"{spacing}cm" /
                                conn
                            )
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_csv = out_dir / f"{shape}_{spacing}cm_{conn}_Week_{week:02d}.csv"

                            # ---- PV CALCULATIONS
                            G_eff = Gx * sconf["irr"]
                            T_cell = Tx + 0.025 * G_eff + soff + sconf["temp"]
                            dT = T_cell - T_stc

                            Isc = (
                                Isc_stc * (G_eff / 1000)
                                * (1 + alpha_isc * dT)
                                * current_loss
                                * n_parallel
                                * conn_gain
                            )

                            Voc = (
                                Voc_stc * voltage_loss
                                * (1 + beta_voc * dT)
                                * n_series
                            )

                            Vm = vm_ratio * Voc
                            Pm = FF * Voc * Isc
                            Im = Pm / Vm if Vm > 0 else 0

                            row = [
                                dt.strftime("%Y-%m-%d"),
                                dt.strftime("%H:%M"),
                                week,
                                dt.strftime("%B"),
                                shape,
                                spacing,
                                conn,
                                round(G_eff, 2),
                                round(Tx, 2),
                                round(T_cell, 2),
                                round(Voc, 2),
                                round(Isc, 2),
                                round(Vm, 2),
                                round(Im, 2),
                                round(Pm, 2),
                                FF
                            ]

                            write_header = not out_csv.exists()
                            with open(out_csv, "a", newline="") as f:
                                w = csv.writer(f)
                                if write_header:
                                    w.writerow(HEADER)
                                w.writerow(row)

                            mw.writerow(row)

    print("✅ COMPLETE: Week-wise dataset for all shapes, spacings, and connections")

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    generate_dataset()
