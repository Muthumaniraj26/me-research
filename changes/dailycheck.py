import csv
import requests
from datetime import datetime, timedelta
from pathlib import Path

print("🚀 DAILY PV DATASET (NASA DIRECT | 365 DAYS)")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
OUT_DIR = Path("PV_DAILY_NASA_DIRECT")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "PV_DAILY_ALL_SHAPES_SPACINGS.csv"

# --------------------------------------------------
# LOCATION (India – Chittoor region)
# --------------------------------------------------
LAT, LON = 9.6735, 77.9647

# --------------------------------------------------
# SHAPES & SPACINGS
# --------------------------------------------------
SHAPES = {
    "Flat": {"irr": 1.00, "temp": 0},
    "V-Shape": {"irr": 1.05, "temp": -1},
    "Inverted-V": {"irr": 1.08, "temp": -2},
}

SPACINGS = {
    62: 4,    # hotter
    77: 0,    # reference
    93: -3,   # cooler
}

# --------------------------------------------------
# PV CONSTANTS (UNCHANGED)
# --------------------------------------------------
FF = 0.67
Voc_stc = 24.8
Isc_stc = 0.60
T_stc = 25

beta_voc = -0.0038
alpha_isc = 0.001

n_series = 4
n_parallel = 3

current_loss = 0.82
voltage_loss = 0.84
vm_ratio = 0.80

# --------------------------------------------------
# NASA POWER (DAILY)
# --------------------------------------------------
def fetch_nasa_daily(start, end):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        "parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE"
        f"&longitude={LON}&latitude={LAT}"
        f"&start={start}&end={end}&format=JSON"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    return p["ALLSKY_SFC_SW_DWN"], p["T2M"]

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def generate_daily_dataset():

    G, T = fetch_nasa_daily("20240201", "20250131")
    dates = sorted(G.keys())

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date",
            "Shape",
            "Spacing_cm",
            "Irradiance_Wm2",
            "AmbientTemp_C",
            "CellTemp_C",
            "Voc_V",
            "Isc_A",
            "Vm_V",
            "Im_A",
            "Pm_W",
            "FF"
        ])

        for d in dates:
            G_day = G[d]
            T_day = T[d]

            if G_day <= 5:
                continue

            for shape, sc in SHAPES.items():
                for spacing, soff in SPACINGS.items():

                    # Effective irradiance
                    G_eff = G_day * sc["irr"]

                    # Cell temperature
                    T_cell = T_day + 0.025 * G_eff + soff + sc["temp"]
                    dT = T_cell - T_stc

                    # Electrical parameters
                    Isc = (
                        Isc_stc
                        * (G_eff / 1000)
                        * (1 + alpha_isc * dT)
                        * current_loss
                        * n_parallel
                    )

                    Voc = (
                        Voc_stc
                        * voltage_loss
                        * (1 + beta_voc * dT)
                        * n_series
                    )

                    Vm = vm_ratio * Voc
                    Pm = FF * Voc * Isc
                    Im = Pm / Vm if Vm > 0 else 0

                    writer.writerow([
                        datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d"),
                        shape,
                        spacing,
                        round(G_eff, 2),
                        round(T_day, 2),
                        round(T_cell, 2),
                        round(Voc, 2),
                        round(Isc, 3),
                        round(Vm, 2),
                        round(Im, 3),
                        round(Pm, 2),
                        FF
                    ])

    print("✅ DAILY NASA DATASET READY")
    print(f"📁 Saved at: {OUT_CSV}")

# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    generate_daily_dataset()
