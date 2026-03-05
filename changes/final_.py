import csv
import requests
from datetime import datetime, timedelta
from pathlib import Path

print("🚀 FULL PV DATASET GENERATION (Apr 2024 → Mar 2025 | 15-min)")

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path("Final _PV_FULL_DATASET")
BASE_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FILE = BASE_DIR / "PV_MASTER_15MIN_2024_2025.csv"


# ============================================================
# LOCATION (Sivakasi / Tamil Nadu region)
# ============================================================

LAT = 9.6735
LON = 77.9647


# ============================================================
# CONFIGURATIONS
# ============================================================

# Shapes
SHAPES = {
    "Flat":        {"irr": 1.00, "temp": 0},
    "V-Shape":     {"irr": 1.05, "temp": -1},
    "Inverted-V":  {"irr": 1.08, "temp": -2}
}

# Spacing (cm → temp offset)
SPACINGS = {
    62:  4,
    77:  0,
    93: -3
}

# Connections (gain factor)
CONNECTIONS = {
    "Se-P": 1.00,
    "TCT":  1.07,
    "TSSC": 1.10
}


# ============================================================
# PV CONSTANTS (STC BASED MODEL)
# ============================================================

FF = 0.67

Voc_stc = 24.8
Isc_stc = 0.60
T_stc   = 25

beta_voc  = -0.0038
alpha_isc =  0.001

n_series   = 4
n_parallel = 3

current_loss = 0.82
voltage_loss = 0.84

vm_ratio = 0.80


# ============================================================
# CSV HEADER
# ============================================================

HEADER = [
    "Date", "Time",
    "Year", "Month", "Day",

    "Shape", "Spacing_cm", "Connection",

    "Irradiance_Wm2",
    "AmbientTemp_C",
    "CellTemp_C",

    "Voc_V",
    "Isc_A",
    "Vm_V",
    "Im_A",
    "Pm_W",

    "FF"
]


# ============================================================
# NASA POWER DATA FETCH
# ============================================================

def fetch_nasa(start, end):

    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        "parameters=ALLSKY_SFC_SW_DWN,T2M"
        "&community=RE"
        f"&longitude={LON}&latitude={LAT}"
        f"&start={start}&end={end}"
        "&format=JSON"
    )

    r = requests.get(url, timeout=40)
    r.raise_for_status()

    p = r.json()["properties"]["parameter"]

    return p["ALLSKY_SFC_SW_DWN"], p["T2M"]


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_dataset():

    print("📡 Fetching NASA POWER data...")

    # Apr–Dec 2024
    g1, t1 = fetch_nasa("20240401", "20241231")

    # Jan–Mar 2025
    g2, t2 = fetch_nasa("20250101", "20250331")

    # Merge
    G = {**g1, **g2}
    T = {**t1, **t2}

    timestamps = sorted(G.keys())

    print("📊 Total hours:", len(timestamps))

    with open(MASTER_FILE, "w", newline="") as master:

        mw = csv.writer(master)
        mw.writerow(HEADER)

        # ====================================================
        # LOOP ALL COMBINATIONS
        # ====================================================

        for shape, sconf in SHAPES.items():

            for spacing, soff in SPACINGS.items():

                for conn, gain in CONNECTIONS.items():

                    print(f"➡️ {shape} | {spacing}cm | {conn}")

                    out_dir = BASE_DIR / shape / f"{spacing}cm" / conn
                    out_dir.mkdir(parents=True, exist_ok=True)

                    out_file = out_dir / f"{shape}_{spacing}cm_{conn}_15min.csv"

                    with open(out_file, "w", newline="") as sub:

                        sw = csv.writer(sub)
                        sw.writerow(HEADER)

                        # ============================================
                        # TIME LOOP
                        # ============================================

                        for i in range(len(timestamps) - 1):

                            ts1 = timestamps[i]
                            ts2 = timestamps[i + 1]

                            dt1 = datetime.strptime(ts1, "%Y%m%d%H")

                            G1, G2 = G[ts1], G[ts2]
                            T1, T2 = T[ts1], T[ts2]

                            # 15-minute interpolation
                            for m in [0, 15, 30, 45]:

                                f = m / 60

                                Gx = G1 + (G2 - G1) * f
                                Tx = T1 + (T2 - T1) * f

                                if Gx <= 5:
                                    continue

                                dt = dt1 + timedelta(minutes=m)

                                # ====================================
                                # PHYSICS MODEL
                                # ====================================

                                G_eff = Gx * sconf["irr"]

                                T_cell = (
                                    Tx +
                                    0.025 * G_eff +
                                    soff +
                                    sconf["temp"]
                                )

                                dT = T_cell - T_stc

                                Isc = (
                                    Isc_stc *
                                    (G_eff / 1000) *
                                    (1 + alpha_isc * dT) *
                                    current_loss *
                                    n_parallel *
                                    gain
                                )

                                Voc = (
                                    Voc_stc *
                                    voltage_loss *
                                    (1 + beta_voc * dT) *
                                    n_series
                                )

                                Vm = vm_ratio * Voc

                                Pm = FF * Voc * Isc

                                Im = Pm / Vm if Vm > 0 else 0

                                # ====================================
                                # WRITE
                                # ====================================

                                row = [

                                    dt.strftime("%Y-%m-%d"),
                                    dt.strftime("%H:%M"),

                                    dt.year,
                                    dt.strftime("%B"),
                                    dt.day,

                                    shape,
                                    spacing,
                                    conn,

                                    round(G_eff, 2),
                                    round(Tx, 2),
                                    round(T_cell, 2),

                                    round(Voc, 3),
                                    round(Isc, 3),
                                    round(Vm, 3),
                                    round(Im, 3),
                                    round(Pm, 3),

                                    FF
                                ]

                                sw.writerow(row)
                                mw.writerow(row)


    print("\n✅ DATASET COMPLETE")
    print(f"📁 Saved in: {BASE_DIR.resolve()}")
    print(f"📄 Master file: {MASTER_FILE.name}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    generate_dataset()
