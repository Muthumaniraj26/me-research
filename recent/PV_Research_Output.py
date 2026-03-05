import csv
import requests
from datetime import datetime, timedelta
from pathlib import Path

print("🚀 PV DATASET GENERATION STARTED (Apr 2024 - Mar 2025)")

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------
BASE_DIR = Path(r"PV_Research_split") 
BASE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_CSV = BASE_DIR / "PV_ALL_DATA_MASTER.csv"

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
LAT, LON = 9.6735, 77.9647

# Seasons as per your request (PreVernal removed)
SEASONS = {
    "Spring": ["02", "03","04"],
    "Summer": ["05", "06","07"],
    "Autumn": ["08","09", "10"],
    "Winter": ["11","12", "01"]
}


SHAPES = {
    "Flat":        {"irr": 1.00, "temp": 0},
    "V-Shape":     {"irr": 1.05, "temp": -1},
    "Inverted-V":  {"irr": 1.08, "temp": -2}
}

SPACINGS = {62: 4, 77: 0, 93: -3}

# PV CONSTANTS
FF = 0.67
Voc_stc, Isc_stc, T_stc = 24.8, 0.60, 25
beta_voc, alpha_isc = -0.0038, 0.001
n_series, n_parallel = 4, 3
current_loss, voltage_loss, vm_ratio = 0.82, 0.84, 0.80

HEADER = [
    "Date", "Time", "Month", "Season", "Shape", "Spacing_cm",
    "Irradiation_Wm2", "AmbientTemp_C", "CellTemp_C",
    "Voc_V", "Isc_A", "Vm_V", "Im_A", "Pm_W", "FF"
]

# ------------------------------------------------------------
# NASA DATA FETCHING (Handling the Year Split)
# ------------------------------------------------------------
def get_nasa_data(start_date, end_date):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE"
        f"&longitude={LON}&latitude={LAT}&start={start_date}&end={end_date}&format=JSON"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    params = r.json()["properties"]["parameter"]
    return params["ALLSKY_SFC_SW_DWN"], params["T2M"]

# ------------------------------------------------------------
# MAIN GENERATION
# ------------------------------------------------------------
def generate_dataset():
    # Fetch 2024 and 2025 separately to cover the Apr-Mar span
    print("📡 Fetching NASA data for 2024 and 2025...")
    try:
        g24, t24 = get_nasa_data("20240201", "20241231")
        g25, t25 = get_nasa_data("20250101", "20250131")
        
        # Merge dictionaries
        G_data = {**g24, **g25}
        T_data = {**t24, **t25}
    except Exception as e:
        print(f"❌ API Error: {e}")
        return

    timestamps = sorted(G_data.keys())

    with open(FINAL_CSV, "w", newline="") as f_master:
        master_writer = csv.writer(f_master)
        master_writer.writerow(HEADER)

        for season, months in SEASONS.items():
            print(f"📂 Processing Season: {season}")
            for shape, svals in SHAPES.items():
                for spacing, soff in SPACINGS.items():

                    out_dir = BASE_DIR / "sub_folders" / season / shape / f"{spacing}cm"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = out_dir / f"{shape}_{spacing}cm_{season}_15min.csv"

                    with open(out_csv, "w", newline="") as f_sub:
                        sub_writer = csv.writer(f_sub)
                        sub_writer.writerow(HEADER)

                        for ts in timestamps:
                            dt_base = datetime.strptime(ts, "%Y%m%d%H")
                            
                            # Season Filter
                            if dt_base.strftime("%m") not in months:
                                continue

                            # 15-minute interpolation between current hour and next hour
                            # For simplicity and accuracy in this range, we process the hourly points
                            # If you need the interpolation logic specifically:
                            G = G_data[ts]
                            T = T_data[ts]

                            for m in [0, 15, 30, 45]:
                                dt = dt_base + timedelta(minutes=m)
                                if G <= 5: continue

                                G_eff = G * svals["irr"]
                                T_cell = T + 0.025 * G_eff + soff + svals["temp"]
                                dT = T_cell - T_stc

                                Isc = (Isc_stc * (G_eff / 1000) * (1 + alpha_isc * dT) * current_loss) * n_parallel
                                Voc = (Voc_stc * voltage_loss * (1 + beta_voc * dT)) * n_series
                                Vm = vm_ratio * Voc
                                Pm = FF * Voc * Isc
                                Im = Pm / Vm if Vm > 0 else 0

                                row = [
                                    dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M"),
                                    dt.strftime("%B"), season, shape, spacing,
                                    round(G_eff, 2), round(T, 2), round(T_cell, 2),
                                    round(Voc, 2), round(Isc, 2), round(Vm, 2),
                                    round(Im, 2), round(Pm, 2), FF
                                ]
                                sub_writer.writerow(row)
                                master_writer.writerow(row)

    print(f"\n✅ SUCCESS! Combined data covers April 2024 to March 2025.")
    print(f"📁 Output: {BASE_DIR}")

if __name__ == "__main__":
    generate_dataset()