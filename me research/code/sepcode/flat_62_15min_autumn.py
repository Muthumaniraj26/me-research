# ============================================================
# FLAT | 62 cm | SERIES–PARALLEL (Se-P)
# Autumn (September–November) | 15-minute resolution
# Array-level PV modeling
# ============================================================

print("✅ Flat | 62 cm | Se-P | Autumn simulation started")

import csv
import requests
from datetime import datetime, timedelta
from pathlib import Path

# ------------------------------------------------------------
# FETCH NASA POWER DATA (HOURLY)
# ------------------------------------------------------------
def fetch_nasa_power(lat, lon, start_date, end_date):
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start_date}&end={end_date}&format=JSON"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()["properties"]["parameter"]
    return data["ALLSKY_SFC_SW_DWN"], data["T2M"]

# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------
def run_flat_62cm_sep_autumn():

    # ------------------ LOCATION
    lat, lon = 9.6735, 77.9647

    # ------------------ AUTUMN PERIOD
    start_date = "20240901"
    end_date   = "20241130"

    # ------------------ MODULE CONSTANTS
    FF = 0.67
    Voc_stc = 24.8
    Isc_stc = 0.60
    T_stc = 25

    # ------------------ ARRAY CONFIGURATION (Se-P)
    n_series = 4
    n_parallel = 3

    beta_voc = -0.0038
    alpha_isc = 0.001

    current_loss = 0.82
    voltage_loss = 0.84
    vm_ratio = 0.80

    # ------------------ SHAPE: FLAT
    irr_factor = 1.00
    shape_temp_offset = 0.0

    # ------------------ SPACING: 62 cm
    spacing_offset = 4.0

    # ------------------ Se-P CONNECTION
    sep_gain = 1.00

    # ------------------ FETCH DATA
    G_data, T_data = fetch_nasa_power(
        lat, lon, start_date, end_date
    )
    timestamps = sorted(G_data.keys())

    # ------------------ OUTPUT
    out_dir = Path(r"C:\Users\muthumaniraj\Documents\me research\me research\sep\flat")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "Flat_62cm_SeP_Autumn_15min.csv"

    with open(out_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Date", "Time",
            "Shape", "Spacing_cm", "Connection",
            "Irradiation_Wm2", "AmbientTemp_C", "CellTemp_C",
            "Voc_V", "Isc_A", "Vm_V", "Im_A", "Pm_W", "FF"
        ])

        for i in range(len(timestamps) - 1):
            ts1, ts2 = timestamps[i], timestamps[i + 1]
            G1, G2 = G_data[ts1], G_data[ts2]
            T1, T2 = T_data[ts1], T_data[ts2]

            base_dt = datetime.strptime(ts1, "%Y%m%d%H")

            for minute in [0, 15, 30, 45]:
                frac = minute / 60
                G = G1 + (G2 - G1) * frac
                T = T1 + (T2 - T1) * frac
                dt = base_dt + timedelta(minutes=minute)

                if G <= 5:
                    continue

                G_eff = G * irr_factor

                T_cell = (
                    T
                    + 0.025 * G_eff
                    + spacing_offset
                    + shape_temp_offset
                )
                delta_T = T_cell - T_stc

                Isc_panel = (
                    Isc_stc * (G_eff / 1000)
                    * (1 + alpha_isc * delta_T)
                    * current_loss
                )
                Isc_array = Isc_panel * n_parallel * sep_gain

                Voc_panel = (
                    Voc_stc * voltage_loss
                    * (1 + beta_voc * delta_T)
                )
                Voc_array = Voc_panel * n_series

                Vm = vm_ratio * Voc_array
                Pm = FF * Voc_array * Isc_array
                Im = Pm / Vm if Vm > 0 else 0

                writer.writerow([
                    dt.strftime("%Y-%m-%d"),
                    dt.strftime("%H:%M"),
                    "Flat", 62, "Se-P",
                    round(G_eff, 1),
                    round(T, 1),
                    round(T_cell, 1),
                    round(Voc_array, 2),
                    round(Isc_array, 2),
                    round(Vm, 2),
                    round(Im, 2),
                    round(Pm, 1),
                    FF
                ])

    print("✅ CSV generated successfully")
    print(f"📁 File saved at: {out_file}")

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    run_flat_62cm_sep_autumn()
