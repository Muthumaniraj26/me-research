# ============================================================
# INVERTED-V | 93 cm | TOTAL CROSS-TIED (TCT)
# Winter (November–January) | 15-minute resolution
# Real NASA POWER data | Array-level PV modeling
# ============================================================

print("✅ Inverted-V | 93 cm | TCT Winter simulation started")

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
def run_invertedv_93cm_tct_winter():

    # ------------------ LOCATION (Chittoor region)
    lat, lon = 9.6735, 77.9647

    # ------------------ WINTER PERIOD
    start_date = "20241101"
    end_date   = "20250131"

    # ------------------ PV MODULE CONSTANTS
    FF = 0.67
    Voc_stc = 24.8     # V
    Isc_stc = 0.60     # A
    T_stc = 25         # °C

    # ------------------ ARRAY CONFIGURATION
    n_series = 4
    n_parallel = 3

    beta_voc = -0.0038
    alpha_isc = 0.001

    current_loss = 0.82
    voltage_loss = 0.84
    vm_ratio = 0.80

    # ------------------ SHAPE: INVERTED-V
    irr_factor = 1.08          # East–West + sky-view gain
    shape_temp_offset = -2.0   # strong chimney cooling

    # ------------------ SPACING: 93 cm (MAX CLEARANCE)
    spacing_offset = -3.0      # lowest operating temperature

    # ------------------ CONNECTION: TCT
    tct_gain = 1.07            # mismatch mitigation

    # ------------------ FETCH NASA DATA
    G_data, T_data = fetch_nasa_power(
        lat, lon, start_date, end_date
    )
    timestamps = sorted(G_data.keys())

    # ------------------ OUTPUT FILE
    out_dir = Path(r"C:\Users\muthumaniraj\Documents\me research\me research\tct\inverted-v\csv")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "InvertedV_93cm_TCT_Winter_15min.csv"

    with open(out_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Date", "Time",
            "Shape", "Spacing_cm", "Connection",
            "Irradiation_Wm2", "AmbientTemp_C", "CellTemp_C",
            "Voc_V", "Isc_A", "Vm_V", "Im_A", "Pm_W", "FF"
        ])

        # ------------------ TIME LOOP
        for i in range(len(timestamps) - 1):
            ts1, ts2 = timestamps[i], timestamps[i + 1]
            G1, G2 = G_data[ts1], G_data[ts2]
            T1, T2 = T_data[ts1], T_data[ts2]

            base_dt = datetime.strptime(ts1, "%Y%m%d%H")

            # ---- 15-minute interpolation
            for minute in [0, 15, 30, 45]:
                frac = minute / 60
                G = G1 + (G2 - G1) * frac
                T = T1 + (T2 - T1) * frac
                dt = base_dt + timedelta(minutes=minute)

                if G <= 5:
                    continue

                # ------------------ EFFECTIVE IRRADIANCE
                G_eff = G * irr_factor

                # ------------------ CELL TEMPERATURE
                T_cell = (
                    T
                    + 0.025 * G_eff
                    + spacing_offset
                    + shape_temp_offset
                )
                delta_T = T_cell - T_stc

                # ------------------ CURRENT (TCT applied)
                Isc_panel = (
                    Isc_stc * (G_eff / 1000)
                    * (1 + alpha_isc * delta_T)
                    * current_loss
                )
                Isc_array = Isc_panel * n_parallel * tct_gain

                # ------------------ VOLTAGE
                Voc_panel = (
                    Voc_stc * voltage_loss
                    * (1 + beta_voc * delta_T)
                )
                Voc_array = Voc_panel * n_series

                # ------------------ POWER
                Vm = vm_ratio * Voc_array
                Pm = FF * Voc_array * Isc_array
                Im = Pm / Vm if Vm > 0 else 0

                # ------------------ WRITE ROW
                writer.writerow([
                    dt.strftime("%Y-%m-%d"),
                    dt.strftime("%H:%M"),
                    "Inverted-V", 93, "TCT",
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
    run_invertedv_93cm_tct_winter()
