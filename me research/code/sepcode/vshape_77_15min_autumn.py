# ============================================================
# V-SHAPE | 77 cm | SERIES-PARALLEL (Se-P)
# Autumn (September–October–November) | 15-minute resolution
# Array-level PV modeling (no wire-level simulation)
# ============================================================

print("✅ V-Shape | 77 cm | Se-P | Autumn simulation started")

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
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()["properties"]["parameter"]
    return d["ALLSKY_SFC_SW_DWN"], d["T2M"]

# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------
def run_vshape_77cm_sep_autumn():

    # -------- Location (Chittoor region)
    lat, lon = 9.6735, 77.9647

    # -------- Autumn period (Sep–Oct–Nov)
    start_date = "20250901"
    end_date   = "20251130"

    # -------- PV module constants
    FF = 0.67
    Voc_stc = 24.8     # V
    Isc_stc = 0.60     # A
    T_stc   = 25       # °C

    # -------- Array configuration (Se-P)
    n_series   = 4
    n_parallel = 3

    beta_voc  = -0.0038
    alpha_isc = 0.001

    current_loss = 0.82
    voltage_loss = 0.84
    vm_ratio     = 0.80

    # -------- Shape: V-SHAPE
    irr_factor = 1.05          # diffuse + ground reflection
    shape_temp_offset = -1.0   # partial cooling

    # -------- Spacing: 77 cm (baseline airflow)
    spacing_offset = 0.0

    # -------- Fetch NASA data
    G_data, T_data = fetch_nasa_power(lat, lon, start_date, end_date)
    timestamps = sorted(G_data.keys())

    # -------- Output
    out_dir = Path(r"C:\Users\muthumaniraj\Documents\me research\me research\sep\v-shape")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "VShape_77cm_SeP_Autumn_15min.csv"

    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Date","Time",
            "Shape","Spacing_cm","Connection","Season",
            "Irradiation_Wm2","AmbientTemp_C","CellTemp_C",
            "Voc_V","Isc_A","Vm_V","Im_A","Pm_W","FF"
        ])

        # -------- Time loop (15-min interpolation)
        for i in range(len(timestamps)-1):
            ts1, ts2 = timestamps[i], timestamps[i+1]
            G1, G2 = G_data[ts1], G_data[ts2]
            T1, T2 = T_data[ts1], T_data[ts2]
            base = datetime.strptime(ts1, "%Y%m%d%H")

            for m in [0, 15, 30, 45]:
                frac = m / 60
                G = G1 + (G2 - G1) * frac
                T = T1 + (T2 - T1) * frac
                if G <= 5:
                    continue

                # ---- Effective irradiance
                G_eff = G * irr_factor

                # ---- Cell temperature
                T_cell = T + 0.025 * G_eff + spacing_offset + shape_temp_offset
                dT = T_cell - T_stc

                # ---- Electrical calculations
                Isc_array = (
                    Isc_stc * (G_eff/1000)
                    * (1 + alpha_isc*dT)
                    * current_loss
                ) * n_parallel

                Voc_array = (
                    Voc_stc * voltage_loss
                    * (1 + beta_voc*dT)
                ) * n_series

                Vm = vm_ratio * Voc_array
                Pm = FF * Voc_array * Isc_array
                Im = Pm / Vm if Vm > 0 else 0

                # ---- Write row
                w.writerow([
                    base.strftime("%Y-%m-%d"),
                    (base + timedelta(minutes=m)).strftime("%H:%M"),
                    "V-Shape", 77, "Se-P", "Autumn",
                    round(G_eff,1),
                    round(T,1),
                    round(T_cell,1),
                    round(Voc_array,2),
                    round(Isc_array,2),
                    round(Vm,2),
                    round(Im,2),
                    round(Pm,1),
                    FF
                ])

    print("✅ CSV generated successfully")
    print(f"📁 File saved at: {out_file}")

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    run_vshape_77cm_sep_autumn()
