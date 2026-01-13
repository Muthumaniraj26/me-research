# ============================================================
# INVERTED-V | 62 / 77 / 93 cm | TOTAL CROSS-TIED (TCT)
# Spring (February–March–April) | 15-minute resolution
# Real NASA POWER data + standard PV equations
# ============================================================

print("✅ Inverted-V | TCT | Spring simulation started")

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
def run_inverted_v_tct_spring():

    # -------- Location (Chittoor region)
    lat, lon = 9.6735, 77.9647

    # -------- Spring season (Feb–Mar–Apr)
    start_date = "20250201"
    end_date   = "20250430"

    # -------- PV module constants
    FF = 0.67
    Voc_stc = 24.8
    Isc_stc = 0.60
    T_stc   = 25

    # -------- Array configuration
    n_series   = 4
    n_parallel = 3

    beta_voc  = -0.0038
    alpha_isc = 0.001

    current_loss = 0.82
    voltage_loss = 0.84
    vm_ratio     = 0.80

    # -------- Shape: INVERTED-V
    irr_factor = 1.08          # East–West + sky-view gain
    shape_temp_offset = -2.0   # chimney cooling effect

    # -------- TCT mismatch gain
    tct_gain = 1.07

    # -------- Spacing cases
    spacing_cases = {
        62:  4.0,    # low clearance
        77:  0.0,    # baseline
        93: -3.0     # best airflow
    }

    # -------- Fetch NASA data
    G_data, T_data = fetch_nasa_power(lat, lon, start_date, end_date)
    timestamps = sorted(G_data.keys())

    # -------- Output directory
    out_dir = Path(r"C:\Users\muthumaniraj\Documents\me research\me research\tct\inverted-v\csv")
    out_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # LOOP THROUGH ALL SPACINGS
    # --------------------------------------------------------
    for spacing_cm, spacing_offset in spacing_cases.items():

        out_file = out_dir / f"InvertedV_{spacing_cm}cm_TCT_Spring_15min.csv"

        with open(out_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "Date","Time",
                "Shape","Spacing_cm","Connection","Season",
                "Irradiation_Wm2","AmbientTemp_C","CellTemp_C",
                "Voc_V","Isc_A","Vm_V","Im_A","Pm_W","FF"
            ])

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

                    # ---- Electrical calculations (TCT)
                    Isc_array = (
                        Isc_stc * (G_eff / 1000)
                        * (1 + alpha_isc * dT)
                        * current_loss
                    ) * n_parallel * tct_gain

                    Voc_array = (
                        Voc_stc * voltage_loss
                        * (1 + beta_voc * dT)
                    ) * n_series

                    Vm = vm_ratio * Voc_array
                    Pm = FF * Voc_array * Isc_array
                    Im = Pm / Vm if Vm > 0 else 0

                    w.writerow([
                        base.strftime("%Y-%m-%d"),
                        (base + timedelta(minutes=m)).strftime("%H:%M"),
                        "Inverted-V", spacing_cm, "TCT", "Spring",
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

        print(f"✅ Generated: {out_file}")

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    run_inverted_v_tct_spring()
