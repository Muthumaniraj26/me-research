# ============================================================
# INVERTED-V | 93 cm | TOTAL CROSS TIED (TCT)
# Summer (Apr–Jun) | 15-minute resolution
# ============================================================

print("✅ Inverted-V | 93 cm | TCT simulation started")

import csv
import requests
from datetime import datetime, timedelta
from pathlib import Path

# ------------------------------------------------------------
# NASA POWER FETCH
# ------------------------------------------------------------
def fetch_nasa(lat, lon, start_date, end_date):
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start_date}&end={end_date}&format=JSON"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    return p["ALLSKY_SFC_SW_DWN"], p["T2M"]

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def run():

    # -------- Location
    lat, lon = 9.6735, 77.9647

    # -------- Summer months
    start_date = "20250401"
    end_date   = "20250630"

    # -------- PV constants
    FF = 0.67
    Voc_stc = 24.8
    Isc_stc = 0.60
    T_stc = 25

    n_series = 4
    n_parallel = 3

    beta_voc = -0.0038
    alpha_isc = 0.001

    current_loss = 0.82
    voltage_loss = 0.84
    vm_ratio = 0.80

    # -------- SHAPE: Inverted-V
    irr_factor = 1.08
    shape_temp_offset = -2

    # -------- SPACING: 93 cm
    spacing_offset = -3

    # -------- CONNECTION: TCT
    tct_gain = 1.07   # mismatch reduction factor

    # -------- Fetch data
    G_data, T_data = fetch_nasa(lat, lon, start_date, end_date)
    timestamps = sorted(G_data.keys())

    # -------- Output
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "InvertedV_93cm_TCT_Summer_15min.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date","Time",
            "Shape","Spacing_cm","Connection",
            "Irradiation_Wm2","AmbientTemp_C","CellTemp_C",
            "Voc_V","Isc_A","Vm_V","Im_A","Pm_W","FF"
        ])

        for i in range(len(timestamps)-1):
            ts1, ts2 = timestamps[i], timestamps[i+1]
            G1, G2 = G_data[ts1], G_data[ts2]
            T1, T2 = T_data[ts1], T_data[ts2]

            base_dt = datetime.strptime(ts1, "%Y%m%d%H")

            for m in [0,15,30,45]:
                frac = m / 60
                G = G1 + (G2 - G1)*frac
                T = T1 + (T2 - T1)*frac
                dt = base_dt + timedelta(minutes=m)

                if G <= 5:
                    continue

                G_eff = G * irr_factor

                T_cell = (
                    T + 0.025*G_eff
                    + spacing_offset
                    + shape_temp_offset
                )
                dT = T_cell - T_stc

                # -------- Current (TCT gain applied)
                Isc_panel = (
                    Isc_stc*(G_eff/1000)
                    *(1 + alpha_isc*dT)
                    * current_loss
                )
                Isc_array = Isc_panel * n_parallel * tct_gain

                # -------- Voltage
                Voc_panel = (
                    Voc_stc * voltage_loss
                    * (1 + beta_voc*dT)
                )
                Voc_array = Voc_panel * n_series

                Vm = vm_ratio * Voc_array
                Pm = FF * Voc_array * Isc_array
                Im = Pm / Vm if Vm > 0 else 0

                writer.writerow([
                    dt.strftime("%Y-%m-%d"),
                    dt.strftime("%H:%M"),
                    "Inverted-V",93,"TCT",
                    round(G_eff,1),round(T,1),round(T_cell,1),
                    round(Voc_array,2),round(Isc_array,2),
                    round(Vm,2),round(Im,2),
                    round(Pm,1),FF
                ])

    print("✅ CSV generated:", out_file)

# ------------------------------------------------------------
if __name__ == "__main__":
    run()
