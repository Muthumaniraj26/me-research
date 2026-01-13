# ===============================
# REAL-TIME SUMMER PV DATA REPORT
# ===============================

print("✅ Script started...")

import csv
import requests
from datetime import datetime

# -------------------------------------------------
# FETCH REAL HOURLY DATA FROM NASA POWER
# -------------------------------------------------
def fetch_nasa_power_data(lat, lon, start_date, end_date):
    """
    Fetch hourly solar irradiation (W/m²) and temperature (°C)
    from NASA POWER API
    """
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters=ALLSKY_SFC_SW_DWN,T2M&"
        f"community=RE&longitude={lon}&latitude={lat}&"
        f"start={start_date}&end={end_date}&format=JSON"
    )

    print("📡 Fetching real meteorological data from NASA POWER...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()["properties"]["parameter"]
    return data["ALLSKY_SFC_SW_DWN"], data["T2M"]


# -------------------------------------------------
# GENERATE REAL SUMMER PV CSV (APR–JUN)
# -------------------------------------------------
def generate_real_summer_pv_csv(
    lat=9.673502,                  # Tamil Nadu latitude in chittot
    lon=77.964719,                 # Tamil Nadu longitude in chittor
    start_date="20250401",     # YYYYMMDD
    end_date="20250630",
    panel_spacing_cm=93,       # 62 / 77 / 93
    output_file="summer_pv_real.csv"
):
    # -------------------------------
    # PV SYSTEM CONSTANTS (PAPER)
    # -------------------------------
    FF = 0.67
    Voc_stc = 24.8     # V
    Isc_stc = 0.60     # A
    T_stc = 25         # °C

    n_series = 4
    n_parallel = 3

    beta_voc = -0.0038    # −0.38 % / °C
    alpha_isc = 0.001     # +0.1 % / °C

    current_loss_factor = 0.82
    voltage_factor = 0.84
    vm_ratio = 0.80

    # -------------------------------
    # PANEL SPACING → COOLING EFFECT
    # -------------------------------
    spacing_offset = {
        93: -3.0,   # best cooling
        77:  0.0,
        62:  4.0    # hottest
    }

    if panel_spacing_cm not in spacing_offset:
        raise ValueError("Panel spacing must be 62, 77, or 93 cm")

    # -------------------------------
    # FETCH REAL DATA
    # -------------------------------
    irradiation_data, temp_data = fetch_nasa_power_data(
        lat, lon, start_date, end_date
    )

    print("🧮 Computing PV electrical parameters...")
    print(f"📐 Panel spacing: {panel_spacing_cm} cm")

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Date", "Time", "PanelSpacing_cm",
            "Irradiation_Wm2", "AmbientTemp_C", "CellTemp_C",
            "Voc_V", "Isc_A", "Vm_V", "Im_A", "Pm_W", "FF"
        ])

        for timestamp in irradiation_data:
            irradiation = irradiation_data[timestamp]
            ambient_temp = temp_data[timestamp]

            # Skip night hours
            if irradiation <= 0:
                continue

            dt = datetime.strptime(timestamp, "%Y%m%d%H")

            # -------------------------------
            # CELL TEMPERATURE
            # -------------------------------
            T_cell = (
                ambient_temp +
                (0.025 * irradiation) +
                spacing_offset[panel_spacing_cm]
            )
            delta_T = T_cell - T_stc

            # -------------------------------
            # CURRENT
            # -------------------------------
            Isc_panel = (
                Isc_stc *
                (irradiation / 1000) *
                (1 + alpha_isc * delta_T) *
                current_loss_factor
            )
            Isc_array = Isc_panel * n_parallel

            # -------------------------------
            # VOLTAGE
            # -------------------------------
            Voc_panel = (
                Voc_stc *
                voltage_factor *
                (1 + beta_voc * delta_T)
            )
            Voc_array = Voc_panel * n_series

            # -------------------------------
            # POWER
            # -------------------------------
            Vm_array = vm_ratio * Voc_array
            Pm_array = FF * Voc_array * Isc_array
            Im_array = Pm_array / Vm_array if Vm_array > 0 else 0

            writer.writerow([
                dt.strftime("%Y-%m-%d"),
                dt.strftime("%H:%M"),
                panel_spacing_cm,
                round(irradiation, 1),
                round(ambient_temp, 1),
                round(T_cell, 1),
                round(Voc_array, 2),
                round(Isc_array, 2),
                round(Vm_array, 2),
                round(Im_array, 2),
                round(Pm_array, 1),
                FF
            ])

    print(f"✅ CSV FILE CREATED SUCCESSFULLY")
    print(f"📁 File name: {output_file}")
    print("🎯 Real summer data (April–June) completed.")


# -------------------------------------------------
# MAIN EXECUTION (THIS MAKES IT RUN)
# -------------------------------------------------
if __name__ == "__main__":
    try:
        generate_real_summer_pv_csv(
            lat=9.67,
            lon=77.96,
            start_date="20250401",
            end_date="20250630",
            panel_spacing_cm=93,
            output_file="summer_93cm_real_chittor.csv"
        )
    except Exception as e:
        print("❌ ERROR OCCURRED:", e)
