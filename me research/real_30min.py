# ==========================================
# REAL-TIME SUMMER PV DATA (30-MIN RESOLUTION)
# ==========================================

print("...Script started...")

import csv
import requests
from datetime import datetime, timedelta

# -------------------------------------------------
# FETCH HOURLY REAL DATA FROM NASA POWER
# -------------------------------------------------
def fetch_nasa_power_data(lat, lon, start_date, end_date):
    """
    Fetch hourly solar irradiation (W/m²) and ambient temperature (°C)
    from NASA POWER API
    """
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters=ALLSKY_SFC_SW_DWN,T2M&"
        f"community=RE&longitude={lon}&latitude={lat}&"
        f"start={start_date}&end={end_date}&format=JSON"
    )

    print("--------------------> Fetching real meteorological data from NASA POWER...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()["properties"]["parameter"]
    return data["ALLSKY_SFC_SW_DWN"], data["T2M"]


# -------------------------------------------------
# GENERATE REAL SUMMER PV CSV (30-MIN GAP)
# -------------------------------------------------
def generate_real_summer_pv_csv_30min(
    lat=9.673502,
    lon=77.964719,
    start_date="20250401",
    end_date="20250630",
    panel_spacing_cm=93,          # 62 / 77 / 93
    output_file="summer_30min_real.csv"
):
    # -------------------------------
    # PV SYSTEM CONSTANTS (PAPER)
    # -------------------------------
    FF = 0.67
    Voc_stc = 24.8      # V
    Isc_stc = 0.60      # A
    T_stc = 25          # °C

    n_series = 4
    n_parallel = 3

    beta_voc = -0.0038   # −0.38 % / °C
    alpha_isc = 0.001    # +0.1 % / °C

    current_loss_factor = 0.82
    voltage_factor = 0.84
    vm_ratio = 0.80

    # -------------------------------
    # PANEL SPACING → COOLING EFFECT
    # -------------------------------
    spacing_offset = {
        93: -3.0,   # high clearance
        77:  0.0,   # baseline
        62:  4.0    # low clearance
    }

    if panel_spacing_cm not in spacing_offset:
        raise ValueError("Panel spacing must be 62, 77, or 93 cm")

    # -------------------------------
    # FETCH REAL DATA
    # -------------------------------
    irradiation_data, temp_data = fetch_nasa_power_data(
        lat, lon, start_date, end_date
    )

    timestamps = sorted(irradiation_data.keys())

    print("-------------------->Computing PV electrical parameters (30-min resolution)...")
    print(f"-----------> Panel spacing: {panel_spacing_cm} cm")

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Date", "Time", "PanelSpacing_cm",
            "Irradiation_Wm2", "AmbientTemp_C", "CellTemp_C",
            "Voc_V", "Isc_A", "Vm_V", "Im_A", "Pm_W", "FF"
        ])

        # -----------------------------------------
        # LOOP THROUGH HOURLY DATA + INTERPOLATION
        # -----------------------------------------
        for i in range(len(timestamps) - 1):
            ts1 = timestamps[i]
            ts2 = timestamps[i + 1]

            G1 = irradiation_data[ts1]
            G2 = irradiation_data[ts2]

            T1 = temp_data[ts1]
            T2 = temp_data[ts2]

            dt1 = datetime.strptime(ts1, "%Y%m%d%H")

            # ---- FULL HOUR POINT (e.g., 09:00)
            write_pv_row(
                writer, dt1, G1, T1,
                panel_spacing_cm, spacing_offset,
                Voc_stc, Isc_stc, T_stc,
                beta_voc, alpha_isc,
                current_loss_factor, voltage_factor,
                vm_ratio, n_series, n_parallel, FF
            )

            # ---- 30-MIN INTERPOLATED POINT (e.g., 09:30)
            G_mid = (G1 + G2) / 2
            T_mid = (T1 + T2) / 2
            dt_mid = dt1 + timedelta(minutes=30)

            write_pv_row(
                writer, dt_mid, G_mid, T_mid,
                panel_spacing_cm, spacing_offset,
                Voc_stc, Isc_stc, T_stc,
                beta_voc, alpha_isc,
                current_loss_factor, voltage_factor,
                vm_ratio, n_series, n_parallel, FF
            )

    print("=================> CSV FILE CREATED SUCCESSFULLY")
    print(f"=================> File name: {output_file}")
    print("=================> Real summer data with 30-minute gap completed.")


# -------------------------------------------------
# HELPER FUNCTION (CALC + WRITE)
# -------------------------------------------------
def write_pv_row(
    writer, dt, irradiation, ambient_temp,
    panel_spacing_cm, spacing_offset,
    Voc_stc, Isc_stc, T_stc,
    beta_voc, alpha_isc,
    current_loss_factor, voltage_factor,
    vm_ratio, n_series, n_parallel, FF
):
    if irradiation <= 0:
        return

    # Cell temperature (rooftop + spacing)
    T_cell = (
        ambient_temp +
        (0.025 * irradiation) +
        spacing_offset[panel_spacing_cm]
    )
    delta_T = T_cell - T_stc

    # Current
    Isc_panel = (
        Isc_stc *
        (irradiation / 1000) *
        (1 + alpha_isc * delta_T) *
        current_loss_factor
    )
    Isc_array = Isc_panel * n_parallel

    # Voltage
    Voc_panel = (
        Voc_stc *
        voltage_factor *
        (1 + beta_voc * delta_T)
    )
    Voc_array = Voc_panel * n_series

    # Power
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


# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    try:
        generate_real_summer_pv_csv_30min(
            lat=9.673502,
            lon=77.964719,
            start_date="20250401",
            end_date="20250630",
            panel_spacing_cm=93,
            output_file="summer_93cm_real_30min.csv"
        )
    except Exception as e:
        print("XXXXXXXXXXXXXX ERROR OCCURRED XXXXXXXXXXXXXXXX:", e)
