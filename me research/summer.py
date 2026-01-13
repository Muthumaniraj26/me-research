import csv
from datetime import datetime, timedelta
import math
import random

def generate_summer_pv_csv(
    start_date="2025-04-01",
    months=3,
    output_file="pv_summer_apr_may_jun.csv"
):
    # ===============================
    # SYSTEM CONSTANTS (from paper)
    # ===============================
    FF = 0.67
    Voc_stc = 24.8
    Isc_stc = 0.60
    T_stc = 25

    n_series = 4
    n_parallel = 3

    beta_voc = -0.0038     # −0.38 % / °C
    alpha_isc = 0.001      # +0.1 % / °C

    current_loss_factor = 0.82
    voltage_factor = 0.84
    vm_ratio = 0.80

    # Time slots: 08:30 – 18:30 (summer longer sun)
    time_slots = [
        (datetime.strptime("08:30", "%H:%M") + timedelta(minutes=30*i)).strftime("%H:%M")
        for i in range(21)
    ]

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=months * 30)

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Date", "Time", "Irradiation_Wm2", "AmbientTemp_C", "CellTemp_C",
            "Voc_V", "Isc_A", "Vm_V", "Im_A", "Pm_W", "FF"
        ])

        current_date = start_dt

        while current_date < end_dt:
            # Summer ambient temperature
            T_ambient = random.uniform(32, 42)

            # Higher summer peak irradiation
            daily_peak = random.randint(1050, 1150)

            for idx, time_str in enumerate(time_slots):
                # Summer irradiation curve (wider bell)
                angle = (idx / (len(time_slots) - 1)) * math.pi
                irradiation = max(
                    250,
                    daily_peak * math.sin(angle) + random.uniform(-40, 40)
                )

                # Cell temperature (summer realistic)
                T_cell = T_ambient + (0.025 * irradiation / 10)
                delta_T = T_cell - T_stc

                # -------------------------------
                # ELECTRICAL CALCULATIONS
                # -------------------------------
                Isc_panel = (
                    Isc_stc *
                    (irradiation / 1000) *
                    (1 + alpha_isc * delta_T) *
                    current_loss_factor
                )
                Isc_array = Isc_panel * n_parallel

                Voc_panel = (
                    Voc_stc *
                    voltage_factor *
                    (1 + beta_voc * delta_T)
                )
                Voc_array = Voc_panel * n_series

                Vm_array = vm_ratio * Voc_array
                Pm_array = FF * Voc_array * Isc_array
                Im_array = Pm_array / Vm_array if Vm_array > 0 else 0

                writer.writerow([
                    current_date.strftime("%Y-%m-%d"),
                    time_str,
                    int(irradiation),
                    round(T_ambient, 1),
                    round(T_cell, 1),
                    round(Voc_array, 2),
                    round(Isc_array, 2),
                    round(Vm_array, 2),
                    round(Im_array, 2),
                    round(Pm_array, 0),
                    FF
                ])

            current_date += timedelta(days=1)

    print(f"✅ Summer CSV generated: {output_file}")

# RUN
generate_summer_pv_csv()
