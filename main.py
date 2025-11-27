# main.py
import os
import numpy as np
import pandas as pd

from testdata import build_test_system_from_excel
from power_flow import run_power_flow
from gas_flow import run_gas_flow
from energy_hub import run_energy_hub
from heat_flow import run_heat_flow
from pit_storage import PitThermalStorage, PitThermalStorageConfig
from io_utils import save_results_to_excel


# 你给出的数据目录（可以按需修改/传参）
DATASET_DIR = r"F:\25秋科研资料\IES_critical_modelling\dataset"
OUTPUT_DIR  = os.path.join(DATASET_DIR, "outputs")

def main():
    # === 读取 Excel 构建系统 ===
    TS = build_test_system_from_excel(DATASET_DIR)

    # —— 新增：构造储热对象（可按需改参数）
    pit_cfg = PitThermalStorageConfig(
        H=16.0, W_top=90.0, W_bot=26.0, n_layers=20,
        U_side=111.0, U_top=26.6, U_bot=74.9,
        T_top_env=35.0, T_side_env=25.0, T_bot_env=25.0,
        T_ref=25.0, T_initial=25.0, dt_int=15.0,
        w_es0=None
    )
    pit = PitThermalStorage(pit_cfg)


    print(">>> [1/4] Power Flow + CEF (PYPOWER style)")
    TS = run_power_flow(TS)
    print("Node NCI (tCO2/MWh):", np.round(TS['PowerSystem']['NCI'], 4))

    print("\n>>> [2/4] Gas Flow + CEFR")
    TS = run_gas_flow(TS)
    print("Gas node injection (km3/h):", np.round(TS['GasSystem']['nodes']['injection'].values, 3))

    print("\n>>> [3/4] Energy Hub coupling")
    TS = run_energy_hub(TS)
    print(TS['EnergyHubs']['outputs'][['type','flow','PCI','CEFR']])

    print("\n>>> [4/4] Heat Network (steady placeholder)")
    TS = run_heat_flow(TS)
    print(f"Heat total out (MW est.): {TS['HeatSystem']['Q_total_MW']:.2f} | "
          f"CEFR_total (tCO2/h): {TS['HeatSystem']['CEFR_total']:.2f}")

    # === 统一导出结果 ===
    save_results_to_excel(TS, OUTPUT_DIR)
    print(f"\nAll done. Results saved to: {OUTPUT_DIR}")

from scenario_runner import run_timeseries_day_wide
if __name__ == "__main__":
    # 单时刻仍可运行
    main()

    # 逐时序列运行（输出在 dataset\outputs\ts 目录）
    OUT_TS = os.path.join(OUTPUT_DIR, "ts")
    path_ts = run_timeseries_day_wide(DATASET_DIR, OUT_TS, hours=24, convert_heat_load_to_injection=False)
    print("Timeseries (wide) results:", path_ts)

