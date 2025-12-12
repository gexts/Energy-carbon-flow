# plot_energy_balance.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def build_energy_balance_from_ts(ts_xlsx: str):
    """
    从 results_timeseries_wide.xlsx 构建系统分时能量输入/输出数据框。

    返回: df_inputs, df_outputs, hours
        - df_inputs: 行 = 小时(0..23), 列 = 输入端功率 [MW]
        - df_outputs: 行 = 小时(0..23), 列 = 输出端功率 [MW]
        - hours: list[int] 小时索引
    """
    ts_path = Path(ts_xlsx)
    xls = pd.ExcelFile(ts_path, engine="openpyxl")

    # 1) 电负荷：__snap_PD_MW__
    snap_PD = xls.parse("__snap_PD_MW__")
    hour_cols = [c for c in snap_PD.columns if c.startswith("H")]
    n_hours = len(hour_cols)
    hours = list(range(n_hours))

    snap_PD = snap_PD.set_index("bus_id")
    elec_load = snap_PD[hour_cols].sum(axis=0)   # Series(Hxx)

    # 2) EH 燃气输入：__snap_EH_inputs_MW__
    try:
        snap_eh_in = xls.parse("__snap_EH_inputs_MW__").set_index("type")
        if "gas" in snap_eh_in.index:
            gas_to_EH = snap_eh_in.loc["gas", hour_cols]
        else:
            gas_to_EH = pd.Series(0.0, index=hour_cols)
    except Exception:
        gas_to_EH = pd.Series(0.0, index=hour_cols)

    # 3) 热侧：heat_source_mix_MW
    heat_mix = xls.parse("heat_source_mix_MW").set_index("metric")
    heat_demand = heat_mix.loc["Demand",        hour_cols]
    pit_out     = heat_mix.loc["Pit_out",       hour_cols]
    pit_charge  = heat_mix.loc["Charge_to_pit", hour_cols]

    # 4) 电储能充/放电：storage_log
    try:
        storage_log = xls.parse("storage_log")
        grp = storage_log.groupby("t", as_index=True)
        batt_pch = grp["Pch_use"].sum().reindex(range(n_hours)).fillna(0.0)
        batt_pdc = grp["Pdc_use"].sum().reindex(range(n_hours)).fillna(0.0)
    except Exception:
        batt_pch = pd.Series(0.0, index=range(n_hours))
        batt_pdc = pd.Series(0.0, index=range(n_hours))

    # 5) 内部机组 / 外部电网发电：power_gen_mix_MW（若没有则退化为“全由电网供给”）
    try:
        gen_mix = xls.parse("power_gen_mix_MW").set_index("metric")
        internal_gen = gen_mix.loc["Internal_gen", hour_cols]
        grid_import  = gen_mix.loc["Grid_import",  hour_cols]
    except Exception:
        internal_gen = pd.Series(0.0, index=hour_cols)
        grid_import  = elec_load.copy()

    # 小工具：H00..H23 -> 0..23
    def _to_series_per_hour(v):
        arr = np.asarray(v, dtype=float).reshape(-1)
        return pd.Series(arr, index=range(len(arr)))

    s_elec_load     = _to_series_per_hour(elec_load)
    s_heat_demand   = _to_series_per_hour(heat_demand)
    s_gas_to_EH     = _to_series_per_hour(gas_to_EH)
    s_pit_out       = _to_series_per_hour(pit_out)
    s_pit_charge    = _to_series_per_hour(pit_charge)
    s_batt_pch      = _to_series_per_hour(batt_pch)
    s_batt_pdc      = _to_series_per_hour(batt_pdc)
    s_internal_gen  = _to_series_per_hour(internal_gen)
    s_grid_import   = _to_series_per_hour(grid_import)

    df_inputs = pd.DataFrame(
        {
            "Grid_electricity_MW":    s_grid_import,
            "Internal_generation_MW": s_internal_gen,
            "Natural_gas_to_EH_MW":   s_gas_to_EH,
            "Battery_discharge_MW":   s_batt_pdc,
            "Pit_discharge_MW":       s_pit_out,
        },
        index=hours,
    )

    df_outputs = pd.DataFrame(
        {
            "Electric_load_MW":  s_elec_load,
            "Heat_load_MW":      s_heat_demand,
            "Battery_charge_MW": s_batt_pch,
            "Pit_charge_MW":     s_pit_charge,
        },
        index=hours,
    )

    return df_inputs, df_outputs, hours


def plot_system_energy_balance(ts_xlsx, out_png=None, show=True):
    """
    绘制“综合能源系统分时能量平衡图”（输入在上、输出在下）。
    """
    df_in, df_out, hours = build_energy_balance_from_ts(ts_xlsx)

    # 保证列顺序稳定，便于多次对比
    df_in = df_in[["Grid_electricity_MW",
                   "Internal_generation_MW",
                   "Natural_gas_to_EH_MW",
                   "Battery_discharge_MW",
                   "Pit_discharge_MW"]]


    df_out = df_out[
        [
            "Electric_load_MW",
            "Heat_load_MW",
            "Battery_charge_MW",
            "Pit_charge_MW",
        ]
    ]

    x = np.arange(len(hours))

    fig, ax = plt.subplots(figsize=(12, 5))

    # 1) 画输入（x 轴上方，正值累积）
    bottom_in = np.zeros_like(x, dtype=float)
    for col in df_in.columns:
        values = df_in[col].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom_in, label=f"Input: {col}")
        bottom_in += values

    # 2) 画输出（x 轴下方，负值累积）
    bottom_out = np.zeros_like(x, dtype=float)
    for col in df_out.columns:
        values = df_out[col].to_numpy(dtype=float)
        ax.bar(x, -values, bottom=-bottom_out, label=f"Output: {col}")
        bottom_out += values

    # 3) 装饰
    ax.axhline(0.0, linewidth=1.0)  # x 轴

    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in hours])  # 0~23
    ax.set_xlim(-0.5, len(hours) - 0.5)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power / Energy Rate [MW]")
    ax.set_title("System-level Time-slice Energy Balance")

    # 图例放在右侧，避免遮挡图形
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.tight_layout()

    if out_png is not None:
        out_path = Path(out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    ts_path = r"F:\25秋科研资料\IES_critical_modelling\dataset\outputs\ts\results_timeseries_wide.xlsx"
    out_fig = r"F:\25秋科研资料\IES_critical_modelling\dataset\outputs\ts\energy_balance.png"
    plot_system_energy_balance(ts_path, out_fig, show=False)
