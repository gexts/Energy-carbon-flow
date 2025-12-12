# plot_energy_balance.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ----------------------------
# Helpers
# ----------------------------
def _hour_cols(df: pd.DataFrame):
    return [c for c in df.columns if isinstance(c, str) and c.startswith("H")]

def _series_from_row(df: pd.DataFrame, key_col: str, key: str, hour_cols):
    """df: wide table, one row per key; returns Series indexed by hour index 0..n-1"""
    if df is None or df.empty:
        return pd.Series(0.0, index=range(len(hour_cols)))
    if key_col not in df.columns:
        return pd.Series(0.0, index=range(len(hour_cols)))
    hit = df[df[key_col].astype(str) == str(key)]
    if hit.empty:
        return pd.Series(0.0, index=range(len(hour_cols)))
    row = hit.iloc[0]
    vals = [0.0 if pd.isna(row[c]) else float(row[c]) for c in hour_cols]
    return pd.Series(vals, index=range(len(hour_cols)))

def _to_series(v, n_hours: int):
    """accept Series indexed by Hxx or 0..23; return Series indexed by 0..n-1"""
    if isinstance(v, pd.Series):
        if all(isinstance(i, str) and i.startswith("H") for i in v.index):
            vals = [float(v.get(f"H{h:02d}", 0.0)) for h in range(n_hours)]
            return pd.Series(vals, index=range(n_hours))
        # assume already hour index
        return v.reindex(range(n_hours)).fillna(0.0).astype(float)
    arr = np.asarray(v, dtype=float).reshape(-1)
    return pd.Series(arr[:n_hours], index=range(n_hours)).fillna(0.0)

def _stacked_balance_bar(df_in: pd.DataFrame, df_out: pd.DataFrame, title: str, out_png: str, show: bool):
    hours = list(df_in.index)
    x = np.arange(len(hours))

    fig, ax = plt.subplots(figsize=(13, 5))

    # inputs (positive)
    bottom_in = np.zeros_like(x, dtype=float)
    for col in df_in.columns:
        y = df_in[col].to_numpy(dtype=float)
        ax.bar(x, y, bottom=bottom_in, label=f"In: {col}")
        bottom_in += y

    # outputs (negative)
    bottom_out = np.zeros_like(x, dtype=float)
    for col in df_out.columns:
        y = df_out[col].to_numpy(dtype=float)
        ax.bar(x, -y, bottom=-bottom_out, label=f"Out: {col}")
        bottom_out += y

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in hours])
    ax.set_xlim(-0.5, len(hours) - 0.5)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power / Energy Rate [MW]")
    ax.set_title(title)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()

    if out_png:
        out_path = Path(out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------
# 1) Network-side balance
# ----------------------------
def build_network_balance_from_ts(ts_xlsx: str):
    """
    图1：园区外边界（电网+气网+电池）能量平衡。
    - 输入：Grid_import + Internal_generation(剔除电池) + Gas_supply + Battery_discharge
    - 输出：Electric_load + Gas_load + Battery_charge + Grid_export(若有) + Gas_compressor(若有)
    注意：不把热网/储热计入这张图（EH 热转换已体现在电/气侧负荷中）。
    """
    xls = pd.ExcelFile(Path(ts_xlsx), engine="openpyxl")

    # Electric load: __snap_PD_MW__ (不含储能充电/售电额外负荷，因为快照在 overlay 阶段取)
    snap_PD = xls.parse("__snap_PD_MW__")
    hour_cols = _hour_cols(snap_PD)
    n_hours = len(hour_cols)
    hours = list(range(n_hours))
    elec_load = snap_PD.set_index(snap_PD.columns[0])[hour_cols].sum(axis=0)  # Series(Hxx)

    # Battery charge/discharge: storage_log (Pch_use/Pdc_use, MW)
    try:
        storage_log = xls.parse("storage_log")
        grp = storage_log.groupby("t", as_index=True)
        batt_pch = grp["Pch_use"].sum().reindex(range(n_hours)).fillna(0.0)
        batt_pdc = grp["Pdc_use"].sum().reindex(range(n_hours)).fillna(0.0)
    except Exception:
        batt_pch = pd.Series(0.0, index=range(n_hours))
        batt_pdc = pd.Series(0.0, index=range(n_hours))

    # Power generation mix: power_gen_mix_MW (Internal_gen, Grid_import)
    try:
        gen_mix = xls.parse("power_gen_mix_MW")
        # columns: metric, H00..H23
        internal_gen = _series_from_row(gen_mix, "metric", "Internal_gen", hour_cols)
        grid_import  = _series_from_row(gen_mix, "metric", "Grid_import",  hour_cols)
    except Exception:
        internal_gen = pd.Series(0.0, index=range(n_hours))
        grid_import  = _to_series(elec_load, n_hours)

    # Grid trade buy/sell (建议由 scenario_runner 导出；若无则默认 0)
    try:
        grid_trade = xls.parse("grid_trade_mix_MW")
        grid_buy  = _series_from_row(grid_trade, "metric", "Buy_MW",  hour_cols)
        grid_sell = _series_from_row(grid_trade, "metric", "Sell_MW", hour_cols)
        # 若你希望“外部电网输入”严格用 buy 而非潮流里算出来的 Grid_import，可在这里替换
        # grid_import = grid_buy.copy()
    except Exception:
        grid_sell = pd.Series(0.0, index=range(n_hours))

    # Gas network mix (建议由 scenario_runner 导出；若无则做降级：只用 EH gas 输入近似)
    try:
        gas_mix = xls.parse("gas_network_mix_MW")
        gas_supply = _series_from_row(gas_mix, "metric", "Supply_MW", hour_cols)
        gas_load   = _series_from_row(gas_mix, "metric", "Load_MW",   hour_cols)
        gas_comp   = _series_from_row(gas_mix, "metric", "Compressor_MW", hour_cols)
    except Exception:
        # fallback: use __snap_EH_inputs_MW__ gas as proxy (只代表EH用气，不含其他气负荷)
        try:
            snap_eh = xls.parse("__snap_EH_inputs_MW__")
            gas_to_eh = _series_from_row(snap_eh, "type", "gas", hour_cols)
        except Exception:
            gas_to_eh = pd.Series(0.0, index=range(n_hours))
        gas_supply = gas_to_eh.copy()
        gas_load   = gas_to_eh.copy()
        gas_comp   = pd.Series(0.0, index=range(n_hours))
        print("[plot_energy_balance] WARN: missing sheet 'gas_network_mix_MW'; using EH gas input as proxy for gas network.")

    # Avoid double count: internal_gen includes battery discharge (因电池放电作为 extra generator 进入潮流)
    internal_gen_no_batt = (internal_gen - batt_pdc).clip(lower=0.0)

    df_in = pd.DataFrame(
        {
            "Grid_import_MW":        _to_series(grid_import, n_hours),
            "Internal_gen_noBatt_MW": _to_series(internal_gen_no_batt, n_hours),
            "Gas_supply_MW":         _to_series(gas_supply, n_hours),
            "Battery_discharge_MW":  _to_series(batt_pdc, n_hours),
        },
        index=hours,
    )

    df_out = pd.DataFrame(
        {
            "Electric_load_MW":   _to_series(elec_load, n_hours),
            "Gas_load_MW":        _to_series(gas_load, n_hours),
            "Gas_compressor_MW":  _to_series(gas_comp, n_hours),
            "Battery_charge_MW":  _to_series(batt_pch, n_hours),
            "Grid_export_MW":     _to_series(grid_sell, n_hours),
        },
        index=hours,
    )

    # residual diagnostics (do not force-close)
    resid = df_in.sum(axis=1) - df_out.sum(axis=1)
    print(f"[Network balance] max |residual| = {float(resid.abs().max()):.4f} MW ; mean residual = {float(resid.mean()):.4f} MW")

    return df_in, df_out


def plot_network_energy_balance(ts_xlsx: str, out_png: str, show: bool = False):
    df_in, df_out = build_network_balance_from_ts(ts_xlsx)
    _stacked_balance_bar(
        df_in,
        df_out,
        title="Network-side Energy Balance (Electric + Gas + Battery)",
        out_png=out_png,
        show=show,
    )


# ----------------------------
# 2) EH-side balance
# ----------------------------
def build_eh_balance_from_ts(ts_xlsx: str):
    """
    图2：EH 子系统能量平衡（包含储热池）。
    - 输入：Electricity_in + Gas_in + Pit_discharge
    - 输出：Electric_out + Heat_to_load + Heat_to_pit + Cooling_out
    其中 Heat_to_load/Heat_to_pit 来自 heat_source_mix_MW（EH并联系统逻辑），Electric_out/Cooling_out 建议由 EH_output_mix_MW 提供。
    """
    xls = pd.ExcelFile(Path(ts_xlsx), engine="openpyxl")

    # hours basis: __snap_EH_inputs_MW__
    snap_eh = xls.parse("__snap_EH_inputs_MW__")
    hour_cols = _hour_cols(snap_eh)
    n_hours = len(hour_cols)
    hours = list(range(n_hours))

    elec_in = _series_from_row(snap_eh, "type", "electricity", hour_cols)
    gas_in  = _series_from_row(snap_eh, "type", "gas",         hour_cols)

    # heat_source_mix_MW provides: Demand, EH_total, EH_direct, Pit_out, Charge_to_pit
    try:
        heat_mix = xls.parse("heat_source_mix_MW")
        pit_out     = _series_from_row(heat_mix, "metric", "Pit_out",       hour_cols)
        pit_charge  = _series_from_row(heat_mix, "metric", "Charge_to_pit", hour_cols)

        # 你这套统计里，“对外供热/热负荷”更可靠的是 Demand 或 ToHeat_total
        heat_demand = _series_from_row(heat_mix, "metric", "Demand",        hour_cols)
        to_heat     = _series_from_row(heat_mix, "metric", "ToHeat_total",  hour_cols)

        # 二选一：推荐先用 Demand（热负荷）作为 EH 子系统对外“热输出”
        heat_to_load = heat_demand
        # 如果你更想体现“注入热网的能量”，就用：
        # heat_to_load = to_heat

    except Exception:
        pit_out    = pd.Series(0.0, index=range(n_hours))
        eh_direct  = pd.Series(0.0, index=range(n_hours))
        pit_charge = pd.Series(0.0, index=range(n_hours))

    # EH output mix (建议由 scenario_runner 导出)
    try:
        eh_out_mix = xls.parse("EH_output_mix_MW")
        elec_out = _series_from_row(eh_out_mix, "metric", "Electric_out_MW", hour_cols)
        cool_out = _series_from_row(eh_out_mix, "metric", "Cooling_out_MW",  hour_cols)
        # total_heat_out can be read for checking but not required
    except Exception:
        elec_out = pd.Series(0.0, index=range(n_hours))
        cool_out = pd.Series(0.0, index=range(n_hours))
        print("[plot_energy_balance] WARN: missing sheet 'EH_output_mix_MW'; set EH Electric/Cooling outputs to 0.")

    df_in = pd.DataFrame(
        {
            "EH_electricity_in_MW": _to_series(elec_in, n_hours),
            "EH_gas_in_MW":         _to_series(gas_in,  n_hours),
            "Pit_discharge_MW":     _to_series(pit_out, n_hours),
        },
        index=hours,
    )

    df_out = pd.DataFrame(
    {
        "EH_electric_out_MW":      _to_series(elec_out,    n_hours),
        "Heat_to_load_MW":         _to_series(heat_to_load,n_hours),
        "Heat_to_pit_storage_MW":  _to_series(pit_charge,  n_hours),
        "Cooling_out_MW":          _to_series(cool_out,    n_hours),
    },
    index=hours,
)
    print("[EH DEBUG] out max:", df_out.max().to_dict())
    print("[EH DEBUG] out mean:", df_out.mean().to_dict())

    resid = df_in.sum(axis=1) - df_out.sum(axis=1)
    print(f"[EH balance] max |residual| = {float(resid.abs().max()):.4f} MW ; mean residual = {float(resid.mean()):.4f} MW")
    return df_in, df_out


def plot_eh_energy_balance(ts_xlsx: str, out_png: str, show: bool = False):
    df_in, df_out = build_eh_balance_from_ts(ts_xlsx)
    _stacked_balance_bar(
        df_in,
        df_out,
        title="Energy Hub Balance (incl. Pit Thermal Storage)",
        out_png=out_png,
        show=show,
    )


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent / "dataset" / "outputs" / "ts"
    ts_path = ROOT / "results_timeseries_wide.xlsx"

    out1 = ROOT / "energy_balance_network.png"
    out2 = ROOT / "energy_balance_EH.png"

    plot_network_energy_balance(str(ts_path), str(out1), show=False)
    plot_eh_energy_balance(str(ts_path), str(out2), show=False)

    print("[plot_energy_balance] saved:", out1)
    print("[plot_energy_balance] saved:", out2)
