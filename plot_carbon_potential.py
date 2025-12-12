# plot_carbon_potential.py  (v2: matches results_timeseries_wide.xlsx)
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.transforms import Bbox

def _save_fig_3d_no_crop(fig, out_png: str, pad_inches: float = 0.65):
    """
    3D 图 tight bbox 经常不包含 zlabel / colorbar label，导致裁切。
    这里先 draw，再用 fig.get_tightbbox(renderer) 手动扩 padding 保存。
    """
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer).padded(pad_inches)
    fig.savefig(out_png, dpi=300, bbox_inches=bbox)
    plt.close(fig)

def _upsample_Z_linear(t, y, Z, t_new, y_new):
    """
    不依赖 SciPy 的二维线性插值：
    - 先按时间 t 对每条 bus 曲线插值
    - 再按 bus y 对每个时刻插值
    Z shape: (len(y), len(t))
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    t_new = np.asarray(t_new, dtype=float)
    y_new = np.asarray(y_new, dtype=float)

    # 1) time interpolation
    Zt = np.zeros((len(y), len(t_new)), dtype=float)
    for i in range(len(y)):
        Zt[i, :] = np.interp(t_new, t, Z[i, :])

    # 2) y interpolation
    Znew = np.zeros((len(y_new), len(t_new)), dtype=float)
    for j in range(len(t_new)):
        Znew[:, j] = np.interp(y_new, y, Zt[:, j])
    return Znew


TS_XLSX_DEFAULT = os.path.join("dataset", "outputs", "ts", "results_timeseries_wide.xlsx")
OUTDIR_DEFAULT  = os.path.join("dataset", "outputs", "figs_carbon")


def _hcols(hours=24):
    return [f"H{h:02d}" for h in range(hours)]


def _read(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    try:
        return xls.parse(sheet)
    except Exception as e:
        print(f"[plot_carbon] WARN: cannot read '{sheet}': {e}")
        return pd.DataFrame()


def _wide_to_Z(df: pd.DataFrame, id_col: str, hours=24):
    if df is None or df.empty or id_col not in df.columns:
        return None
    hc = [c for c in _hcols(hours) if c in df.columns]
    if not hc:
        return None
    ids = df[id_col].to_numpy()
    Z = df[hc].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return ids, np.arange(len(hc)), Z, hc


def plot_power_nci_3d(ts_xlsx: str, out_png: str, hours=24):
    xls = pd.ExcelFile(ts_xlsx, engine="openpyxl")
    df = _read(xls, "power_NCI")
    got = _wide_to_Z(df, "bus_id", hours=hours)
    if got is None:
        print("[plot_carbon] WARN: power_NCI missing/empty -> skip")
        return
    bus_ids, t, Z, hc = got

    # tCO2/MWh -> kg/MWh
    Z = Z * 1000.0

    # y axis
    try:
        y = bus_ids.astype(float)
    except Exception:
        y = np.arange(1, len(bus_ids) + 1, dtype=float)

    # ---- upsample for smooth surface ----
    t_new = np.linspace(t.min(), t.max(), 240)      # 时间加密
    y_new = np.linspace(y.min(), y.max(), 140)      # 节点加密
    Z_smooth = _upsample_Z_linear(t, y, Z, t_new, y_new)

    X, Y = np.meshgrid(t_new, y_new)
    zmin = float(np.nanmin(Z_smooth))
    zmax = float(np.nanmax(Z_smooth))

    fig = plt.figure(figsize=(16.0, 8.8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z_smooth,
        linewidth=0, antialiased=True, alpha=0.98,
        cmap="jet"
    )
    # 底部投影（参考图风格）
    ax.contourf(X, Y, Z_smooth, zdir="z", offset=zmin, levels=30, cmap="jet", alpha=0.90)

    ax.set_title("Power Grid Node Carbon Potential (NCI)", pad=18)
    ax.set_xlabel("Hour", labelpad=14)
    ax.set_ylabel("Bus ID", labelpad=14)
    ax.set_zlabel("Carbon Potential (kg/MWh)", labelpad=18)
    ax.set_zlim(zmin, zmax)

    # 让视角更“长条”一点
    ax.view_init(elev=25, azim=-135)

    # colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.10)
    cbar.set_label("kg/MWh")

    # 不用 tight_layout；用“手动 bbox 保存”避免裁切
    _save_fig_3d_no_crop(fig, out_png, pad_inches=0.75)
    print(f"[plot_carbon] saved: {out_png}")



def plot_heat_source_ci_line(ts_xlsx: str, out_png: str, hours=24):
    xls = pd.ExcelFile(ts_xlsx, engine="openpyxl")
    df = _read(xls, "heat_input_GCI_tperMWh")
    got = _wide_to_Z(df, "source_id", hours=hours)
    if got is None:
        print("[plot_carbon] WARN: heat_input_GCI_tperMWh missing/empty -> skip")
        return
    source_ids, t, Z, hc = got  # tCO2/MWh

    # tCO2/MWh -> kg/MWh
    Z = Z * 1000.0

    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=160)
    for i, sid in enumerate(source_ids):
        ax.plot(t, Z[i, :], marker="o", linewidth=1.8, label=f"Source {sid}")

    ax.set_title("Heat Source Carbon Potential vs Time")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Carbon Potential (kg/MWh)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[plot_carbon] saved: {out_png}")



def plot_battery_storage_ci(ts_xlsx: str, out_png: str):
    xls = pd.ExcelFile(ts_xlsx, engine="openpyxl")
    df = _read(xls, "storage_log")
    if df.empty:
        print("[plot_carbon] WARN: storage_log empty -> skip battery plot")
        return
    need = {"t", "id", "w_es_t", "w_es_next"}
    if not need.issubset(set(df.columns)):
        print(f"[plot_carbon] WARN: storage_log missing columns {need - set(df.columns)} -> skip")
        return

    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=160)

    for sid, g in df.groupby("id"):
        g = g.sort_values("t")
        ax.plot(g["t"], g["w_es_t"], label=f"{sid}: w_es_t")
        ax.plot(g["t"], g["w_es_next"], linestyle="--", label=f"{sid}: w_es_next")

    ax.set_title("Battery Storage Carbon Potential (tCO2/MWh)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Carbon Potential (tCO2/MWh)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(f"[plot_carbon] saved: {out_png}")


def plot_pit_storage_ci(ts_xlsx: str, out_png: str, hours=24):
    """
    你当前 results_timeseries_wide.xlsx 里：
      storage_heat_wes_tperMWh / storage_heat_e_MWh / storage_heat_E_t 都是空表
    所以这里会给出“空表提示”，避免画空白 0~1 图误导。
    """
    xls = pd.ExcelFile(ts_xlsx, engine="openpyxl")
    df = _read(xls, "storage_heat_wes_tperMWh")
    if df.empty or len(df.columns) <= 1:
        print("[plot_carbon] WARN: storage_heat_wes_tperMWh is EMPTY. "
              "This means pit CI was NOT exported. Apply scenario_runner patch (see below).")
        return

    # 正常情况下应为：scope + H00..H23
    id_col = "scope" if "scope" in df.columns else df.columns[0]
    got = _wide_to_Z(df, id_col, hours=hours)
    if got is None:
        print("[plot_carbon] WARN: pit CI sheet exists but has no Hxx columns -> skip")
        return
    scopes, t, Z, hc = got

    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=160)
    for i, sc in enumerate(scopes):
        ax.plot(t, Z[i, :], label=str(sc))
    ax.set_title("Pit Thermal Storage Carbon Potential (tCO2/MWh)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Carbon Potential (tCO2/MWh)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(f"[plot_carbon] saved: {out_png}")


def main(ts_xlsx=TS_XLSX_DEFAULT, outdir=OUTDIR_DEFAULT):
    print(f"[plot_carbon] using: {ts_xlsx}")
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)

    plot_power_nci_3d(ts_xlsx, os.path.join(outdir, "power_NCI_3d.png"))
    plot_heat_source_ci_line(ts_xlsx, os.path.join(outdir, "heat_source_CI_line.png"), hours=24)
    plot_battery_storage_ci(ts_xlsx, os.path.join(outdir, "battery_CI.png"))
    plot_pit_storage_ci(ts_xlsx, os.path.join(outdir, "pit_CI.png"))

    print(f"[plot_carbon] saved to: {outdir}")


if __name__ == "__main__":
    main()
