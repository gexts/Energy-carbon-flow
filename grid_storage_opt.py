# grid_storage_opt.py
"""
电力系统与电储能的日尺度购售电 + 储能调度优化模块

功能定位：
- 作为现有 IES 仿真体系的“上游优化层”，
  在给定内部机组出力、系统负荷、电价曲线和储能参数的条件下，
  求解 24 小时的：
    * 外部购电功率 P_buy(t)
    * 外部售电功率 P_sell(t)
    * 储能充电功率 P_ch(t)
    * 储能放电功率 P_dc(t)
- 优化目标：24h 电费最小化（不显式加入碳价）
- 优化结果写回 Excel：
    * storage_dispatch_ch / storage_dispatch_dc
    * grid_trade_buy / grid_trade_sell

注意：
- 当前版本假定只有一台电储能（storage_system 表中的第一行 / 指定 id）。
- 优化在系统“能量平衡”层面进行，不直接包含 AC 潮流约束；
  潮流与碳流仍由原有 scenario_runner + power_flow + gas/heat/EH 模块负责。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog


HOURS = 24
H_COLS = [f"H{h:02d}" for h in range(HOURS)]


@dataclass
class StorageParams:
    id: str
    Pch_max: float
    Pdc_max: float
    eta_ch: float
    eta_dc: float
    kappa: float
    e_min: float
    e_max: float
    e0: float


@dataclass
class OptimizationInputs:
    L: np.ndarray           # shape (T,), 系统电负荷总量 (MW)
    G_local: np.ndarray     # shape (T,), 内部机组总出力 (MW)
    price_buy: np.ndarray   # shape (T,), 购电电价 (元/MWh)
    price_sell: np.ndarray  # shape (T,), 售电电价 (元/MWh)
    storage: StorageParams
    dt: float = 1.0         # 小时步长


@dataclass
class OptimizationResult:
    P_buy: np.ndarray       # (T,)
    P_sell: np.ndarray      # (T,)
    P_ch: np.ndarray        # (T,)
    P_dc: np.ndarray        # (T,)
    e: np.ndarray           # (T+1,), e[0] = e0
    obj_value: float        # 最小化目标值（总电费）
    status: int
    message: str


def _load_optimization_inputs(dataset_path: str,
                              storage_id: Optional[str] = None,
                              tariff_id: Optional[str] = None
                              ) -> OptimizationInputs:
    """
    从 ies_dataset_extgrid.xlsx 加载优化所需的聚合曲线与储能参数。
    - storage_id: storage_system.id（若为 None，则取第一行）
    - tariff_id: elec_price_buy / sell 中的 tariff_id（若无此列则取第一行）
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"未找到数据文件：{dataset_path}")
    xls = pd.ExcelFile(dataset_path, engine="openpyxl")

    # 1) 系统总负荷 L_t = sum_bus PD_bus(t)
    df_PD = xls.parse("power_PD")
    if not set(H_COLS).issubset(df_PD.columns):
        raise ValueError("power_PD 缺少 Hxx 列，请检查。")
    L = df_PD[H_COLS].to_numpy(dtype=float).sum(axis=0)  # (T,)

    # 2) 内部机组总出力 G_local_t = sum_gen PG_gen(t)
    #    注意：ext_grid 不在 gen_PG 表中；该表只包含内部机组
    df_genPG = xls.parse("gen_PG")
    if not set(H_COLS).issubset(df_genPG.columns):
        raise ValueError("gen_PG 缺少 Hxx 列，请检查。")
    G_local = df_genPG[H_COLS].to_numpy(dtype=float).sum(axis=0)

    # 3) 储能参数（暂时假定一台）
    df_stor = xls.parse("storage_system")
    if df_stor.empty:
        raise ValueError("storage_system 为空，至少需要一台电储能。")
    if storage_id is None:
        row = df_stor.iloc[0]
    else:
        hits = df_stor[df_stor["id"].astype(str) == str(storage_id)]
        if hits.empty:
            raise ValueError(f"storage_system 中未找到 id={storage_id} 的行。")
        row = hits.iloc[0]

    sp = StorageParams(
        id=str(row["id"]),
        Pch_max=float(row["Pch_max"]),
        Pdc_max=float(row["Pdc_max"]),
        eta_ch=float(row["eta_ch"]),
        eta_dc=float(row["eta_dc"]),
        kappa=float(row["kappa"]),
        e_min=float(row["e_min"]),
        e_max=float(row["e_max"]),
        e0=float(row["e0"]),
    )

    # 4) 电价曲线
    df_buy = xls.parse("elec_price_buy")
    df_sell = xls.parse("elec_price_sell")

    def _pick_row(df: pd.DataFrame) -> pd.Series:
        if "tariff_id" in df.columns:
            tid = tariff_id or str(df["tariff_id"].iloc[0])
            hits = df[df["tariff_id"].astype(str) == str(tid)]
            if hits.empty:
                raise ValueError(f"电价表中未找到 tariff_id={tid} 的行。")
            return hits.iloc[0]
        else:
            return df.iloc[0]

    row_buy = _pick_row(df_buy)
    row_sell = _pick_row(df_sell)

    if not set(H_COLS).issubset(row_buy.index) or not set(H_COLS).issubset(row_sell.index):
        raise ValueError("elec_price_buy/sell 缺少 Hxx 列，请检查。")

    price_buy = row_buy[H_COLS].to_numpy(dtype=float)
    price_sell = row_sell[H_COLS].to_numpy(dtype=float)

    # 简单检查：售电价不应高于购电价，否则会导致无穷套利
    if np.any(price_sell - price_buy > 1e-9):
        raise ValueError("检测到某些小时 price_sell > price_buy，"
                         "请检查电价设置以避免无穷套利。")

    return OptimizationInputs(
        L=L,
        G_local=G_local,
        price_buy=price_buy,
        price_sell=price_sell,
        storage=sp,
        dt=1.0,
    )


def _solve_day_ahead_lp(inp: OptimizationInputs) -> OptimizationResult:
    """
    基于给定的曲线和储能参数，求解 24 小时购售电 + 储能调度线性规划。
    决策变量顺序：
        x = [Pbuy_0..23, Psell_0..23, Pch_0..23, Pdc_0..23, e_1..e_24]
    """
    T = HOURS
    dt = float(inp.dt)
    sp = inp.storage

    nvars = 4 * T + T  # 4*T for powers + T for e_1..e_T

    # 1) 变量边界
    bounds: List[Tuple[float, float]] = []

    # P_buy, P_sell：设一个较大的上限，避免无界；实际可改为接口容量
    P_buy_max = max(inp.L.max() - inp.G_local.min(), 0.0) + max(sp.Pdc_max, 10.0)
    P_sell_max = max(inp.G_local.max() + sp.Pdc_max - inp.L.min(), 0.0) + 10.0
    P_buy_max = max(P_buy_max, 50.0)
    P_sell_max = max(P_sell_max, 50.0)

    for _ in range(T):
        bounds.append((0.0, P_buy_max))   # P_buy
    for _ in range(T):
        bounds.append((0.0, P_sell_max))  # P_sell
    for _ in range(T):
        bounds.append((0.0, sp.Pch_max))  # P_ch
    for _ in range(T):
        bounds.append((0.0, sp.Pdc_max))  # P_dc
    for _ in range(T):
        bounds.append((sp.e_min, sp.e_max))  # e_1..e_T

    # 2) 目标函数：min Σ λ_buy * P_buy - λ_sell * P_sell
    c = np.zeros(nvars, dtype=float)
    for t in range(T):
        c[t] = inp.price_buy[t]          # P_buy
        c[T + t] = -inp.price_sell[t]    # P_sell
    # P_ch, P_dc, e_* 不直接出现在电费中

    # 3) 约束：A_eq x = b_eq

    Aeq_rows: List[np.ndarray] = []
    beq_vals: List[float] = []

    # (a) 系统电能平衡：
    #     G_local_t + P_buy - P_sell + P_dc - P_ch = L_t
    #     => P_buy - P_sell - P_ch + P_dc = L_t - G_local_t
    for t in range(T):
        row = np.zeros(nvars, dtype=float)
        row[t] = 1.0          # P_buy_t
        row[T + t] = -1.0     # P_sell_t
        row[2*T + t] = -1.0   # P_ch_t
        row[3*T + t] = 1.0    # P_dc_t
        Aeq_rows.append(row)
        beq_vals.append(float(inp.L[t] - inp.G_local[t]))

    # (b) 储能水箱动力学：
    #     e_{t+1} = κ e_t + dt (η_ch P_ch - 1/η_dc P_dc)
    #   用 e_1..e_T 作变量，e_0 已知：
    #   t=0:  e_1 - κ e_0 - dt(η_ch P_ch0 - 1/η_dc P_dc0) = 0
    #   t>0:  e_{t+1} - κ e_t - dt(η_ch P_cht - 1/η_dc P_dct) = 0
    for t in range(T):
        row = np.zeros(nvars, dtype=float)
        # P_ch_t, P_dc_t
        row[2*T + t] = -sp.eta_ch * dt          # P_ch_t
        row[3*T + t] = (1.0 / sp.eta_dc) * dt   # P_dc_t
        # e_{t+1}
        row[4*T + t] = 1.0
        if t == 0:
            # e_1 - κ e0 - dt(...) = 0  -> RHS = κ e0
            beq = sp.kappa * sp.e0
        else:
            # e_{t+1} - κ e_t - dt(...) = 0
            row[4*T + (t - 1)] = -sp.kappa
            beq = 0.0
        Aeq_rows.append(row)
        beq_vals.append(beq)

    # (c) 期末 SoC 约束：e_T = e0
    row = np.zeros(nvars, dtype=float)
    row[4*T + (T - 1)] = 1.0
    Aeq_rows.append(row)
    beq_vals.append(sp.e0)

    Aeq = np.vstack(Aeq_rows)
    beq = np.asarray(beq_vals, dtype=float)

    # 4) 求解 LP
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")

    if res.status != 0:
        return OptimizationResult(
            P_buy=np.zeros(T),
            P_sell=np.zeros(T),
            P_ch=np.zeros(T),
            P_dc=np.zeros(T),
            e=np.full(T+1, sp.e0),
            obj_value=float("nan"),
            status=res.status,
            message=res.message,
        )

    x = res.x
    P_buy = x[0:T]
    P_sell = x[T:2*T]
    P_ch = x[2*T:3*T]
    P_dc = x[3*T:4*T]
    e_next = x[4*T:5*T]
    e = np.concatenate(([sp.e0], e_next))

    return OptimizationResult(
        P_buy=P_buy,
        P_sell=P_sell,
        P_ch=P_ch,
        P_dc=P_dc,
        e=e,
        obj_value=float(res.fun),
        status=res.status,
        message=res.message,
    )


def optimize_grid_and_storage(dataset_path: str,
                              storage_id: Optional[str] = None,
                              tariff_id: Optional[str] = None
                              ) -> OptimizationResult:
    """
    对给定 ies_dataset_extgrid.xlsx 做 24h 购售电 + 储能优化。
    返回 OptimizationResult，其中包含四个时间序列与 SOC。
    """
    inp = _load_optimization_inputs(dataset_path, storage_id=storage_id, tariff_id=tariff_id)
    return _solve_day_ahead_lp(inp)


def write_optimized_dispatch(dataset_in: str,
                             dataset_out: Optional[str] = None,
                             storage_id: Optional[str] = None,
                             tariff_id: Optional[str] = None
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    读取 dataset_in（如 ies_dataset_extgrid.xlsx），
    运行优化，生成并写回以下四个横表：
      - storage_dispatch_ch
      - storage_dispatch_dc
      - grid_trade_buy
      - grid_trade_sell

    若 dataset_out 为 None，则覆盖原文件；否则写入新文件。
    返回四个 DataFrame，便于调试和画图。
    """
    res = optimize_grid_and_storage(dataset_in, storage_id=storage_id, tariff_id=tariff_id)
    if res.status != 0:
        raise RuntimeError(f"优化失败：status={res.status}, message={res.message}")

    # 构造四个横表
    xls = pd.ExcelFile(dataset_in, engine="openpyxl")
    df_stor = xls.parse("storage_system")
    if df_stor.empty:
        raise ValueError("storage_system 为空。")
    if storage_id is None:
        sid = str(df_stor.iloc[0]["id"])
    else:
        sid = str(storage_id)

    row_ch = {"id": sid}
    row_dc = {"id": sid}
    for i, col in enumerate(H_COLS):
        row_ch[col] = float(res.P_ch[i])
        row_dc[col] = float(res.P_dc[i])

    df_ch = pd.DataFrame([row_ch])
    df_dc = pd.DataFrame([row_dc])

    row_buy = {"profile_id": "ext_grid"}
    row_sell = {"profile_id": "ext_grid"}
    for i, col in enumerate(H_COLS):
        row_buy[col] = float(res.P_buy[i])
        row_sell[col] = float(res.P_sell[i])

    df_buy = pd.DataFrame([row_buy])
    df_sell = pd.DataFrame([row_sell])

    # 写回 Excel
    target_path = dataset_out or dataset_in
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    with pd.ExcelWriter(target_path, engine="openpyxl") as w:
        for sheet in xls.sheet_names:
            if sheet == "storage_dispatch_ch":
                df_ch.to_excel(w, sheet_name=sheet, index=False)
            elif sheet == "storage_dispatch_dc":
                df_dc.to_excel(w, sheet_name=sheet, index=False)
            elif sheet == "grid_trade_buy":
                df_buy.to_excel(w, sheet_name=sheet, index=False)
            elif sheet == "grid_trade_sell":
                df_sell.to_excel(w, sheet_name=sheet, index=False)
            else:
                xls.parse(sheet).to_excel(w, sheet_name=sheet, index=False)

    return df_ch, df_dc, df_buy, df_sell


if __name__ == "__main__":
    # 把 ROOT 设置为 dataset 目录
    ROOT = r"F:\25秋科研资料\IES_critical_modelling\dataset"
    dataset_in = os.path.join(ROOT, "ies_dataset_extgrid.xlsx")
    dataset_out = os.path.join(ROOT, "ies_dataset.xlsx")

    if os.path.exists(dataset_in):
        print(f"[INFO] 使用数据文件: {dataset_in}")
        df_ch, df_dc, df_buy, df_sell = write_optimized_dispatch(dataset_in, dataset_out)
        print("[INFO] 优化完成并写回 Excel：", dataset_out)
        print("storage_dispatch_ch:")
        print(df_ch)
        print("grid_trade_buy:")
        print(df_buy)
    else:
        print(f"[WARN] 未找到 {dataset_in}，请检查路径。")

