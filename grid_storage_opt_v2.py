
# grid_storage_opt.py (iterated version)
"""
电力系统与电储能的日尺度购售电 + 储能调度优化模块（迭代版）

本模块在现有项目框架中的定位：
- 作为 IES 时序仿真的“上游优化层”，给出 24h 外部购/售电与电储能充放电计划；
- 目标函数采用“电碳综合购电价”口径：
    * 购电侧：电价 + 碳价 * 外部电网碳强度
    * 售电侧：仅电价（不抵扣碳成本）
- 同时计算并报告气网 slack 源购气成本（来自气网数据的“实际购气量”），
  该部分当前不影响电侧优化解（常数项），但用于完整的系统成本核算与后续耦合扩展。

单位约定（与项目其余模块保持一致）：
- 功率：MW
- 能量：MWh（dt=1h 时 MW*h = MWh）
- 电价：Excel 输入为 元/kWh；程序读入时统一换算为 元/MWh（×1000）
- 外部电网碳强度：tCO2/MWh
- 碳价：元/tCO2
- 气网流率：km^3/h（千标方/小时），气价：元/m^3

依赖：
- numpy, pandas
- scipy.optimize.linprog (HiGHS)
- openpyxl（仅用于 Excel 读写）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from scipy.optimize import linprog

H_COLS = [f"H{i:02d}" for i in range(24)]


# ========================= 数据结构 =========================

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
    # 电侧（用于优化）
    L: np.ndarray           # (T,) 系统电负荷总量 (MW)
    G_local: np.ndarray     # (T,) 内部机组总出力预测 (MW)（gen_PG 汇总）
    price_buy: np.ndarray   # (T,) 购电价 (元/MWh)
    price_sell: np.ndarray  # (T,) 售电价 (元/MWh)
    CI_ext: np.ndarray      # (T,) 外部电网碳强度 (tCO2/MWh)
    carbon_price: float     # 元/tCO2
    storage: StorageParams
    dt: float = 1.0

    # 若生成了新的 gen_PG 预测，保存下来，便于 write_optimized_dispatch 写回
    genPG_used: Optional[pd.DataFrame] = None

    # 气侧（仅用于成本核算）
    gas_price_y_per_m3: Optional[float] = None   # slack 气源价格（元/m3）
    gas_slack_km3ph: Optional[np.ndarray] = None # (T,) slack 购气量（km3/h）


@dataclass
class OptimizationResult:
    P_buy: np.ndarray   # (T,) MW
    P_sell: np.ndarray  # (T,) MW
    P_ch: np.ndarray    # (T,) MW
    P_dc: np.ndarray    # (T,) MW
    e: np.ndarray       # (T+1,) MWh

    status: int
    message: str
    obj_electric_yuan: float
    obj_gas_yuan: float
    obj_total_yuan: float

    # 便于分析
    price_buy_ec: Optional[np.ndarray] = None  # (T,) 元/MWh
    gas_slack_km3ph: Optional[np.ndarray] = None
    gas_price_y_per_m3: Optional[float] = None


# ========================= 预测出力生成（用于修正 gen_PG 过小问题） =========================

def _smooth_1d(x: np.ndarray, win: int = 3) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win, dtype=float) / win
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, k, mode="valid")


def generate_gen_forecast_24h(peak_g1_mw: float = 5.0, peak_g2_mw: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 24h 预测出力（MW）
    - G1：燃气轮机（可调性强，承担早晚峰；夜间保底）
    - G2：可再生（默认光伏型：日出后爬坡，中午峰值，傍晚归零）
    """
    h = np.arange(24, dtype=float)

    # G1：两峰 + 夜间保底（分段插值 + 平滑）
    h_pts = np.array([0, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 19, 20, 22, 23], dtype=float)
    frac = np.array([0.40, 0.36, 0.46, 0.60, 0.76, 0.92, 1.00, 0.88, 0.80, 0.84, 1.00, 0.96, 0.86, 0.56, 0.46], dtype=float)
    g1 = np.interp(h, h_pts, peak_g1_mw * frac)
    g1 = _smooth_1d(g1, win=3)
    g1 = g1 * (peak_g1_mw / max(float(g1.max()), 1e-12))
    g1 = np.clip(g1, 0.0, peak_g1_mw)

    # G2：PV-like（白天为正，夜晚为 0）
    pv = np.sin((h - 6.0) / 12.0 * np.pi)
    pv = np.clip(pv, 0.0, None) ** 1.6
    cloud_dip = 1.0 - 0.12 * np.exp(-0.5 * ((h - 14.0) / 1.7) ** 2)
    g2 = peak_g2_mw * pv * cloud_dip
    if float(g2.max()) > 1e-12:
        g2 = g2 * (peak_g2_mw / float(g2.max()))
    g2 = np.clip(g2, 0.0, peak_g2_mw)

    return g1, g2


def build_genPG_dataframe(g1: np.ndarray, g2: np.ndarray, gen_ids: Tuple[int, int] = (1, 2)) -> pd.DataFrame:
    row1 = {"gen_id": int(gen_ids[0])}
    row2 = {"gen_id": int(gen_ids[1])}
    for i, col in enumerate(H_COLS):
        row1[col] = float(g1[i])
        row2[col] = float(g2[i])
    return pd.DataFrame([row1, row2])


# ========================= 气网购气成本核算（不影响电侧优化解） =========================

def compute_gas_slack_purchase_from_dataset(
    xls_path: str,
    dt: float = 1.0,
    slack_source_id: Optional[int] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    从数据表直接核算每小时 slack 气源购气量（km3/h）与购气总成本（元）。

    口径尽量与 gas_flow.run_gas_flow 中 slack_need 计算一致：
      slack_need(t) = total_demand(t) - total_fixed_supply(t)

    - total_demand(t) = sum(gas_load_demand[:,t]) + sum(gas_compressors.gasconsumption)
    - total_fixed_supply(t) = sum(非slack源 output 或 gas_source_output_fixed[:,t] )

    气价来源：
      gas_sources.price_y_per_m3（若不存在则返回 nan，调用侧可设默认）

    返回：
      slack_km3ph: (24,)
      gas_price_y_per_m3: float (slack 源价格)
      gas_cost_total_yuan: float
    """
    xls = pd.ExcelFile(xls_path, engine="openpyxl")

    sources = pd.read_excel(xls, sheet_name="gas_sources")
    if "is_slack" in sources.columns:
        slack_mask = sources["is_slack"].fillna(0).astype(int) == 1
    else:
        slack_mask = pd.Series([True] + [False] * (len(sources) - 1))

    if slack_source_id is not None and "id" in sources.columns:
        slack_mask = sources["id"].astype(int) == int(slack_source_id)

    if slack_mask.sum() != 1:
        raise ValueError(f"[GasCost] 需要且仅需要一个 slack 气源。当前 slack 数={int(slack_mask.sum())}")

    slack_row = sources.loc[slack_mask].iloc[0]
    gas_price = float(slack_row.get("price_y_per_m3", np.nan))

    # 非 slack 源：固定供给（km3/h）
    fixed_sources = sources.loc[~slack_mask].copy()
    fixed_supply_const = float(fixed_sources["output"].fillna(0.0).astype(float).sum()) if len(fixed_sources) else 0.0

    # 负荷需求（km3/h）按小时
    gw = pd.read_excel(xls, sheet_name="gas_load_demand")
    if not set(H_COLS).issubset(gw.columns):
        raise KeyError("[GasCost] gas_load_demand 缺少 H00~H23 列")
    demand_load = gw[H_COLS].fillna(0.0).astype(float).to_numpy().sum(axis=0)  # (24,)

    # 压缩机耗气（km3/h），目前表内为常数（不随时间）
    comps = pd.read_excel(xls, sheet_name="gas_compressors")
    demand_comp = float(comps.get("gasconsumption", pd.Series(dtype=float)).fillna(0.0).astype(float).sum()) if len(comps) else 0.0

    total_demand = demand_load + demand_comp

    # （可选）固定源出力时序覆盖
    fixed_supply_ts = None
    if "gas_source_output_fixed" in xls.sheet_names:
        sw = pd.read_excel(xls, sheet_name="gas_source_output_fixed")
        if len(sw):
            if not set(H_COLS).issubset(sw.columns) or "source_id" not in sw.columns:
                raise KeyError("[GasCost] gas_source_output_fixed 需包含 source_id + H00~H23")
            # 只统计非slack源
            fixed_ids = set(fixed_sources["id"].astype(int).tolist()) if len(fixed_sources) else set()
            hit = sw["source_id"].astype(int).isin(fixed_ids)
            if hit.any():
                fixed_supply_ts = sw.loc[hit, H_COLS].fillna(0.0).astype(float).to_numpy().sum(axis=0)

    if fixed_supply_ts is None:
        fixed_supply = np.full(24, fixed_supply_const, dtype=float)
    else:
        fixed_supply = fixed_supply_ts

    slack_need = total_demand - fixed_supply
    if np.any(slack_need < -1e-8):
        # 负 slack 表示固定供给>需求：当前气网未建模外送/弃气，直接报错更安全
        raise ValueError(
            "[GasCost] 检测到 slack_need<0：固定气源供给大于总需求。"
            "请调小固定源 output 或引入外送/弃气建模。"
        )

    gas_cost = float(np.nansum(slack_need * 1000.0 * gas_price * dt))  # km3/h -> m3/h: *1000

    return slack_need, gas_price, gas_cost


# ========================= 读入优化输入 =========================

def _load_optimization_inputs(
    dataset_path: str,
    storage_id: Optional[str] = None,
    tariff_id: Optional[str] = None,
    ci_profile_id: str = "ext_grid",
    carbon_price: float = 0.0,
    gen_peak_mw: float = 5.0,
    gen_replace_if_max_below_mw: float = 1.0,
    gas_default_price_y_per_m3: float = 3.55,
) -> OptimizationInputs:
    """
    读取 Excel，构造 OptimizationInputs。

    说明：
    - 电价：elec_price_buy/sell 输入为元/kWh，本函数在读入时统一×1000变为元/MWh
    - gen_PG：若最大值过小（<gen_replace_if_max_below_mw），则用生成的 G1/G2 预测曲线替换（峰值=gen_peak_mw）
    - 气网购气量：从 gas_sources/gas_load_demand/gas_compressors 等表计算 slack 购气量与购气成本（不影响优化解）
    """
    xls = pd.ExcelFile(dataset_path, engine="openpyxl")

    # 1) 负荷曲线 L(t) = sum(power_PD)
    pdw = pd.read_excel(xls, sheet_name="power_PD")
    if not set(H_COLS).issubset(pdw.columns):
        raise ValueError("power_PD 缺少 H00~H23 列")
    L = pdw[H_COLS].fillna(0.0).astype(float).to_numpy().sum(axis=0)

    # 2) 内部机组出力预测（gen_PG）
    gen_pg = pd.read_excel(xls, sheet_name="gen_PG")
    if not set(H_COLS).issubset(gen_pg.columns):
        raise ValueError("gen_PG 缺少 H00~H23 列")
    gen_max = float(np.nanmax(gen_pg[H_COLS].to_numpy(dtype=float)))
    genPG_used = None
    if gen_max < gen_replace_if_max_below_mw:
        g1, g2 = generate_gen_forecast_24h(peak_g1_mw=gen_peak_mw, peak_g2_mw=gen_peak_mw)
        genPG_used = build_genPG_dataframe(g1, g2, gen_ids=(1, 2))
        G_local = genPG_used[H_COLS].to_numpy(dtype=float).sum(axis=0)
    else:
        G_local = gen_pg[H_COLS].fillna(0.0).astype(float).to_numpy().sum(axis=0)

    # 3) 电价（元/kWh -> 元/MWh）
    def _pick_row(df: pd.DataFrame, id_col: str, want: Optional[str], where: str) -> pd.Series:
        if id_col in df.columns:
            key = want or str(df[id_col].iloc[0])
            hit = df[df[id_col].astype(str) == str(key)]
            if hit.empty:
                raise ValueError(f"{where} 中未找到 {id_col}={key}")
            return hit.iloc[0]
        return df.iloc[0]

    df_buy = pd.read_excel(xls, sheet_name="elec_price_buy")
    df_sell = pd.read_excel(xls, sheet_name="elec_price_sell")
    row_buy = _pick_row(df_buy, "tariff_id", tariff_id, "elec_price_buy")
    row_sell = _pick_row(df_sell, "tariff_id", tariff_id, "elec_price_sell")

    for hc in H_COLS:
        if hc not in row_buy.index or hc not in row_sell.index:
            raise KeyError("elec_price_buy/sell 缺少 H00~H23 列")

    price_buy = row_buy[H_COLS].astype(float).to_numpy() * 1000.0   # ✅元/MWh
    price_sell = row_sell[H_COLS].astype(float).to_numpy() * 1000.0 # ✅元/MWh

    # 4) external_CI（tCO2/MWh）
    df_ci = pd.read_excel(xls, sheet_name="external_CI")
    row_ci = _pick_row(df_ci, "profile_id", ci_profile_id, "external_CI")
    CI_ext = row_ci[H_COLS].astype(float).to_numpy()

    # 5) 储能参数
    df_es = pd.read_excel(xls, sheet_name="storage_system")
    if "id" in df_es.columns:
        sid = storage_id or str(df_es["id"].iloc[0])
        hit = df_es[df_es["id"].astype(str) == str(sid)]
        if hit.empty:
            raise ValueError(f"storage_system 中未找到 id={sid}")
        row_es = hit.iloc[0]
    else:
        row_es = df_es.iloc[0]
        sid = str(row_es.get("id", "ES1"))

    sp = StorageParams(
        id=sid,
        Pch_max=float(row_es["Pch_max"]),
        Pdc_max=float(row_es["Pdc_max"]),
        eta_ch=float(row_es["eta_ch"]),
        eta_dc=float(row_es["eta_dc"]),
        kappa=float(row_es["kappa"]),
        e_min=float(row_es["e_min"]),
        e_max=float(row_es["e_max"]),
        e0=float(row_es["e0"]),
    )

    # 6) 气网 slack 购气量与价格（用于成本核算）
    try:
        gas_slack_km3ph, gas_price, gas_cost = compute_gas_slack_purchase_from_dataset(dataset_path, dt=1.0)
        if not np.isfinite(gas_price):
            gas_price = float(gas_default_price_y_per_m3)
    except Exception:
        gas_slack_km3ph = None
        gas_price = float(gas_default_price_y_per_m3)
        gas_cost = 0.0

    # 7) 防止无界套利：sell 不得高于 buy(含碳)
    price_buy_ec = price_buy + carbon_price * CI_ext
    if np.any(price_sell > price_buy_ec + 1e-9):
        raise ValueError("检测到某些小时 售电价 > 购电价(含碳成本)，可能导致无界/套利。")

    return OptimizationInputs(
        L=L,
        G_local=G_local,
        price_buy=price_buy,
        price_sell=price_sell,
        CI_ext=CI_ext,
        carbon_price=float(carbon_price),
        storage=sp,
        dt=1.0,
        genPG_used=genPG_used,
        gas_price_y_per_m3=float(gas_price) if gas_price is not None else None,
        gas_slack_km3ph=gas_slack_km3ph,
    )


# ========================= 求解 LP =========================

def _solve_day_ahead_lp(inp: OptimizationInputs) -> OptimizationResult:
    """
    LP 变量（按小时展开）：
      x = [P_buy(0..T-1), P_sell(0..T-1), P_ch(0..T-1), P_dc(0..T-1), e(1..T)]
    其中 e(0)=e0 为已知。
    """
    T = 24
    dt = float(inp.dt)
    nvars = 4 * T + T  # e(1..T)

    idx_buy0 = 0
    idx_sell0 = T
    idx_ch0 = 2 * T
    idx_dc0 = 3 * T
    idx_e1 = 4 * T

    # 目标函数：购电（含碳） - 售电（不抵扣碳）
    price_buy_ec = inp.price_buy + inp.carbon_price * inp.CI_ext   # 元/MWh
    price_sell_ec = inp.price_sell                                 # 元/MWh

    c = np.zeros(nvars, dtype=float)
    for t in range(T):
        c[idx_buy0 + t] = price_buy_ec[t] * dt
        c[idx_sell0 + t] = -price_sell_ec[t] * dt

    # 等式约束：Aeq x = beq
    # 1) 电力平衡：G + buy - sell + dc - ch = L
    #    => buy - sell - ch + dc = L - G
    Aeq = []
    beq = []

    for t in range(T):
        row = np.zeros(nvars, dtype=float)
        row[idx_buy0 + t] = 1.0
        row[idx_sell0 + t] = -1.0
        row[idx_ch0 + t] = -1.0
        row[idx_dc0 + t] = 1.0
        Aeq.append(row)
        beq.append(float(inp.L[t] - inp.G_local[t]))

    # 2) 储能能量动态：e(t+1) = kappa e(t) + dt*(eta_ch Pch - Pdc/eta_dc)
    #    变量仅包含 e(1..T)，e(0)=e0 常数
    for t in range(T):
        row = np.zeros(nvars, dtype=float)
        # e_{t+1}
        row[idx_e1 + t] = 1.0
        # -kappa * e_t
        if t >= 1:
            row[idx_e1 + (t - 1)] = -inp.storage.kappa
        # -dt*eta_ch*Pch + dt*(1/eta_dc)*Pdc
        row[idx_ch0 + t] = -dt * inp.storage.eta_ch
        row[idx_dc0 + t] = dt * (1.0 / inp.storage.eta_dc)

        rhs = 0.0
        if t == 0:
            rhs = inp.storage.kappa * inp.storage.e0
        Aeq.append(row)
        beq.append(float(rhs))

    # 3) 日内循环：e(T) = e0
    row = np.zeros(nvars, dtype=float)
    row[idx_e1 + (T - 1)] = 1.0
    Aeq.append(row)
    beq.append(float(inp.storage.e0))

    Aeq = np.vstack(Aeq)
    beq = np.array(beq, dtype=float)

    # 变量边界
    # 外部接口上界：经验估计（后续可替换为接口容量）
    # 取负荷峰值 + 储能功率峰值 作为粗上界，并给一个下限避免过小上界导致不可行
    peak_load = float(np.nanmax(inp.L))
    peak_gen = float(np.nanmax(inp.G_local))
    P_trade_max = max(50.0, peak_load + inp.storage.Pch_max + 5.0)  # MW
    P_sell_max = max(50.0, peak_gen + inp.storage.Pdc_max + 5.0)    # MW

    bounds = []
    for t in range(T):
        bounds.append((0.0, P_trade_max))  # P_buy
    for t in range(T):
        bounds.append((0.0, P_sell_max))   # P_sell
    for t in range(T):
        bounds.append((0.0, float(inp.storage.Pch_max)))  # P_ch
    for t in range(T):
        bounds.append((0.0, float(inp.storage.Pdc_max)))  # P_dc
    for t in range(T):
        bounds.append((float(inp.storage.e_min), float(inp.storage.e_max)))  # e(1..T)

    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")

    if not res.success:
        return OptimizationResult(
            P_buy=np.zeros(T),
            P_sell=np.zeros(T),
            P_ch=np.zeros(T),
            P_dc=np.zeros(T),
            e=np.r_[inp.storage.e0, np.zeros(T)],
            status=int(res.status),
            message=str(res.message),
            obj_electric_yuan=float("nan"),
            obj_gas_yuan=float("nan"),
            obj_total_yuan=float("nan"),
            price_buy_ec=price_buy_ec,
            gas_slack_km3ph=inp.gas_slack_km3ph,
            gas_price_y_per_m3=inp.gas_price_y_per_m3,
        )

    x = res.x
    P_buy = x[idx_buy0: idx_buy0 + T]
    P_sell = x[idx_sell0: idx_sell0 + T]
    P_ch = x[idx_ch0: idx_ch0 + T]
    P_dc = x[idx_dc0: idx_dc0 + T]
    e1T = x[idx_e1: idx_e1 + T]
    e = np.r_[inp.storage.e0, e1T]

    obj_elec = float(res.fun)

    # 气网购气成本（常数项）
    gas_cost = 0.0
    if inp.gas_slack_km3ph is not None and inp.gas_price_y_per_m3 is not None:
        gas_cost = float(np.nansum(inp.gas_slack_km3ph * 1000.0 * inp.gas_price_y_per_m3 * dt))

    return OptimizationResult(
        P_buy=P_buy,
        P_sell=P_sell,
        P_ch=P_ch,
        P_dc=P_dc,
        e=e,
        status=int(res.status),
        message=str(res.message),
        obj_electric_yuan=obj_elec,
        obj_gas_yuan=gas_cost,
        obj_total_yuan=obj_elec + gas_cost,
        price_buy_ec=price_buy_ec,
        gas_slack_km3ph=inp.gas_slack_km3ph,
        gas_price_y_per_m3=inp.gas_price_y_per_m3,
    )


# ========================= 对外接口 =========================

def optimize_grid_and_storage(
    dataset_path: str,
    storage_id: Optional[str] = None,
    tariff_id: Optional[str] = None,
    ci_profile_id: str = "ext_grid",
    carbon_price: float = 0.0,
    gen_peak_mw: float = 5.0,
    gen_replace_if_max_below_mw: float = 1.0,
    gas_default_price_y_per_m3: float = 3.55,
) -> OptimizationResult:
    """
    一键优化入口（返回 OptimizationResult）。
    """
    inp = _load_optimization_inputs(
        dataset_path=dataset_path,
        storage_id=storage_id,
        tariff_id=tariff_id,
        ci_profile_id=ci_profile_id,
        carbon_price=carbon_price,
        gen_peak_mw=gen_peak_mw,
        gen_replace_if_max_below_mw=gen_replace_if_max_below_mw,
        gas_default_price_y_per_m3=gas_default_price_y_per_m3,
    )
    return _solve_day_ahead_lp(inp)


def write_optimized_dispatch(
    dataset_in: str,
    dataset_out: str,
    storage_id: Optional[str] = None,
    tariff_id: Optional[str] = None,
    ci_profile_id: str = "ext_grid",
    carbon_price: float = 0.0,
    gen_peak_mw: float = 5.0,
    gen_replace_if_max_below_mw: float = 1.0,
    gas_default_price_y_per_m3: float = 3.55,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, OptimizationResult]:
    """
    运行优化并写回 Excel（输出横表）：
      - storage_dispatch_ch / storage_dispatch_dc
      - grid_trade_buy / grid_trade_sell
    另外：若生成了 gen_PG 预测曲线，会覆盖写回 gen_PG（保持优化输入与后续分析一致）

    返回：四张表 + result
    """
    inp = _load_optimization_inputs(
        dataset_path=dataset_in,
        storage_id=storage_id,
        tariff_id=tariff_id,
        ci_profile_id=ci_profile_id,
        carbon_price=carbon_price,
        gen_peak_mw=gen_peak_mw,
        gen_replace_if_max_below_mw=gen_replace_if_max_below_mw,
        gas_default_price_y_per_m3=gas_default_price_y_per_m3,
    )
    res = _solve_day_ahead_lp(inp)
    if res.status != 0 or not np.isfinite(res.obj_electric_yuan):
        raise RuntimeError(f"[grid_storage_opt] 优化失败：status={res.status} | {res.message}")

    # 组织写回表
    df_ch = pd.DataFrame([{"profile_id": "default", **{H_COLS[i]: float(res.P_ch[i]) for i in range(24)}}])
    df_dc = pd.DataFrame([{"profile_id": "default", **{H_COLS[i]: float(res.P_dc[i]) for i in range(24)}}])
    df_buy = pd.DataFrame([{"profile_id": "default", **{H_COLS[i]: float(res.P_buy[i]) for i in range(24)}}])
    df_sell = pd.DataFrame([{"profile_id": "default", **{H_COLS[i]: float(res.P_sell[i]) for i in range(24)}}])

    # 读入原文件，写出新文件（保留其他 sheet）
    with pd.ExcelWriter(dataset_out, engine="openpyxl") as w:
        xls = pd.ExcelFile(dataset_in, engine="openpyxl")
        for sh in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sh)
            if sh == "storage_dispatch_ch":
                df_ch.to_excel(w, sheet_name=sh, index=False)
            elif sh == "storage_dispatch_dc":
                df_dc.to_excel(w, sheet_name=sh, index=False)
            elif sh == "grid_trade_buy":
                df_buy.to_excel(w, sheet_name=sh, index=False)
            elif sh == "grid_trade_sell":
                df_sell.to_excel(w, sheet_name=sh, index=False)
            elif sh == "gen_PG" and inp.genPG_used is not None:
                inp.genPG_used.to_excel(w, sheet_name=sh, index=False)
            else:
                df.to_excel(w, sheet_name=sh, index=False)

    return df_ch, df_dc, df_buy, df_sell, res


if __name__ == "__main__":
    import os
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent
    DATASET_DIR = (ROOT / "dataset") if (ROOT / "dataset").exists() else ROOT
    dataset_in = str(DATASET_DIR / "ies_dataset_extgrid.xlsx")
    dataset_out = str(DATASET_DIR / "ies_dataset.xlsx")

    if os.path.exists(dataset_in):
        print(f"[INFO] 使用数据文件: {dataset_in}")
        df_ch, df_dc, df_buy, df_sell, res = write_optimized_dispatch(
            dataset_in=dataset_in,
            dataset_out=dataset_out,
            carbon_price=0.0,
        )
        print("[INFO] 写回完成：", dataset_out)
        print("[INFO] 电侧目标(元):", res.obj_electric_yuan)
        print("[INFO] 气网购气成本(元):", res.obj_gas_yuan)
        print("[INFO] 总成本(元):", res.obj_total_yuan)
    else:
        print(f"[WARN] 未找到 {dataset_in}，请检查路径。")
