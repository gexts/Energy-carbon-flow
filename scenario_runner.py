# -*- coding: utf-8 -*-
"""
from storage import build_runtime_from_TS
import copy

scenario_runner.py —— 逐时仿真（读取“横表输入”，逐小时覆盖，再运行四大模块）
新增导出（行=ID，列=H00..H23）：
  - power_NCI                     tCO2/MWh（电网节点碳势）
  - power_CFR_branch_tph          tCO2/h  （电网支路碳流率）
  - power_CFR_load_bus_tph        tCO2/h  （电网负荷碳流率，按母线）
  - gas_CFR_pipes_tph             tCO2/h  （气网支路碳流率）
  - heat_input_GCI_tperMWh        tCO2/MWh（热网输入端口碳势，按热源）
  - heat_CFR_pipes_tph            tCO2/h  （热网支路碳流率）
  - heat_CFR_loads_tph            tCO2/h  （热网负荷碳流率）

保留：
  - heat_Q_total_MW               MW
  - heat_CEFR_total_tph           tCO2/h
  - __snap_PD_MW__                MW（潮流入口快照）
  - __snap_EH_inputs_MW__         MW（EH入口快照）
  - kpi_hourly                    长表KPI
"""
from storage import build_runtime_from_TS
import copy


import os
import copy
import numpy as np
import pandas as pd
from pypower.idx_brch import F_BUS, T_BUS, PF, PT  # 用于支路潮流方向与功率

from testdata import build_test_system_from_excel
from power_flow import run_power_flow
from gas_flow import run_gas_flow
from energy_hub import run_energy_hub
from heat_flow import run_heat_flow
from pit_storage import PitThermalStorage, PitThermalStorageConfig



# ========================= 基础工具 =========================

def _hcol(h: int) -> str:
    return f"H{int(h):02d}"

def _require_cols(df: pd.DataFrame, needed, where: str):
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise KeyError(f"[输入缺列] {where} 缺少列：{miss}")


from storage import build_runtime_from_TS
import copy

def _load_inputs_wide(xls_path: str):
    """读取横表输入；存在则校验必需列，缺失返回 None。"""
    if not os.path.exists(xls_path):
        raise FileNotFoundError(f"未找到数据文件：{xls_path}")
    xls = pd.ExcelFile(xls_path, engine="openpyxl")

    def read_sheet(name: str):
        try:
            df = xls.parse(name)
            if isinstance(df, pd.DataFrame) and df.empty:
                return None
            return df
        except Exception:
            return None

    inputs = {
        "power_PD": read_sheet("power_PD"),                         # BUS_I + Hxx
        "storage_dispatch_ch": read_sheet("storage_dispatch_ch"),   # id + Hxx
        "storage_dispatch_dc": read_sheet("storage_dispatch_dc"),   # id + Hxx
        "power_QD": read_sheet("power_QD"),                         # BUS_I + Hxx
        "eh_flow_electricity": read_sheet("eh_flow_electricity"),
        "eh_flow_gas": read_sheet("eh_flow_gas"),
        "eh_PCI_gas": read_sheet("eh_PCI_gas"),
        "heat_Q_demand": read_sheet("heat_Q_demand"),
        "gas_load_demand": read_sheet("gas_load_demand"),                 # load_id + Hxx  (km3/h)
        "gas_source_output_fixed": read_sheet("gas_source_output_fixed"), # source_id + Hxx (km3/h) 可选
        "gas_load_demand": read_sheet("gas_load_demand"),


        # 新增：外部购售电 + 外部碳强度
        "grid_trade_buy": read_sheet("grid_trade_buy"),             # profile_id + Hxx
        "grid_trade_sell": read_sheet("grid_trade_sell"),           # profile_id + Hxx
        "external_CI": read_sheet("external_CI"),                   # profile_id + Hxx
    }

    

    hours_cols = [f"H{h:02d}" for h in range(24)]

    # 必需表与列的校验
    if inputs["power_PD"] is None:
        raise FileNotFoundError("缺少横表 power_PD，请在 ies_dataset.xlsx 中提供。")
    _require_cols(inputs["power_PD"], ["BUS_I"] + hours_cols, "power_PD")

    if inputs["power_QD"] is not None:
        _require_cols(inputs["power_QD"], ["BUS_I"] + hours_cols, "power_QD")

    if inputs["eh_flow_electricity"] is not None:
        _require_cols(inputs["eh_flow_electricity"], ["type"] + hours_cols, "eh_flow_electricity")

    if inputs["eh_flow_gas"] is not None:
        _require_cols(inputs["eh_flow_gas"], ["type"] + hours_cols, "eh_flow_gas")

    if inputs["eh_PCI_gas"] is not None:
        _require_cols(inputs["eh_PCI_gas"], ["type", "PCI_ref"], "eh_PCI_gas")

    if inputs["heat_Q_demand"] is not None:
        _require_cols(inputs["heat_Q_demand"], ["load_id"] + hours_cols, "heat_Q_demand")
    
    if inputs["gas_load_demand"] is not None:
        if ("id" not in inputs["gas_load_demand"].columns) and ("load_id" not in inputs["gas_load_demand"].columns):
            raise KeyError("gas_load_demand 需包含 id 或 load_id 列")
        _require_cols(inputs["gas_load_demand"], [("id" if "id" in inputs["gas_load_demand"].columns else "load_id")] + hours_cols, "gas_load_demand")

    
    if inputs["gas_load_demand"] is not None:
        _require_cols(inputs["gas_load_demand"], ["load_id"] + hours_cols, "gas_load_demand")

    if inputs["gas_source_output_fixed"] is not None:
        _require_cols(inputs["gas_source_output_fixed"], ["source_id"] + hours_cols, "gas_source_output_fixed")

    if inputs["grid_trade_buy"] is not None:
        _require_cols(inputs["grid_trade_buy"], ["profile_id"] + hours_cols, "grid_trade_buy")

    if inputs["grid_trade_sell"] is not None:
        _require_cols(inputs["grid_trade_sell"], ["profile_id"] + hours_cols, "grid_trade_sell")

    if inputs["external_CI"] is not None:
        _require_cols(inputs["external_CI"], ["profile_id"] + hours_cols, "external_CI")

    return inputs


def _find_bus_row_index(bus, bus_id: int):
    try:
        bid = int(bus_id)
        if hasattr(bus, "iloc"):  # DataFrame
            if "BUS_I" in bus.columns:
                hits = bus.index[bus["BUS_I"].astype(int) == bid]
                return int(hits[0]) if len(hits) else None
            else:
                hits = bus.index[bus.iloc[:, 0].astype(int) == bid]
                return int(hits[0]) if len(hits) else None
        else:  # ndarray
            col0 = bus[:, 0].astype(int)
            idxs = np.where(col0 == bid)[0]
            return int(idxs[0]) if len(idxs) else None
    except Exception:
        return None

def _set_bus_pd_qd(bus, bus_id: int, PD=None, QD=None):
    r = _find_bus_row_index(bus, bus_id)
    if r is None:
        return
    if hasattr(bus, "iloc"):  # DataFrame
        if PD is not None:
            if "PD" in bus.columns: bus.at[r, "PD"] = float(PD)
            else: bus.iat[r, 2] = float(PD)
        if QD is not None:
            if "QD" in bus.columns: bus.at[r, "QD"] = float(QD)
            elif bus.shape[1] >= 4: bus.iat[r, 3] = float(QD)
    else:  # ndarray
        if PD is not None: bus[r, 2] = float(PD)
        if QD is not None and bus.shape[1] >= 4: bus[r, 3] = float(QD)

def _ensure_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

def _wide_from_series_dict(series_per_hour: dict, id_name: str) -> pd.DataFrame:
    if not series_per_hour:
        return pd.DataFrame(columns=[id_name])
    hours_cols = sorted(series_per_hour.keys())
    all_ids = None
    for s in series_per_hour.values():
        if s is None: continue
        all_ids = s.index if all_ids is None else all_ids.union(s.index)
    if all_ids is None:
        return pd.DataFrame(columns=[id_name] + hours_cols)
    all_ids = pd.Index(sorted(all_ids))
    cols = []
    for hc in hours_cols:
        s = series_per_hour.get(hc, None)
        if s is None:
            s = pd.Series([np.nan] * len(all_ids), index=all_ids)
        col = s.reindex(all_ids); col.name = hc
        cols.append(col)
    wide = pd.concat(cols, axis=1)
    wide.insert(0, id_name, all_ids)
    return wide

def _build_power_gen_mix_from_dataset(dataset_dir: str) -> pd.DataFrame:
    """
    从 ies_dataset(_extgrid).xlsx 中读取 gen_PG，按照“内部机组 + 外部电网”拆分，
    组装成宽表 power_gen_mix_MW（metric × H00..H23）。

    返回 DataFrame:
        metric, H00, H01, ..., H23
        metric 行：Internal_gen, Grid_import
    如果找不到数据文件，则返回空 DataFrame。
    """
    # 1) 确定数据集路径：优先使用 ies_dataset_extgrid.xlsx
    cand1 = os.path.join(dataset_dir, "ies_dataset_extgrid.xlsx")
    cand2 = os.path.join(dataset_dir, "ies_dataset.xlsx")
    if os.path.exists(cand1):
        ds_path = cand1
    elif os.path.exists(cand2):
        ds_path = cand2
    else:
        return pd.DataFrame(columns=["metric"])

    # 2) 读入 bus / gen / gen_PG
    bus = pd.read_excel(ds_path, sheet_name="power_bus", engine="openpyxl")
    gen = pd.read_excel(ds_path, sheet_name="power_gen", engine="openpyxl")
    gen_pg = pd.read_excel(ds_path, sheet_name="gen_PG", engine="openpyxl")

    # bus: 获取 BUS_I, PD
    if "BUS_I" in bus.columns:
        bus_ids = bus["BUS_I"].astype(int)
        PD = bus["PD"].astype(float)
    else:
        bus_ids = bus.iloc[:, 0].astype(int)
        PD = bus.iloc[:, 2].astype(float)

    # gen: 每台机组所在母线（与 gen_PG 行顺序一致）
    gen_bus = gen.iloc[:, 0].astype(int)

    # 3) 判定“外部电网母线”：有机组，且负荷几乎为 0
    zero_load_buses = set(bus_ids[np.isclose(PD, 0.0)])
    cand = sorted(zero_load_buses.intersection(set(gen_bus)))
    extgrid_bus_id = int(cand[0]) if len(cand) >= 1 else None

    # 4) 按小时累加：内部机组 / 外部电网
    hour_cols = [c for c in gen_pg.columns if c.startswith("H")]
    internal = pd.Series(0.0, index=hour_cols, dtype=float)
    grid_import = pd.Series(0.0, index=hour_cols, dtype=float)

    for i, row in gen_pg.iterrows():
        bus_id = int(gen_bus.iloc[i])
        P = row[hour_cols].astype(float)
        if extgrid_bus_id is not None and bus_id == extgrid_bus_id:
            grid_import += P
        else:
            internal += P

    data = {"metric": ["Internal_gen", "Grid_import"]}
    for hc in hour_cols:
        data[hc] = [internal[hc], grid_import[hc]]

    return pd.DataFrame(data)


# ========================= 覆盖逻辑（把“横表值”写入 TS） =========================

def _overlay_power_loads(TS, inputs_wide, hour: int, strict_assert=True, snap_PD=None):
    TS2 = TS
    h = _hcol(hour)
    pdw = inputs_wide["power_PD"]
    qdw = inputs_wide.get("power_QD", None)

    for _, r in pdw.iterrows():
        bus_i = int(r["BUS_I"])
        PD = None if pd.isna(r[h]) else float(r[h])
        QD = None
        if qdw is not None:
            qrow = qdw[qdw["BUS_I"] == bus_i]
            if not qrow.empty and h in qrow.columns and not pd.isna(qrow.iloc[0][h]):
                QD = float(qrow.iloc[0][h])
        _set_bus_pd_qd(TS2["PowerSystem"]["bus"], bus_i, PD=PD, QD=QD)

    # 一致性断言 + 快照
    bus = TS2["PowerSystem"]["bus"]
    if hasattr(bus, "iloc"):
        try:
            bus_ids = bus["BUS_I"].astype(int).to_numpy()
            PD_vec = bus["PD"].astype(float).to_numpy()
        except Exception:
            bus_ids = bus.iloc[:, 0].astype(int).to_numpy()
            PD_vec = bus.iloc[:, 2].astype(float).to_numpy()
    else:
        bus_ids = bus[:, 0].astype(int)
        PD_vec = bus[:, 2].astype(float)
    expect = pd.Series(pdw.set_index("BUS_I")[h]).fillna(0.0)
    actual = pd.Series(PD_vec, index=bus_ids).groupby(level=0).first()
    if snap_PD is not None:
        snap_PD[h] = actual.copy()
    if strict_assert:
        denom = max(1e-6, float(expect.sum()))
        l1err = float(np.abs(expect.sort_index() - actual.reindex(expect.index).fillna(0.0)).sum()) / denom
        if l1err > 1e-6:
            raise AssertionError(f"[{h}] PD覆盖不一致：L1相对误差={l1err:.3e}，请检查 BUS_I 匹配/列名/单位。")
    return TS2

def _overlay_gas_timeseries(TS, inputs_wide, hour: int):
    TS2 = TS
    h = _hcol(hour)

    # 1) 覆盖负荷 demand（km3/h）
    gw = inputs_wide.get("gas_load_demand", None)
    if gw is not None:
        loads = TS2["GasSystem"]["loads"].copy()
        # loads: id, node, demand
        for _, r in gw.iterrows():
            lid = int(r["load_id"])
            if h in r.index and pd.notna(r[h]):
                val = float(r[h])
                hit = loads["id"].astype(int) == lid
                if hit.any():
                    loads.loc[hit, "demand"] = val
        TS2["GasSystem"]["loads"] = loads

    # 2) （可选）覆盖“非 slack 源”的固定出力 output（km3/h）
    sw = inputs_wide.get("gas_source_output_fixed", None)
    if sw is not None:
        src = TS2["GasSystem"]["sources"].copy()
        for _, r in sw.iterrows():
            sid = int(r["source_id"])
            if h in r.index and pd.notna(r[h]):
                val = float(r[h])
                hit = src["id"].astype(int) == sid
                if hit.any():
                    src.loc[hit, "output"] = val
                    # 固定源强制非 slack（避免误配）
                    if "is_slack" in src.columns:
                        src.loc[hit, "is_slack"] = 0
        TS2["GasSystem"]["sources"] = src

    return TS2


def _overlay_eh_inputs(TS, inputs_wide, hour: int, snap_EH=None):
    TS2 = TS
    h = _hcol(hour)
    EH = TS2.get("EnergyHubs", {})
    inp = EH.get("inputs", None)
    if inp is None or not hasattr(inp, "copy"):
        inp = pd.DataFrame({"type": ["electricity", "gas"], "flow": [0.0, 0.0], "PCI": [np.nan, np.nan]})
    inp = inp.copy()
    _ensure_cols(inp, ["type", "flow", "PCI"])
    inp["type_norm"] = inp["type"].astype(str).str.lower()

    ew = inputs_wide.get("eh_flow_electricity", None)
    if ew is not None and "type" in ew.columns and h in ew.columns:
        val = ew[ew["type"].astype(str).str.lower() == "electricity"]
        eflow = float(val[h].fillna(0.0).sum()) if not val.empty else 0.0
        hit = inp["type_norm"] == "electricity"
        if hit.any(): inp.loc[hit, "flow"] = eflow
        else: inp = pd.concat([inp, pd.DataFrame([{"type":"electricity","flow":eflow,"PCI":np.nan,"type_norm":"electricity"}])])

    gw = inputs_wide.get("eh_flow_gas", None)
    if gw is not None and "type" in gw.columns and h in gw.columns:
        val = gw[gw["type"].astype(str).str.lower() == "gas"]
        gflow = float(val[h].fillna(0.0).sum()) if not val.empty else 0.0
        hit = inp["type_norm"] == "gas"
        if hit.any(): inp.loc[hit, "flow"] = gflow
        else: inp = pd.concat([inp, pd.DataFrame([{"type":"gas","flow":gflow,"PCI":np.nan,"type_norm":"gas"}])])

    pci_tbl = inputs_wide.get("eh_PCI_gas", None)
    if pci_tbl is not None and "type" in pci_tbl.columns and "PCI_ref" in pci_tbl.columns:
        pci_vals = pci_tbl[pci_tbl["type"].astype(str).str.lower() == "gas"]["PCI_ref"].dropna()
        if len(pci_vals):
            inp.loc[inp["type_norm"] == "gas", "PCI"] = float(pci_vals.iloc[0])

    EH["inputs"] = inp.drop(columns=["type_norm"], errors="ignore")
    EH["respect_inputs"] = bool(
    (ew is not None) or (gw is not None)
)
    TS2["EnergyHubs"] = EH

    if snap_EH is not None:
        g = (EH["inputs"][["type", "flow"]]
             .assign(type=EH["inputs"]["type"].astype(str).str.lower())
             .groupby("type", as_index=True)["flow"].sum())
        snap_EH[h] = g.reindex(["electricity", "gas"])
    return TS2

def _overlay_heat_loads(TS, inputs_wide, hour: int, convert_to_injection=False):
    TS2 = TS
    h = _hcol(hour)
    hw = inputs_wide.get("heat_Q_demand", None)
    if hw is None or "load_id" not in hw.columns or h not in hw.columns:
        return TS2
    HS = TS2.get("HeatSystem", {})
    loads = HS.get("loads", None)
    if loads is None or not hasattr(loads, "copy"):
        return TS2
    loads = loads.copy()
    if "Q_demand" not in loads.columns:
        loads["Q_demand"] = 0.0
    id2row = {}
    if "load_id" in loads.columns:
        for idx, lid in enumerate(loads["load_id"].fillna(method="ffill").fillna(method="bfill").fillna(-1).astype(int)):
            id2row[int(lid)] = loads.index[idx]
    else:
        for idx in range(len(loads)):
            id2row[idx + 1] = loads.index[idx]
    for _, r in hw.iterrows():
        lid = int(r["load_id"])
        if lid in id2row and not pd.isna(r[h]):
            loads.loc[id2row[lid], "Q_demand"] = float(r[h])
    TS2["HeatSystem"]["loads"] = loads

    if convert_to_injection:
        nodes = HS.get("nodes", None)
        if isinstance(nodes, pd.DataFrame) and "node" in loads.columns and "id" in nodes.columns:
            dT = 15.0
            nodes = nodes.copy()
            inj = nodes.get("injection", pd.Series([0.0] * len(nodes))).to_numpy(dtype=float)
            idlist = nodes["id"].astype(int).tolist()
            for _, rr in loads.iterrows():
                if pd.isna(rr.get("Q_demand")) or pd.isna(rr.get("node")): continue
                nid = int(rr["node"])
                if nid in idlist:
                    Qw = float(rr["Q_demand"]) * 1e6
                    m = Qw / (4200.0 * dT)
                    inj[idlist.index(nid)] += m
            nodes["injection"] = inj
            TS2["HeatSystem"]["nodes"] = nodes
    return TS2

def _overlay_gas_loads(TS, inputs_wide, hour: int):
    TS2 = TS
    h = _hcol(hour)
    gw = inputs_wide.get("gas_load_demand", None)
    if gw is None or h not in gw.columns:
        return TS2

    GS = TS2.get("GasSystem", {})
    loads = GS.get("loads", None)
    if loads is None or not hasattr(loads, "copy"):
        return TS2
    loads = loads.copy()

    # 兼容 id / load_id
    key_col = "id" if "id" in loads.columns else ("load_id" if "load_id" in loads.columns else None)
    if key_col is None:
        return TS2

    key_in = "id" if "id" in gw.columns else ("load_id" if "load_id" in gw.columns else None)
    if key_in is None:
        raise KeyError("gas_load_demand 缺少 id/load_id 列")

    id2row = {int(v): i for i, v in enumerate(loads[key_col].astype(int).tolist())}

    for _, r in gw.iterrows():
        lid = int(r[key_in])
        if lid in id2row and not pd.isna(r[h]):
            loads.loc[id2row[lid], "demand"] = float(r[h])

    TS2["GasSystem"]["loads"] = loads
    return TS2


def _heat_build_injection_with_slack(HS, dT: float = 15.0, cp: float = 4200.0):
    """
    强制热网满足 Demand：由 loads.Q_demand 构造 nodes.injection（kg/s），并用 sources[0] 作为 slack 平衡。
    约定（与现有热网数据表的符号一致）：
      - injection > 0：向网络注入（热源）
      - injection < 0：从网络抽取（热负荷）
    """
    nodes   = HS["nodes"].copy()
    loads   = HS.get("loads", pd.DataFrame()).copy()
    sources = HS.get("sources", pd.DataFrame()).copy()

    if len(sources) == 0:
        raise ValueError("[HeatSlack] HeatSystem.sources 为空，无法设置 slack 热源。")
    if "node" not in sources.columns:
        raise ValueError("[HeatSlack] HeatSystem.sources 缺少 node 列。")
    if "id" not in nodes.columns:
        raise ValueError("[HeatSlack] HeatSystem.nodes 缺少 id 列。")

    idlist = nodes["id"].astype(int).tolist()
    inj = np.zeros(len(nodes), dtype=float)

    # 1) 负荷：injection 设为负（抽取）
    if isinstance(loads, pd.DataFrame) and len(loads) > 0:
        if "Q_demand" not in loads.columns:
            loads["Q_demand"] = 0.0
        if "node" not in loads.columns:
            raise ValueError("[HeatSlack] HeatSystem.loads 缺少 node 列，无法定位负荷节点。")

        for _, rr in loads.iterrows():
            if pd.isna(rr.get("Q_demand")) or pd.isna(rr.get("node")):
                continue
            nid = int(rr["node"])
            if nid not in idlist:
                continue
            Q_MW = float(rr["Q_demand"])
            m = (Q_MW * 1e6) / (cp * dT)  # kg/s
            inj[idlist.index(nid)] -= m   # 负荷是抽取：负号

    # 2) slack 热源：sources 第一行作为 slack（注入）
    src_node = int(sources.loc[sources.index[0], "node"])
    if src_node not in idlist:
        raise ValueError(f"[HeatSlack] slack source node={src_node} 不在 heat_nodes.id 中")
    src_pos = idlist.index(src_node)

    inj[src_pos] = -inj.sum()   # 保证 sum(injection)=0

    # 3) 把 slack 注入换算回 MW，写回 sources.output，保证 MW↔kg/s 一致
    Q_src_MW = inj[src_pos] * cp * dT / 1e6
    sources.loc[sources.index[0], "output"] = float(Q_src_MW)

    nodes["injection"] = inj
    HS["nodes"] = nodes
    HS["sources"] = sources
    HS["loads"] = loads
    return HS

# ========================= 逐时主流程 =========================

def run_timeseries_day_wide(dataset_dir: str, out_dir: str, hours: int = 24,
                            convert_heat_load_to_injection: bool = True) -> str:
    TS0 = build_test_system_from_excel(dataset_dir)
    # —— 新增：构造储热对象（可按需改参数），后续放在excel中配置

    # ---- 识别“外部电网”机组（默认：接在零负荷母线上的机组） ----
    pw0 = TS0.get("PowerSystem", {})
    bus0 = np.asarray(pw0.get("bus"))
    gen0 = np.asarray(pw0.get("gen"))

    extgrid_bus_id = None
    internal_gen_mask = None
    if bus0 is not None and gen0 is not None and bus0.size > 0 and gen0.size > 0:
        try:
            bus_ids0 = bus0[:, 0].astype(int)
            PD0      = bus0[:, 2].astype(float)
            gen_bus0 = gen0[:, 0].astype(int)

            zero_load_buses = set(bus_ids0[np.isclose(PD0, 0.0)])
            gen_buses       = set(gen_bus0.tolist())
            cand = sorted(zero_load_buses.intersection(gen_buses))

            # 目前数据中应只有一个“外部电网”机组母线（例如 bus 7）
            if len(cand) == 1:
                extgrid_bus_id = int(cand[0])
            # 生成内部机组掩码（True = 内部机组）
            if extgrid_bus_id is not None:
                internal_gen_mask = (gen_bus0 != extgrid_bus_id)
            else:
                internal_gen_mask = np.ones(gen0.shape[0], dtype=bool)
        except Exception:
            internal_gen_mask = None

    pit_cfg = PitThermalStorageConfig(
    H=16.0, W_top=90.0, W_bot=26.0, n_layers=20,
    U_side=111.0, U_top=26.6, U_bot=74.9,
    T_top_env=35.0, T_side_env=25.0, T_bot_env=25.0,
    T_ref=25.0, T_initial=25.0, dt_int=15.0,
    w_es0=None
    )
    pit = PitThermalStorage(pit_cfg)
    # 24小时收集器（新增）
    perh_pit_e = {}      # 储热能量（MWh）
    perh_pit_wes = {}    # 储热碳势（tCO2/MWh）
    perh_pit_E = {}      # 储热碳库存（tCO2）
    # —— 并联系统分流统计（可选导出，便于复核）
    perh_heat_source_mix = {}  # index=['EH_total','EH_direct','Pit_out','ToHeat_total','Demand','Charge_to_pit']
    # [STORAGE HOOK] build runtime from storage tables if present
    # 电源出力分解（内部机组 vs 外部电网）
    # index = ['Internal_gen', 'Grid_import']，单位 MW
    perh_power_gen_mix = {}
    perh_gas_network_mix = {}   # metric -> Hxx
    perh_grid_trade_mix  = {}   # metric -> Hxx
    perh_EH_output_mix   = {}   # metric -> Hxx


    try:
        stor = build_runtime_from_TS(TS0, delta_t_hours=1.0)
    except Exception:
        stor = None
    xls_path = os.path.join(dataset_dir, "ies_dataset.xlsx")
    inputs_wide = _load_inputs_wide(xls_path)
    os.makedirs(out_dir, exist_ok=True)
    # 外部接口 / 购售电表 & 碳强度
    grid_buy_tbl = inputs_wide.get("grid_trade_buy")
    grid_sell_tbl = inputs_wide.get("grid_trade_sell")
    ext_ci_tbl = inputs_wide.get("external_CI")

    # 选取 external_CI 中 profile_id = "ext_grid" 的那一行（若无则取第一行）
    ext_ci_row = None
    if isinstance(ext_ci_tbl, pd.DataFrame) and not ext_ci_tbl.empty:
        if "profile_id" in ext_ci_tbl.columns:
            mask = ext_ci_tbl["profile_id"].astype(str) == "ext_grid"
            if mask.any():
                ext_ci_row = ext_ci_tbl.loc[mask].iloc[0]
            else:
                ext_ci_row = ext_ci_tbl.iloc[0]
        else:
            ext_ci_row = ext_ci_tbl.iloc[0]

    # 外部接口母线 ID（我们在 Excel 里约定为 7）
    ext_bus_id = 7

    # 收集器（已有）
    perh_NCI = {}
    perh_heat_Q = {}
    perh_heat_CEFR = {}
    perh_EH_flow_by_type = {}

    snap_PD = {}
    snap_EH = {}

    kpi = {"hour": [], "NCI_avg_tCO2_per_MWh": [], "Heat_Q_total_MW": [], "Heat_CEFR_total_tph": [], "EH_heat_flow_MW": []}

    # 新增收集器（你需求的 7 项）
    perh_power_CFR_branch = {}       # index=branch_id, tCO2/h
    perh_power_CFR_load_bus = {}     # index=bus_id, tCO2/h
    perh_gas_CFR_pipes = {}          # index=pipe_id, tCO2/h
    perh_heat_input_GCI = {}         # index=source_id, tCO2/MWh
    perh_heat_CFR_pipes = {}         # index=pipe_id, tCO2/h
    perh_heat_CFR_loads = {}         # index=load_id, tCO2/h

    for h in range(hours):
        TS_h = copy.deepcopy(TS0)
        TS_h = _overlay_power_loads(TS_h, inputs_wide, h, strict_assert=True, snap_PD=snap_PD)
        TS_h = _overlay_eh_inputs(TS_h, inputs_wide, h, snap_EH=snap_EH)
        TS_h = _overlay_heat_loads(TS_h, inputs_wide, h, convert_to_injection=convert_heat_load_to_injection)

        # 四大模块
        # [STORAGE HOOK] build extra injections from dispatch
        # [STORAGE + GRID HOOK] build extra injections from dispatch & grid trades
        extra_loads = []
        extra_gens = []

        # 1) 电储能注入（充电->负荷，放电->带 CI 的机组）
        if stor is not None:
            ch_tbl = inputs_wide.get("storage_dispatch_ch")
            dc_tbl = inputs_wide.get("storage_dispatch_dc")
            cmd_ch = {}; cmd_dc = {}
            if isinstance(ch_tbl, pd.DataFrame) and _hcol(h) in ch_tbl.columns and "id" in ch_tbl.columns:
                cmd_ch = {
                    str(r["id"]): float(r[_hcol(h)])
                    for _, r in ch_tbl.iterrows()
                    if pd.notna(r.get(_hcol(h)))
                }
            if isinstance(dc_tbl, pd.DataFrame) and _hcol(h) in dc_tbl.columns and "id" in dc_tbl.columns:
                cmd_dc = {
                    str(r["id"]): float(r[_hcol(h)])
                    for _, r in dc_tbl.iterrows()
                    if pd.notna(r.get(_hcol(h)))
                }
            add_loads, add_gens = stor.plan_injections(h, cmd_ch, cmd_dc)
            extra_loads.extend(add_loads)
            extra_gens.extend(add_gens)

        # 2) 外部购售电注入（统一加在 BUS7）
        P_buy = 0.0
        P_sell = 0.0

        # 2.1 购电
        if isinstance(grid_buy_tbl, pd.DataFrame) and _hcol(h) in grid_buy_tbl.columns:
            df = grid_buy_tbl
            if "profile_id" in df.columns:
                mask = df["profile_id"].astype(str) == "ext_grid"
                row = df.loc[mask].iloc[0] if mask.any() else df.iloc[0]
            else:
                row = df.iloc[0]
            val = row.get(_hcol(h), np.nan)
            if pd.notna(val):
                P_buy = float(val)

        # 2.2 售电
        if isinstance(grid_sell_tbl, pd.DataFrame) and _hcol(h) in grid_sell_tbl.columns:
            df = grid_sell_tbl
            if "profile_id" in df.columns:
                mask = df["profile_id"].astype(str) == "ext_grid"
                row = df.loc[mask].iloc[0] if mask.any() else df.iloc[0]
            else:
                row = df.iloc[0]
            val = row.get(_hcol(h), np.nan)
            if pd.notna(val):
                P_sell = float(val)

        # 2.3 把购电作为新增机组注入（带外部碳强度 CI）
        if P_buy > 1e-6:
            if ext_ci_row is not None and _hcol(h) in ext_ci_row.index:
                w_ext = float(ext_ci_row[_hcol(h)])
            else:
                w_ext = 0.0
            extra_gens.append((ext_bus_id, P_buy, w_ext))

        # 2.4 把售电作为 BUS7 的额外负荷
        if P_sell > 1e-6:
            extra_loads.append((ext_bus_id, P_sell))
        # --- 记录购售电（用于绘图/核算）---
        perh_grid_trade_mix[_hcol(h)] = pd.Series({"Buy_MW": float(P_buy), "Sell_MW": float(P_sell)})

        # 3) 运行潮流（把储能 + 外部购售电一起带进去）
        TS_h = run_power_flow(
            TS_h,
            extra_loads=extra_loads if extra_loads else None,
            extra_gens=extra_gens if extra_gens else None,
        )
        
        # [STORAGE + GRID HOOK] build extra injections from dispatch & grid trades
        extra_loads = []
        extra_gens = []

        # 1) 电储能注入（充电->负荷，放电->带 CI 的机组）
        if stor is not None:
            ch_tbl = inputs_wide.get("storage_dispatch_ch")
            dc_tbl = inputs_wide.get("storage_dispatch_dc")
            cmd_ch = {}; cmd_dc = {}
            if isinstance(ch_tbl, pd.DataFrame) and _hcol(h) in ch_tbl.columns and "id" in ch_tbl.columns:
                cmd_ch = {
                    str(r["id"]): float(r[_hcol(h)])
                    for _, r in ch_tbl.iterrows()
                    if pd.notna(r.get(_hcol(h)))
                }
            if isinstance(dc_tbl, pd.DataFrame) and _hcol(h) in dc_tbl.columns and "id" in dc_tbl.columns:
                cmd_dc = {
                    str(r["id"]): float(r[_hcol(h)])
                    for _, r in dc_tbl.iterrows()
                    if pd.notna(r.get(_hcol(h)))
                }
            add_loads, add_gens = stor.plan_injections(h, cmd_ch, cmd_dc)
            extra_loads.extend(add_loads)
            extra_gens.extend(add_gens)

        # 2) 外部购售电注入（统一加在 BUS7）
        P_buy = 0.0
        P_sell = 0.0

        # 2.1 购电
        if isinstance(grid_buy_tbl, pd.DataFrame) and _hcol(h) in grid_buy_tbl.columns:
            df = grid_buy_tbl
            if "profile_id" in df.columns:
                mask = df["profile_id"].astype(str) == "ext_grid"
                row = df.loc[mask].iloc[0] if mask.any() else df.iloc[0]
            else:
                row = df.iloc[0]
            val = row.get(_hcol(h), np.nan)
            if pd.notna(val):
                P_buy = float(val)

        # 2.2 售电
        if isinstance(grid_sell_tbl, pd.DataFrame) and _hcol(h) in grid_sell_tbl.columns:
            df = grid_sell_tbl
            if "profile_id" in df.columns:
                mask = df["profile_id"].astype(str) == "ext_grid"
                row = df.loc[mask].iloc[0] if mask.any() else df.iloc[0]
            else:
                row = df.iloc[0]
            val = row.get(_hcol(h), np.nan)
            if pd.notna(val):
                P_sell = float(val)

        # 2.3 把购电作为新增机组注入（带外部碳强度 CI）
        if P_buy > 1e-6:
            if ext_ci_row is not None and _hcol(h) in ext_ci_row.index:
                w_ext = float(ext_ci_row[_hcol(h)])
            else:
                w_ext = 0.0
            extra_gens.append((ext_bus_id, P_buy, w_ext))

        # 2.4 把售电作为 BUS7 的额外负荷
        if P_sell > 1e-6:
            extra_loads.append((ext_bus_id, P_sell))

        # 3) 运行潮流（把储能 + 外部购售电一起带进去）
        TS_h = run_power_flow(
            TS_h,
            extra_loads=extra_loads if extra_loads else None,
            extra_gens=extra_gens if extra_gens else None,
        )
                # ---- 记录电源出力：内部机组 vs 外部电网（单位 MW）----
               # ---- 记录电源出力：内部机组 vs 外部电网（单位 MW）----
        try:
            pw = TS_h.get("PowerSystem", {})
            res = pw.get("results", {})
            gen_res = np.asarray(res.get("gen"))
            if gen_res is not None and gen_res.size > 0:
                gen_bus = gen_res[:, 0].astype(int)
                PG_MW   = gen_res[:, 1].astype(float)  # 现在直接就是 MW

                if extgrid_bus_id is not None:
                    ext_mask = (gen_bus == int(extgrid_bus_id))
                    PG_ext = float(PG_MW[ext_mask].sum())   # Grid_import [MW]
                    PG_int = float(PG_MW[~ext_mask].sum())  # Internal_gen [MW]
                else:
                    PG_int = float(PG_MW.sum())
                    PG_ext = 0.0

                perh_power_gen_mix[_hcol(h)] = pd.Series(
                    {"Internal_gen": PG_int, "Grid_import": PG_ext}
                )

        except Exception:
            # 出问题时该小时留空，但不要让整个仿真挂掉
            perh_power_gen_mix[_hcol] = None

        # [STORAGE HOOK] update es states based on node CIs
        if stor is not None and isinstance(TS_h.get("PowerSystem",{}).get("NCI", None), np.ndarray):
            EN = np.asarray(TS_h['PowerSystem']['NCI'], dtype=float).ravel()
            bus_ids = TS_h['PowerSystem']['bus'][:,0].astype(int)
            w_map = {int(b): float(EN[i]) for i,b in enumerate(bus_ids)}
            stor.update_states(h, w_map)
            
        TS_h = _overlay_gas_timeseries(TS_h, inputs_wide, h)
        TS_h = _overlay_gas_loads(TS_h, inputs_wide, h)
        TS_h = run_gas_flow(TS_h)
        TS_h = run_energy_hub(TS_h)
        
        # --- 记录气网供需（换算为 MW）---
        try:
            gs = TS_h.get("GasSystem", {})
            B = 10.45  # MWh/(km^3)
            loads_df = gs.get("loads", None)
            src_df   = gs.get("sources", None)
            comp_df  = gs.get("compressors", None)

            load_k = float(loads_df["demand"].fillna(0.0).astype(float).sum()) if isinstance(loads_df, pd.DataFrame) else 0.0
            sup_k  = float(src_df["output"].fillna(0.0).astype(float).sum()) if isinstance(src_df, pd.DataFrame) else 0.0
            comp_k = float(comp_df["gasconsumption"].fillna(0.0).astype(float).sum()) if isinstance(comp_df, pd.DataFrame) and "gasconsumption" in comp_df.columns else 0.0

            slack_k = 0.0
            if isinstance(src_df, pd.DataFrame):
                if "is_slack" in src_df.columns:
                    slack_k = float(src_df.loc[src_df["is_slack"].fillna(0).astype(int) == 1, "output"].fillna(0.0).astype(float).sum())
                else:
                    slack_k = float(src_df["output"].fillna(0.0).astype(float).iloc[0]) if len(src_df) else 0.0

            perh_gas_network_mix[_hcol(h)] = pd.Series({
                "Supply_MW":      sup_k  * B,
                "Load_MW":        load_k * B,
                "Compressor_MW":  comp_k * B,
                "Slack_supply_MW": slack_k * B,
            })
        except Exception:
            pass

        TS_h = run_energy_hub(TS_h)
        # --- 记录 EH 输出汇总（用于 EH 平衡图）---
        try:
            eh = TS_h.get("EnergyHubs", {})
            out = eh.get("outputs", None)
            if isinstance(out, pd.DataFrame) and "type" in out.columns and "flow" in out.columns:
                t = out["type"].astype(str).str.lower()
                f = out["flow"].fillna(0.0).astype(float)

                elec_out = float(f[t.str.startswith("electricity")].sum())
                heat_out = float(f[t.str.startswith("heat")].sum())
                cool_out = float(f[t.str.startswith("cooling")].sum())
                
                perh_EH_output_mix[_hcol(h)] = pd.Series({
                    "Electric_out_MW": elec_out,
                    "Heat_out_MW":     heat_out,
                    "Cooling_out_MW":  cool_out,
                })
        except Exception:
            pass

        eh_out = TS_h['EnergyHubs']['outputs'].copy()
        eh_out['type_norm'] = eh_out['type'].astype(str).str.lower()
        mask_h = eh_out['type_norm'].str.startswith('heat_')
        Q_eh_MW = float(eh_out.loc[mask_h, 'flow'].fillna(0.0).sum())
        if Q_eh_MW > 1e-9:
            w_eh = float((eh_out.loc[mask_h, 'PCI'].fillna(0.0) * eh_out.loc[mask_h, 'flow'].fillna(0.0)).sum() / Q_eh_MW)
        else:
            w_eh = 0.0


        # 当小时热负荷（优先满足）
        HS = TS_h['HeatSystem']
        loads_df = HS.get('loads', None)
        demand_MW = float(loads_df.get('Q_demand', pd.Series([0.0]*len(loads_df))).sum()) if isinstance(loads_df, pd.DataFrame) else 0.0

        # 并联策略拆分：
        Q_eh_direct = min(Q_eh_MW, demand_MW)        # 直供到热网（EH→负荷）
        Q_charge    = max(Q_eh_MW - Q_eh_direct, 0)  # 富余充入池
        Q_out_sp    = max(demand_MW - Q_eh_direct, 0)  # 缺口由池子补足（放热设定值）

        # 反解充热质量流量（按入口温度策略）
        T_inlet = 90.0
        T_top = float(pit.state.T_layers[1])
        cp = pit.cfg.cp_w
        dT = max(1.0, T_inlet - T_top)
        m_flow = (Q_charge * 1e6) / (cp * dT) if Q_charge > 1e-9 else 0.0  # kg/s

        # 逐小时推进储热（充热+按缺口放热）
        res_pit = pit.step(
            dt_seconds=3600,
            T_inlet_C=T_inlet,
            m_flow_kg_s=float(m_flow),
            Q_out_setpoint_MW=float(Q_out_sp),
            w_in_t_per_MWh=w_eh
        )
        # === after pit.step(...) ===
        snap = pit.get_state_snapshot()  # returns e_state_MWh, w_es_t_per_MWh, E_state_tCO2, ...
        perh_pit_e[_hcol(h)] = pd.Series({"PIT": float(snap.get("e_state_MWh", 0.0))})
        perh_pit_wes[_hcol(h)] = pd.Series({"PIT": float(snap.get("w_es_t_per_MWh", 0.0))})
        perh_pit_E[_hcol(h)] = pd.Series({"PIT": float(snap.get("E_state_tCO2", 0.0))})

        # 合并到“热源侧”：总供热=EH直供 + 池放热
        Q_pit_out = float(res_pit['Q_out_MW'])
        Q_to_heat = Q_eh_direct + Q_pit_out

        # 合并碳势（功率加权），池侧 w_out 取当期放热的混合强度（模块已返回）
        w_pit_out = float(res_pit['w_out_t_per_MWh'])
        if Q_to_heat > 1e-9:
            GCI_mix = (Q_eh_direct * w_eh + Q_pit_out * w_pit_out) / Q_to_heat
        else:
            GCI_mix = 0.0

        # 覆盖热源：把“并联合成源”写入 HeatSystem.sources[0]
        # --- 新增：Demand 必须满足 → slack 热源自动补齐 ---
        slack_heat_MW = max(demand_MW - Q_to_heat, 0.0)     # 外部补齐热量（MW）
        Q_to_heat_total = Q_to_heat + slack_heat_MW         # 应等于 demand_MW

        # slack 热源的碳强度（可按需要改成从表读取）
        GCI_slack = 0.222  # tCO2/MWh（例如气锅炉 0.20/0.9≈0.222；也可改成固定值/表内值）

        if Q_to_heat_total > 1e-9:
            GCI_mix_total = (Q_eh_direct * w_eh + Q_pit_out * w_pit_out + slack_heat_MW * GCI_slack) / Q_to_heat_total
        else:
            GCI_mix_total = 0.0

        # 先把“对热网的总供给”和碳强度写入 sources[0]
        HS['sources'].loc[0, 'output'] = float(Q_to_heat_total)   # MW，保证满足需求
        HS['sources'].loc[0, 'GCI']    = float(GCI_mix_total)     # tCO2/MWh

        # 关键：重构 nodes.injection（kg/s）并用 slack 平衡，保证数值一致
        HS = _heat_build_injection_with_slack(HS, dT=15.0, cp=4200.0)

        # --- 立即做一致性校验（强烈建议保留）---
        # 需求 MW
        demand_chk = float(HS['loads']['Q_demand'].fillna(0.0).astype(float).sum()) if isinstance(HS.get('loads', None), pd.DataFrame) else 0.0
        # 由 injection 反算出来的 slack 供给 MW（应≈demand_chk）
        nodes_chk = HS['nodes']
        idlist = nodes_chk['id'].astype(int).tolist()
        src_node = int(HS['sources'].loc[0, 'node'])
        src_pos = idlist.index(src_node)
        inj_src = float(nodes_chk.loc[src_pos, 'injection'])
        Q_src_chk = inj_src * 4200.0 * 15.0 / 1e6

        print(f"[HEAT SLACK CHECK h={h}] Demand={demand_chk:.3f} MW | ToHeat(EH+Pit)={Q_to_heat:.3f} MW | Slack={slack_heat_MW:.3f} MW | Source_from_inj={Q_src_chk:.3f} MW")

        TS_h['HeatSystem'] = HS

        # （可选）并联系统分流统计：建议把 slack 也记进去，后面画图/排查非常有用
        perh_heat_source_mix[_hcol(h)] = pd.Series({
            'EH_total':      Q_eh_MW,
            'EH_direct':     Q_eh_direct,
            'Pit_out':       Q_pit_out,
            'ToHeat_total':  float(Q_to_heat_total),
            'Demand':        demand_MW,
            'Charge_to_pit': Q_charge,
            'Slack_heat':    slack_heat_MW
        })

        # 然后再跑热网
        TS_h = run_heat_flow(TS_h)

        # ----- 电侧：节点碳势（tCO2/MWh）
        pw = TS_h.get("PowerSystem", {})
        # 母线 ID
        if "bus" in pw and isinstance(pw["bus"], np.ndarray):
            bus_ids = pw["bus"][:, 0].astype(int)
        else:
            bus_ids = None

        EN = None
        if isinstance(pw.get("NCI", None), np.ndarray):
            EN = pw["NCI"].ravel()
        if EN is None and "results" in pw and "bus" in pw["results"]:
            # 兜底（不推荐）：如未写回 NCI，可在此按需要重算或置 NaN
            EN = np.full(len(bus_ids), np.nan, dtype=float)

        if bus_ids is not None and EN is not None:
            perh_NCI[_hcol(h)] = pd.Series(EN, index=bus_ids)


        # ----- 电侧：支路碳流率（tCO2/h，按单支路）
        # 规则与 PB 构造一致：方向 f->t 用 |PT|；方向 t->f 用 |PF|；碳强度取“源端节点 EN”
        if "results" in pw and "branch" in pw["results"] and "BCI" in pw:
            branch = pw["results"]["branch"]
            EN = np.asarray(pw.get("NCI", []), dtype=float).ravel()
            n_line = branch.shape[0]
            f_bus = (branch[:, F_BUS].astype(int) - 1)
            t_bus = (branch[:, T_BUS].astype(int) - 1)
            pf_col = branch[:, PF]
            pt_col = branch[:, PT]
            CFR_branch = np.zeros(n_line, dtype=float)
            for i in range(n_line):
                if pf_col[i] >= 0:
                    src = f_bus[i]
                    mw = abs(pt_col[i])
                else:
                    src = t_bus[i]
                    mw = abs(pf_col[i])
                src = np.clip(src, 0, len(EN)-1)
                CFR_branch[i] = EN[src] * mw  # t/MWh * MW = t/h
            # 分配 branch_id（若无 id 列则 1..n）
            if isinstance(TS_h["PowerSystem"]["branch"], np.ndarray):
                ids = pd.Index(range(1, n_line + 1), name="branch_id")
            else:
                df_br = pd.DataFrame(TS_h["PowerSystem"]["branch"])
                ids = df_br.get("branch_id", pd.Series(range(1, n_line + 1))).astype(int)
            perh_power_CFR_branch[_hcol(h)] = pd.Series(CFR_branch, index=np.asarray(ids))

        # ----- 电侧：负荷碳流率（按母线，tCO2/h）≈ max(PD,0) * NCI
        if bus_ids is not None and "NCI" in pw:
            bus = TS_h["PowerSystem"]["bus"]
            PD = (bus["PD"].to_numpy() if hasattr(bus, "iloc") and "PD" in bus.columns else bus[:, 2]).astype(float)
            RL_bus = np.maximum(PD, 0.0) * np.asarray(pw["NCI"], dtype=float).ravel()
            bus_idx = pd.Index(np.asarray(bus_ids, dtype=int))   # 保证是 pandas Index
            perh_power_CFR_load_bus[_hcol(h)] = pd.Series(np.asarray(RL_bus, dtype=float), index=bus_idx)

        # ----- 气侧：支路碳流率（tCO2/h）
        gs = TS_h.get("GasSystem", {})
        pipes = gs.get("pipelines", None)
        if isinstance(pipes, pd.DataFrame) and "CEFR" in pipes.columns:
            ids = (pipes.get("id", pd.Series(range(1, len(pipes) + 1))).astype(int)).values
            perh_gas_CFR_pipes[_hcol(h)] = pd.Series(pipes["CEFR"].to_numpy(dtype=float), index=ids)

        # ----- 热侧：输入端口碳势（按热源，tCO2/MWh）
        hs = TS_h.get("HeatSystem", {})
        sources = hs.get("sources", None)
        if isinstance(sources, pd.DataFrame) and "GCI" in sources.columns:
            src_ids = (sources.get("id", pd.Series(range(1, len(sources) + 1))).astype(int)).values
            perh_heat_input_GCI[_hcol(h)] = pd.Series(sources["GCI"].to_numpy(dtype=float), index=src_ids)

        # ----- 热侧：支路碳流率（tCO2/h）
        hpipes = hs.get("pipelines", None)
        if isinstance(hpipes, pd.DataFrame) and "id" in hpipes.columns and "CEFR_pipes" in hs:
            # heat_flow.py 写回的是 ndarray：HS['CEFR_pipes'] 与 pipelines 行数对齐
            cfr = np.asarray(hs["CEFR_pipes"], dtype=float).ravel()
            hid = hpipes["id"].astype(int).values if "id" in hpipes.columns else np.arange(1, len(hpipes)+1)
            perh_heat_CFR_pipes[_hcol(h)] = pd.Series(cfr, index=hid)

        # ----- 热侧：负荷碳流率（按热负荷，tCO2/h）≈ GCI_heat * Q_demand
        if isinstance(sources, pd.DataFrame) and "GCI" in sources.columns and isinstance(hs.get("loads", None), pd.DataFrame):
            GCI_heat = float(sources["GCI"].ffill().bfill().iloc[0])
            loads_df = hs["loads"]
            lid = (loads_df.get("load_id", pd.Series(range(1, len(loads_df) + 1))).astype(int)).values
            Qd = loads_df.get("Q_demand", pd.Series([0.0] * len(loads_df))).to_numpy(dtype=float)
            perh_heat_CFR_loads[_hcol(h)] = pd.Series(GCI_heat * Qd, index=lid)

        # ----- KPI
        q_total = float(hs.get("Q_total_MW", np.nan))
        cefr_total = float(hs.get("CEFR_total", np.nan))
        perh_heat_Q[_hcol(h)] = pd.Series([q_total], index=["system"])
        perh_heat_CEFR[_hcol(h)] = pd.Series([cefr_total], index=["system"])

        NCI = np.array(pw.get("NCI", []), dtype=float)
        NCI_avg = float(np.nanmean(NCI)) if NCI.size else np.nan
        # 统计 EH 热端口总流量（MW）
        EH_heat_total = 0.0
        eh = TS_h.get("EnergyHubs", {})
        if "outputs" in eh and hasattr(eh["outputs"], "copy"):
            out = eh["outputs"].copy()
            if "type" in out.columns and "flow" in out.columns:
                out["type_norm"] = out["type"].astype(str).str.lower()
                EH_heat_total = float(out.loc[out["type_norm"].str.startswith("heat_"), "flow"].fillna(0).sum())

        kpi["hour"].append(h)
        kpi["NCI_avg_tCO2_per_MWh"].append(NCI_avg)
        kpi["Heat_Q_total_MW"].append(q_total)
        kpi["Heat_CEFR_total_tph"].append(cefr_total)
        kpi["EH_heat_flow_MW"].append(EH_heat_total)

    # ===== 导出 =====
    out_xlsx = os.path.join(out_dir, "results_timeseries_wide.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        # 已有
        _wide_from_series_dict(perh_NCI, "bus_id").to_excel(w, sheet_name="power_NCI", index=False)
        _wide_from_series_dict(perh_heat_Q, "scope").to_excel(w, sheet_name="heat_Q_total_MW", index=False)
        _wide_from_series_dict(perh_heat_CEFR, "scope").to_excel(w, sheet_name="heat_CEFR_total_tph", index=False)
        # 新增（7项）
        _wide_from_series_dict(perh_power_CFR_branch, "branch_id").to_excel(w, sheet_name="power_CFR_branch_tph", index=False)
        _wide_from_series_dict(perh_power_CFR_load_bus, "bus_id").to_excel(w, sheet_name="power_CFR_load_bus_tph", index=False)
        _wide_from_series_dict(perh_gas_CFR_pipes, "pipe_id").to_excel(w, sheet_name="gas_CFR_pipes_tph", index=False)
        _wide_from_series_dict(perh_heat_input_GCI, "source_id").to_excel(w, sheet_name="heat_input_GCI_tperMWh", index=False)
        _wide_from_series_dict(perh_heat_CFR_pipes, "pipe_id").to_excel(w, sheet_name="heat_CFR_pipes_tph", index=False)
        _wide_from_series_dict(perh_heat_CFR_loads, "load_id").to_excel(w, sheet_name="heat_CFR_loads_tph", index=False)
        _wide_from_series_dict(perh_pit_e,   "scope").to_excel(w, sheet_name="storage_heat_e_MWh", index=False)
        _wide_from_series_dict(perh_pit_wes, "scope").to_excel(w, sheet_name="storage_heat_wes_tperMWh", index=False)
        _wide_from_series_dict(perh_pit_E,   "scope").to_excel(w, sheet_name="storage_heat_E_t", index=False)
        _wide_from_series_dict(perh_heat_source_mix, "metric").to_excel(w, sheet_name="heat_source_mix_MW", index=False)
        # --- 新增：气网供需汇总 / 购售电 / EH输出汇总（用于能量平衡绘图）---
        _wide_from_series_dict(perh_gas_network_mix, "metric").to_excel(w, sheet_name="gas_network_mix_MW", index=False)
        _wide_from_series_dict(perh_grid_trade_mix,  "metric").to_excel(w, sheet_name="grid_trade_mix_MW", index=False)
        _wide_from_series_dict(perh_EH_output_mix,   "metric").to_excel(w, sheet_name="EH_output_mix_MW", index=False)


        # 内部机组 / 外部电网出力（来自逐时潮流结果）
        _wide_from_series_dict(
            perh_power_gen_mix,
            "metric"
        ).to_excel(w, sheet_name="power_gen_mix_MW", index=False)

        # [STORAGE HOOK] export
        try:
            if stor is not None:
                stor.log_dataframe().to_excel(w, sheet_name="storage_log", index=False)
        except Exception:
            pass
        # 入口快照与KPI
        _wide_from_series_dict(snap_PD, "bus_id").to_excel(w, sheet_name="__snap_PD_MW__", index=False)
        _wide_from_series_dict(snap_EH, "type").to_excel(w, sheet_name="__snap_EH_inputs_MW__", index=False)
        pd.DataFrame(kpi).to_excel(w, sheet_name="kpi_hourly", index=False)

    return out_xlsx
