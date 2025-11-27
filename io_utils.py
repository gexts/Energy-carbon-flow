# io_utils.py  —— 统一结果导出（含 ID 与单位 + 自适应列命名 + 关键字 to_excel）
import os
import numpy as np
import pandas as pd

# ===== 工具 =====
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _rename_with_units(df: pd.DataFrame, units: dict) -> pd.DataFrame:
    """按 units 映射把列名改成  name [unit]  的格式。未在映射里的列保持不变。"""
    new_cols = []
    for c in df.columns:
        # 已经带了 [ ] 说明标过单位了，跳过
        if isinstance(c, str) and c.endswith("]") and "[" in c:
            new_cols.append(c)
            continue
        u = units.get(c)
        if u:
            new_cols.append(f"{c} [{u}]")
        else:
            new_cols.append(c)
    df = df.copy()
    df.columns = new_cols
    return df

def _as_df(x):
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, (np.ndarray, list, tuple)):
        return pd.DataFrame(np.array(x))
    if isinstance(x, dict):
        return pd.DataFrame([x])
    return pd.DataFrame()

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return x

def _assign_cols_flexible(df: pd.DataFrame, base_cols: list, extra_cols_known: list) -> pd.DataFrame:
    """
    给定基础列名与“已知扩展列名”，按 df 实际列数进行自适应命名。
    若还有多余列，则以 EXTRA_# 依次命名。
    """
    n = df.shape[1]
    cols = []
    # 先用基础列
    for i in range(min(len(base_cols), n)):
        cols.append(base_cols[i])
    # 追加已知扩展列
    rem = n - len(cols)
    if rem > 0:
        for i in range(min(rem, len(extra_cols_known))):
            cols.append(extra_cols_known[i])
        rem = n - len(cols)
    # 仍有剩余 -> EXTRA_#
    if rem > 0:
        cols.extend([f"EXTRA_{i+1}" for i in range(rem)])
    df2 = df.copy()
    df2.columns = cols
    return df2

# ===== 单位字典（可按需添加/修改）=====
UNITS_POWER_RESULTS = {
    # MATPOWER bus
    "PD": "MW", "QD": "MVAr", "VM": "p.u.", "VA": "deg", "BASE_KV": "kV",
    # MATPOWER gen
    "PG": "MW", "QG": "MVAr", "PMAX": "MW", "PMIN": "MW", "VG": "p.u.",
    # MATPOWER branch（参数）
    "BR_R": "p.u.", "BR_X": "p.u.", "BR_B": "p.u.", "RATE_A": "MVA", "RATE_B": "MVA", "RATE_C": "MVA",
    # branch（潮流结果）
    "PF": "MW", "QF": "MVAr", "PT": "MW", "QT": "MVAr",
    # 常见计算派生量（如有）
    "P": "MW", "Q": "MVAr", "Loss_P": "MW", "Loss_Q": "MVAr",
    # 碳强度
    "NCI": "tCO2/MWh", "BCI": "tCO2/MWh",
    # 你项目里的中间张量（按含义估计单位）
    "PB": "MW", "PBloss": "MW", "PG_map": "MW", "PL": "MW",
    "RB": "tCO2/h", "RBloss": "tCO2/h", "RL": "tCO2/h", "RG": "tCO2/h"
}

UNITS_GAS = {
    "pressure": "MPa", "flowrate": "km3/h", "injection": "km3/h",
    "Kp": "-", "Capacity": "km3/h", "length": "km", "diameter": "mm",
    "CEFR": "tCO2/h"
}

UNITS_HEAT = {
    "temperature": "°C", "T_in": "°C", "T_out": "°C",
    "massflow": "kg/s", "heat_loss": "kW", "Q_total_MW": "MW",
    "CEFR_total": "tCO2/h"
}

UNITS_EH = {
    "flow": "MW", "PCI": "tCO2/MWh", "CEFR": "tCO2/h"
}

def _attach_id_index_vector(vec, id_series, name_value, unit=None, id_name="bus_id"):
    """把一维向量包装为 DataFrame，带 id 与单位化列名"""
    df = pd.DataFrame({
        id_name: list(map(_safe_int, id_series)),
        name_value: np.array(vec).reshape(-1)
    })
    if unit:
        df = _rename_with_units(df, {name_value: unit})
    return df

# ===== 主导出函数 =====
def save_results_to_excel(TS, out_dir):
    _ensure_dir(out_dir)

    # ---------- POWER ----------
    pw = TS.get("PowerSystem", {})
    with pd.ExcelWriter(os.path.join(out_dir, "results_power.xlsx"), engine="openpyxl") as w:
        # (1) 基础潮流结果（若存在）
        results = pw.get("results", {})
        bus_df = _as_df(results.get("bus"))
        gen_df = _as_df(results.get("gen"))
        br_df  = _as_df(results.get("branch"))

        # 识别并自适应命名 + 单位
        # bus: 基础13列 + 常见扩展4列
        if not bus_df.empty:
            bus_base = ["BUS_I","BUS_TYPE","PD","QD","GS","BS","BUS_AREA","VM","VA","BASE_KV","ZONE","VMAX","VMIN"]
            bus_extra = ["LAM_P","LAM_Q","MU_VMIN","MU_VMAX"]
            bus_df = _assign_cols_flexible(bus_df, bus_base, bus_extra)
            bus_df = _rename_with_units(bus_df, UNITS_POWER_RESULTS)
            bus_df.to_excel(w, sheet_name="bus", index=False)

        # gen: 基础21列 + 常见扩展4列
        if not gen_df.empty:
            gen_base = ["GEN_BUS","PG","QG","QMAX","QMIN","VG","MBASE","GEN_STATUS","PMAX","PMIN",
                        "PC1","PC2","QC1MIN","QC1MAX","QC2MIN","QC2MAX","RAMP_AGC","RAMP_10","RAMP_30","RAMP_Q","APF"]
            gen_extra = ["MU_PMAX","MU_PMIN","MU_QMAX","MU_QMIN"]
            gen_df = _assign_cols_flexible(gen_df, gen_base, gen_extra)
            gen_df = _rename_with_units(gen_df, UNITS_POWER_RESULTS)
            # 增加一个 gen_id 便于索引
            if "gen_id" not in gen_df.columns:
                gen_df.insert(0, "gen_id", np.arange(1, len(gen_df)+1))
            gen_df.to_excel(w, sheet_name="gen", index=False)

        # branch: 基础13列 + 常见扩展4列（PF,QF,PT,QT）
        if not br_df.empty:
            br_base = ["F_BUS","T_BUS","BR_R","BR_X","BR_B","RATE_A","RATE_B","RATE_C","TAP","SHIFT","BR_STATUS","ANGMIN","ANGMAX"]
            br_extra = ["PF","QF","PT","QT"]
            br_df = _assign_cols_flexible(br_df, br_base, br_extra)
            br_df = _rename_with_units(br_df, UNITS_POWER_RESULTS)
            if "branch_id" not in br_df.columns:
                br_df.insert(0, "branch_id", np.arange(1, len(br_df)+1))
            br_df.to_excel(w, sheet_name="branch", index=False)

        # (2) 碳强度与相关张量（全部附 ID 与单位）
        # bus_id 序列（如果 bus_df 没有，按长度回推）
        if not bus_df.empty and "BUS_I" in bus_df.columns:
            bus_ids = bus_df["BUS_I"].tolist()
        else:
            nbus = 0
            if not bus_df.empty:
                nbus = len(bus_df)
            elif "NCI" in pw:
                nbus = len(np.array(pw["NCI"]).reshape(-1))
            bus_ids = list(range(1, nbus+1))

        # branch_id 序列
        if not br_df.empty and "branch_id" in br_df.columns:
            branch_ids = br_df["branch_id"].tolist()
        else:
            nbr = 0
            if "BCI" in pw:
                nbr = len(np.array(pw["BCI"]).reshape(-1))
            branch_ids = list(range(1, nbr+1))

        # NCI（按母线）
        if "NCI" in pw:
            df = _attach_id_index_vector(pw["NCI"], bus_ids, "NCI", UNITS_POWER_RESULTS["NCI"], "bus_id")
            df.to_excel(w, sheet_name="NCI", index=False)

        # BCI（长度可能对应 bus 或 branch，自动判定）
        if "BCI" in pw:
            arr = np.array(pw["BCI"]).reshape(-1)
            if len(arr) == len(bus_ids):
                df = _attach_id_index_vector(arr, bus_ids, "BCI", UNITS_POWER_RESULTS["BCI"], "bus_id")
            else:
                df = _attach_id_index_vector(arr, branch_ids, "BCI", UNITS_POWER_RESULTS["BCI"], "branch_id")
            df.to_excel(w, sheet_name="BCI", index=False)

        # 其它张量（PB/PBloss/PG_map/PL/RB/RBloss/RL/RG）
        tensors = ["PB","PBloss","PG_map","PL","RB","RBloss","RL","RG"]
        for t in tensors:
            if t in pw:
                vec = np.array(pw[t])
                if vec.ndim == 1:
                    if len(vec) == len(bus_ids):
                        df = _attach_id_index_vector(vec, bus_ids, t, UNITS_POWER_RESULTS.get(t), "bus_id")
                    elif len(vec) == len(branch_ids):
                        df = _attach_id_index_vector(vec, branch_ids, t, UNITS_POWER_RESULTS.get(t), "branch_id")
                    else:
                        df = pd.DataFrame({t: vec})
                        df = _rename_with_units(df, {t: UNITS_POWER_RESULTS.get(t)})
                    df.to_excel(w, sheet_name=t, index=False)
                else:
                    df = pd.DataFrame(vec)
                    # 给矩阵列加单位（如果有）
                    unit_map = {c: UNITS_POWER_RESULTS.get(t) for c in df.columns}
                    df = _rename_with_units(df, unit_map)
                    df.to_excel(w, sheet_name=t, index=False)

    # ---------- GAS ----------
    gs = TS.get("GasSystem", {})
    with pd.ExcelWriter(os.path.join(out_dir, "results_gas.xlsx"), engine="openpyxl") as w:
        nodes = _as_df(gs.get("nodes"))
        if not nodes.empty and "id" not in nodes.columns:
            nodes.insert(0, "id", np.arange(1, len(nodes)+1))
        nodes = _rename_with_units(nodes, UNITS_GAS)
        nodes.to_excel(w, sheet_name="nodes", index=False)

        pipes = _as_df(gs.get("pipelines"))
        if not pipes.empty and "id" not in pipes.columns:
            pipes.insert(0, "id", np.arange(1, len(pipes)+1))
        pipes = _rename_with_units(pipes, UNITS_GAS)
        pipes.to_excel(w, sheet_name="pipelines", index=False)

        compressors = _as_df(gs.get("compressors"))
        if not compressors.empty and "id" not in compressors.columns:
            compressors.insert(0, "id", np.arange(1, len(compressors)+1))
        compressors = _rename_with_units(compressors, UNITS_GAS)
        compressors.to_excel(w, sheet_name="compressors", index=False)

        sources = _as_df(gs.get("sources"))
        sources = _rename_with_units(sources, UNITS_GAS)
        sources.to_excel(w, sheet_name="sources", index=False)

        loads = _as_df(gs.get("loads"))
        loads = _rename_with_units(loads, UNITS_GAS)
        loads.to_excel(w, sheet_name="loads", index=False)

        # 展示类结果（若存在）
        for k in ("Ag","nodes_CEFR_disp","pipes_CEFR_disp","loads_CEFR_disp","compressors_CEFR_disp"):
            if k in gs:
                df = _as_df(gs[k])
                df = _rename_with_units(df, {"CEFR": "tCO2/h"})
                if "node_id" in df.columns:
                    pass
                elif k.startswith("nodes") and not df.empty and "id" not in df.columns:
                    df.insert(0, "id", np.arange(1, len(df)+1))
                df.to_excel(w, sheet_name=k, index=False)

        if "injection_disp" in gs:
            df = _as_df(gs["injection_disp"])
            df = _rename_with_units(df, {"injection_disp": "km3/h"})
            df.to_excel(w, sheet_name="injection_disp", index=False)
        if "flowrate_disp" in gs:
            df = _as_df(gs["flowrate_disp"])
            df = _rename_with_units(df, {"flowrate_disp": "km3/h"})
            df.to_excel(w, sheet_name="flowrate_disp", index=False)
        for k in ("residual_norm_disp","sum_injection_disp"):
            if k in gs:
                pd.DataFrame({k: [gs[k]]}).to_excel(w, sheet_name=k, index=False)

    # ---------- HEAT ----------
    ht = TS.get("HeatSystem", {})
    with pd.ExcelWriter(os.path.join(out_dir, "results_heat.xlsx"), engine="openpyxl") as w:
        nodes = _as_df(ht.get("nodes"))
        if not nodes.empty and "id" not in nodes.columns:
            nodes.insert(0, "id", np.arange(1, len(nodes)+1))
        nodes = _rename_with_units(nodes, UNITS_HEAT)
        nodes.to_excel(w, sheet_name="nodes", index=False)

        pipes = _as_df(ht.get("pipelines"))
        if not pipes.empty and "id" not in pipes.columns:
            pipes.insert(0, "id", np.arange(1, len(pipes)+1))
        pipes = _rename_with_units(pipes, UNITS_HEAT)
        pipes.to_excel(w, sheet_name="pipelines", index=False)

        valves = _as_df(ht.get("valves"))
        if not valves.empty and "id" not in valves.columns:
            valves.insert(0, "id", np.arange(1, len(valves)+1))
        valves = _rename_with_units(valves, UNITS_HEAT)
        valves.to_excel(w, sheet_name="valves", index=False)

        pumps = _as_df(ht.get("pumps"))
        if not pumps.empty and "id" not in pumps.columns:
            pumps.insert(0, "id", np.arange(1, len(pumps)+1))
        pumps = _rename_with_units(pumps, UNITS_HEAT)
        pumps.to_excel(w, sheet_name="pumps", index=False)

        sources = _as_df(ht.get("sources"))
        sources = _rename_with_units(sources, UNITS_HEAT)
        sources.to_excel(w, sheet_name="sources", index=False)

        loads = _as_df(ht.get("loads"))
        loads = _rename_with_units(loads, UNITS_HEAT)
        loads.to_excel(w, sheet_name="loads", index=False)

        # 汇总类
        scalars = {}
        for k in ("Q_total_MW","CEFR_total","Q_total_MW_disp","CEFR_total_disp"):
            if k in ht:
                scalars[k] = ht[k]
        if scalars:
            pd.DataFrame([scalars]).to_excel(w, sheet_name="summary", index=False)

        for k in ("nodes_disp","pipelines_disp"):
            if k in ht:
                df = _as_df(ht[k])
                df = _rename_with_units(df, UNITS_HEAT)
                if not df.empty and "id" not in df.columns:
                    df.insert(0, "id", np.arange(1, len(df)+1))
                df.to_excel(w, sheet_name=k, index=False)

    # ---------- ENERGY HUB ----------
    eh = TS.get("EnergyHubs", {})
    with pd.ExcelWriter(os.path.join(out_dir, "results_eh.xlsx"), engine="openpyxl") as w:
        meta = {}
        cn = eh.get("connectednodes") or {}
        if "powernode" in cn: meta["powernode"] = cn["powernode"]
        if "gasnode"   in cn: meta["gasnode"]   = cn["gasnode"]
        if meta:
            pd.DataFrame([meta]).to_excel(w, sheet_name="connectednodes", index=False)

        inputs = _as_df(eh.get("inputs"))
        if not inputs.empty and "id" not in inputs.columns:
            inputs.insert(0, "id", np.arange(1, len(inputs)+1))
        inputs = _rename_with_units(inputs, UNITS_EH)
        inputs.to_excel(w, sheet_name="inputs", index=False)

        outputs = _as_df(eh.get("outputs"))
        if not outputs.empty and "id" not in outputs.columns:
            outputs.insert(0, "id", np.arange(1, len(outputs)+1))
        outputs = _rename_with_units(outputs, UNITS_EH)
        outputs.to_excel(w, sheet_name="outputs", index=False)

        outputs_disp = _as_df(eh.get("outputs_disp"))
        if not outputs_disp.empty and "id" not in outputs_disp.columns:
            outputs_disp.insert(0, "id", np.arange(1, len(outputs_disp)+1))
        outputs_disp = _rename_with_units(outputs_disp, UNITS_EH)
        if not outputs_disp.empty:
            outputs_disp.to_excel(w, sheet_name="outputs_disp", index=False)
