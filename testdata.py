# testdata.py  —— 以 Excel 为唯一数据源
import os
import numpy as np
import pandas as pd

def _read_xlsx(path, sheet):
    return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

def _read_power(dataset_dir):
    single = os.path.join(dataset_dir, "ies_dataset.xlsx")
    if os.path.exists(single):
        bus    = _read_xlsx(single, "power_bus").to_numpy(dtype=float)
        gen    = _read_xlsx(single, "power_gen").to_numpy(dtype=float)
        branch = _read_xlsx(single, "power_branch").to_numpy(dtype=float)
        gci    = _read_xlsx(single, "power_gci")["GCI"].to_numpy(dtype=float)
        baseMVA = float(_read_xlsx(single, "power_meta")["baseMVA"].iloc[0])
    else:
        pw = os.path.join(dataset_dir, "power.xlsx")
        bus    = _read_xlsx(pw, "bus").to_numpy(dtype=float)
        gen    = _read_xlsx(pw, "gen").to_numpy(dtype=float)
        branch = _read_xlsx(pw, "branch").to_numpy(dtype=float)
        gci    = _read_xlsx(pw, "gci")["GCI"].to_numpy(dtype=float)
        baseMVA_df = _read_xlsx(pw, "meta")
        baseMVA = float(baseMVA_df.loc[0, "baseMVA"])
    return {
        "baseMVA": baseMVA,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "GCI": gci
    }

def _read_gas(dataset_dir):
    single = os.path.join(dataset_dir, "ies_dataset.xlsx")
    if os.path.exists(single):
        nodes        = _read_xlsx(single, "gas_nodes")
        pipes        = _read_xlsx(single, "gas_pipelines")
        compressors  = _read_xlsx(single, "gas_compressors")
        sources      = _read_xlsx(single, "gas_sources")
        loads        = _read_xlsx(single, "gas_loads")
    else:
        gz = os.path.join(dataset_dir, "gas.xlsx")
        nodes        = _read_xlsx(gz, "nodes")
        pipes        = _read_xlsx(gz, "pipelines")
        compressors  = _read_xlsx(gz, "compressors")
        sources      = _read_xlsx(gz, "sources")
        loads        = _read_xlsx(gz, "loads")
    # 统一列名/类型
    for df in (nodes, pipes, compressors, sources, loads):
        for c in df.columns:
            if df[c].dtype == "O":
                # 尽量转数值，无法转再保留
                try:
                    df[c] = pd.to_numeric(df[c])
                except Exception:
                    pass
    return {
        "nodes": nodes,
        "pipelines": pipes,
        "compressors": compressors,
        "sources": sources,
        "loads": loads
    }

def _read_heat(dataset_dir):
    single = os.path.join(dataset_dir, "ies_dataset.xlsx")
    if os.path.exists(single):
        nodes   = _read_xlsx(single, "heat_nodes")
        pipes   = _read_xlsx(single, "heat_pipelines")
        valves  = _read_xlsx(single, "heat_valves")
        pumps   = _read_xlsx(single, "heat_pumps")
        sources = _read_xlsx(single, "heat_sources")
        loads   = _read_xlsx(single, "heat_loads")
    else:
        ht = os.path.join(dataset_dir, "heat.xlsx")
        nodes   = _read_xlsx(ht, "nodes")
        pipes   = _read_xlsx(ht, "pipelines")
        valves  = _read_xlsx(ht, "valves")
        pumps   = _read_xlsx(ht, "pumps")
        sources = _read_xlsx(ht, "sources")
        loads   = _read_xlsx(ht, "loads")
    # 基本字段兜底
    if "massflow" not in pipes: pipes["massflow"] = 0.0
    if "T_in"     not in pipes: pipes["T_in"]     = 0.0
    if "T_out"    not in pipes: pipes["T_out"]    = 0.0
    if "heat_loss"not in pipes: pipes["heat_loss"]= 0.0
    if "temperature" not in nodes: nodes["temperature"] = 0.0
    if "injection"   not in nodes: nodes["injection"]   = 0.0
    return {
        "nodes": nodes,
        "pipelines": pipes,
        "valves": valves,
        "pumps": pumps,
        "sources": sources,
        "loads": loads
    }

def _read_storage(dataset_dir):
    single = os.path.join(dataset_dir, "ies_dataset.xlsx")
    if not os.path.exists(single):
        return {}
    try:
        sys_df = pd.read_excel(single, sheet_name="storage_system", engine="openpyxl")
    except Exception:
        sys_df = pd.DataFrame()
    # optional wide dispatch sheets
    try:
        ch = pd.read_excel(single, sheet_name="storage_dispatch_ch", engine="openpyxl")
    except Exception:
        ch = pd.DataFrame()
    try:
        dc = pd.read_excel(single, sheet_name="storage_dispatch_dc", engine="openpyxl")
    except Exception:
        dc = pd.DataFrame()
    return {
        "system": sys_df,
        "dispatch_ch": ch,
        "dispatch_dc": dc
    }

def _read_eh(dataset_dir):
    single = os.path.join(dataset_dir, "ies_dataset.xlsx")
    if os.path.exists(single):
        meta    = _read_xlsx(single, "eh_meta")      # 应含 powernode, gasnode
        inputs  = _read_xlsx(single, "eh_inputs")    # type, flow(opt), PCI(opt)
        outputs = _read_xlsx(single, "eh_outputs")   # type, flow(opt), PCI(opt), CEFR(opt)
    else:
        eh = os.path.join(dataset_dir, "energy_hub.xlsx")
        meta    = _read_xlsx(eh, "meta")
        inputs  = _read_xlsx(eh, "inputs")
        outputs = _read_xlsx(eh, "outputs")
    # 规范字段
    assert {"powernode","gasnode"}.issubset(set(meta.columns)), "eh_meta/meta 需包含 powernode, gasnode"
    connectednodes = {
        "powernode": int(meta["powernode"].iloc[0]),
        "gasnode": int(meta["gasnode"].iloc[0])
    }
    # 最少要有 type 列
    if "type" not in inputs:  inputs["type"]  = ["electricity","gas"]
    if "flow" not in inputs:  inputs["flow"]  = [None]*len(inputs)
    if "PCI"  not in inputs:  inputs["PCI"]   = [None]*len(inputs)
    if "type" not in outputs: outputs["type"] = ["electricity_o1","electricity_o2","heat_o1","heat_o2","cooling_o1","cooling_o2"]
    for c in ("flow","PCI","CEFR"):
        if c not in outputs: outputs[c] = [None]*len(outputs)
    return {
        "id": 1,
        "connectednodes": connectednodes,
        "inputs": inputs,
        "outputs": outputs
    }

def build_test_system_from_excel(dataset_dir):
    """
    从 Excel 构建 TS（替代原先硬编码的 build_test_system）
    """
    TS = {}
    TS["PowerSystem"] = _read_power(dataset_dir)
    TS["GasSystem"]   = _read_gas(dataset_dir)
    TS["HeatSystem"]  = _read_heat(dataset_dir)
    TS["EnergyHubs"]  = _read_eh(dataset_dir)
    TS["StorageSystem"] = _read_storage(dataset_dir)
    return TS
