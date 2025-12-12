import numpy as np

def _safe_div(a, b, eps=1e-12):
    b2 = np.where(np.abs(b) < eps, np.sign(b) * eps + (np.abs(b) < eps) * eps, b)
    return a / b2

def run_energy_hub(TS):
    """
    严格复刻 MATLAB EnergyHub 逻辑：
      - 用 C 矩阵计算能量输出 flows = C @ inputs.flow
      - 用 D 矩阵计算输出端 PCI = D @ inputs.PCI
      - CEFR = PCI .* flow
    口径与单位：
      - 电输入：取电网该母线的 Pd（MW，正值为负荷/消耗）
      - 气输入：取气网该节点 injection（km^3/h，正=本节点消耗），乘 B=10.45 => MW
      - PCI_e = Power NCI (t/MWh)；PCI_g = 0.20 t/MWh
    """
    EH = TS['EnergyHubs']
    pnode = int(EH['connectednodes']['powernode'])
    gnode = int(EH['connectednodes']['gasnode'])

    # -------- 输入能流（MW）--------
    B = 10.45  # MWh/(km^3)
    use_inputs = bool(EH.get("respect_inputs", False))

    if use_inputs and "inputs" in EH and hasattr(EH["inputs"], "copy"):
        inp = EH["inputs"].copy()
        inp["type_norm"] = inp["type"].astype(str).str.lower()
        e_row = inp[inp["type_norm"] == "electricity"]
        g_row = inp[inp["type_norm"] == "gas"]
        E = float(e_row["flow"].iloc[0]) if len(e_row) else 0.0
        G = float(g_row["flow"].iloc[0]) if len(g_row) else 0.0
    else:
        # 电：来自电网母线PD（节点5时序负荷）
        Pd = TS['PowerSystem']['bus'][:, 2]     # PD (MW)
        E = float(Pd[pnode - 1])

        # 气：来自气网节点 injection（节点5时序负荷聚合后的注入）
        gas_nodes = TS['GasSystem']['nodes']
        inj_series = gas_nodes.loc[gas_nodes['id'] == gnode, 'injection']
        if inj_series.empty:
            raise ValueError(f"Gas node id={gnode} not found in GasSystem.nodes")
        G = float(inj_series.iloc[0]) * B       # MW


    # 写回 inputs
    EH['inputs'].loc[EH['inputs']['type'] == 'electricity', 'flow'] = E
    EH['inputs'].loc[EH['inputs']['type'] == 'gas',         'flow'] = G

    # -------- 输入 PCI（t/MWh）--------
    PCI_e = float(TS['PowerSystem']['NCI'][pnode - 1])   # 节点电碳强度
    PCI_g = 0.20                                         # 天然气碳强度
    EH['inputs'].loc[EH['inputs']['type'] == 'electricity', 'PCI'] = PCI_e
    EH['inputs'].loc[EH['inputs']['type'] == 'gas',         'PCI'] = PCI_g

    # -------- 设备效率与比例（支持从 Excel eh_meta 读取 + 供热优先模式）--------
    params = EH.get("params", {}) or {}
    mode = str(params.get("mode", "")).strip().lower()

    # 默认（保持你原来的）
    etaCHPW = float(params.get("etaCHPW", 0.30))
    etaCHPQ = float(params.get("etaCHPQ", 0.45))
    etaAB   = float(params.get("etaAB",   0.75))
    etaCERG = float(params.get("etaCERG", 3.0))
    etaWARG = float(params.get("etaWARG", 0.70))
    X = float(params.get("X", 0.214286))
    Y = float(params.get("Y", 0.206608))
    Z = float(params.get("Z", 1/6))
    W = float(params.get("W", 2/3))
    U = float(params.get("U", 0.739633))

    # 供热优先：减少“去制冷”的分流，把气尽量用于供热
    # 这一套在你 h=23 的 E=3.79, G=12.42 条件下，热输出会从 ~2.30 MW 提升到 ~10.09 MW
    if mode in ("heat_priority", "heat-first", "heat_first"):
        etaCHPW = float(params.get("etaCHPW", 0.33))
        etaCHPQ = float(params.get("etaCHPQ", 0.55))
        etaAB   = float(params.get("etaAB",   0.90))
        X       = float(params.get("X",       0.08))
        Y       = float(params.get("Y",       0.25))
        Z       = float(params.get("Z",       0.05))
        W       = float(params.get("W",       0.0))
        U       = float(params.get("U",       0.0))


    # -------- 能量耦合矩阵 C（严格按 MATLAB）--------
    # 输出顺序： [elec_o1, elec_o2, heat_o1, heat_o2, cool_o1, cool_o2]
    # 输入顺序： [E, G]
    C = np.array([
        [1 - X,         0],
        [0,        (1 - Z) * Y * etaCHPW],
        [0,        (1 - W) * Y * etaCHPQ],
        [0,        (1 - U) * (1 - Y) * etaAB],
        [X * etaCERG,   Z * Y * etaCERG * etaCHPW],
        [0,        etaWARG * (W * Y * etaCHPQ + U * (1 - Y) * etaAB)]
    ], dtype=float)

    inputs_flow = np.array([E, G], dtype=float)
    flows = C @ inputs_flow   # MW

    # -------- 碳耦合矩阵 D（严格按 MATLAB）--------
    # 先准备常数项
    denom = (etaCHPW ** 2 + etaCHPQ ** 2)
    # 需要用到 flow(5) 与 flow(6)
    f5 = flows[4]
    f6 = flows[5]

    # 注意：第5、6行含有除以 outputs.flow 的项，做安全除法
    D = np.array([
        [1.0, 0.0],
        [0.0, etaCHPW / denom],
        [0.0, etaCHPQ / denom],
        [0.0, 1.0 / etaAB],
        [
            _safe_div(X * E, f5),
            _safe_div(Z * Y * etaCHPW * G, f5) * (etaCHPW / denom)
        ],
        [
            0.0,
            _safe_div(etaCHPQ / denom * W * Y * etaCHPQ * G + U * (1 - Y) * G, f6)
        ]
    ], dtype=float)

    inputs_PCI = np.array([PCI_e, PCI_g], dtype=float)
    PCIs = D @ inputs_PCI                # t/MWh
    CEFRs = PCIs * flows                 # t/h

    # -------- 写回 EH.outputs（两位小数展示副本也一并写回）--------
    out = EH['outputs'].copy()
    out['flow'] = flows
    out['PCI']  = PCIs
    out['CEFR'] = CEFRs
    EH['outputs'] = out

    # 展示副本（不改变内部精度）
    EH['outputs_disp'] = out.copy()
    EH['outputs_disp']['flow'] = np.round(out['flow'].to_numpy(), 2)
    EH['outputs_disp']['PCI']  = np.round(out['PCI'].to_numpy(), 2)
    EH['outputs_disp']['CEFR'] = np.round(out['CEFR'].to_numpy(), 2)


    # -------- 控制台打印，两位小数 --------
    def _fmt2(a):
        return np.array2string(np.asarray(a), formatter={'float_kind': lambda x: f'{x:.2f}'},
                               max_line_width=200, threshold=1e6)

    print('EH 输入 flow (MW) [E, G]:', _fmt2(inputs_flow))
    print('EH 输出 flow (MW) [e1, e2, q1, q2, c1, c2]:\n', _fmt2(flows))
    print('EH 输出 PCI (tCO2/MWh):\n', _fmt2(PCIs))
    print('EH 输出 CEFR (tCO2/h):\n', _fmt2(CEFRs))

    TS['EnergyHubs'] = EH
    return TS
