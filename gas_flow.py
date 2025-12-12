import numpy as np

def run_gas_flow(TS):
    """
    气网潮流 + CEF
    约定：
      - 流率：km^3/h
      - nodes.injection：正=消耗(负荷/压缩机)，负=注入(气源)
      - 若 gas_sources 含 is_slack=1：该源 output 由本函数按气量平衡自动求解
    """

    nodes        = TS['GasSystem']['nodes'].copy()
    pipes        = TS['GasSystem']['pipelines'].copy()
    compressors  = TS['GasSystem']['compressors'].copy()
    sources      = TS['GasSystem']['sources'].copy()
    loads        = TS['GasSystem']['loads'].copy()

    B  = 10.45  # MWh/(km^3)
    CI = 0.20   # tCO2/MWh

    # ---------- 节点 id -> 位置索引（避免 “-1” 假设 node 必从 1 连续编号） ----------
    node_ids = nodes['id'].astype(int).to_numpy()
    pos = {int(nid): i for i, nid in enumerate(node_ids)}
    N_node = len(node_ids)
    N_pipe = len(pipes)

    def _must_pos(nid: int, where: str):
        if int(nid) not in pos:
            raise KeyError(f"[GasFlow] {where}: node id={nid} 不在 gas_nodes.id 中")
        return pos[int(nid)]

    # ---------- 1) 统计总需求（正向消耗） ----------
    demand_load = float(loads['demand'].fillna(0.0).astype(float).sum()) if len(loads) else 0.0
    demand_comp = float(compressors['gasconsumption'].fillna(0.0).astype(float).sum()) if len(compressors) else 0.0
    total_demand = demand_load + demand_comp  # km3/h

    # ---------- 2) slack 源识别 ----------
    if 'is_slack' in sources.columns:
        slack_mask = sources['is_slack'].fillna(0).astype(int).to_numpy() == 1
    else:
        # 兼容旧表：默认第一台源为 slack
        slack_mask = np.zeros(len(sources), dtype=bool)
        if len(sources) > 0:
            slack_mask[0] = True
            sources['is_slack'] = 0
            sources.loc[sources.index[0], 'is_slack'] = 1

    if slack_mask.sum() != 1:
        raise ValueError(f"[GasFlow] 目前实现为“单 slack 源”，请确保 gas_sources.is_slack 恰好有且仅有一个 1；当前={int(slack_mask.sum())}")

    # 非 slack 源视为“固定源”（output 已在 TS 中，可能被横表覆盖过）
    fixed_mask = ~slack_mask
    fixed_out = sources.loc[fixed_mask, 'output'].fillna(0.0).astype(float).to_numpy()
    total_fixed_supply = float(fixed_out.sum())  # km3/h

    # slack 需要承担的供气量
    slack_need = total_demand - total_fixed_supply
    if slack_need < -1e-8:
        raise ValueError(
            f"[GasFlow] 固定气源供给({total_fixed_supply:.6f}) > 总需求({total_demand:.6f})，"
            "会导致 slack_need 为负。请调小固定源或允许外送/弃气（当前未建模）。"
        )
    slack_idx = int(np.where(slack_mask)[0][0])
    sources.loc[sources.index[slack_idx], 'output'] = float(slack_need)

    # ---------- 3) 组装 injection（正消耗、负注入） ----------
    inject = np.zeros(N_node, dtype=float)

    # 气源：负注入
    for r in sources.itertuples(index=False):
        nid = int(getattr(r, 'node'))
        out_val = float(getattr(r, 'output'))
        inject[_must_pos(nid, "gas_sources")] -= out_val

    # 负荷：正消耗
    for r in loads.itertuples(index=False):
        nid = int(getattr(r, 'node'))
        dem_val = float(getattr(r, 'demand'))
        inject[_must_pos(nid, "gas_loads")] += dem_val

    # 压缩机：正消耗（location 是节点 id）
    for r in compressors.itertuples(index=False):
        nid = int(getattr(r, 'location'))
        cons_val = float(getattr(r, 'gasconsumption'))
        inject[_must_pos(nid, "gas_compressors")] += cons_val

    nodes['injection'] = inject
    TS['GasSystem']['sources'] = sources  # 关键：把 slack 计算后的 output 写回

    # ---------- 4) 构造 Ag（N_node × N_pipe） ----------
    Ag = np.zeros((N_node, N_pipe), dtype=float)
    for j in range(N_pipe):
        f_id = int(pipes.iloc[j]['from'])
        t_id = int(pipes.iloc[j]['to'])
        f = _must_pos(f_id, "gas_pipelines.from")
        t = _must_pos(t_id, "gas_pipelines.to")
        Ag[f, j] = -1.0
        Ag[t, j] =  1.0

    # ---------- 5) 求管段流率 ----------
    # 当 sum(injection)=0 时，理论可解；这里仍用最小二乘与原逻辑一致
    m_dot, *_ = np.linalg.lstsq(Ag, inject, rcond=None)
    pipes['flowrate'] = m_dot

    # ---------- 6) CEFR ----------
    CEFR_nodes = B * CI * inject
    CEFR_pipes = B * CI * m_dot
    CEFR_loads = B * CI * loads['demand'].to_numpy(dtype=float)
    CEFR_compressors_direct = B * CI * compressors['gasconsumption'].to_numpy(dtype=float)

    nodes['CEFR'] = CEFR_nodes
    pipes['CEFR'] = CEFR_pipes

    res_vec = Ag @ m_dot - inject
    res_norm = float(np.linalg.norm(res_vec, ord=2))
    sum_inject = float(inject.sum())

    TS['GasSystem']['nodes'] = nodes
    TS['GasSystem']['pipelines'] = pipes
    TS['GasSystem']['Ag'] = Ag
    TS['GasSystem']['CEFR_loads'] = CEFR_loads
    TS['GasSystem']['CEFR_compressors'] = CEFR_compressors_direct
    TS['GasSystem']['residual_norm'] = res_norm
    TS['GasSystem']['sum_injection'] = sum_inject

    # display
    TS['GasSystem']['nodes_CEFR_disp']        = np.round(CEFR_nodes, 2)
    TS['GasSystem']['pipes_CEFR_disp']        = np.round(CEFR_pipes, 2)
    TS['GasSystem']['loads_CEFR_disp']        = np.round(CEFR_loads, 2)
    TS['GasSystem']['compressors_CEFR_disp']  = np.round(CEFR_compressors_direct, 2)
    TS['GasSystem']['injection_disp']         = np.round(inject, 2)
    TS['GasSystem']['flowrate_disp']          = np.round(m_dot, 2)
    TS['GasSystem']['Ag_disp']                = np.round(Ag, 2)
    TS['GasSystem']['residual_norm_disp']     = round(res_norm, 6)
    TS['GasSystem']['sum_injection_disp']     = round(sum_inject, 6)

    print(f"[GasFlow] total_demand={total_demand:.6f} km3/h | fixed_supply={total_fixed_supply:.6f} | slack={slack_need:.6f}")
    print(f"[GasFlow] ||Ag*m_dot - injection||_2 = {res_norm:.6e} ; sum(injection) = {sum_inject:.6e}")

    return TS
