# -*- coding: utf-8 -*-
import numpy as np

def run_heat_flow(TS):
    """
    Heat network: steady hydraulic + thermal + CEF
    与给定 MATLAB 代码等价：
      - 定压力：固定按 MPa→Pa (×1e6)
      - 不做总注入守恒修正（沿用 nodes.injection 原值）
      - 热力段权重与指数均使用“带符号的 m”
    """

    # ===== 读取 =====
    HS = TS['HeatSystem']
    nodes = HS['nodes'].copy()
    # ---- ensure float dtype (avoid int64 -> float assignment issues) ----
    if 'temperature' in nodes.columns:
        nodes['temperature'] = nodes['temperature'].astype(float)
    else:
        nodes['temperature'] = 80.0
    pipes = HS['pipelines'].copy()
    sources = HS['sources'].copy()
    valve_params = HS.get('valves', None)
    pump_params  = HS.get('pumps',  None)

   # ===== 热源 GCI（由 scenario_runner 覆盖写入）=====
    GCI_heat = float(sources.loc[0, 'GCI']) if (len(sources) > 0 and 'GCI' in sources.columns) else 0.0


    # ===== 常数 =====
    rho = 1000.0  # kg/m^3
    c   = 4200.0  # J/(kg·°C)

    # ===== id→位置映射 =====
    node_ids = nodes['id'].astype(int).to_numpy()
    node_pos = {nid: idx for idx, nid in enumerate(node_ids)}
    nnodes = len(nodes)

    npipe = len(pipes)
    L   = pipes['length'          ].to_numpy(dtype=float)  # m
    D   = pipes['diameter'        ].to_numpy(dtype=float)  # m
    lam = pipes['friction_factor' ].to_numpy(dtype=float)  # -
    miu = pipes['dissipation'     ].to_numpy(dtype=float)  # 热损系数
    As  = np.pi * (D**2) / 4.0

    mb0 = pipes['massflow'].to_numpy(dtype=float) if 'massflow' in pipes else None
    mb  = np.where(np.isfinite(mb0), mb0, 250.0) if mb0 is not None else np.full(npipe, 250.0)  # kg/s 初值

    # ===== Ah（节点-支路关联）：from→-1, to→+1（与 MATLAB 等价）=====
    Ah = np.zeros((nnodes, npipe), dtype=float)
    f_idx = np.empty(npipe, dtype=int)
    t_idx = np.empty(npipe, dtype=int)
    for j in range(npipe):
        f_id = int(pipes.iloc[j]['from'])
        t_id = int(pipes.iloc[j]['to'])
        fi = node_pos[f_id]
        ti = node_pos[t_id]
        f_idx[j], t_idx[j] = fi, ti
        Ah[fi, j] = -1.0
        Ah[ti, j] =  1.0

    # ===== 节点类型 =====
    ntype = nodes['type'].astype(str).str.strip()
    fix_p = np.where(ntype == "定压力")[0]
    fix_G = np.where(ntype == "定注入")[0]
    if fix_p.size == 0 or fix_G.size == 0:
        raise ValueError("需要至少一个“定压力”和一个“定注入”节点")

    # ===== 注入向量（kg/s）=====
    if 'injection' not in nodes or not np.isfinite(nodes['injection']).all():
        nodes['injection'] = 0.0
    inj = nodes['injection'].to_numpy(dtype=float)

    # ===== 稳态水力迭代（完全按 MATLAB 公式与 0.2/0.8 松弛）=====
    iter_max = 100
    tol = 1e-3
    converged = False

    # 设备位置与参数（按出现顺序对齐）
    pump_locs  = np.where(pipes.get('pump',  0).to_numpy(dtype=float) > 0)[0]
    valve_locs = np.where(pipes.get('valve', 0).to_numpy(dtype=float) > 0)[0]

    def _param(df, name, size):
        if df is None or name not in df:
            return np.zeros(size, dtype=float)
        v = df[name].to_numpy(dtype=float)
        return v[:size] if v.size >= size else np.pad(v, (0, size - v.size), constant_values=0.0)

    kp1 = _param(pump_params,  'kp1', pump_locs.size)
    kp2 = _param(pump_params,  'kp2', pump_locs.size)
    kp3 = _param(pump_params,  'kp3', pump_locs.size)
    wpm = _param(pump_params,  'w',   pump_locs.size)
    kv  = _param(valve_params, 'kv',  valve_locs.size)

    for iter in range(1, iter_max + 1):
        # 支路参数
        R = lam * mb / rho / (As**2) / D * L
        E = -lam * (mb**2) / 2.0 / rho / (As**2) / D * L

        # 极小防护（仅避免 1./R 爆掉；不改变算法）
        R = np.where(np.abs(R) < 1e-12, 1e-12, R)

        # 泵修正
        for k, i in enumerate(pump_locs):
            R[i] -= (2.0 * kp1[k] * mb[i] + kp2[k] * wpm[k])
            E[i] += (kp1[k] * mb[i]**2 - kp3[k] * wpm[k]**2)

        # 阀修正
        for k, i in enumerate(valve_locs):
            R[i] += 2.0 * kv[k] * mb[i]
            E[i] += kv[k] * mb[i]**2

        # 导纳与分块
        yb = np.diag(1.0 / R)
        Y  = Ah @ yb @ Ah.T
        Ygg = Y[np.ix_(fix_G, fix_G)]
        Ygp = Y[np.ix_(fix_G, fix_p)]

        # 定压力：固定按 MPa → Pa（与 MATLAB 一致）
        pp_raw = nodes.loc[fix_p, 'pressure'].to_numpy(dtype=float)
        pp = pp_raw * 1e6  # MPa → Pa

        # 注入项
        G  = inj - (Ah @ yb @ E)
        Gg = G[fix_G]

        # 解压：pg = Ygg \ (Gg - Ygp * pp)
        try:
            pg = np.linalg.solve(Ygg, (Gg - Ygp @ pp))
        except np.linalg.LinAlgError:
            pg = np.linalg.lstsq(Ygg, (Gg - Ygp @ pp), rcond=None)[0]

        pn = np.zeros(nnodes, dtype=float)
        pn[fix_p] = pp
        pn[fix_G] = pg

        # 更新流量：Gb = yb * ((-Ah') * pn - E)
        Gb = (yb @ ((-Ah.T) @ pn - E)).ravel()

        # 收敛与松弛（0.2/0.8）
        err = np.linalg.norm(Gb - mb)
        new_mb = 0.2 * mb + 0.8 * Gb
        if err < tol:
            converged = True
            mb = Gb
            break
        mb = new_mb

    if not converged:
        raise RuntimeError(f"水力计算未在 {iter_max} 次迭代内收敛")
    print(f"水力计算在 {iter} 次迭代后收敛，最终误差：{err:.3f}")

    # ===== 稳态热力计算（数值稳定版：热损&混合用 |m|；小流量保护）=====
    m = mb.copy()  # kg/s (signed)
    m_eps = 1e-6
    m_abs = np.abs(m)
    m_abs_safe = np.where(m_abs < m_eps, m_eps, m_abs)  # ONLY for thermal calculations
    m_eff = np.where(m_abs < m_eps, 0.0, m)             # used in energy/CEFR to avoid tiny*huge

    A  = -Ah
    Af = np.zeros_like(A);  Af[A > 0] = 1.0
    At = np.zeros_like(A);  At[A < 0] = 1.0

    # At_：按“|m|”加权并行归一（物理混合权重必须为正）
    At_ = At.copy()
    for i in range(npipe):
        At_[:, i] = At_[:, i] * m_abs_safe[i]
    row_sum = At_.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    At_ = At_ / row_sum

    Tin   = nodes['temperature'].to_numpy(dtype=float)
    Etemp = pipes['deltaT'].to_numpy(dtype=float) if 'deltaT' in pipes else np.zeros(npipe)

    # 热损因子 K 必须在 (0,1]，指数使用 |m|
    Rh   = miu / (c**2 * (m_abs_safe**2))
    expo = -c * m_abs_safe * Rh * L
    expo = np.clip(expo, -50.0, 0.0)
    Kdiag = np.exp(expo)
    Kdiag_safe = np.where(Kdiag < 1e-12, 1e-12, Kdiag)

    I = np.eye(npipe)
    Aheat = I - np.diag(Kdiag) @ Af.T @ At_
    rhs   = np.diag(Kdiag) @ Af.T @ Tin + Etemp

    try:
        Tt = np.linalg.solve(Aheat, rhs)
    except np.linalg.LinAlgError:
        Tt = np.linalg.lstsq(Aheat, rhs, rcond=None)[0]

    Tf = (Tt - Etemp) / Kdiag_safe

    # 对近零流量的管段：视为无输运，冻结温度以避免数值噪声
    zero_mask = (m_abs < m_eps)
    if np.any(zero_mask):
        Tin_from = (Af.T @ Tin)
        Tf[zero_mask] = Tin_from[zero_mask]
        Tt[zero_mask] = Tin_from[zero_mask]

# ===== 回写 =====
    pipes['massflow'] = mb
    pipes['T_in']  = Tf
    pipes['T_out'] = Tt
    pipes['deltaT']= Tf - Tt

    # 覆盖式回写节点温度（与 MATLAB 一致）
    for j in range(npipe):
        nodes.at[t_idx[j], 'temperature'] = Tt[j]
    for j in range(npipe):
        nodes.at[f_idx[j], 'temperature'] = Tf[j]

    # ===== CEF =====
    Q_pipes_W  = c * m_eff * Tt
    Q_pipes_MW = Q_pipes_W / 1e6
    CEFR_pipes = Q_pipes_MW * GCI_heat   # tCO2/h

    Q_out_W  = c * m_eff * (Tf - Tt)
    Q_out_MW = Q_out_W / 1e6
    CEFR_pipes_output = GCI_heat * Q_out_MW
    Q_total_MW = float(Q_out_MW.sum())
    CEFR_total = float(CEFR_pipes_output.sum())

    # ===== 写回 =====
    HS['nodes'] = nodes
    HS['pipelines'] = pipes
    HS['sources'] = sources
    HS['Q_total_MW'] = Q_total_MW
    HS['CEFR_total'] = CEFR_total
    HS['CEFR_pipes'] = CEFR_pipes
    HS['CEFR_pipes_output'] = CEFR_pipes_output
    TS['HeatSystem'] = HS

    # ===== 打印（两位小数）=====
    def _fmt2(a):
        return np.array2string(np.asarray(a),
                               formatter={'float_kind': lambda x: f'{x:.2f}'},
                               max_line_width=200, threshold=1e6)

    print("热网节点温度(°C):\n", _fmt2(nodes['temperature'].to_numpy(dtype=float)))
    print("热网管段质量流率(kg/s):\n", _fmt2(mb))
    print("热网各管段温度变化(°C) Tf - Tt:\n", _fmt2((Tf - Tt)))
    print("热网总输出能量(MW): ", f"{Q_total_MW:.2f}")
    print("热网总输出碳流率(tCO2/h): ", f"{CEFR_total:.2f}")

    # ===== 展示副本 =====
    TS['HeatSystem']['nodes_disp'] = nodes.copy()
    TS['HeatSystem']['nodes_disp']['temperature'] = np.round(nodes['temperature'].to_numpy(), 2)

    pipes_disp = pipes.copy()
    for col in ['massflow', 'T_in', 'T_out', 'deltaT']:
        if col in pipes_disp:
            pipes_disp[col] = np.round(pipes_disp[col].to_numpy(dtype=float), 2)
    TS['HeatSystem']['pipelines_disp'] = pipes_disp
    TS['HeatSystem']['Q_total_MW_disp'] = round(Q_total_MW, 2)
    TS['HeatSystem']['CEFR_total_disp'] = round(CEFR_total, 2)

    return TS
