import numpy as np

def run_gas_flow(TS):
    """
    气网潮流 + CEF（与给定 MATLAB 逻辑一致 or 更优）
    约定：
      - 所有流率单位：km^3/h
      - B = 10.45 MWh/(km^3), CI = 0.20 tCO2/MWh
      - nodes.injection：正=从节点流出（负荷/压缩机消耗），负=流入（气源注入）
      - Ag(i,j) = 1 表示管段 j 流入节点 i；Ag(i,j) = -1 表示管段 j 从节点 i 流出
      - Ag @ m_dot ≈ injection
    输出：
      - nodes['injection'], pipes['flowrate'], nodes/pipes/loads/compressors 的 CEFR
      - 统一两位小数的 *_disp 字段
      - TS['GasSystem']['Ag']
    """

    # ---- 读取并复制（不污染原 DF）----
    nodes        = TS['GasSystem']['nodes'].copy()
    pipes        = TS['GasSystem']['pipelines'].copy()
    compressors  = TS['GasSystem']['compressors'].copy()
    sources      = TS['GasSystem']['sources'].copy()
    loads        = TS['GasSystem']['loads'].copy()

    # ---- 基本参数 ----
    B  = 10.45  # MWh/(km^3)
    CI = 0.20   # tCO2/MWh

    # ---- 节点/管段数量，注意用行数而不是 index 值 ----
    N_node = len(nodes)
    N_pipe = len(pipes)

    # ---- 统一使用“位置索引 iloc”保证与矩阵列对齐（避免 index 不连续导致错位）----
    # 构造注入向量：正出负入
    inject = np.zeros(N_node, dtype=float)

    # 气源：流入节点（负值）
    for r in sources.itertuples(index=False):
        node_idx = int(getattr(r, 'node')) - 1   # 1-based -> 0-based
        out_val  = float(getattr(r, 'output'))
        inject[node_idx] -= out_val

    # 负荷：从节点流出（正值）
    for r in loads.itertuples(index=False):
        node_idx = int(getattr(r, 'node')) - 1
        dem_val  = float(getattr(r, 'demand'))
        inject[node_idx] += dem_val

    # 压缩机自耗气：从所在节点流出（正值）
    # 若多台压缩机位于同一节点，累加
    for r in compressors.itertuples(index=False):
        node_idx = int(getattr(r, 'location')) - 1
        cons_val = float(getattr(r, 'gasconsumption'))
        inject[node_idx] += cons_val

    nodes['injection'] = inject

    # ---- 关联系数矩阵 Ag（N_node × N_pipe）----
    Ag = np.zeros((N_node, N_pipe), dtype=float)
    # 严格用 iloc 的列序构造，以保证与 m_dot 列对应
    for j in range(N_pipe):
        f = int(pipes.iloc[j]['from']) - 1   # 1-based -> 0-based
        t = int(pipes.iloc[j]['to'])   - 1
        Ag[f, j] = -1.0   # 从 f 流出
        Ag[t, j] =  1.0   # 流入 t

    # ---- 管段流率 m_dot：Ag * m_dot = inject ----
    # MATLAB 的 "\" 为最小二乘求解；这里用 lstsq 等价
    m_dot, residuals, rank, s = np.linalg.lstsq(Ag, inject, rcond=None)
    pipes['flowrate'] = m_dot

    # ---- CEF 计算（与 MATLAB 一致）----
    CEFR_nodes = B * CI * inject           # tCO2/h
    CEFR_pipes = B * CI * m_dot            # tCO2/h
    CEFR_loads = B * CI * loads['demand'].to_numpy(dtype=float)  # tCO2/h

    nodes['CEFR'] = CEFR_nodes
    pipes['CEFR'] = CEFR_pipes

    # 压缩机碳流率（“更优”口径）：直接按自耗气量计算
    # 说明：你给的 MATLAB 代码里：CEFR_compressor = nodes.CEFR(compressors.location(1)) - pipes.CEFR(compressors.location(1))
    # 该写法把“管段 CEFR”按“节点索引”取值，存在口径/下标不一致问题。
    # 这里给出两种：优先推荐“直接按自耗”法；若你坚持复刻，可解开下方 MATLAB 风格的近似。
    CEFR_compressors_direct = B * CI * compressors['gasconsumption'].to_numpy(dtype=float)  # tCO2/h（推荐）
    # —— 可选：粗复刻 MATLAB 近似（不建议）——
    # CEFR_compressors_matlab_like = []
    # for r in compressors.itertuples(index=False):
    #     node_idx = int(getattr(r, 'location')) - 1
    #     # 如果硬要用“节点 CEFR - 同号管段 CEFR”，那这里只能勉强取“与该节点相连的任一入段或出段”做近似
    #     # 为避免强行错位，这里就不实现该不严谨写法了。

    # ---- 自检：平衡与残差（可留作日志）----
    # 1) 方程残差 ||Ag*m_dot - inject||
    res_vec = Ag @ m_dot - inject
    res_norm = float(np.linalg.norm(res_vec, ord=2))
    # 2) 简单守恒：Σ(injection) ≈ 0（孤岛/泄漏等情况下可能非零）
    sum_inject = float(inject.sum())

    # ---- 写回 TS ----
    TS['GasSystem']['nodes'] = nodes
    TS['GasSystem']['pipelines'] = pipes
    TS['GasSystem']['Ag'] = Ag
    TS['GasSystem']['CEFR_loads'] = CEFR_loads
    TS['GasSystem']['CEFR_compressors'] = CEFR_compressors_direct
    TS['GasSystem']['residual_norm'] = res_norm
    TS['GasSystem']['sum_injection'] = sum_inject

    # ---- 展示用：两位小数副本（不影响内部精度）----
    TS['GasSystem']['nodes_CEFR_disp']        = np.round(CEFR_nodes, 2)
    TS['GasSystem']['pipes_CEFR_disp']        = np.round(CEFR_pipes, 2)
    TS['GasSystem']['loads_CEFR_disp']        = np.round(CEFR_loads, 2)
    TS['GasSystem']['compressors_CEFR_disp']  = np.round(CEFR_compressors_direct, 2)
    TS['GasSystem']['injection_disp']         = np.round(inject, 2)
    TS['GasSystem']['flowrate_disp']          = np.round(m_dot, 2)
    TS['GasSystem']['Ag_disp']                = np.round(Ag, 2)
    TS['GasSystem']['residual_norm_disp']     = round(res_norm, 6)
    TS['GasSystem']['sum_injection_disp']     = round(sum_inject, 6)

    # ---- 控制台打印（两位小数）----
    def _fmt2(arr):  # 仅用于打印
        return np.array2string(np.asarray(arr),
                               formatter={'float_kind': lambda x: f"{x:.2f}"},
                               max_line_width=200, threshold=1e6)
    print('气网节点流率向量 mq_dot (km^3/h):\n', _fmt2(inject))
    print('气网关联矩阵 Ag:\n', _fmt2(Ag))
    print('气网管段流率向量 m_dot (km^3/h):\n', _fmt2(m_dot))
    print('气网节点碳流率 CEFR_nodes (tCO2/h):\n', _fmt2(CEFR_nodes))
    print('气网管段碳流率 CEFR_pipes (tCO2/h):\n', _fmt2(CEFR_pipes))
    print('气网负荷碳流率 CEFR_loads (tCO2/h):\n', _fmt2(CEFR_loads))
    print('气网压缩机碳流率 CEFR_compressors (tCO2/h, direct by consumption):\n',
          _fmt2(CEFR_compressors_direct))
    print(f'[诊断] ||Ag*m_dot - injection||_2 = {res_norm:.6e} ; sum(injection) = {sum_inject:.6e}')

    return TS
