# power_flow.py — 稳健版（不改库）：
# 1) 使用 DC 初值 + 连续同伦（60%→80%→90%→100%）以增强收敛；
# 2) 全程关闭库内 Q 限分支（ENFORCE_Q_LIMS=0），绕开浮点索引/括号优先级问题；
# 3) CEF 计算使用 p.u. 功率矩阵（PB/PG/PL/RBloss 全部除以 baseMVA），
#    与 MATLAB 结果数值一一对齐；同时也回写物理量 t/h 版本供后续使用。

import numpy as np
from pypower.api import runpf, ppoption

def _build_and_sanitize(TS):
    ppc = {
        'version': '2',
        'baseMVA': float(TS['PowerSystem']['baseMVA']),
        'bus': TS['PowerSystem']['bus'].astype(float).copy(),
        'gen': TS['PowerSystem']['gen'].astype(float).copy(),
        'branch': TS['PowerSystem']['branch'].astype(float).copy(),
        'gencost': None
    }
    bus, gen, branch = ppc['bus'], ppc['gen'], ppc['branch']

    # 规范索引列（MATPOWER 1-based）
    bus[:, 0] = np.arange(1, bus.shape[0] + 1)
    gen[:, 0] = np.rint(gen[:, 0]).astype(int)
    branch[:, 0:2] = np.rint(branch[:, 0:2]).astype(int)
    bus[:, 1] = np.rint(bus[:, 1]).astype(int)  # BUS_TYPE: 1 PQ, 2 PV, 3 REF

    # gencost（占位即可，避免 runpf 缺字段）
    ng = gen.shape[0]
    gencost = np.zeros((ng, 7), float)
    gencost[:, 0] = 2  # polynomial
    gencost[:, 3] = 3
    gencost[:, 4] = 1e-2
    gencost[:, 5] = 1.0
    ppc['gencost'] = gencost

    # 清 NaN / 合理默认
    # bus: PD,QD,GS,BS,VM,VA
    for col in [2, 3, 4, 5, 7, 8]:
        nan = np.isnan(bus[:, col])
        if col == 7:  # VM
            bus[nan, col] = 1.0
        elif col == 8:  # VA
            bus[nan, col] = 0.0
        else:
            bus[nan, col] = 0.0
    # gen: PG,QG,VG
    for col in [1, 2, 5]:
        nan = np.isnan(gen[:, col])
        gen[nan, col] = 1.0 if col == 5 else 0.0

    # 支路默认与开启：branch: R,X,B,RATEA,STATUS
    for col in [2, 3, 4, 5, 10]:
        nan = np.isnan(branch[:, col])
        if col in (2, 3):      # R/X
            branch[nan, col] = 1e-4
        elif col == 4:         # B
            branch[nan, col] = 0.0
        elif col == 5:         # RATEA
            branch[nan, col] = 1e3
        elif col == 10:        # STATUS
            branch[nan, col] = 1.0
    branch[:, 10] = np.where(branch[:, 10] == 0, 1, branch[:, 10])

    # 确保恰好 1 个 REF
    TYPE = 1
    ref = np.where(bus[:, TYPE] == 3)[0]
    if ref.size == 0:
        rb = int(gen[0, 0]) - 1
        bus[rb, TYPE] = 3
    elif ref.size > 1:
        keep = ref[0]
        bus[:, TYPE] = np.where(np.arange(bus.shape[0]) == keep, 3,
                                np.where(bus[:, TYPE] == 3, 2, bus[:, TYPE]))

    # 给机组母线 VM 赋 VG；其它母线 VM=1.0，VA=0
    VM, VA, VG = 7, 8, 5
    bus[:, VM] = np.where(bus[:, VM] > 0, bus[:, VM], 1.0)
    bus[:, VA] = 0.0
    for i in range(ng):
        b = int(gen[i, 0]) - 1
        vset = gen[i, VG] if gen[i, VG] > 0 else 1.0
        bus[b, VM] = vset

    # （可选）把 PV/REF 母线初值略抬，利于收敛
    pv_or_ref = (bus[:, TYPE] != 1)
    bus[pv_or_ref, VM] = np.maximum(bus[pv_or_ref, VM], 1.02)

    # PG 初值裁剪
    PG, PMAX, PMIN = 1, 8, 9
    gen[:, PG] = np.minimum(gen[:, PMAX], np.maximum(gen[:, PG], gen[:, PMIN]))

    # 粗平衡：若总 PG 初值过小，抬 slack 机组
    PD = 2
    total_load = bus[:, PD].sum()
    total_pg0 = gen[:, PG].sum()
    if total_pg0 < 0.95 * total_load:
        ref_bus = np.where(bus[:, TYPE] == 3)[0][0] + 1
        idx = np.where(gen[:, 0] == ref_bus)[0]
        i0 = int(idx[0] if idx.size else 0)
        need = 1.05 * total_load - total_pg0
        gen[i0, PG] = min(gen[i0, PMAX], gen[i0, PG] + need)

    ppc['bus'], ppc['gen'], ppc['branch'] = bus, gen, branch
    # --- 若输入为 MW，则统一转换成 p.u. ---
    cols_power_bus = [2, 3]       # PD, QD
    cols_power_gen = [1, 2, 3, 4, 8, 9]  # PG, QG, QMAX, QMIN, PMAX, PMIN
    bus[:, cols_power_bus] /= ppc['baseMVA']
    gen[:, cols_power_gen] /= ppc['baseMVA']
    return ppc

def _ppopt(ac_method=1, enforce_q=0, init_from_dc=True, verbose=2):
    # ac_method: 1=NR, 2=FDBX
    # enforce_q=0 避开库内有问题的 Q 限制分支
    return ppoption(
        VERBOSE=verbose, OUT_ALL=1,
        ENFORCE_Q_LIMS=int(bool(enforce_q)),
        PF_ALG=ac_method,
        PF_TOL=1e-8, PF_MAX_IT=100,
        INIT=2 if init_from_dc else 1  # 2=用 DC 作为初值（更稳）
    )

def _run_pf(ppc, ac_method=1, init_from_dc=True, verbose=2):
    return runpf(ppc, _ppopt(ac_method=ac_method, enforce_q=0, init_from_dc=init_from_dc, verbose=verbose))

def _continuation_pf(ppc, verbose=2):
    """
    简单双同伦：把负荷从 60% -> 80% -> 90% -> 100% 逐步加上，
    每一步用上一步结果作为初值。
    """
    PD, QD, VM, VA = 2, 3, 7, 8
    base_bus = ppc['bus'].copy()

    for lam in [0.60, 0.80, 0.90, 1.00]:
        ppc['bus'][:, PD] = base_bus[:, PD] * lam
        ppc['bus'][:, QD] = base_bus[:, QD] * lam

        # 先 FDBX（更稳），失败再 NR；INIT=2 用 DC 初值
        res, ok = _run_pf(ppc, ac_method=2, init_from_dc=True, verbose=verbose)
        if not ok:
            res, ok = _run_pf(ppc, ac_method=1, init_from_dc=True, verbose=verbose)
        if not ok:
            return res, ok

        # 用本步解作下一步初值
        ppc['bus'][:, VM] = res['bus'][:, VM]
        ppc['bus'][:, VA] = res['bus'][:, VA]

    # 恢复 100% 负荷后，再用 NR 精化（INIT=1 用上一步解作初值）
    return runpf(ppc, _ppopt(ac_method=1, enforce_q=0, init_from_dc=False, verbose=verbose))

def run_power_flow(TS, extra_loads=None, extra_gens=None, extra_ci=None, verbose=2):
    # 1) 构造并体检
    ppc = _build_and_sanitize(TS)

    # === [STORAGE HOOK] apply extra injections before solving (in p.u. domain) ===
    if extra_loads or extra_gens:
        baseMVA = ppc['baseMVA']

        # 充电 -> 负荷（MW -> p.u. 叠加到 PD 列，索引 2）
        if extra_loads:
            for (bus_id, P_MW) in extra_loads:
                rows = np.where(ppc['bus'][:, 0].astype(int) == int(bus_id))[0]
                if rows.size > 0:
                    r = int(rows[0])
                    ppc['bus'][r, 2] += float(P_MW) / baseMVA  # PD

        # 放电 -> 新增机组（MW -> p.u.），并为每个新增机组追加一行 gencost
        if extra_gens:
            new_gencost_rows = []  # 收集待追加的 gencost 行
            for item in extra_gens:
                if len(item) == 3:
                    bus_id, P_MW, ci_val = item
                else:
                    bus_id, P_MW = item
                    ci_val = 0.0

                new = np.zeros((1, ppc['gen'].shape[1]), dtype=float)
                new[0, 0] = int(bus_id)                 # GEN_BUS
                new[0, 1] = float(P_MW) / baseMVA       # PG (p.u.)
                new[0, 2] = 0.0                         # QG
                new[0, 3] = 1e3                         # QMAX
                new[0, 4] = -1e3                        # QMIN
                new[0, 5] = 1.0                         # VG
                new[0, 6] = baseMVA                     # MBASE
                new[0, 7] = 1.0                         # GEN_STATUS
                new[0, 8] = max(new[0, 1] * 1.2, 0.001) # PMAX (p.u.)
                new[0, 9] = 0.0                         # PMIN
                ppc['gen'] = np.vstack([ppc['gen'], new])

                # 记录对应的碳强度，供 CEF 拼接
                if '__extra_gci__' not in TS:
                    TS['__extra_gci__'] = np.asarray([[float(ci_val)]], dtype=float)
                else:
                    TS['__extra_gci__'] = np.vstack([TS['__extra_gci__'], [[float(ci_val)]]])

                # 追加一行默认 gencost（与 _build_and_sanitize 里保持一致的 7 列格式）
                gc_cols = ppc['gencost'].shape[1] if 'gencost' in ppc else 7
                row = np.zeros((1, gc_cols), dtype=float)
                row[0, 0] = 2      # MODEL=polynomial
                # STARTUP(1) & SHUTDOWN(2) 留 0
                row[0, 3] = 3      # NCOST
                row[0, 4] = 1e-2   # c2
                row[0, 5] = 1.0    # c1
                # c0(6) 留 0
                new_gencost_rows.append(row)

            if 'gencost' in ppc:
                ppc['gencost'] = np.vstack([ppc['gencost']] + new_gencost_rows)
            else:
                ppc['gencost'] = np.vstack(new_gencost_rows)
    # === [STORAGE HOOK END] ===


# 2) 连续同伦求解（多步加载）
    # —— 求解 ——
    res, ok = _continuation_pf(ppc, verbose=verbose)
    if not ok:
        TS['PowerSystem']['results'] = {'ok': False}
        return TS

    # 记录原机组行数（不包含储能新增机组）
    n_gen0 = TS['PowerSystem']['gen'].shape[0]

    # 回写（仅写回与 TS 尺寸一致的字段）
    TS['PowerSystem']['results'] = {
        'ok': True,
        'bus': res['bus'],
        'gen': res['gen'],
        'branch': res['branch'],
    }

    # VM/VA（母线数不变）
    try:
        VM, VA = 7, 8
        TS['PowerSystem']['bus'][:, VM] = res['bus'][:, VM]
        TS['PowerSystem']['bus'][:, VA] = res['bus'][:, VA]
    except Exception:
        pass

    # 仅更新“原有机组”的 PG（列索引 1）
    try:
        pg_res = res['gen'][:, 1]
        n_take = min(n_gen0, pg_res.shape[0])
        TS['PowerSystem']['gen'][:n_take, 1] = pg_res[:n_take]
        TS['PowerSystem']['gen_pg_all'] = pg_res.copy()  # 可选：保留含“新增机组”的全量PG
    except Exception:
        pass

    # 先计算 CEF（此时 __extra_gci__ 仍存在，EG 会正确拼接）
    TS = compute_cef_like_matlab(TS, res)

    # 再清理临时字段，避免跨时段残留
    if '__extra_gci__' in TS:
        try:
            del TS['__extra_gci__']
        except Exception:
            pass

    return TS

## Define a function to compute CEF like MATLAB version
import numpy as np
from pypower.idx_bus import VM, PD
from pypower.idx_brch import F_BUS, T_BUS, PF, PT
from pypower.idx_gen import GEN_BUS, PG as GEN_PG

def compute_cef_like_matlab(TS, results):
    """
    严格按 MATLAB 代码思路进行 CEF 计算（MW 口径）。
    需要 TS 中包含:
      - TS['PowerSystem']['GCI'] : 发电机碳强度向量 (tCO2/MWh), shape=(N_gen,)
    并将以下结果回写到 TS['PowerSystem']：
      - 'NCI' (EN, tCO2/MWh), 'BCI' (tCO2/MWh)
      - 'RB','RBloss','RL','RG' (tCO2/h)，以及中间矩阵 'PB','PL','PBloss' (MW)
    """

    # ===== 读入潮流结果（与 MATLAB 一致的口径：MW） =====
    bus    = results['bus']        # [:, VM], [:, PD]
    gen    = results['gen']        # [:, GEN_BUS], [:, PG]
    branch = results['branch']     # [:, F_BUS], [:, T_BUS], [:, PF], [:, PT]

    # 基本量
    n_node = bus.shape[0]
    n_line = branch.shape[0]
    n_gen  = gen.shape[0]

    # 节点编号（PYPOWER/Matpower 数据里是以 1 开始的物理号，转为 0 基索引）
    f_bus = branch[:, F_BUS].astype(int) - 1
    t_bus = branch[:, T_BUS].astype(int) - 1

    # 机组接入节点（0 基索引）
    g_bus = gen[:, GEN_BUS].astype(int) - 1

    # 功率（MW，注意：PYPOWER 的 PG、PF、PT 就是 MW，无需乘 baseMVA）
    pg_col  = gen[:, GEN_PG]              # 发电 MW
    pd_col  = bus[:, PD]                  # 负荷 MW（按节点聚合）
    pf_col  = branch[:, PF]               # 由 from 端注入（MW）, 正值表示从 from 端流出
    pt_col  = branch[:, PT]               # 由 to   端注入（MW）, 负值表示流入 to 端
    # ===== 1) 支路潮流分布矩阵 PB(i,j) = 从 i→j 的“受端功率”(MW) =====
    # 规则：方向 f->t 时，用 abs(PT) 写到 PB(f,t)；方向 t->f 时，用 abs(PF) 写到 PB(t,f)
    PB = np.zeros((n_node, n_node), dtype=float)
    tol = 1e-9
    for k in range(n_line):
        f = f_bus[k]; t = t_bus[k]
        pf = pf_col[k]; pt = pt_col[k]
        if pf >  tol:            # f -> t
            PB[f, t] += abs(pt)  # 受端(t端)有功，等价于 MATLAB 用 |PT|
        elif pt > tol:           # t -> f
            PB[t, f] += abs(pt)  # 受端(f端)有功，取 |PF|
        # 两端都<=tol 则忽略（极小值）
    print("PB matrix:\n", PB)

    # ===== 2) 机组注入分布矩阵 PG（N_gen × N_node, MW）=====
    PG = np.zeros((n_gen, n_node), dtype=float)
    for i in range(n_gen):
        PG[i, g_bus[i]] = pg_col[i]
    print("PG matrix:\n", PG)

    # ===== 3) 负荷分布矩阵 PL（N_load × N_node, MW）=====
    load_idx = np.where(pd_col > 0)[0]
    n_load = load_idx.size
    PL = np.zeros((n_load, n_node), dtype=float)
    for i, j in enumerate(load_idx):
        PL[i, j] = pd_col[j]
    print("PL matrix:\n", PL)

    # ===== 4) 节点有功通量对角阵 PN（MW）=====
    # ep = ones(1, N_node + N_gen)
    # PZ = [PB; PG]  (竖直拼接)
    # PN = diag(ep * PZ) -> 得到每个节点“流入量之和”，做成对角阵
    ep = np.ones((1, n_node + n_gen))
    PZ = np.vstack([PB, PG])               # (N_node+N_gen) × N_node
    PN_vec = (ep @ PZ).ravel()             # 1×N_node -> 向量
    PN = np.diag(PN_vec)                   # N_node×N_node
    print("PN matrix:\n", PN)


    # ===== 5) 发电机碳强度向量 EG（tCO2/MWh）=====
    EG = np.asarray(TS['PowerSystem']['GCI'], dtype=float).reshape(-1, 1)
    extra = TS.get('__extra_gci__', None)
    if extra is not None and len(np.asarray(extra).ravel()) > 0:
        EG = np.vstack([EG, np.asarray(extra, dtype=float).reshape(-1, 1)])

    # ……此处已构造好：
    #   n_node, n_gen
    #   g_bus : 每台机组接入节点（0基索引）
    #   pg_col: 每台机组的出力（MW）
    #   PG    : (n_gen × n_node) 的机组注入分布矩阵（第 i 行只有 g_bus[i] 那列为 pg_col[i]）

    # === [STORAGE MIX @ NODE] 分类混合：若某节点同时存在“原机组”和“储能放电”，
    #     则将二者按出力混合成该节点的“整体发电碳强度”，覆盖该节点上所有相关机组的 EG；
    #     若某节点只有储能（无原机组），保留其 EG 原值（等价于作为独立机组参与 CEF）。
    n_gen = gen.shape[0]
    n_extra = 0
    if extra is not None:
        n_extra = int(np.asarray(extra).size)
    n_gen0 = n_gen - n_extra  # 原机组行数（不含储能放电临时机组）

    EG_vec   = EG.ravel()              # shape: (n_gen,)
    pg_vec   = np.asarray(pg_col).ravel()
    g_bus_vec= np.asarray(g_bus).ravel()

    idx_all  = np.arange(n_gen)

    for i_node in range(n_node):
        idx_orig  = idx_all[(idx_all <  n_gen0) & (g_bus_vec == i_node) & (pg_vec > 0)]
        idx_extra = idx_all[(idx_all >= n_gen0) & (g_bus_vec == i_node) & (pg_vec > 0)]
        # 分类：只有当“该节点同时存在原机组和储能放电”时才混合；否则不动
        if idx_orig.size == 0 or idx_extra.size == 0:
            continue
        P_sum = float(pg_vec[idx_orig].sum() + pg_vec[idx_extra].sum())
        if P_sum <= 0:
            continue
        # 节点整体发电碳强度（功率加权）
        E_mix = float( (pg_vec[idx_orig] @ EG_vec[idx_orig] + pg_vec[idx_extra] @ EG_vec[idx_extra]) / P_sum )
        # 用该“节点整体发电碳强度”覆盖该节点上的所有相关机组（原机组 + 储能机组）
        EG_vec[idx_orig]  = E_mix
        EG_vec[idx_extra] = E_mix

    EG = EG_vec.reshape(-1, 1)
    # === [STORAGE MIX @ NODE END] ===

    # 仍使用标准的逐机组聚合：B = PG.T @ EG  （现在 EG 已按节点内混合后的数值）
    B = PG.T @ EG                           # N_node×1



    # ===== 6) 节点碳势 EN（tCO2/MWh）=====
    # EN = (PN - PB') \ (PG' * EG)
    # 注意：PB' 是转置
    A = PN - PB.T
    B = PG.T @ EG                           # N_node×1
    # 优先线性求解；若病态就最小二乘兜底
    try:
        EN = np.linalg.solve(A, B).ravel()
    except np.linalg.LinAlgError:
        EN = np.linalg.lstsq(A, B, rcond=None)[0].ravel()

    # 回写节点 NCI
    TS['PowerSystem']['NCI'] = EN  # tCO2/MWh

    # ===== 7) 支路碳强度 BCI 与碳流率 RB（tCO2/h）=====
    # BCI 取“源端” NCI（与 MATLAB BCI = EN(lines.from) 一致）
    BCI = EN[f_bus]
    TS['PowerSystem']['BCI'] = BCI  # tCO2/MWh

    # RB(i,j) = EN(i) * PB(i,j)  -> 源端 NCI × MW = t/h
    RB = EN.reshape(-1, 1) * PB
    TS['PowerSystem']['RB'] = RB  # tCO2/h

    # ===== 8) 负荷碳流率 RL（tCO2/h）=====
    RL = PL @ EN
    TS['PowerSystem']['RL'] = RL  # tCO2/h

    # ===== 9) 线路损失分配矩阵 PBloss（MW）=====
    PBloss = np.zeros((n_node, n_node), dtype=float)
    loss_col = pf_col + pt_col
    for i in range(n_line):
        f = f_bus[i]; t = t_bus[i]
        loss = max(loss_col[i], 0.0)   # 钳位非负
        if pf_col[i] >= 0:             # f -> t 方向
            PBloss[f, t] += loss
        else:                          # t -> f 方向
            PBloss[t, f] += loss
    TS['PowerSystem']['PBloss'] = PBloss  # MW

    # ===== 10) 线路损失碳流率 RBloss（tCO2/h）=====
    RBloss = EN.reshape(-1, 1) * PBloss
    TS['PowerSystem']['RBloss'] = RBloss  # tCO2/h

    # ===== 11) 发电机组碳流注入 RG（tCO2/h）=====
    RG = (EG.ravel() * pg_col.ravel()).reshape(-1, 1)  # t/MWh × MW = t/h
    TS['PowerSystem']['RG'] = RG

    # ===== 附：把 PB/PG/PL 也回写，便于对照/调试 =====
    TS['PowerSystem']['PB'] = PB
    TS['PowerSystem']['PG_map'] = PG
    TS['PowerSystem']['PL'] = PL

    # ===== 控制台展示（对齐 MATLAB 输出与单位）=====
    # ===== 统一展示为小数点后两位（不影响内部精度）=====

    def _fmt2(arr: np.ndarray) -> str:
        """将数组以两位小数的方式格式化成字符串，仅用于打印。"""
        return np.array2string(
            np.asarray(arr, dtype=float),
            formatter={'float_kind': lambda x: f"{x:.2f}"},
            threshold=1e6,  # 防止大矩阵被省略
            max_line_width=200
        )

    # 控制台统一两位小数打印
    print('电网节点碳强度分布向量 EN (tCO2/MWh)：\n', _fmt2(EN))
    print('电网支路碳强度分布向量 BCI (tCO2/MWh)：\n', _fmt2(BCI))
    print('发电机组碳流率注入矩阵 RG (tCO2/h)：\n', _fmt2(RG))
    print('电网支路碳流率分布矩阵 RB (tCO2/h)：\n', _fmt2(RB))
    print('电网支路碳流率损失分布矩阵 RBloss (tCO2/h)：\n', _fmt2(RBloss))
    print('电网负荷碳流率分布向量 RL (tCO2/h)：\n', _fmt2(RL))

    return TS

