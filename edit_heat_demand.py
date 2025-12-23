import pandas as pd

xls = r"F:\工位宿舍共享文件夹\IES_critical_modelling\dataset\ies_dataset_extgrid.xlsx"
out = r"F:\工位宿舍共享文件夹\IES_critical_modelling\dataset\ies_dataset_extgrid.xlsx"  # 覆盖写（谨慎）
ts  = r"F:\工位宿舍共享文件夹\IES_critical_modelling\dataset\outputs\ts\results_timeseries_wide.xlsx"

# 1) 读原热需求
Q = pd.read_excel(xls, sheet_name="heat_Q_demand")
hours = [c for c in Q.columns if isinstance(c,str) and c.startswith("H")]

# 2) 读 ToHeat(EH+Pit) 能力（来自时序结果 heat_source_mix_MW 的 ToHeat_total）
mix = pd.read_excel(ts, sheet_name="heat_source_mix_MW")
mix = mix.set_index("metric")
toheat = mix.loc["ToHeat_total", hours].astype(float)  # Series indexed by Hxx

# 3) 把每个热负荷节点按同一个比例缩放（保留空间分布），确保总需求<=toheat
Qh = Q[hours].astype(float)
tot_old = Qh.sum(axis=0)
ratio = (toheat / tot_old).clip(upper=1.0).fillna(0.0)

Q_new = Qh.mul(ratio, axis=1)
Q[hours] = Q_new

with pd.ExcelWriter(out, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
    Q.to_excel(w, sheet_name="heat_Q_demand", index=False)

print("done. max ratio applied:", ratio.min(), ratio.max())
