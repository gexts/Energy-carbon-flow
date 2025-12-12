import pandas as pd
from pathlib import Path

# 电侧缩放系数：0.01 ≈ 把 1.3 GW 级电负荷变成 13 MW 级
alpha = 0.01

path_in  = Path(r"F:\25秋科研资料\IES_critical_modelling\dataset\ies_dataset_extgrid.xlsx")
path_out = Path(r"F:\25秋科研资料\IES_critical_modelling\dataset\ies_dataset_extgrid_park.xlsx")

xls = pd.ExcelFile(path_in, engine="openpyxl")

with pd.ExcelWriter(path_out, engine="openpyxl") as w:
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)

        # 1) 电负荷 / 无功负荷时序：power_PD, power_QD
        if sheet in ["power_PD", "power_QD"]:
            hour_cols = [c for c in df.columns if c.startswith("H")]
            df[hour_cols] = df[hour_cols] * alpha

        # 2) 发电机时序出力：gen_PG（只包含内部机组，不包括 ext_grid）
        if sheet == "gen_PG":
            hour_cols = [c for c in df.columns if c.startswith("H")]
            df[hour_cols] = df[hour_cols] * alpha

        # 3) 发电机静态参数（Matpower 格式）：power_gen
        #    列一般是: BUS, PG, QG, QMAX, QMIN, VG, MBASE, STATUS, PMAX, PMIN
        if sheet == "power_gen":
            # 有些列名可能不是英文，这里按位置缩放第二、9、10列
            # 安全起见先检查列数>=10
            if df.shape[1] >= 10:
                df.iloc[:, 1] = df.iloc[:, 1] * alpha   # PG 初始出力
                df.iloc[:, 8] = df.iloc[:, 8] * alpha   # PMAX
                df.iloc[:, 9] = df.iloc[:, 9] * alpha   # PMIN

        # 4) 电储能功率能力：storage_system（按需一起缩放）
        if sheet == "storage_system":
            if "Pch_max" in df.columns:
                df["Pch_max"] = df["Pch_max"] * alpha
            if "Pdc_max" in df.columns:
                df["Pdc_max"] = df["Pdc_max"] * alpha
            # e_min/e_max/e0 是否缩放看你希望的SOC绝对量级，这里先不动

        # 其它所有 sheet（热 / 气 / EH / 电价 / external_CI / grid_trade_* 等）原样写回
        df.to_excel(w, sheet_name=sheet, index=False)

print("已生成缩放后的母版数据集:", path_out)
