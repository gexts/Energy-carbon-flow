import os, re
import pandas as pd

def hcols(df):
    return [c for c in df.columns if re.match(r"H\d\d", str(c))]

def check(dataset_dir, fname="ies_dataset_extgrid.xlsx"):
    xls = os.path.join(dataset_dir, fname)
    print("Using:", xls)

    # --- Power ---
    pd_w = pd.read_excel(xls, sheet_name="power_PD")
    qd_w = pd.read_excel(xls, sheet_name="power_QD")
    gen  = pd.read_excel(xls, sheet_name="power_gen", header=None)

    P = pd_w[hcols(pd_w)].sum(axis=0)
    Q = qd_w[hcols(qd_w)].sum(axis=0)
    print("\n[Power] total PD peak/avg (MW):", float(P.max()), float(P.mean()))
    print("[Power] total QD peak/avg (MVAr):", float(Q.max()), float(Q.mean()))
    print("[Power] PF approx at peak:", float(P.max()) / ((float(P.max())**2 + float(Q.max())**2) ** 0.5 + 1e-12))
    print("[Power] gen PMAX sum (MW):", float(gen.iloc[:,8].sum()))

    # --- Heat ---
    hq = pd.read_excel(xls, sheet_name="heat_Q_demand")
    H = hq[hcols(hq)].sum(axis=0)
    print("\n[Heat] total Q peak/avg (MWth):", float(H.max()), float(H.mean()))

    # --- Gas (km^3/h ≈ 10^3 m^3/h) ---
    gl = pd.read_excel(xls, sheet_name="gas_loads")
    gs = pd.read_excel(xls, sheet_name="gas_sources")
    B = 10.45  # MWh/(km^3)
    g_load = float(gl["demand"].sum())
    g_src  = float(gs["output"].sum())
    print("\n[Gas] total load (km^3/h):", g_load, " => MW equiv:", g_load * B)
    print("[Gas] total source (km^3/h):", g_src,  " => MW equiv:", g_src * B)

if __name__ == "__main__":
    check(r"F:\工位宿舍共享文件夹\IES_critical_modelling\dataset", fname="ies_dataset_extgrid.xlsx")
