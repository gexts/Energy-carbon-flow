
# storage.py
# Water-tank storage model for electricity with carbon accounting.
# Units convention:
#   Power P in MW; Energy e in MWh; Time step Δt in hours
#   Carbon intensity w in tCO2/MWh; Embodied carbon E in tCO2
#
# The model follows the "water-tank" approach:
#   e_{t+1} = kappa * e_t + Δt * (eta_ch * P_ch - (1/eta_dc) * P_dc)
#   E_{t+1} = kappa * E_t + Δt * (w_bus * eta_ch * P_ch - (w_es_t / eta_dc) * P_dc)
#   w_es_t = E_t / max(e_t, eps)   (defined only if e_t > 0)
#
# Implementation goals:
#   - Provide injection equivalents for power flow & carbon flow:
#       * Charging -> additional load +P_ch at the bus
#       * Discharging -> additional generator +P_dc at the bus with CI = w_es_t
#   - Enforce mutual exclusivity of charge/discharge each step
#   - Enforce energy bounds via feasible clipping of P_ch / P_dc
#   - Keep a detailed per-step log for auditing
#
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

EPS_E = 1e-6

@dataclass
class StorageUnit:
    id: str
    bus: int                      # connected bus id (as used by the power module)
    Pch_max: float                # MW
    Pdc_max: float                # MW
    eta_ch: float                 # (0,1]
    eta_dc: float                 # (0,1]
    kappa: float                  # self-discharge retention per step, e.g. 0.999
    e_min: float                  # MWh, recommend > 0 to keep w_es well-defined
    e_max: float                  # MWh
    e0: float                     # initial energy MWh
    w_es0: float                  # initial internal CI tCO2/MWh (used to derive E0)
    model: str = "water_tank"     # reserved for future models
    # optional initial embodied carbon (overrides w_es0 if provided)
    E0: Optional[float] = None

    # runtime states (filled by init)
    e_t: float = field(init=False, default=0.0)
    E_t: float = field(init=False, default=0.0)
    w_es_t: float = field(init=False, default=0.0)

def _coerce_positive(x: float, name: str) -> float:
    if not np.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be >=0")
    return float(x)

def build_units_from_table(df: pd.DataFrame) -> List[StorageUnit]:
    req = ["id","bus","Pch_max","Pdc_max","eta_ch","eta_dc","kappa","e_min","e_max","e0","w_es0"]
    for c in req:
        if c not in df.columns:
            raise KeyError(f"Storage table is missing column: {c}")
    units: List[StorageUnit] = []
    for _, r in df.iterrows():
        u = StorageUnit(
            id=str(r["id"]),
            bus=int(r["bus"]),
            Pch_max=float(r["Pch_max"]),
            Pdc_max=float(r["Pdc_max"]),
            eta_ch=float(r["eta_ch"]),
            eta_dc=float(r["eta_dc"]),
            kappa=float(r["kappa"]),
            e_min=float(r["e_min"]),
            e_max=float(r["e_max"]),
            e0=float(r["e0"]),
            w_es0=float(r["w_es0"]),
            model=str(r.get("model","water_tank")),
            E0 = None if pd.isna(r.get("E0", np.nan)) else float(r.get("E0"))
        )
        units.append(u)
    return units

def init_states(units: List[StorageUnit]) -> None:
    for u in units:
        if not (0 < u.eta_ch <= 1 and 0 < u.eta_dc <= 1):
            raise ValueError(f"{u.id}: eta_ch/eta_dc must be in (0,1].")
        if not (0 < u.kappa <= 1):
            raise ValueError(f"{u.id}: kappa must be in (0,1].")
        if not (u.e_min >= 0 and u.e_max > u.e_min):
            raise ValueError(f"{u.id}: require e_max > e_min >= 0.")
        u.e_t = float(np.clip(u.e0, u.e_min, u.e_max))
        if u.E0 is not None:
            u.E_t = float(u.E0)
        else:
            u.E_t = float(u.w_es0 * max(u.e_t, EPS_E))
        u.w_es_t = float(u.E_t / max(u.e_t, EPS_E))

def _mutual_exclusive(Pch_cmd: float, Pdc_cmd: float) -> Tuple[float,float,str]:
    # if both positive, keep the larger command; zero the other
    if Pch_cmd > 0 and Pdc_cmd > 0:
        if Pch_cmd >= Pdc_cmd:
            return Pch_cmd, 0.0, "conflict:kept_charge"
        else:
            return 0.0, Pdc_cmd, "conflict:kept_discharge"
    return max(0.0, Pch_cmd), max(0.0, Pdc_cmd), "ok"

def _feasible_clip(u: StorageUnit, Pch: float, Pdc: float, dt: float) -> Tuple[float,float,str]:
    # power bounds
    Pch = float(np.clip(Pch, 0.0, u.Pch_max))
    Pdc = float(np.clip(Pdc, 0.0, u.Pdc_max))
    # energy feasibility:
    # e_next = kappa*e_t + dt*(eta_ch*Pch - (1/eta_dc)*Pdc)
    # Enforce e_min <= e_next <= e_max by scaling down Pch or Pdc
    e_keep = u.kappa * u.e_t
    # upper bound
    if Pch > 0 and Pdc == 0:
        # max increase allowed
        max_dE = u.e_max - e_keep
        if max_dE < 0:
            Pch_feas = 0.0
            reason = "saturated_high"
        else:
            Pch_feas = min(Pch, max_dE / (dt * u.eta_ch + 1e-12))
            reason = "ok"
        return Pch_feas, 0.0, reason
    # lower bound
    if Pdc > 0 and Pch == 0:
        max_down = e_keep - u.e_min
        if max_down < 0:
            Pdc_feas = 0.0
            reason = "saturated_low"
        else:
            Pdc_feas = min(Pdc, max_down * u.eta_dc / (dt + 1e-12))
            reason = "ok"
        return 0.0, Pdc_feas, reason
    # idle or both zero
    return 0.0, 0.0, "idle"

class StorageRuntime:
    """
    Manages a fleet of StorageUnit across time with given power commands.
    """
    def __init__(self, units: List[StorageUnit], delta_t_hours: float = 1.0):
        self.units = units
        self.dt = float(delta_t_hours)
        init_states(self.units)
        self.log_rows: List[dict] = []

    def plan_injections(self, t: int, cmd_Pch: Dict[str,float], cmd_Pdc: Dict[str,float]) -> Tuple[List[Tuple[int,float]], List[Tuple[int,float,float]]]:
        """
        From commands, produce injection equivalents for time t.
        Returns:
            add_loads: list of (bus_id, P_load_MW) for charging
            add_gens:  list of (bus_id, P_gen_MW, CI_tCO2_per_MWh) for discharging
        """
        add_loads = []
        add_gens = []
        for u in self.units:
            Pch_cmd = float(cmd_Pch.get(u.id, 0.0))
            Pdc_cmd = float(cmd_Pdc.get(u.id, 0.0))
            Pch_cmd, Pdc_cmd, flag_mux = _mutual_exclusive(Pch_cmd, Pdc_cmd)
            Pch_use, Pdc_use, flag_feas = _feasible_clip(u, Pch_cmd, Pdc_cmd, self.dt)
            if Pch_use > 0:
                add_loads.append((u.bus, Pch_use))
            if Pdc_use > 0:
                add_gens.append((u.bus, Pdc_use, float(u.w_es_t)))
            # log command vs used; state will be updated after we know node CI
            self.log_rows.append({
                "t": t, "id": u.id, "bus": u.bus,
                "Pch_cmd": Pch_cmd, "Pdc_cmd": Pdc_cmd,
                "Pch_use": Pch_use, "Pdc_use": Pdc_use,
                "flag_mux": flag_mux, "flag_feas": flag_feas,
                "w_es_t": u.w_es_t, "e_t": u.e_t, "E_t": u.E_t
            })
        return add_loads, add_gens

    def update_states(self, t: int, node_ci_at_bus: Dict[int,float]) -> None:
        """
        Given node carbon intensities at the buses for this time step (t),
        update each storage unit's (e,E,w) to time t+1 and complete per-step logs.
        node_ci_at_bus: mapping {bus_id: w_node_t}
        """
        dt = self.dt
        for u in self.units:
            # fetch the last row we logged for this (t,u)
            m = [r for r in self.log_rows if r["t"]==t and r["id"]==u.id]
            if not m:
                raise RuntimeError(f"Missing plan log for t={t}, unit={u.id}")
            row = m[-1]
            Pch = row["Pch_use"]; Pdc = row["Pdc_use"]
            w_node = float(node_ci_at_bus.get(u.bus, 0.0))
            # energy & carbon recursion
            e_next = u.kappa * u.e_t + dt * (u.eta_ch * Pch - (1.0/u.eta_dc) * Pdc)
            # clamp numerically inside bounds (after feasible clipping this should be within, but keep guard)
            e_next = float(np.clip(e_next, u.e_min, u.e_max))
            E_next = u.kappa * u.E_t + dt * (w_node * u.eta_ch * Pch - (u.w_es_t / u.eta_dc) * Pdc)
            w_next = float(E_next / max(e_next, EPS_E))
            # owner accounting (two viewpoints)
            scope2_charge = dt * (w_node * Pch)  # grid-carbon charged (no efficiency)
            embodied_delta = E_next - (u.kappa * u.E_t)  # equals dt*(w_node*eta_ch*Pch - w_es/eta_dc*Pdc)
            leak_carbon = (1.0 - u.kappa) * u.E_t       # carbon lost due to self-discharge (not assigned)
            # commit state
            u.e_t, u.E_t, u.w_es_t = e_next, E_next, w_next
            # enrich log
            row.update({
                "w_node_t": w_node,
                "e_next": e_next, "E_next": E_next, "w_es_next": w_next,
                "scope2_charge": scope2_charge,
                "embodied_delta": embodied_delta,
                "leak_carbon": leak_carbon
            })

    def log_dataframe(self) -> pd.DataFrame:
        if not self.log_rows:
            return pd.DataFrame(columns=[
                "t","id","bus","Pch_cmd","Pdc_cmd","Pch_use","Pdc_use",
                "flag_mux","flag_feas","w_es_t","e_t","E_t",
                "w_node_t","e_next","E_next","w_es_next",
                "scope2_charge","embodied_delta","leak_carbon"
            ])
        df = pd.DataFrame(self.log_rows)
        # enforce sorted order
        return df.sort_values(["t","id"]).reset_index(drop=True)

# storage.py  —— 替换/更新这个函数
def build_runtime_from_TS(TS, delta_t_hours: float = 1.0):
    """
    从 TS 中构建储能运行时。
    兼容两种输入：
      1) TS['StorageSystem'] 是 dict，且包含 key 'system'（设备台账 DataFrame）
      2) TS['StorageSystem'] 直接就是设备台账 DataFrame
    """
    import pandas as pd

    key = "StorageSystem" if "StorageSystem" in TS else ("storage" if "storage" in TS else None)
    if key is None:
        raise KeyError("TS must contain 'StorageSystem' or 'storage'.")

    obj = TS[key]
    if isinstance(obj, dict):
        df = obj.get("system", None)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("TS['StorageSystem']['system'] must be a non-empty DataFrame.")
    elif isinstance(obj, pd.DataFrame):
        df = obj
    else:
        try:
            df = pd.DataFrame(obj)
        except Exception as e:
            raise ValueError(f"Unsupported StorageSystem format: {type(obj)}") from e

    units = build_units_from_table(df)
    return StorageRuntime(units, delta_t_hours=delta_t_hours)

