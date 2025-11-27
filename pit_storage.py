
"""
pit_storage.py — Modular pit (pool) long-duration thermal storage model with carbon state tracking.

Design goals
------------
1) Modular class `PitThermalStorage` with a simple `step(...)` interface
   so it can be called hourly from your scenario runner (IES).
2) Physics-preserving but numerically-stable layered tank model (1D stratified),
   with optional soil-side losses (parameterized via side/top/bottom conductances).
3) Carbon-state tracking analogous to battery storage:
      - e_state (MWh): stored thermal energy relative to T_ref
      - E_state (tCO2): embodied carbon in the stored heat
      - w_es (tCO2/MWh): carbon intensity of stored heat = E_state / e_state
   Carbon updates consider charge, discharge, and thermal losses.
4) Heat-source-side integration:
   The module returns discharge heat flow and its carbon intensity `w_out` so you can
   map it to HeatSystem.sources (output, GCI). Charging is handled via inlet T / m_flow.

Notes
-----
- Units: SI inside; report energy in MWh, power in MW, temperature in °C.
- Integration: explicitly sub-steps from your outer dt (e.g., 3600 s) down to dt_int (default 15 s).
- Soil 2D field is NOT simulated here for runtime; use an effective side loss via U_side and T_env_side.
  (You can extend with a soil grid later if needed.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


# ------------------------ Utilities ------------------------

J_PER_MWH = 3.6e9   # J in 1 MWh
MW_PER_W  = 1e-6    # MW per W (J/s)


def _frustum_layer_geometry(H: float, W_top: float, W_bot: float, n_layers: int, dz: Optional[float]=None):
    """
    Build per-layer geometry for a square frustum-like pit:
    - Height H (m), top width W_top (m), bottom width W_bot (m)
    - n_layers discretization; dz optional (if provided, overrides H/n_layers)
    Returns dict with arrays:
      V[j]     : volume of layer j (m^3), j=1..n_layers (index 0 unused)
      A_side[j]: side area of layer j (m^2)
      A_top    : top area (m^2)
      A_bot    : bottom area of the very bottom (m^2)
      s_at_z(z): helper for side length vs. depth
    """
    if dz is None:
        dz = H / n_layers
    # helper: linear side length vs depth z (0=top, H=bottom)
    def side_len(z):
        return W_top - (W_top - W_bot) * (z / H)

    V = np.zeros(n_layers + 1, dtype=float)
    A_side = np.zeros(n_layers + 1, dtype=float)
    A_btm_layer = np.zeros(n_layers + 1, dtype=float)

    for j in range(1, n_layers + 1):
        z0 = (j - 1) * dz
        z1 = j * dz
        s0 = side_len(z0)
        s1 = side_len(z1)
        A0 = s0 * s0    # top area of the layer (square)
        A1 = s1 * s1    # bottom area of the layer (square)
        # volume of a frustum between z0 and z1 (exact)
        V[j] = dz / 3.0 * (A0 + A1 + np.sqrt(A0 * A1))
        # approximate side area of this band (perimeter*(slant height)/2 for square -> use π-approx as in user's code)
        A_side[j] = np.pi * np.sqrt(dz**2 + (s0 - s1)**2 / 4.0) * (s0 + s1) / 2.0
        A_btm_layer[j] = A1

    A_top = W_top * W_top
    A_bot = W_bot * W_bot
    return {
        "dz": dz,
        "V": V,
        "A_side": A_side,
        "A_top": A_top,
        "A_bot": A_bot,
        "side_len": side_len,
    }


@dataclass
class PitThermalStorageConfig:
    # Geometry
    H: float = 16.0           # height (m)
    W_top: float = 90.0       # top width (m)
    W_bot: float = 26.0       # bottom width (m)
    n_layers: int = 20

    # Water/Material properties
    rho_w: float = 1000.0     # kg/m^3
    cp_w: float = 4187.0      # J/(kg*K)
    lambda_w: float = 0.63    # W/(m*K)

    # Loss / boundary conductances (effective)
    U_side: float = 111.0     # W/(m^2*K), side effective U
    U_top: float = 26.6       # W/(m^2*K), top effective U
    U_bot: float = 74.9       # W/(m^2*K), bottom effective U

    # Boundary/environment temperatures (°C)
    T_top_env: float = 35.0   # ambient air
    T_side_env: float = 25.0  # soil effective side temperature
    T_bot_env: float = 25.0   # soil beneath

    # Numerics
    dt_int: float = 15.0      # internal integrator dt (s)
    mix_advection: bool = True   # include simple advection due to m_flow

    # Reference temperature for SoC and energy accounting (°C)
    T_ref: float = 25.0

    # Initial conditions
    T_initial: float = 25.0
    w_es0: Optional[float] = None  # tCO2/MWh; if None, init from w_in of first charge

    # Safety
    T_min: float = 0.0
    T_max: float = 120.0


@dataclass
class PitThermalStorageState:
    # Layer temperatures with 2 ghost nodes: [0] top ghost, [1..n] layers, [n+1] bottom ghost
    T_layers: np.ndarray
    # Carbon store
    e_state_MWh: float = 0.0      # stored energy (relative to T_ref)
    E_state_t: float = 0.0        # stored embodied carbon (tCO2)
    w_es: float = 0.0             # E_state / max(e_state,eps)


class PitThermalStorage:
    """
    A stratified pit thermal storage with simple conduction + side/top/bottom losses
    and optional advective mixing driven by inlet massflow.
    Carbon state tracking analogous to battery storage.
    """

    def __init__(self, cfg: PitThermalStorageConfig):
        self.cfg = cfg
        geo = _frustum_layer_geometry(cfg.H, cfg.W_top, cfg.W_bot, cfg.n_layers)
        self.dz = geo["dz"]
        self.V = geo["V"]              # m^3 per layer j=1..n
        self.A_side = geo["A_side"]    # m^2 per layer
        self.A_top = geo["A_top"]
        self.A_bot = geo["A_bot"]

        # vertical conductances between layers and to boundaries
        # Using a simple area*lambda/dz form; we approximate the conduction area per layer as bottom area (A_btm_layer).
        # You can refine this with harmonic averages of A0/A1 if needed.
        # For numerical stability and to keep it aligned with user's code, we use a circular-approximation factor π/4.
        A_eq_per_layer = np.zeros(cfg.n_layers + 1, dtype=float)
        for j in range(1, cfg.n_layers + 1):
            # approximate cross-sectional area at the bottom of layer j:
            s = cfg.W_bot + (cfg.W_top - cfg.W_bot) * (j * self.dz / cfg.H)
            A_eq_per_layer[j] = (np.pi / 4.0) * s * s   # circle approx as in user's snippet

        self.K_vert = np.zeros(cfg.n_layers + 1, dtype=float)
        for j in range(1, cfg.n_layers + 1):
            self.K_vert[j] = cfg.lambda_w * A_eq_per_layer[j] / self.dz  # W/K

        # top/bottom effective conductances to environment:
        self.K_top = cfg.U_top * self.A_top  # W/K
        self.K_bot = cfg.U_bot * self.A_bot  # W/K

        # Initialize temperature profile with ghosts
        T = np.full(cfg.n_layers + 2, cfg.T_initial, dtype=float)
        # Ghosts
        T[0] = cfg.T_top_env
        T[-1] = cfg.T_bot_env

        self.state = PitThermalStorageState(T_layers=T)

        # Initialize carbon state based on initial energy (which is zero by definition relative to T_ref)
        self._recompute_energy_and_update_w()

    # -------------- Energy & Carbon helpers --------------
    def _layer_heat_capacity_J_per_K(self, j: int) -> float:
        return self.cfg.rho_w * self.cfg.cp_w * self.V[j]  # J/K

    def _energy_J_from_T(self, T: np.ndarray) -> float:
        # energy above reference T_ref
        dT = (T[1:-1] - self.cfg.T_ref)  # layers only
        C = np.array([self._layer_heat_capacity_J_per_K(j) for j in range(1, self.cfg.n_layers + 1)])
        return float(np.sum(C * dT))

    def _recompute_energy_and_update_w(self):
        e_J = self._energy_J_from_T(self.state.T_layers)
        self.state.e_state_MWh = e_J / J_PER_MWH
        if self.state.e_state_MWh <= 1e-9:
            self.state.w_es = 0.0 if self.state.E_state_t <= 0 else self.state.E_state_t / 1e-9
        else:
            self.state.w_es = self.state.E_state_t / self.state.e_state_MWh

    # -------------- Thermal update core --------------
    def _advance_thermal(self, dt: float, T_inlet_C: float, m_flow_kg_s: float):
        """
        Explicit Euler update of layer temperatures over dt (s).
        Includes:
          - vertical conduction between adjacent layers
          - top conduction to ambient and bottom to soil
          - side losses to side_env
          - optional advective mixing along downward flow driven by m_flow (from top->down)
        """
        cfg = self.cfg
        T = self.state.T_layers.copy()
        n = cfg.n_layers

        # prepare ghost nodes for this step
        T[0] = cfg.T_top_env
        T[-1] = cfg.T_bot_env

        # constants per layer
        Cj = np.array([self._layer_heat_capacity_J_per_K(j) for j in range(1, n + 1)])  # J/K
        # per-layer side loss conductance (W/K)
        K_side = cfg.U_side * self.A_side[1: n+1]

        # Precompute advective term coefficient alpha_j = (m_flow * cp) / Cj [1/s]
        adv_alpha = 0.0
        if cfg.mix_advection and m_flow_kg_s > 0.0:
            adv_alpha = m_flow_kg_s * cfg.cp_w  # W/K (J/s/K)
            adv_alpha = adv_alpha / Cj          # 1/s per layer

        dTdt = np.zeros(n, dtype=float)

        for j in range(1, n + 1):
            # conduction to neighbors
            up_term = self.K_vert[j-1] * (T[j-1] - T[j]) if j-1 >= 1 else self.K_top * (cfg.T_top_env - T[j])
            dn_term = self.K_vert[j]   * (T[j+1] - T[j]) if j   <= n   else self.K_bot * (cfg.T_bot_env - T[j])
            side_term = K_side[j-1] * (cfg.T_side_env - T[j])

            # sum as power (W), divide by Cj to get dT/dt
            base = (up_term + dn_term + side_term) / Cj[j-1]

            # advection: flow from layer (j-1) into j; for j=1, inlet boundary uses T_inlet
            if cfg.mix_advection and m_flow_kg_s > 0.0:
                T_in = (T_inlet_C if j == 1 else T[j-1])
                base += adv_alpha[j-1] * (T_in - T[j])

            dTdt[j-1] = base

        # explicit Euler
        T_new = T.copy()
        T_new[1:-1] = np.clip(T[1:-1] + dt * dTdt, cfg.T_min, cfg.T_max)

        self.state.T_layers = T_new

    def _withdraw_energy_from_top(self, E_out_J: float) -> Tuple[float, float]:
        """
        Remove thermal energy from the top layers to serve the heat network.
        Returns a tuple (E_delivered_J, T_out_C_before_withdraw).
        We assume the outlet draws from the topmost hot layer; T_out is the top layer temp before withdrawal.
        """
        if E_out_J <= 0.0:
            T_out = float(self.state.T_layers[1])
            return 0.0, T_out

        T = self.state.T_layers.copy()
        cfg = self.cfg
        E_needed = E_out_J
        T_out = float(T[1])  # temperature offered to the network (before withdrawal)

        for j in range(1, cfg.n_layers + 1):
            if E_needed <= 0.0:
                break
            Cj = self._layer_heat_capacity_J_per_K(j)
            # how much we can drop this layer to T_ref at most
            dE_max = max(0.0, Cj * (T[j] - cfg.T_ref))
            dE = min(E_needed, dE_max)
            if dE > 0:
                dT = dE / Cj
                T[j] -= dT
                E_needed -= dE

        E_delivered = E_out_J - E_needed
        self.state.T_layers = T
        return E_delivered, T_out

    # -------------- Public API --------------
    def get_state_snapshot(self) -> Dict:
        T = self.state.T_layers
        avg_T = float(np.mean(T[1:-1]))
        top_T = float(T[1])
        return {
            "avg_T_C": avg_T,
            "top_T_C": top_T,
            "e_state_MWh": float(self.state.e_state_MWh),
            "E_state_t": float(self.state.E_state_t),
            "w_es_t_per_MWh": float(self.state.w_es),
            "n_layers": self.cfg.n_layers,
        }

    def step(self,
             dt_seconds: float,
             T_inlet_C: float,
             m_flow_kg_s: float,
             Q_out_setpoint_MW: float,
             w_in_t_per_MWh: Optional[float] = None) -> Dict:
        """
        Advance the internal model by dt_seconds.
        Inputs:
          - T_inlet_C: charge inlet temperature at top (°C)
          - m_flow_kg_s: mass flow rate (kg/s); use >0 for charging/mixing; 0 to disable advection
          - Q_out_setpoint_MW: requested discharge power to heat network (MW), >=0
          - w_in_t_per_MWh: carbon intensity of the charging heat (tCO2/MWh); if None and storage is empty,
            w_es will stay at its previous value (or cfg.w_es0 if set).

        Returns a dict:
          {
            "Q_out_MW": delivered discharge power (may be < setpoint if energy limited),
            "w_out_t_per_MWh": carbon intensity of delivered heat,
            "T_out_C": outlet temperature (top layer before withdrawal),
            "avg_T_C": average water temperature,
            "e_state_MWh": energy state after step,
            "E_state_t": carbon state after step,
            "w_es_t_per_MWh": storage carbon intensity after step,
          }
        """
        cfg = self.cfg
        st = self.state

        # Carbon intensity for the first-time charge if needed
        if st.e_state_MWh <= 1e-9 and st.E_state_t <= 1e-12 and cfg.w_es0 is not None and w_in_t_per_MWh is None:
            w_in_t_per_MWh = cfg.w_es0

        dt = float(dt_seconds)
        n_sub = max(1, int(np.ceil(dt / cfg.dt_int)))
        dt_sub = dt / n_sub

        # Pre-step energy/carbon snapshot
        e_before_MWh = self._energy_J_from_T(st.T_layers) / J_PER_MWH
        w_es_before = float(st.w_es)

        # Estimate charging energy from inlet flow for carbon accounting (independent of conduction losses)
        # Use the top-layer temperature at the *beginning* as reference
        T_top_before = float(st.T_layers[1])
        Q_in_W_est = max(0.0, m_flow_kg_s * cfg.cp_w * (T_inlet_C - T_top_before))  # J/s
        E_in_J_est = Q_in_W_est * dt
        E_in_MWh_est = E_in_J_est / J_PER_MWH

        # Thermal evolution (charging) over substeps
        for _ in range(n_sub):
            self._advance_thermal(dt_sub, T_inlet_C, m_flow_kg_s)

        # Discharge by withdrawing energy from the stratified layers (top-down)
        E_req_out_J = max(0.0, Q_out_setpoint_MW) * dt * 1e6
        E_delivered_J, T_out_C = self._withdraw_energy_from_top(E_req_out_J)
        Q_out_MW_eff = (E_delivered_J / dt) * MW_PER_W

        # Energy after evolution & withdrawal
        e_after_MWh = self._energy_J_from_T(self.state.T_layers) / J_PER_MWH

        # Thermal loss over the period = e_after - e_before - E_in + E_out (MWh)
        Q_loss_MWh = e_after_MWh - e_before_MWh - E_in_MWh_est + (E_delivered_J / J_PER_MWH)

        # ---- Carbon state updates ----
        # 1) Charge
        if E_in_MWh_est > 0.0 and w_in_t_per_MWh is not None:
            st.E_state_t += w_in_t_per_MWh * E_in_MWh_est

        # 2) Losses carry the storage mixture carbon intensity (proportional removal)
        if abs(Q_loss_MWh) > 1e-12:
            # Q_loss_MWh is usually negative; removing energy reduces carbon proportionally
            st.E_state_t += st.w_es * Q_loss_MWh

        # 3) Discharge: delivered carbon is from storage mixture BEFORE withdrawal
        if E_delivered_J > 0.0:
            E_out_MWh = E_delivered_J / J_PER_MWH
            st.E_state_t -= w_es_before * E_out_MWh

        # Recompute e_state from temperature and update w_es
        self._recompute_energy_and_update_w()

        return {
            "Q_out_MW": float(Q_out_MW_eff),
            "w_out_t_per_MWh": float(w_es_before if E_delivered_J > 0.0 else self.state.w_es),
            "T_out_C": float(T_out_C),
            "avg_T_C": float(np.mean(self.state.T_layers[1:-1])),
            "e_state_MWh": float(self.state.e_state_MWh),
            "E_state_t": float(self.state.E_state_t),
            "w_es_t_per_MWh": float(self.state.w_es),
        }


# ------------------------ Simple factory ------------------------

def make_default_pit_storage() -> PitThermalStorage:
    cfg = PitThermalStorageConfig()
    return PitThermalStorage(cfg)
