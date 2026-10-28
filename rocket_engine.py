"""
================================================================================
  ROCKET TRAJECTORY & LANDING PREDICTION PHYSICS ENGINE
  -------------------------------------------------------
  Course  : Numerical Methods & Analysis
  Methods : Runge-Kutta 4th Order (RK4)  +  Newton-Raphson Root Finding
  Author  : [Your Name]
  Date    : 2026-10-28
================================================================================

PHYSICS BACKGROUND
------------------
We model 2-D projectile motion with air drag.

State vector:  S = [x, y, vx, vy]

Drag force magnitude:
    F_drag = 0.5 * rho * v^2 * Cd * A          (v = speed = sqrt(vx²+vy²))

Velocity angle:
    theta = atan2(vy, vx)                       (angle of velocity vector)

Equations of motion (ODEs):
    dx/dt  =  vx
    dy/dt  =  vy
    dvx/dt = -(F_drag / m) * cos(theta)
    dvy/dt = -g - (F_drag / m) * sin(theta)

NUMERICAL METHODS
-----------------
1. RK4  - advances the state S step by step with time-step dt.
2. Newton-Raphson - once Y goes below 0, finds exact landing time t* where Y(t*)=0.
   Update rule:  t_new = t_old - Y(t_old) / Vy(t_old)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PHYSICS PARAMETERS  (tweak these to experiment)
# ──────────────────────────────────────────────────────────────────────────────

GRAVITY   = 9.81          # m/s²  — gravitational acceleration
RHO       = 1.225         # kg/m³ — air density at sea level
CD        = 0.3           # —      drag coefficient (dimensionless)
AREA      = 0.05          # m²    — rocket cross-sectional area
MASS      = 50.0          # kg    — rocket mass
V0        = 150.0         # m/s   — initial speed
ANGLE_DEG = 45.0          # °     — launch angle above horizontal
DT        = 0.05          # s     — RK4 time-step
NR_TOL    = 1e-10         # m     — Newton-Raphson convergence tolerance
NR_MAXITER= 100           # —     — safety cap on N-R iterations


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DRAG HELPER
# ──────────────────────────────────────────────────────────────────────────────

def aerodynamic_drag(vx: float, vy: float,
                     rho: float = RHO, cd: float = CD, area: float = AREA) -> float:
    """
    Compute the magnitude of the aerodynamic drag force [N].

        F_drag = 0.5 * rho * v² * Cd * A

    Parameters
    ----------
    vx, vy : float  — velocity components [m/s]
    rho    : float  — air density [kg/m³]
    cd     : float  — drag coefficient
    area   : float  — reference area [m²]

    Returns
    -------
    float — drag force magnitude [N]
    """
    v_squared = vx**2 + vy**2          # v² (saves a sqrt for force calc)
    return 0.5 * rho * v_squared * cd * area


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ODE RIGHT-HAND SIDE  —  dS/dt = f(S)
# ──────────────────────────────────────────────────────────────────────────────

def odes(state: np.ndarray, mass: float = MASS) -> np.ndarray:
    """
    Evaluate the right-hand side of the system of ODEs.

    State vector S = [x, y, vx, vy]

    Returns dS/dt = [vx,  vy,  dvx/dt,  dvy/dt]

    Parameters
    ----------
    state : np.ndarray — current [x, y, vx, vy]
    mass  : float      — rocket mass [kg]

    Returns
    -------
    np.ndarray — derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    x, y, vx, vy = state

    # ── drag force magnitude ──────────────────────────────────────────────────
    f_drag = aerodynamic_drag(vx, vy)

    # ── velocity direction angle  theta = atan2(vy, vx) ──────────────────────
    theta = np.arctan2(vy, vx)

    # ── accelerations ─────────────────────────────────────────────────────────
    ax = -(f_drag / mass) * np.cos(theta)           # drag opposes motion in x
    ay = -GRAVITY - (f_drag / mass) * np.sin(theta) # gravity + drag in y

    return np.array([vx, vy, ax, ay])


# ──────────────────────────────────────────────────────────────────────────────
# 4.  RUNGE-KUTTA 4th ORDER  (single step)
# ──────────────────────────────────────────────────────────────────────────────

def rk4_step(state: np.ndarray, dt: float, mass: float = MASS) -> np.ndarray:
    """
    Advance the state vector by ONE time step dt using the classical RK4 scheme.

    The four-stage formula:
        k1 = f( S )
        k2 = f( S + dt/2 * k1 )
        k3 = f( S + dt/2 * k2 )
        k4 = f( S + dt   * k3 )

        S_new = S + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Parameters
    ----------
    state : np.ndarray — current state [x, y, vx, vy]
    dt    : float      — time step [s]
    mass  : float      — rocket mass [kg]

    Returns
    -------
    np.ndarray — updated state after one dt
    """
    k1 = odes(state,                 mass)
    k2 = odes(state + 0.5 * dt * k1, mass)
    k3 = odes(state + 0.5 * dt * k2, mass)
    k4 = odes(state +       dt * k3, mass)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  NEWTON-RAPHSON  —  find exact landing time where Y(t) = 0
# ──────────────────────────────────────────────────────────────────────────────

def newton_raphson_landing(state_prev: np.ndarray,
                            t_prev: float,
                            dt: float,
                            mass: float = MASS,
                            tol: float = NR_TOL,
                            max_iter: int = NR_MAXITER):
    """
    Use Newton-Raphson iteration to find the exact time t* at which Y(t*) = 0,
    starting from the last valid RK4 state just before ground crossing.

    Update rule:
        t_new = t_old - Y(t_old) / Vy(t_old)

    where  Y(t_old) is the altitude   and
           Vy(t_old) is the vertical velocity (= dY/dt — our analytic derivative).

    We use RK4 with small sub-steps from t_prev to t_old to obtain Y and Vy
    at each Newton iteration point.

    Parameters
    ----------
    state_prev : np.ndarray — RK4 state at the last step BEFORE ground crossing
    t_prev     : float      — time at state_prev [s]
    dt         : float      — the full RK4 step that crossed the ground [s]
    mass       : float      — rocket mass [kg]
    tol        : float      — convergence tolerance on |Y| [m]
    max_iter   : int        — maximum Newton-Raphson iterations

    Returns
    -------
    t_land   : float        — exact landing time [s]
    x_land   : float        — exact landing x-position [m]
    iterations : int        — number of Newton-Raphson iterations performed
    """
    # Initial guess: midway between t_prev and t_prev + dt
    t_guess = t_prev + dt / 2.0

    print("\n  Newton-Raphson Iteration Log")
    print(f"  {'Iter':>4}  {'t_guess (s)':>14}  {'Y (m)':>14}  {'Vy (m/s)':>12}")
    print("  " + "-" * 52)

    for iteration in range(1, max_iter + 1):
        # ── Integrate from t_prev → t_guess to get state at the guess time ──
        delta_t   = t_guess - t_prev    # time span from last known state
        sub_steps = max(1, int(delta_t / (dt * 0.1)))  # ≥1 fine sub-steps
        sub_dt    = delta_t / sub_steps

        state_at_guess = state_prev.copy()
        for _ in range(sub_steps):
            state_at_guess = rk4_step(state_at_guess, sub_dt, mass)

        y_val  = state_at_guess[1]    # altitude  Y(t_guess)
        vy_val = state_at_guess[3]    # vertical velocity  Vy(t_guess) = dY/dt

        print(f"  {iteration:>4}  {t_guess:>14.8f}  {y_val:>14.8f}  {vy_val:>12.6f}")

        # ── Newton-Raphson update ─────────────────────────────────────────────
        if abs(vy_val) < 1e-15:       # guard against zero-division
            print("  [WARNING] Vy ≈ 0 — Newton-Raphson aborted.")
            break

        t_new = t_guess - y_val / vy_val

        # ── Convergence check ─────────────────────────────────────────────────
        if abs(y_val) < tol:
            t_guess = t_new
            print(f"\n  ✔ Converged after {iteration} iteration(s)  |Y| = {abs(y_val):.2e} m")
            break

        t_guess = t_new

    else:
        print(f"  [WARNING] Newton-Raphson did not converge in {max_iter} iterations.")

    # ── Get final state at t_land ─────────────────────────────────────────────
    t_land   = t_guess
    delta_t  = t_land - t_prev
    sub_steps = max(1, int(delta_t / (dt * 0.1)))
    sub_dt   = delta_t / sub_steps

    final_state = state_prev.copy()
    for _ in range(sub_steps):
        final_state = rk4_step(final_state, sub_dt, mass)

    x_land = final_state[0]

    return t_land, x_land, iteration


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN SIMULATION  —  orchestrates RK4 + Newton-Raphson
# ──────────────────────────────────────────────────────────────────────────────

def simulate_trajectory(v0: float = V0,
                        angle_deg: float = ANGLE_DEG,
                        mass: float = MASS,
                        dt: float = DT) -> dict:
    """
    Run the full rocket trajectory simulation.

    Steps:
        1. Initialise launch conditions.
        2. Advance state with RK4 until Y < 0 (ground crossing detected).
        3. Step back one step; call Newton-Raphson for the exact landing.
        4. Record and return all trajectory data for plotting.

    Parameters
    ----------
    v0        : float — initial speed [m/s]
    angle_deg : float — launch angle [degrees]
    mass      : float — rocket mass [kg]
    dt        : float — RK4 time step [s]

    Returns
    -------
    dict with keys:
        'times'      — list of time stamps
        'x_data'     — list of x positions [m]
        'y_data'     — list of y positions [m]
        'vx_data'    — list of vx values   [m/s]
        'vy_data'    — list of vy values   [m/s]
        't_land'     — exact landing time  [s]
        'x_land'     — exact landing x     [m]
        'nr_iters'   — Newton-Raphson iteration count
    """
    # ── Initial conditions ────────────────────────────────────────────────────
    angle_rad = np.radians(angle_deg)
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)

    state = np.array([0.0, 0.0, vx0, vy0])   # S = [x, y, vx, vy]
    t     = 0.0

    # ── Storage lists ─────────────────────────────────────────────────────────
    times   = [t]
    x_data  = [state[0]]
    y_data  = [state[1]]
    vx_data = [state[2]]
    vy_data = [state[3]]

    print("=" * 70)
    print("  ROCKET TRAJECTORY SIMULATION  —  RK4 + Newton-Raphson")
    print("=" * 70)
    print(f"\n  Launch Parameters:")
    print(f"    Initial Speed    v₀  = {v0:.1f}  m/s")
    print(f"    Launch Angle     θ   = {angle_deg:.1f}  °")
    print(f"    Mass             m   = {mass:.1f}  kg")
    print(f"    Drag Coefficient Cd  = {CD}")
    print(f"    Air Density      ρ   = {RHO}  kg/m³")
    print(f"    Cross-Sect. Area A   = {AREA}  m²")
    print(f"    Time Step        dt  = {dt}  s")
    print(f"\n  Running RK4 integration …")

    # ── RK4 MAIN LOOP ─────────────────────────────────────────────────────────
    state_prev = state.copy()
    t_prev     = t

    while True:
        new_state = rk4_step(state, dt, mass)
        t_new     = t + dt

        # ── Did we cross the ground? ──────────────────────────────────────────
        if new_state[1] < 0 and t > 0:
            # Y went negative → ground crossing detected
            # We do NOT append this state; instead, we refine with Newton-Raphson
            print(f"  Ground crossing detected at RK4 step t = {t_new:.4f} s  "
                  f"(Y = {new_state[1]:.4f} m)")
            break

        # ── Accept step ───────────────────────────────────────────────────────
        state_prev = state.copy()
        t_prev     = t

        state = new_state
        t     = t_new

        times.append(t)
        x_data.append(state[0])
        y_data.append(state[1])
        vx_data.append(state[2])
        vy_data.append(state[3])

    # ── NEWTON-RAPHSON  refinement ────────────────────────────────────────────
    t_land, x_land, nr_iters = newton_raphson_landing(
        state_prev=state_prev,
        t_prev=t_prev,
        dt=dt,
        mass=mass
    )

    # Append the precise landing point
    times.append(t_land)
    x_data.append(x_land)
    y_data.append(0.0)          # by definition, landing Y = 0

    return {
        "times"   : times,
        "x_data"  : x_data,
        "y_data"  : y_data,
        "vx_data" : vx_data,
        "vy_data" : vy_data,
        "t_land"  : t_land,
        "x_land"  : x_land,
        "nr_iters": nr_iters,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectory(results: dict,
                    angle_deg: float = ANGLE_DEG,
                    v0: float = V0) -> None:
    """
    Generate a publication-quality 2-D trajectory plot.

    Shows:
      • The rocket path (coloured by speed)
      • Launch point marker
      • Exact landing point marker (from Newton-Raphson)
      • Annotated landing stats

    Parameters
    ----------
    results   : dict  — output of simulate_trajectory()
    angle_deg : float — launch angle (for title)
    v0        : float — initial speed (for title)
    """
    x_arr = np.array(results["x_data"])
    y_arr = np.array(results["y_data"])
    t_arr = np.array(results["times"])

    # ── Compute speed at each recorded point for colour map ───────────────────
    vx_arr = np.array(results["vx_data"] + [results["vx_data"][-1]])  # pad last
    vy_arr = np.array(results["vy_data"] + [results["vy_data"][-1]])
    speed  = np.sqrt(vx_arr**2 + vy_arr**2)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")

    ax  = axes[0]   # main trajectory panel
    ax2 = axes[1]   # altitude vs time panel

    for a in axes:
        a.set_facecolor("#161b22")
        a.tick_params(colors="white")
        a.xaxis.label.set_color("white")
        a.yaxis.label.set_color("white")
        a.title.set_color("white")
        for spine in a.spines.values():
            spine.set_edgecolor("#30363d")

    # ── (A)  Trajectory coloured by speed ────────────────────────────────────
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # Build segments for LineCollection (speed-colour gradient)
    points  = np.array([x_arr, y_arr]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    speed_seg = 0.5 * (speed[:-1] + speed[1:])

    norm = Normalize(vmin=speed.min(), vmax=speed.max())
    lc   = LineCollection(segs, cmap="plasma", norm=norm, linewidth=2.8, alpha=0.9)
    lc.set_array(speed_seg)
    ax.add_collection(lc)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="plasma"),
                        ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label("Speed  [m/s]", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # ── Ground line ────────────────────────────────────────────────────────────
    ax.axhline(0, color="#58a6ff", linewidth=1.0, linestyle="--", alpha=0.5,
               label="Ground level")

    # ── Launch point ──────────────────────────────────────────────────────────
    ax.scatter(0, 0, s=180, color="#3fb950", zorder=6, marker="^",
               label="Launch Point  (0, 0)")
    ax.annotate("[LAUNCH]\n(0, 0)", xy=(0, 0),
                xytext=(results["x_land"] * 0.04, results["y_data"][len(y_arr)//6]),
                color="#3fb950", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#3fb950", lw=1.4))

    # ── Exact landing point (Newton-Raphson result) ───────────────────────────
    x_land = results["x_land"]
    ax.scatter(x_land, 0, s=220, color="#f78166", zorder=7, marker="X",
               label=f"Landing Point  ({x_land:.1f} m, 0)")
    ax.annotate(f"[LANDING]\n({x_land:.2f} m, 0 m)\nt = {results['t_land']:.3f} s",
                xy=(x_land, 0),
                xytext=(x_land * 0.78, max(y_arr) * 0.35),
                color="#f78166", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#f78166", lw=1.4))

    # ── Apogee marker ─────────────────────────────────────────────────────────
    apogee_idx = np.argmax(y_arr)
    ax.scatter(x_arr[apogee_idx], y_arr[apogee_idx],
               s=150, color="#ffa657", zorder=5, marker="*",
               label=f"Apogee  ({x_arr[apogee_idx]:.1f} m, {y_arr[apogee_idx]:.1f} m)")
    ax.annotate(f"  Apogee\n  y = {y_arr[apogee_idx]:.1f} m",
                xy=(x_arr[apogee_idx], y_arr[apogee_idx]),
                xytext=(x_arr[apogee_idx] * 0.55, y_arr[apogee_idx] * 1.02),
                color="#ffa657", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#ffa657", lw=1.2))

    ax.set_xlim(-x_land * 0.03, x_land * 1.08)
    ax.set_ylim(-max(y_arr) * 0.08, max(y_arr) * 1.15)
    ax.set_xlabel("Horizontal Distance  [m]", fontsize=10)
    ax.set_ylabel("Altitude  [m]", fontsize=10)
    ax.set_title(
        f"Rocket Trajectory  |  v₀ = {v0} m/s, θ = {angle_deg}°, m = {MASS} kg, "
        f"Cd = {CD}, A = {AREA} m²\n"
        f"Numerical Method: RK4 (dt = {DT} s)  +  Newton-Raphson Landing Refinement",
        fontsize=10, pad=10
    )
    ax.legend(loc="upper left", fontsize=8.5,
              framealpha=0.25, edgecolor="#30363d", labelcolor="white")
    ax.grid(True, color="#21262d", linewidth=0.6)

    # ── (B)  Altitude vs Time ─────────────────────────────────────────────────
    ax2.plot(t_arr, y_arr, color="#58a6ff", linewidth=1.8, label="Altitude Y(t)")
    ax2.axhline(0, color="#f78166", linewidth=1.0, linestyle="--")
    ax2.scatter(results["t_land"], 0, s=100, color="#f78166", zorder=5)
    ax2.set_xlabel("Time  [s]", fontsize=9)
    ax2.set_ylabel("Altitude  [m]", fontsize=9)
    ax2.set_title("Altitude vs. Time", fontsize=9)
    ax2.legend(fontsize=8, framealpha=0.2, edgecolor="#30363d", labelcolor="white")
    ax2.grid(True, color="#21262d", linewidth=0.6)
    ax2.set_facecolor("#161b22")

    plt.tight_layout(pad=2.0)

    # ── Stats text box ─────────────────────────────────────────────────────────
    stats = (
        f"Landing Time      : {results['t_land']:.6f}  s\n"
        f"Landing Distance  : {results['x_land']:.4f}  m\n"
        f"N-R Iterations    : {results['nr_iters']}\n"
        f"Max Altitude       : {max(y_arr):.2f}  m"
    )
    fig.text(0.76, 0.60, stats, transform=fig.transFigure,
             fontsize=9, color="white", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                       edgecolor="#58a6ff", alpha=0.85))

    plt.savefig("rocket_trajectory.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\n  Plot saved → rocket_trajectory.png")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Run simulation ─────────────────────────────────────────────────────────
    results = simulate_trajectory(
        v0        = V0,
        angle_deg = ANGLE_DEG,
        mass      = MASS,
        dt        = DT,
    )

    # ── Print final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  ✔ Exact Landing Time      : {results['t_land']:.6f}  s")
    print(f"  ✔ Exact Landing Distance  : {results['x_land']:.4f}  m")
    print(f"  ✔ Newton-Raphson Iters    : {results['nr_iters']}")
    print(f"  ✔ Max Altitude (Apogee)   : {max(results['y_data']):.4f}  m")
    print("=" * 70)

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_trajectory(results, angle_deg=ANGLE_DEG, v0=V0)


if __name__ == "__main__":
    main()
