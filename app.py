# app.py
# Streamlit + Pyomo DC Power Flow (auto-layout diagram, compact styling, smart labels)

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

import pyomo.environ as pyo
from pyomo.environ import inequality, minimize
from pyomo.contrib.appsi.solvers import Highs  # pure-Python HiGHS (deploy-friendly)

# Try to import adjustText (optional). If missing, we just skip collision-avoidance.
try:
    from adjustText import adjust_text
    _HAS_ADJUST = False
except Exception:
    _HAS_ADJUST = False

st.set_page_config(page_title="DC Power Flow", layout="wide")
st.title("DC Power Flow Optimization")
col1, col2, col3 = st.columns([1,2,1])   # middle column is wider
with col2:
    st.image(
        "formulation.png",   # path to your image file
        caption="Optimization Model Formulation",
        width=1000                 # set a fixed width in pixels
        # use_container_width=True  # alternatively stretch to fit column width
    )

st.caption("Edit the data and click **Solve**. The compact diagram shows Pg*, Pl*, θ and the Optimal Cost.")

# -------------------------------------------------------------------
# Starter data (no x,y needed; layout computed automatically)
# -------------------------------------------------------------------
def example_data():
    buses = pd.DataFrame({
        "bus":  [0, 1, 2],
        "name": ["B0", "B1", "B2"],
    })
    generations = pd.DataFrame({
        "gen":   [0, 1],
        "bus":   [0, 1],
        "pgmax": [20.0, 30.0],
        "cost":  [0.2,  0.5],
    })
    loads = pd.DataFrame({
        "load_id": [0],
        "bus":     [2],
        "load":    [25.0],
    })
    lines = pd.DataFrame({
        "line":     [0, 1, 2],
        "from_bus": [0, 0, 1],   # L0: B0->B1, L1: B0->B2, L2: B1->B2
        "to_bus":   [1, 2, 2],
        "plmax":    [15.0, 15.0, 15.0],
        "Bl":       [1000.0, 1000.0, 1000.0],
    })
    return buses, generations, loads, lines

# ---------------------------
# UI – editable parameter tables
# ---------------------------
st.subheader("Input Data")

buses, generations, loads, lines = example_data()

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Buses** (columns: `bus`, `name`)")
    buses = st.data_editor(buses, num_rows="dynamic", key="buses_editor", use_container_width=True)
with c2:
    st.markdown("**Generators** (columns: `gen`, `bus`, `pgmax`, `cost`)")
    generations = st.data_editor(generations, num_rows="dynamic", key="gens_editor", use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown("**Loads** (columns: `load_id`, `bus`, `load`)")
    loads = st.data_editor(loads, num_rows="dynamic", key="loads_editor", use_container_width=True)
with c4:
    st.markdown("**Lines** (columns: `line`, `from_bus`, `to_bus`, `plmax`, `Bl`)")
    lines = st.data_editor(lines, num_rows="dynamic", key="lines_editor", use_container_width=True)

# Slack bus choice (external id)
candidate_slacks = sorted(buses["bus"].unique()) if "bus" in buses.columns else []
slack_bus = st.selectbox("Slack bus (external id)", candidate_slacks, index=0 if candidate_slacks else None)

solver_choice = st.selectbox("Solver", ["GLPK","HiGHS"], index=0)

# Toggle: show Pg* in blue to stand out (optional)
show_pg_star_blue = st.toggle("Highlight Pg* in blue", value=True)

# ---------------------------
# Auto-layout (no x,y required)
# ---------------------------
def compute_positions(buses_df: pd.DataFrame, lines_df: pd.DataFrame):
    """Return dict {bus_id: (x,y)}. If x,y present, use them; else spring layout (rescaled)."""
    if {"x", "y"}.issubset(buses_df.columns):
        return {int(r["bus"]): (float(r["x"]), float(r["y"])) for _, r in buses_df.iterrows()}
    G = nx.Graph()
    for b in buses_df["bus"]:
        G.add_node(int(b))
    for _, row in lines_df.iterrows():
        G.add_edge(int(row["from_bus"]), int(row["to_bus"]))
    raw = nx.spring_layout(G, seed=42)

    xs = np.array([p[0] for p in raw.values()])
    ys = np.array([p[1] for p in raw.values()])
    def scale(v, vmin, vmax, a, b):
        return a + (v - vmin) * (b - a) / (vmax - vmin if vmax > vmin else 1.0)

    # Compact rectangle ~ (0..3) x (0.3..2.1)
    pos = {b: (scale(x, xs.min(), xs.max(), 0.0, 3.0),
               scale(y, ys.min(), ys.max(), 0.3, 2.1)) for b, (x, y) in raw.items()}
    return pos

# ---------------------------
# Build + solve function
# ---------------------------
def build_and_solve(buses, generations, loads, lines, slack_bus_ext, solver_choice):
    # Schema checks
    required_cols = {
        "buses": {"bus", "name"},
        "generations": {"gen", "bus", "pgmax", "cost"},
        "loads": {"load_id", "bus", "load"},
        "lines": {"line", "from_bus", "to_bus", "plmax", "Bl"},
    }
    for tab, req in required_cols.items():
        df = {"buses": buses, "generations": generations, "loads": loads, "lines": lines}[tab]
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"{tab}: missing columns {sorted(miss)}")

    buses = buses.reset_index(drop=True)
    generations = generations.reset_index(drop=True)
    loads = loads.reset_index(drop=True)
    lines = lines.reset_index(drop=True)

    Nb, Ng, Nl = len(buses), len(generations), len(lines)
    ext2int = {int(buses.loc[i, "bus"]): i for i in range(Nb)}
    def b_int(ext):
        if ext not in ext2int: raise ValueError(f"Unknown bus id {ext}")
        return ext2int[ext]

    g_bus  = {g: b_int(int(generations.loc[g, "bus"])) for g in range(Ng)}
    pgmax  = {g: float(generations.loc[g, "pgmax"]) for g in range(Ng)}
    gcost  = {g: float(generations.loc[g, "cost"])  for g in range(Ng)}
    line_f = {l: b_int(int(lines.loc[l, "from_bus"])) for l in range(Nl)}
    line_t = {l: b_int(int(lines.loc[l, "to_bus"]))   for l in range(Nl)}
    plmax  = {l: float(lines.loc[l, "plmax"]) for l in range(Nl)}
    Bl     = {l: float(lines.loc[l, "Bl"])    for l in range(Nl)}

    Pd = {i: 0.0 for i in range(Nb)}
    for _, row in loads.iterrows():
        Pd[b_int(int(row["bus"]))] += float(row["load"])

    slack_int = b_int(int(slack_bus_ext))

    # Pyomo model
    model = pyo.ConcreteModel()
    model.Pg = pyo.Var(range(Ng), within=pyo.Reals, bounds=(0, None))
    model.Pl = pyo.Var(range(Nl), within=pyo.Reals)
    model.theta = pyo.Var(range(Nb), within=pyo.Reals, bounds=(-np.pi, np.pi))

    Pg, Pl, theta = model.Pg, model.Pl, model.theta
    model.obj = pyo.Objective(expr=sum(Pg[g]*gcost[g] for g in range(Ng)), sense=minimize)

    model.balance = pyo.ConstraintList()
    for n in range(Nb):
        sum_Pg  = sum(Pg[g] for g in range(Ng) if g_bus[g]==n)
        sum_Pls = sum(Pl[l] for l in range(Nl) if line_f[l]==n)
        sum_Plr = sum(Pl[l] for l in range(Nl) if line_t[l]==n)
        model.balance.add(expr=sum_Pg - sum_Pls + sum_Plr == Pd[n])

    model.flux = pyo.ConstraintList()
    for l in range(Nl):
        model.flux.add(expr = Pl[l] == Bl[l]*(theta[line_f[l]] - theta[line_t[l]]))

    model.limger = pyo.ConstraintList()
    for g in range(Ng):
        model.limger.add(inequality(0, Pg[g], pgmax[g]))

    model.limflux = pyo.ConstraintList()
    for l in range(Nl):
        model.limflux.add(inequality(-plmax[l], Pl[l], plmax[l]))

    model.ref = pyo.Constraint(expr = theta[slack_int] == 0.0)

    # Solve
    if solver_choice.startswith("GLPK"):
        opt = pyo.SolverFactory("glpk")
        res = opt.solve(model, tee=False)
        status = str(res.solver.termination_condition)
    else:
        opt = Highs()
        res = opt.solve(model)
        status = str(res.solver.termination_condition)

    # Results (external ids)
    obj = float(pyo.value(model.obj))
    Pg_sol = {}
    for g in range(Ng):
        bus_ext = int(generations.loc[g, "bus"])
        Pg_sol[bus_ext] = Pg_sol.get(bus_ext, 0.0) + float(pyo.value(Pg[g]))
    Pl_sol = {int(lines.loc[l, "line"]): float(pyo.value(Pl[l])) for l in range(Nl)}
    theta_sol = {int(buses.loc[n, "bus"]): float(pyo.value(theta[n])) for n in range(Nb)}

    return status, obj, Pg_sol, Pl_sol, theta_sol

# ---------------------------
# Helper: base font size from free space
# ---------------------------
def auto_base_font(pos):
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max((max(xs) - min(xs)), (max(ys) - min(ys)), 1e-6)
    # Smaller span => larger font; cap between 6 and 11
    base = int(np.clip(10.0 / span, 6, 11))
    return base

# ---------------------------
# Diagram (buses = circles, thin arrows, color-coded, smart labels)
# ---------------------------
def draw_diagram(buses, generations, loads, lines, Pg_sol, Pl_sol, theta_sol, obj, pos, show_pg_star_blue=True):
    names = {int(r["bus"]): str(r["name"]) for _, r in buses.iterrows()}
    gen_at_bus = {int(row["bus"]): (float(row["pgmax"]), float(row["cost"]))
                  for _, row in generations.iterrows()}
    Pd_at_bus = loads.groupby("bus")["load"].sum().to_dict()

    Plmax_on_line = {int(row["line"]): float(row["plmax"]) for _, row in lines.iterrows()}
    ends_on_line  = {int(row["line"]): (int(row["from_bus"]), int(row["to_bus"]))
                     for _, row in lines.iterrows()}

    # Compact figure
    fig = plt.figure(figsize=(5.0, 3.0), dpi=170)
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
    ax.axis("off")

    # Auto font size from free space
    base_font = auto_base_font(pos)

    # Thinner arrows
    abs_flows = [abs(v) for v in Pl_sol.values()] or [1.0]
    fmax = max(abs_flows)
    def lw(flow):
        return 1.0 + 1.0 * (abs(flow) / (fmax if fmax > 0 else 1.0))

    # Collect texts for optional adjustText
    texts = []

    # Lines with orientation-aware label offsets + bbox
    for lid, flow in sorted(Pl_sol.items()):
        i, j = ends_on_line[lid]
        (x0, y0), (x1, y1) = pos[i], pos[j]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=lw(flow), color="#DD571C"))

        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = (x1 - x0), (y1 - y0)

        # Offset labels depending on line orientation
        if abs(dx) >= abs(dy):   # mostly horizontal
            offs_name = (0.0, +0.12)
            offs_pl   = (0.0, +0.02)
            offs_lim  = (0.0, -0.10)
        else:                    # mostly vertical
            offs_name = (+0.12, 0.0)
            offs_pl   = (+0.12, -0.12)
            offs_lim  = (+0.12, -0.24)

        t1 = ax.text(xm + offs_name[0], ym + offs_name[1], f"L{lid}",
                     ha="center", va="bottom", fontsize=base_font,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
        t2 = ax.text(xm + offs_pl[0],   ym + offs_pl[1],   f"Pl = {flow:.2f} kW",
                     ha="center", va="center",color="teal", fontsize=base_font,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
        t3 = ax.text(xm + offs_lim[0],  ym + offs_lim[1],  f"limit = {Plmax_on_line[lid]:.0f} kW",
                     ha="center", va="top", fontsize=max(6, base_font-1), color="red",
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
        texts += [t1, t2, t3]

    # Buses as circles + labels + angles (with small bbox)
    for b, (x, y) in pos.items():
        node = plt.Circle((x, y), 0.07, color="black")
        ax.add_patch(node)

        tb = ax.text(x - 0.14, y - 0.02, names.get(b, f"B{b}"),
                     fontsize=base_font, ha="right", va="center",
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))
        tth = ax.text(x + 0.12, y + 0.02, f"θ = {theta_sol.get(b, 0):.3f}",
                      fontsize=max(6, base_font-1), ha="left", va="bottom",
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))
        texts += [tb, tth]

        # Generator constraints (green) + Pg* (optionally blue)
        if b in gen_at_bus:
            pgmax, cg = gen_at_bus[b]
            t_pgmax = ax.text(x, y + 0.45, f"Pg ≤ {pgmax:.0f} kW",
                              ha="center", fontsize=base_font, color="red",
                              bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))
            t_cg = ax.text(x, y + 0.32, f"Cg = {cg:.2f} $/kWh",
                           ha="center", fontsize=base_font, color="brown",
                           bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))
            color_pg = "#0a68cc" if show_pg_star_blue else "black"
            t_pgstar = ax.text(x, y + 0.20, f"Pg* = {Pg_sol.get(b, 0):.2f} kW",
                               ha="center", fontsize=base_font, color=color_pg,
                               bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))
            texts += [t_pgmax, t_cg, t_pgstar]

        # Load arrow and Pd (dark red), with small connector bar
        Pd = float(Pd_at_bus.get(b, 0.0))
        if Pd > 0:
            ax.plot([x - 0.10, x + 0.10], [y - 0.12, y - 0.12], color="black", lw=1.3)
            ax.annotate("", xy=(x, y - 0.35), xytext=(x, y - 0.12),
                        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))
            t_pd = ax.text(x, y - 0.45, f"Pd = {Pd:.0f} kW",
                           ha="center", fontsize=base_font, color="#b23a48",
                           bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))
            texts.append(t_pd)

    # Optimal cost (top-right)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    t_cost = ax.text(max(xs) + 0.35, max(ys) + 0.35, f"Optimal Cost:  ${obj:,.4f}",
                     ha="right", va="top", fontsize=base_font+1, color="#1f4b99",
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    # Optional: collision-avoidance if adjustText is available
    if _HAS_ADJUST:
        try:
            adjust_text(texts + [t_cost], ax=ax, expand_points=(1.2, 1.2), force_text=(0.2, 0.2),
                        arrowprops=dict(arrowstyle="-", lw=0.5, color='gray'))
        except Exception:
            pass

    plt.tight_layout(pad=0.4)
    st.pyplot(fig, clear_figure=True)

# ---------------------------
# Solve
# ---------------------------
solve_btn = st.button("Solve")
st.markdown(
    '<a href="https://github.com/KaziArman/Optimal-Power-Supply/blob/a8861ea5741cd8181dbd93f089b6dc8766ffd720/power%20flow%20solution%20gurobipy.py" target="_blank">**Gurobipy Code**</a>'
    ' '
    '<a href="https://github.com/KaziArman/Optimal-Power-Supply/blob/a8861ea5741cd8181dbd93f089b6dc8766ffd720/power%20flow%20solution%20pyomo.py" target="_blank">**Pyomo Code**</a>'
    ' '
    '<a href="https://en.wikipedia.org/wiki/Optimal_power_flow" target="_blank">Data File</a>',
    unsafe_allow_html=True
)

if solve_btn:
    try:
        with st.spinner("Solving…"):
            status, obj, Pg_sol, Pl_sol, theta_sol = build_and_solve(
                buses, generations, loads, lines, slack_bus_ext=slack_bus, solver_choice=solver_choice
            )

        st.success(f"Solve status: {status}")
        st.info(f"Optimal objective (total generation cost): **${obj:,.4f}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Pg* (kW)")
            st.dataframe(pd.DataFrame(sorted(Pg_sol.items()), columns=["Bus", "Pg*"]), use_container_width=True)
        with col2:
            st.subheader("Pl* (kW)")
            st.dataframe(pd.DataFrame(sorted(Pl_sol.items()), columns=["Line", "Pl*"]), use_container_width=True)
        with col3:
            st.subheader("θ* (rad)")
            st.dataframe(pd.DataFrame(sorted(theta_sol.items()), columns=["Bus", "θ*"]), use_container_width=True)

        # Auto positions (or use buses.x,y if provided)
        pos = compute_positions(buses, lines)

        st.subheader("Diagram with Optimal Values")
        draw_diagram(buses, generations, loads, lines, Pg_sol, Pl_sol, theta_sol, obj, pos,
                     show_pg_star_blue=show_pg_star_blue)

    except Exception as e:
        st.error("Failed to solve. Please check your inputs.")
        st.exception(e)
else:
    st.info("Adjust the tables above and click **Solve**.")
