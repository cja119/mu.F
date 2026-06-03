"""Plot node-wise rollout trajectories (the live uncertain paths) against the monolithic optimum."""

import math
import logging
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import ListConfig


# Per-series plotting style: label, colour, marker, linestyle.
_SERIES = {
    'rollout':    ('Rollout',    'b', 'o', '-'),
    'monolithic': ('Monolithic', 'r', 's', '--'),
}

# Saved live rollout tensor and the fallback single-policy files.
_TRAJECTORY_FILE = 'rollout_trajectories_iterate_0.npz'
_POLICY_FILES = {
    'rollout':    ('rollout_policy_iterate_0.xlsx',),
    'monolithic': ('monolithic_policy.xlsx',),
}

# Most trajectories drawn for an uncertain rollout.
_MAX_TRAJECTORIES = 10


# ---------------------------------------------------------------------------
# Policy / trajectory loading
# ---------------------------------------------------------------------------

def _policy_path(run_dir, names):
    """Return the first existing candidate file in run_dir, or None."""
    for name in names:
        path = Path(run_dir) / name
        if path.exists():
            return path
    return None


def _parse_policy(path, n_design_per_node):
    """Split a one-row policy xlsx into per-node decision vectors by design-arg count."""
    values = pd.read_excel(path, index_col=0).iloc[0].to_numpy(dtype=float)   # design-space order
    per_node, offset = {}, 0
    for node, n_d in enumerate(n_design_per_node):
        per_node[node] = values[offset:offset + n_d]
        offset += n_d
    return per_node


def _root_state(cfg):
    """Flat initial state from the root node inputs, falling back to zeros when unset."""
    root = cfg.model.root_node_inputs
    root = root[0] if isinstance(root[0], (list, tuple, ListConfig)) else root
    if root in (None, 'None'):
        return np.zeros(int(cfg.case_study.sizes.F_SIZE))
    return np.ravel(np.asarray(root, dtype=float))


def _subsample(n, limit):
    """Evenly-spaced trajectory indices, at most `limit` of the available n."""
    return np.linspace(0, n - 1, min(limit, n)).astype(int)


def _captured_trajectories(path):
    """Load the live per-trajectory rollout, subsampled to at most _MAX_TRAJECTORIES paths."""
    data = np.load(path)
    idx = _subsample(data['states'].shape[0], _MAX_TRAJECTORIES)
    return {'decisions': data['decisions'][idx], 'states': data['states'][idx],
            'constraints': data['constraints'][idx], 'costs': data['costs'][idx][..., None]}


def _simulated_trajectory(graph, cfg, x0, policy_path, n_design, n_nodes):
    """Forward-simulate one saved policy into the shared (1, n_nodes, .) layout."""
    traj = _simulate_policy(graph, cfg, x0, _parse_policy(policy_path, n_design), n_nodes)
    return {key: np.asarray(value)[None, ...] for key, value in traj.items()}


def _load_trajectories(cfg, graph, run_dir):
    """Build per-series tensors: the live rollout (npz, else re-simulated) and the monolithic optimum."""
    n_nodes = int(cfg.case_study.num_nodes)
    n_design = [int(graph.nodes[node]['n_design_args']) for node in range(n_nodes)]
    x0 = _root_state(cfg)

    trajs = {}
    captured = _policy_path(run_dir, (_TRAJECTORY_FILE,))
    rollout_policy = _policy_path(run_dir, _POLICY_FILES['rollout'])
    if captured is not None:
        trajs['rollout'] = _captured_trajectories(captured)
    elif rollout_policy is not None:
        trajs['rollout'] = _simulated_trajectory(graph, cfg, x0, rollout_policy, n_design, n_nodes)

    monolithic_policy = _policy_path(run_dir, _POLICY_FILES['monolithic'])
    if monolithic_policy is not None:
        trajs['monolithic'] = _simulated_trajectory(graph, cfg, x0, monolithic_policy, n_design, n_nodes)
    return trajs


# ---------------------------------------------------------------------------
# Forward re-simulation
# ---------------------------------------------------------------------------

def _simulate_node(graph, node, state, decision, sizes):
    """Evaluate one node's forward model; returns (next_state, constraints, stage_cost)."""
    f_size, g_size, l_size = sizes
    design_args = jnp.asarray(decision, dtype=float).reshape(1, -1)           # (1, n_design)
    input_args = jnp.asarray(state, dtype=float).reshape(1, 1, -1)            # (1, 1, F_SIZE)
    aux_args = jnp.empty((1, 0))
    unc = jnp.asarray(graph.nodes[node]['parameters_best_estimate']).reshape(1, -1)

    flat = jnp.ravel(graph.nodes[node]['forward_evaluator'].evaluate(design_args, input_args, aux_args, unc))
    next_state = np.asarray(flat[:f_size])                                    # F block
    constraints = np.asarray(flat[f_size:f_size + g_size])                    # G block
    stage_cost = float(flat[f_size + g_size]) if l_size > 0 else 0.0          # first L entry
    return next_state, constraints, stage_cost


def _simulate_policy(graph, cfg, x0, per_node_decisions, n_nodes):
    """Roll one policy forward through every node and collect the node-wise trajectory."""
    sizes = (int(cfg.case_study.sizes.F_SIZE),
             int(cfg.case_study.sizes.G_SIZE),
             int(cfg.case_study.sizes.L_SIZE))
    state = np.asarray(x0, dtype=float)
    decisions, states, constraints, costs = [], [], [], []

    for node in range(n_nodes):
        states.append(state)
        decisions.append(np.asarray(per_node_decisions[node], dtype=float))
        state, g_vals, cost = _simulate_node(graph, node, state, per_node_decisions[node], sizes)
        constraints.append(g_vals)
        costs.append(cost)

    return {'decisions': np.array(decisions), 'states': np.array(states),
            'constraints': np.array(constraints), 'costs': np.array(costs).reshape(-1, 1)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _decision_labels(cfg, u):
    """Decision-axis labels from process_space_names, cycled to u entries."""
    names = list(cfg.case_study.process_space_names)
    if len(names) >= u:
        return names[:u]
    return (names * (u // len(names) + 1))[:u]


def _panel_specs(trajs, cfg):
    """Ordered (key, column, ylabel, title, is_constraint) tuples, one per trajectory variable."""
    sample = next(iter(trajs.values()))
    u, f, g = sample['decisions'].shape[2], sample['states'].shape[2], sample['constraints'].shape[2]
    labels = _decision_labels(cfg, u)

    specs = [('decisions', i, labels[i], 'Decisions' if i == 0 else None, False) for i in range(u)]
    specs += [('states', i, f"$x_{{{i}}}$", 'State variables' if i == 0 else None, False) for i in range(f)]
    specs += [('constraints', i, f"$g_{{{i}}}$",
               'Constraint values  (feasible ≥ 0)' if i == 0 else None, True) for i in range(g)]
    specs += [('costs', 0, 'Stage cost', 'Stage cost', False)]
    return specs


def _draw_series(ax, nodes, lines, style):
    """Plot one series' trajectories under a single legend label; markers only when a single path."""
    label, colour, marker, linestyle = style
    multi = lines.shape[0] > 1
    for index, row in enumerate(lines):
        ax.plot(nodes, row, color=colour, linewidth=1.0 if multi else 1.8,
                marker=None if multi else marker, markersize=4, linestyle=linestyle,
                alpha=0.45 if multi else 1.0, label=label if index == 0 else None)


def _draw_panel(ax, nodes, series, ylabel, title, is_constraint):
    """Overlay every present series (each one or more trajectories) on a single axis."""
    for name, lines in series.items():
        _draw_series(ax, nodes, lines, _SERIES[name])
    if is_constraint:
        ax.axhline(0.0, color='k', linewidth=0.8, linestyle=':')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=7, loc='best')
    if title is not None:
        ax.set_title(title, fontsize=9, loc='left', pad=2)


def _build_figure(cfg, trajs):
    """Lay out one panel per variable and overlay every present series."""
    specs = _panel_specs(trajs, cfg)
    nodes = np.arange(int(cfg.case_study.num_nodes))

    n_cols = 2
    n_rows = math.ceil(len(specs) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.0 * n_cols, 2.2 * n_rows), sharex=True)
    axes_2d = np.asarray(axes).reshape(n_rows, n_cols)
    axes_flat = axes_2d.reshape(-1).tolist()

    for ax, (key, col, ylabel, title, is_con) in zip(axes_flat, specs):
        series = {name: traj[key][:, :, col] for name, traj in trajs.items()}
        _draw_panel(ax, nodes, series, ylabel, title, is_con)

    for ax in axes_flat[len(specs):]:
        ax.set_visible(False)
    for col in range(n_cols):
        axes_2d[n_rows - 1][col].set_xlabel('Node (time step)', fontsize=9)

    fig.suptitle('Rollout trajectory', fontsize=11, y=1.002)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_rollout_trajectory(cfg, graph, run_dir='.', save_path='rollout_trajectory.png'):
    """
    Plot the node-wise decision / state / constraint / stage-cost trajectories,
    overlaying up to ten uncertain rollout paths and the monolithic optimum.
    """
    trajs = _load_trajectories(cfg, graph, run_dir)
    if not trajs:
        return None

    fig = _build_figure(cfg, trajs)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    logging.info(f"Saved rollout trajectory plot to {save_path} ({', '.join(trajs)})")
    return fig
