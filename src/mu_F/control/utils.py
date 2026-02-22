"""
Utils   
"""

# Standard Library Imports
from typing import List, Optional
from pathlib import Path

# Third Party Imports
import jax.numpy as jnp
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# Local Application Imports
from mu_F.visualisation.methods import plotting_format, reconstruction_plot, add_policy

# -------------------------------------------------------------------------------- #
# ------------------------------ Global Constants -------------------------------- #
# -------------------------------------------------------------------------------- #

OUTPUTS_DIR = str(Path(__file__).parent.parent) + "/outputs/{solve_date}/{solve_id}/"

PICKLE_FILE = "graph_{mode}_iterate_{i}_node_{n}.pickle"

HYDRA_CONFIG_FILE = ".hydra/config.yaml"

XSLX_file = "inside_samples_{mode}_iterate_{i}.xlsx"

COST_XLSX_FILE = "inside_costs_{mode}_iterate_{i}.xlsx"

POOL = "mp-ms"

CFG_UPDATE = {
    "parallelised": False,
    "n_starts": 10,
    "n_rejects": 100,
    "rejection_margin": 100,
    "casadi_ipopt_options": {
        "maxiter": 2000,
        "verbose": False,
        "tol": 1e-4,
        "options": {
            "verbose": 0,
            "maxiter": 2000,
            "disp": False,
        },
    },
}

ALL_CONSTANTS = {
    "OUTPUTS_DIR": OUTPUTS_DIR,
    "PICKLE_FILE": PICKLE_FILE,
    "HYDRA_CONFIG_FILE": HYDRA_CONFIG_FILE,
    "XSLX_file": XSLX_file,
    "COST_XLSX_FILE": COST_XLSX_FILE,
    "POOL": POOL,
    "CFG_UPDATE": CFG_UPDATE,
}

# -------------------------------------------------------------------------------- #
# --------------------------------- Agent Utils ---------------------------------- #
# -------------------------------------------------------------------------------- #

def _trajectory_plots(cfg, actions, action_names: Optional[List] = None):
    
    plotting_format()

    if len(actions) == 0:
        raise ValueError("No actions recorded. Call `act` before plotting trajectories.")

    action_dim = np.ravel(np.asarray(actions[0][1])).shape[0]

    y_labels = list(cfg.case_study.process_space_names[0])[:action_dim]
    if len(y_labels) < action_dim:
        y_labels += [f"action_{i}" for i in range(len(y_labels), action_dim)]

    if action_names is not None:
        assert (
            len(action_names) == action_dim
        ), "Length of action namespace must equal environment action dimension."
    else:
        action_names = list(cfg.case_study.process_space_names[0])[:action_dim]
        if len(action_names) < action_dim:
            action_names += [f"action_{i}" for i in range(len(action_names), action_dim)]

    trajectories = []
    action_labels = []

    for node, action in actions:
        action_vec = np.ravel(np.asarray(action, dtype=float))
        if action_vec.shape[0] != action_dim:
            raise ValueError(
                f"Inconsistent action size at node {node}. "
                f"Expected {action_dim}, got {action_vec.shape[0]}."
            )
        trajectories.append(action_vec)
        action_labels.append(f"t{node}")

    lower_bounds = np.full(action_dim, np.nan, dtype=float)
    upper_bounds = np.full(action_dim, np.nan, dtype=float)
    for node_bounds in cfg.case_study.KS_bounds.design_args:
        for i, bounds in enumerate(node_bounds[:action_dim]):
            lb, ub = bounds
            if lb != "None" and ub != "None":
                lb = float(lb)
                ub = float(ub)
                if np.isnan(lower_bounds[i]) or lb < lower_bounds[i]:
                    lower_bounds[i] = lb
                if np.isnan(upper_bounds[i]) or ub > upper_bounds[i]:
                    upper_bounds[i] = ub

    actions = np.vstack(trajectories)
    fig, axs = plt.subplots(action_dim, 1, figsize=(12, 3.5 * action_dim), squeeze=False)
    axs = axs.ravel()

    steps = np.arange(actions.shape[0])
    for i, ax in enumerate(axs):
        ax.plot(steps, actions[:, i], marker="o", linewidth=3, color='r')
        ax.set_xticks(steps)
        ax.set_xticklabels(action_labels, rotation=45, ha="right")
        ax.set_ylabel(str(y_labels[i]))
        ax.set_title(str(action_names[i]))
        if not np.isnan(lower_bounds[i]) and not np.isnan(upper_bounds[i]):
            ax.set_ylim(lower_bounds[i], upper_bounds[i])
            ax.axhline(y=lower_bounds[i], ls="--", linewidth=2, c="black", alpha=0.7)
            ax.axhline(y=upper_bounds[i], ls="--", linewidth=2, c="black", alpha=0.7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig

def _add_policy_to_plot(cfg, graph, actions, out_dir):
    
    data_frame = pd.read_excel(
        out_dir
        + XSLX_file.format(
            mode=cfg.case_study.mode[0],
            i=0,
        ),
        index_col=0,
    )
    data_frame = data_frame[cfg.case_study.design_space_dimensions]
    vis = reconstruction_plot(
        cfg,
        graph,
        data_frame,
        save=False,
        path=out_dir + "reconstructed_with_policy",
        include_decomposition=False,
    )

    cols = list(cfg.case_study.design_space_dimensions)
    col_to_idx = {name: i for i, name in enumerate(cols)}
    policy_vec = np.full(len(cols), np.nan, dtype=float)

    for node, action in actions:
        action_vec = np.ravel(action)
        node_dims = list(cfg.case_study.process_space_names[node])

        for dim_idx, value in enumerate(action_vec):
            if dim_idx >= len(node_dims):
                break

            base_name = node_dims[dim_idx]
            target_name = f"{base_name}_{node}"
            col_idx = col_to_idx.get(target_name, col_to_idx.get(base_name))
            if col_idx is not None:
                policy_vec[col_idx] = float(value)

    vis = add_policy(vis, policy_vec, cfg=cfg, color="r", marker="o", size=60)
    vis.savefig(out_dir + "reconstructed_with_policy.svg", dpi=300)
    return vis