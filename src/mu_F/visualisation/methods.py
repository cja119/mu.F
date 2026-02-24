import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import logging
import re

DEFAULT_DPI = 20

def plotting_format():
    font = {"family": "serif", "weight": "bold", "size": 20}
    plt.rc("font", **font)  # pass in the font dict as kwargs
    plt.rc("axes", labelsize=25)  # fontsize of the x and y label
    plt.rc("axes", linewidth=3)
    plt.rc("axes", labelpad=30)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)

    return


def initializer_cp(df):
    plotting_format()
    fig = plt.figure(figsize=(35, 35))
    cols = df.columns
    pp = sns.PairGrid(
        df[cols],
        aspect=1.4
    )
    pp.map_diag(hide_current_axis)
    pp.map_upper(hide_current_axis)
    
    return pp

def get_ds_bounds(cfg, G):
    DS_bounds = [np.array(G.nodes[unit_index]['KS_bounds']) for unit_index in G.nodes if G.nodes[unit_index]['KS_bounds'][0][0] != 'None']
    if G.graph['aux_bounds'][0][0][0] != 'None': DS_bounds += [np.array(G.graph['aux_bounds']).reshape(np.array(G.graph['aux_bounds']).shape[0], np.array(G.graph['aux_bounds']).shape[2])]
    DS_bounds = np.vstack(DS_bounds)
    DS_bounds = pd.DataFrame(DS_bounds.T, columns=cfg.case_study.design_space_dimensions)
    return DS_bounds

def init_plot(cfg, G, pp= None, init=True, save=True):

    init_df = G.graph['initial_forward_pass']
    DS_bounds = get_ds_bounds(cfg, G)

    if init:
        pp = initializer_cp(init_df)
    else:
        assert pp is not None, "PairGrid object is None. Please provide a valid PairGrid object."
    
    pp.map_lower(sns.scatterplot, data=init_df, edgecolor="k", c="k", size=0.01, alpha=0.05,  linewidth=0.5)
        
    indices = zip(*np.tril_indices_from(pp.axes, -1))

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in DS_bounds.columns and y_var in DS_bounds.columns:
            ax.axvline(x=DS_bounds[x_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axvline(x=DS_bounds[x_var].iloc[1], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[1], ls='--', linewidth=3, c='black')

    if save: pp.savefig("initial_forward_pass.svg", dpi=DEFAULT_DPI)

    return pp

def add_policy(pp, policy_data, cfg=None, color="r", marker="o", size=60):
    """
    Overlay policy points on an existing PairGrid.
    """

    cols = list(cfg.case_study.design_space_dimensions)
    vec = np.ravel(policy_data)
    if vec.shape[0] < len(cols):
        vec = np.hstack([vec, np.full(len(cols) - vec.shape[0], np.nan)])
    else:
        vec = vec[:len(cols)]
    policy_df = pd.DataFrame([vec], columns=cols)

    # Manually overlay a single point per subplot to avoid seaborn re-plotting
    row = policy_df.iloc[0]
    indices = zip(*np.tril_indices_from(pp.axes, -1))
    for i, j in indices:
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        x_val = row.get(x_var, np.nan)
        y_val = row.get(y_var, np.nan)
        if not (np.isnan(x_val) or np.isnan(y_val)):
            ax.scatter(x_val, y_val, c=color, marker=marker, s=size, edgecolor="k", linewidth=0.5, zorder=5)
    return pp

def decompose_call(cfg, G, path, init=True):
    pp = init_plot(cfg, G, init=init, save=False)
    pp = decomposition_plot(cfg, G, pp, save=True, path=path)
    return pp


def decomposition_plot(cfg, G, pp, save=True, path='decomposed_pair_grid_plot'):
    # load live sets for each subproblem from the graph 
    inside_samples_decom = [pd.DataFrame({col:G.nodes[node]['live_set_inner'][:,i] for i, col in enumerate(cfg.case_study.process_space_names[node])}) for node in G.nodes]
    print('cols', [{i: col for i, col in enumerate(cfg.case_study.process_space_names[node])} for node in G.nodes])

    print("inside_samples_decom", inside_samples_decom)
    # just keep those variables with Ui in the column name # TODO update this to also receive the live set probabilities 
    inside_samples_decom = [in_[[col for col in in_.columns if f"N{i+1}" in col]] for (i,in_) in enumerate(inside_samples_decom)]
    
    print("inside_samples_decom", inside_samples_decom)
    if cfg.reconstruction.plot_reconstruction == 'probability_map':
        for i, is_ in enumerate(inside_samples_decom):
            is_['probability'] = G.nodes[i]['live_set_inner_prob'] # TODO update this to also receive the live set probabilities

    # Get the indices of the lower triangle of the pair grid plot
    indices = zip(*np.tril_indices_from(pp.axes, -1))

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        for is_ in inside_samples_decom:
            if x_var in is_.columns and y_var in is_.columns:
                print(f"Plotting {x_var} vs {y_var}")
                sns.scatterplot(x=x_var, y=y_var, data=is_, edgecolor="k", c='r', alpha=0.8, ax=ax)
        
    if save: pp.savefig(path +'.svg', dpi=DEFAULT_DPI)

    return pp

def reconstruction_with_policy_plot(cfg, G, reconstructed_df, policy_data, save=True, path='reconstructed_with_policy'):
    pp = initializer_cp(reconstructed_df)
    pp = init_plot(cfg, G, pp, init=False, save=False)
    pp = decomposition_plot(cfg, G, pp, save=False)
    pp.map_lower(sns.scatterplot, data=reconstructed_df, edgecolor="k", c="b", linewidth=0.5)
    pp = add_policy(pp, policy_data, cfg=cfg, color="r", marker="X", size=80)
    if save:
        pp.savefig(path + ".svg", dpi=DEFAULT_DPI)
    return pp


def _blank_reconstruction_df(cfg):
    cols = list(cfg.case_study.design_space_dimensions)
    return pd.DataFrame([{col: np.nan for col in cols}])


def _load_latest_inside_samples():
    pattern = re.compile(r"^inside_samples_.+_iterate_(\d+)\.xlsx$")
    candidates = []
    for fname in os.listdir("."):
        match = pattern.match(fname)
        if match:
            iterate = int(match.group(1))
            candidates.append((iterate, os.path.getmtime(fname), fname))

    if not candidates:
        return None, None

    _, _, latest_file = max(candidates, key=lambda x: (x[0], x[1]))
    df = pd.read_excel(latest_file, index_col=0)
    return df, latest_file


def _rollout_policy_from_graph(cfg, G):
    cols = list(cfg.case_study.design_space_dimensions)
    col_to_idx = {name: i for i, name in enumerate(cols)}
    policy_vec = np.full(len(cols), np.nan, dtype=float)

    for node in G.nodes:
        if 'rollout_action' not in G.nodes[node]:
            continue

        action_vec = np.ravel(np.asarray(G.nodes[node]['rollout_action'], dtype=float))
        node_dims = cfg.case_study.process_space_names[node]
        if not isinstance(node_dims, (list, tuple)):
            node_dims = [node_dims]

        for dim_idx, value in enumerate(action_vec):
            if dim_idx >= len(node_dims):
                break
            base_name = str(node_dims[dim_idx])
            stripped_name = base_name
            node_prefix = f"n{node}_"
            if stripped_name.startswith(node_prefix):
                stripped_name = stripped_name[len(node_prefix):]
            candidate_cols = [
                f"{base_name}_node_{node}",
                f"{base_name}_{node}",
                base_name,
                f"{stripped_name}_node_{node}",
                f"{stripped_name}_{node}",
                stripped_name,
            ]
            col_idx = next((col_to_idx[c] for c in candidate_cols if c in col_to_idx), None)
            if col_idx is not None:
                policy_vec[col_idx] = value

    return policy_vec


def rollout_with_policy_plot(cfg, G, policy_data=None, save=True, path='rollout_with_policy'):
    reconstructed_df, source = _load_latest_inside_samples()
    if reconstructed_df is None:
        logging.warning("No inside-samples file found. Falling back to blank reconstruction canvas.")
        reconstructed_df = _blank_reconstruction_df(cfg)
    else:
        logging.info(f"Loaded latest inside samples from {source}")

    cols = list(cfg.case_study.design_space_dimensions)
    missing_cols = [c for c in cols if c not in reconstructed_df.columns]
    for c in missing_cols:
        reconstructed_df[c] = np.nan
    reconstructed_df = reconstructed_df[cols]

    pp = initializer_cp(reconstructed_df)
    pp = init_plot(cfg, G, pp, init=False, save=False)

    try:
        pp = decomposition_plot(cfg, G, pp, save=False)
    except Exception as exc:
        logging.warning(f"Could not overlay decomposition live sets on rollout plot: {exc}")

    if not reconstructed_df.isna().all().all():
        pp.map_lower(sns.scatterplot, data=reconstructed_df, edgecolor="k", c="b", linewidth=0.5)

    policy_vec = _rollout_policy_from_graph(cfg, G) if policy_data is None else policy_data
    pp = add_policy(pp, policy_vec, cfg=cfg, color="r", marker="o", size=90)

    if save:
        pp.savefig(path + ".svg", dpi=DEFAULT_DPI)

    return pp

def reconstruction_plot(cfg, G, reconstructed_df, save=True, path='reconstructed_pair_grid_plot'):

    pp = initializer_cp(reconstructed_df)
    pp = init_plot(cfg, G, pp, init=False, save=False)
    pp = decomposition_plot(cfg, G, pp, save =False)
    pp.map_lower(sns.scatterplot, data=reconstructed_df, edgecolor="k", c="b", linewidth=0.5)

    if save: pp.savefig(path + ".svg", dpi=DEFAULT_DPI)

    return pp

def design_space_plot(cfg, G, joint_data_direct, path):

    # Pair-wise Scatter Plots
    pp = initializer_cp(joint_data_direct)
    DS_bounds = get_ds_bounds(cfg, G)
    
    indices = zip(*np.tril_indices_from(pp.axes, -1))

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in DS_bounds.columns and y_var in DS_bounds.columns:
            ax.axvline(x=DS_bounds[x_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axvline(x=DS_bounds[x_var].iloc[1], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[1], ls='--', linewidth=3, c='black')

    pp.map_lower(sns.scatterplot, data=joint_data_direct, edgecolor="k", c="b", linewidth=0.5)
    # Save the updated figure
    pp.savefig(path + ".svg", dpi=DEFAULT_DPI)

    return 


def design_space_plot_plus_polytope(cfg, G, pp, joint_data_direct, path, save=True):

    # Pair-wise Scatter Plots

    DS_bounds = get_ds_bounds(cfg, G)
    
    indices = zip(*np.tril_indices_from(pp.axes, -1))

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in DS_bounds.columns and y_var in DS_bounds.columns:
            ax.axvline(x=DS_bounds[x_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axvline(x=DS_bounds[x_var].iloc[1], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[0], ls='--', linewidth=3, c='black')
            ax.axhline(y=DS_bounds[y_var].iloc[1], ls='--', linewidth=3, c='black')

    pp.map_lower(sns.scatterplot, data=joint_data_direct, edgecolor="k", c="b", linewidth=0.5)
    # Save the updated figure
    if save: pp.savefig(path + ".svg", dpi=DEFAULT_DPI)

    return pp

def polytope_plot(pp, polytope):

    indices = zip(*np.tril_indices_from(pp.axes, -1))

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if (x_var,y_var) in list(polytope.keys()):
            ax.fill(polytope[(x_var,y_var)][0], polytope[(x_var,y_var)][1], alpha=0.5, color='red', edgecolor='black', linewidth=1.5)
    # Save the updated figure
    return pp

def polytope_plot_2(pp, polytope):
    from scipy.spatial import ConvexHull

    indices = zip(*np.tril_indices_from(pp.axes, -1))

    for i, j in indices: 
        x_var = pp.x_vars[j]
        y_var = pp.y_vars[i]
        ax = pp.axes[i, j]
        if x_var in list(polytope.keys()) and y_var in list(polytope.keys()) and x_var[:2] == y_var[:2]:
            points = np.hstack([np.array(polytope[x_var]).reshape(-1,1), np.array(polytope[y_var]).reshape(-1,1)]).reshape(-1, 2)
            hull = ConvexHull(points)
            hull_vertices = points[hull.vertices]
            # Unzip for plotting
            x_hull, y_hull = zip(*hull_vertices)
            ax.fill(x_hull, y_hull, alpha=0.5, color='red', edgecolor='black', linewidth=1.5)
    # Save the updated figure
    return pp

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
    return


def post_process_upper_solution(cfg, G, args):
    """
    Visualise the solution of the post-processing upper-level problem.
    :param cfg: Configuration object
    :param G: Graph object
    :param solution: Solution to be visualised
    :param path: Path to save the visualisation
    """
    solution, value_fn = args
    plotting_format()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Assuming solution is a DataFrame with columns: 'x', 'y', 'z'
    # where 'x' and 'y' are the 2D input variables and 'z' is the mapped 1D output
    xi, yi, zi = solution['x'], solution['y'], solution['z']
    # Plot the contour
    contour = ax.contourf(xi, yi, zi, vmin=0, vmax=np.max(zi), levels=20, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Point-wise error')

    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')

    plt.savefig("post_process_upper_solution.svg", dpi=DEFAULT_DPI)

    logging.info("Difference between predicted max point wise error and actual max point wise error: %s",
                 np.max(zi) - value_fn)
    
    return fig


def plot_contour(func, x_range, y_range, value_fn, path, num_points=200, levels=10):
    """
    Generates a contour plot for a given 2D function over a specified box domain.
    This version is designed for functions that can only be evaluated on a batch
    of points where the batch is the zeroth axis (e.g., func(x_coords, y_coords)
    where x_coords and y_coords are 1D arrays).

    Args:
        func (callable): The function to plot. It should accept two 1D arrays,
                        x and y, and return a single 1D array of results.
        x_range (tuple): A tuple (x_min, x_max) defining the range for the x-axis.
        y_range (tuple): A tuple (y_min, y_max) defining the range for the y-axis.
        num_points (int, optional): The number of points to use for the grid along
                                    each axis. A higher number results in a smoother
                                    plot. Defaults to 200.
        levels (int, optional): The number of contour lines or filled regions to display.
                                Defaults to 10.
    """
    # Create a grid of points for the x and y axes
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    
    # Use numpy.meshgrid to create the 2D grid from the 1D arrays
    X, Y = np.meshgrid(x, y)

    # Flatten the X and Y grids into 1D arrays for the batch evaluation
    # This creates a "batch" of all coordinate pairs to be evaluated
    x_coords_batch = X.ravel()
    y_coords_batch = Y.ravel()

    # Evaluate the input function on the batch of points
    # The function is expected to return a 1D array of results
    z_values_batch = func(x_coords_batch, y_coords_batch)

    # Reshape the 1D result back into a 2D array with the original grid shape
    Z = z_values_batch.reshape(X.shape)

    # Create the plot figure and axes
    fig, ax = plt.subplots(figsize=(15, 10))

    # --- Generate the contour plot ---
    # The 'contourf' function creates filled contour regions.
    # The 'cmap' parameter sets the colormap.
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
    
    # The 'contour' function creates the contour lines on top of the filled regions.
    # We use 'colors='k'' to make the lines black.
    ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.5)

    # --- Customize the plot ---
    # Add a color bar to show the mapping from color to Z-value
    fig.colorbar(cf, ax=ax, vmin=np.min(Z), vmax=np.max(Z), label='Point-wise error')

    # Add titles and labels
    #ax.set_title(f'Contour Plot of {func.__name__}', fontsize=16)
    ax.set_xlabel(f'r$x_1$', fontsize=12)
    ax.set_ylabel(f'r$x_2$', fontsize=12)
    ax.set_aspect('equal', adjustable='box') # Ensure the plot is not stretched

    logging.info("Difference between predicted max point wise error and actual max point wise error: %s",
                 np.max(Z) - value_fn)

    # Display the plot
    plt.savefig(os.path.join(path, "post_process_upper_solution.svg"), dpi=DEFAULT_DPI)
