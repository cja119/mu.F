"""Entry point: build the case study graph and dispatch to the chosen solver method."""
import os
import multiprocessing

# Force spawn before any other imports run.
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# Set JAX/Ray flags before JAX inits
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import logging
import hydra
from omegaconf import DictConfig
import networkx as nx


"""
TODO :
- visualisation of probability maps
- extensive unit tests
- documentation
"""

def _set_log(level):
    """Set the logging level for the application."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s")

def _set_jax(max_devices):
    import jax
    import os
    
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={max_devices}"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    jax.config.update('jax_platform_name', 'cpu')
    n_jax_devices = len(jax.devices())
    logging.info(f"Jax maximum devices set to: {n_jax_devices}")
    return n_jax_devices

_DECOMP_METHODS = {'decomposition', 'decomposition_constraint_tuner'}
_MONO_METHODS   = {'direct', 'single_shooting', 'multiple_shooting'}


def _select_solver_block(cfg: DictConfig) -> DictConfig:
    """
    Promote the method-specific solver block to cfg.solvers so downstream
    consumers read cfg.solvers.* flat.
    """
    if cfg.method in _DECOMP_METHODS:
        cfg.solvers = cfg.solvers.decomposition
    elif cfg.method in _MONO_METHODS:
        cfg.solvers = cfg.solvers.monolithic
    else:
        raise ValueError(
            f"Unknown method {cfg.method!r}; expected one of "
            f"{sorted(_DECOMP_METHODS | _MONO_METHODS)}"
        )
    return cfg


@hydra.main(config_path="config", config_name="integrator")
def main(cfg: DictConfig) -> None:
    import multiprocessing

    total_devices = multiprocessing.cpu_count()
    cfg.max_devices = min(cfg.max_devices, total_devices)
    

    _set_log(cfg.log_level)
    cfg = _select_solver_block(cfg)
    max_devices = _set_jax(cfg.max_devices)

    from mu_F.direct.constructor import apply_direct_method
    from mu_F.decomposition import Decomposition, decomposition_constraint_tuner
    from mu_F.constraints.constructor import ConstraintEvaluator
    from mu_F.cs_assembly import case_study_constructor, make_markov
    from mu_F.utils import save_graph
    

    # Construct the case study graph.
    G = case_study_constructor(cfg)
    cfg = make_markov(cfg)

    save_graph(G.copy(), "initial")

    # Dispatch to the solver method named in the config.
    if cfg.method == 'decomposition':
        mode = cfg.case_study.mode
        precedence_order = list(nx.topological_sort(G))
        G = Decomposition(cfg, G, precedence_order, mode, max_devices).run()
    elif cfg.method in ['direct', 'single_shooting', 'multiple_shooting']:
        outs = apply_direct_method(cfg, G, method=cfg.method)
        logging.info(f"Direct method {cfg.method} completed with outputs: {outs}")
        save_graph(G.copy(), f'{cfg.method}_complete')
    elif cfg.method == 'decomposition_constraint_tuner':
        decomposition_constraint_tuner(cfg, G, max_devices)

    else:
        raise ValueError("Method not recognised")

    # Log the function evaluations for each node in the graph.
    for node in G.nodes():
        logging.info(f"Function evaluations for node {node}: {G.nodes[node]['fn_evals']}")

    return G


if __name__ == "__main__":
    
    
    import sys

    #jax.config.update("jax_enable_x64", True)
    sys.path.append(os.path.join(os.getcwd(),'src'))
    # run the program
    main()
    print("Done")