import os
import multiprocessing

# Force spawn immediately, BEFORE any other imports run
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

def _set_ray(method, max_devices):
    import ray
    from hydra.utils import get_original_cwd
    import jax
    if not method == 'direct'  and not method == 'monolithic':
        ray.init(
            _node_ip_address="127.0.0.1",  
            include_dashboard=False, 
            runtime_env={"working_dir": get_original_cwd(), 'excludes': ['/multirun/', '/outputs/', '/config/', '../.git/']},
            num_cpus=min(max_devices, multiprocessing.cpu_count())) 
        logging.info(f"Ray initialized with {ray.available_resources()['CPU']} CPUs.")
    return None
        

def _kill_ray():
    import ray
    if ray.is_initialized():
        ray.shutdown()
        logging.info("Ray shutdown successfully.")
    return None

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

@hydra.main(config_path="config", config_name="integrator")
def main(cfg: DictConfig) -> None:
    # Configure logging level from config
    _set_log(cfg.log_level)
    _set_ray(cfg.method, cfg.max_devices)
    max_devices = _set_jax(cfg.max_devices)

    from mu_F.direct.constructor import apply_direct_method
    from mu_F.decomposition import decomposition, decomposition_constraint_tuner
    from mu_F.constraints.constructor import constraint_evaluator
    from mu_F.cs_assembly import case_study_constructor, make_markov
    from mu_F.utils import save_graph
    

    # Construct the case study graph
    G = case_study_constructor(cfg)
    cfg = make_markov(cfg)

    # Save the graph to a file
    save_graph(G.copy(), "initial")

    # identify constraint sets
    if cfg.method == 'decomposition':
        # iterate over the modes defined in the config file
        mode = cfg.case_study.mode
        # getting precedence order
        precedence_order = list(nx.topological_sort(G))
        # run the decomposition
        G = decomposition(cfg, G, precedence_order, mode, max_devices).run()
        # finished decomposition                    
    elif cfg.method in ['direct', 'monolithic']:
        # run the decomposition
        outs = apply_direct_method(cfg, G, method=cfg.method)
        logging.info(f"Direct method {cfg.method} completed with outputs: {outs}")
        save_graph(G.copy(), f'{cfg.method}_complete')
    elif cfg.method == 'decomposition_constraint_tuner':
        decomposition_constraint_tuner(cfg, G, max_devices)

    else:
        # raise an error
        raise ValueError("Method not recognised")
    
    _kill_ray()
        
    # Log the function evaluations for each node in the graph.
    for node in G.nodes():
        logging.info(f"Function evaluations for node {node}: {G.nodes[node]['fn_evals']}")

    return G


if __name__ == "__main__":
    
    
    import sys  

    # Enable 64 bit floating point precision
    #jax.config.update("jax_enable_x64", True)
    sys.path.append(os.path.join(os.getcwd(),'src'))
    # run the program
    main()
    print("Done")