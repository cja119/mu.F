"""Logging, JAX backend setup, and config-prep helpers.

set_jax sets the platform env vars before importing jax (they are only read at
backend init), so jax must never be imported at module level in this file.
"""
import copy
import logging
import os

from omegaconf import DictConfig, ListConfig, OmegaConf


def set_log(level):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s")


def set_jax(device, max_devices):
    device = str(device).lower()
    if device == "gpu":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("XLA_FLAGS", None)                              # CPU-only flag
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")  # leave VRAM for torch
    elif device == "cpu":
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={max_devices}"
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        raise ValueError(f"cfg.device must be 'cpu' or 'gpu', got {device!r}")

    import jax
    devices = jax.devices()
    n_devices = min(len(devices), int(max_devices))
    logging.info(f"JAX backend={jax.default_backend()} | devices={devices} | using {n_devices}")
    return n_devices


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


def _as_methods(method):
    """
    Normalise the configured method into a list so a single solve and a
    chained multi-method solve share one driver loop.
    """
    if isinstance(method, (list, ListConfig)):
        return [str(m) for m in method]
    return [str(method)]


def _select_integration_block(cfg: DictConfig) -> DictConfig:
    """
    Promote a method-specific integration override onto cfg.model.integration
    when present (e.g. fixed-step for the monolithic, adaptive for sampling).
    """
    key = 'integration_monolithic' if cfg.method in _MONO_METHODS else 'integration_decomposition'
    override = OmegaConf.select(cfg.model, key)
    if override is not None:
        cfg.model.integration = override
    return cfg


def _prepare_cfg(pristine, method):
    """
    Copy the pristine config, fix the scalar method, and promote that
    method's solver and integration blocks for the downstream consumers.
    """
    cfg = copy.deepcopy(pristine)
    cfg.method = method
    return _select_integration_block(_select_solver_block(cfg))
