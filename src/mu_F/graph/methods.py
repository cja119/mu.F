import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
from omegaconf import DictConfig



""" insert case study specific functions for graph edges here"""


""" tablet press methods """

@jit
def data_IO_1(steady_state_outputs):
    # get mass flowrate of api and exipient
    return (steady_state_outputs[-2:].reshape(1, -1) * steady_state_outputs[-3
    ].reshape(1, -1)).reshape(-1,)


@jit
def data_IO_2(steady_state_outputs):
    # get porosity
    return steady_state_outputs[-1].squeeze()

vmap_data_IO_1 = vmap(vmap(data_IO_1, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)
vmap_data_IO_2 = vmap(vmap(data_IO_2, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)



""" batch reactor methods """
@jit
def data_transform(dynamic_profile):
    x = dynamic_profile[:-1].reshape(
        1, -1
    )
    return x.squeeze()

vmap_data_transform = vmap(vmap(data_transform, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)

""" convex estimator methods """
@jit
def data_transform_cvx(dynamic_profile):
    return dynamic_profile


vmap_data_transform_cvx = vmap(vmap(data_transform_cvx, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)

@jit 
def affine_cs34(dynamic_profile):
    return dynamic_profile[0]

@jit 
def affine_cs35(dynamic_profile):
    return dynamic_profile[1]

vmap_cs34 = vmap(vmap(affine_cs34, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)
vmap_cs35 = vmap(vmap(affine_cs35, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)


def make_markov_edge_cfg(cfg):
    """cfg-driven edge slicer: slice F block ([..., :F_SIZE]) from the bundled rollout."""
    F_size = int(cfg.case_study.sizes.F_SIZE)
    return lambda rollout: rollout[..., :F_size]


def vmap_markov_edge_cfg(cfg):
    fn = make_markov_edge_cfg(cfg)
    return vmap(vmap(fn, in_axes=0, out_axes=0), in_axes=1, out_axes=1)


# -------------------------------------------------------------------------------- #
# ----- Per-case-study edge factories ------------------------------------------- #
# Same return contract as the generic markov edge (rollout -> F_slice).  Each
# factory body is the place to add saturation / clipping rules.  When
# `cfg.case_study.edge_clip` is a list of [lo, hi] per F-state-dim, the
# returned closure clamps the F slice to those bounds; otherwise it's a
# no-op delegate to the generic markov factory.
# -------------------------------------------------------------------------------- #

def _markov_edge_with_clip(cfg):
    """Edge function for case studies with bounds"""

    from omegaconf import OmegaConf, ListConfig, DictConfig

    F_size = int(cfg.case_study.sizes.F_SIZE)
    
    clip = getattr(cfg.case_study, "edge_clip", None)
    
    if clip is None or (isinstance(clip, str) and clip.lower() == "none"):
        return lambda rollout: rollout[..., :F_size]
    
    if isinstance(clip, (ListConfig, DictConfig)):
        clip = OmegaConf.to_container(clip, resolve=True)

    bounds = jnp.asarray(clip, dtype=jnp.float64)

    if bounds.ndim != 2 or bounds.shape != (F_size, 2):
        raise ValueError(
            f"edge_clip for {cfg.case_study.case_study} must be a list of "
            f"{F_size} [lo, hi] pairs; got shape {bounds.shape}"
        )
    lo = bounds[:, 0]
    hi = bounds[:, 1]

    eps = float(getattr(cfg.case_study, "edge_clip_eps", 1.0e-6))

    def _clip(rollout):
        x = rollout[..., :F_size]
        c = jnp.clip(x, lo, hi)
        return c + eps * (x - c)
    
    return _clip


def _vmap_with_clip(cfg):
    fn = _markov_edge_with_clip(cfg)
    return vmap(vmap(fn, in_axes=0, out_axes=0), in_axes=1, out_axes=1)


def make_cstr_edge(cfg):              return _markov_edge_with_clip(cfg)
def vmap_cstr_edge(cfg):              return _vmap_with_clip(cfg)

def make_waste_water_edge(cfg):       return _markov_edge_with_clip(cfg)
def vmap_waste_water_edge(cfg):       return _vmap_with_clip(cfg)

def make_hydrogen_export_edge(cfg):   return _markov_edge_with_clip(cfg)
def vmap_hydrogen_export_edge(cfg):   return _vmap_with_clip(cfg)

def make_biohydrogen_edge(cfg):       return _markov_edge_with_clip(cfg)
def vmap_biohydrogen_edge(cfg):       return _vmap_with_clip(cfg)


""" insert case study specific functions for constraints here"""
CS_edge_holder = {  'tablet_press': {(0,1): data_IO_1, (1,2): data_IO_2}, 'serial_mechanism_batch': {(0,1): data_transform},
                    'convex_estimator': {(0,5): data_transform_cvx, (1,5): affine_cs34, (2,5): affine_cs34,
                                            (3,5): data_transform_cvx, (4,5): data_transform_cvx},
                    'convex_underestimator': {(0,5): data_transform_cvx, (1,5): data_transform_cvx, (2,5): data_transform_cvx,
                                        (3,5): data_transform_cvx, (4,5): data_transform_cvx},
                    'affine_study': {(0,2): data_transform_cvx, (1,2): data_transform_cvx, (2,3): affine_cs34, (2,4): affine_cs35},
                    'cstr': make_cstr_edge,
                    'waste_water': make_waste_water_edge,
                    'hydrogen_export': make_hydrogen_export_edge,
                    'biohydrogen': make_biohydrogen_edge}

vmap_CS_edge_holder = {'tablet_press': {(0,1): vmap_data_IO_1, (1,2): vmap_data_IO_2}, 'serial_mechanism_batch': {(0,1): vmap_data_transform},
                       'estimator': {(0,5): vmap_data_transform_cvx, (1,5): vmap_data_transform_cvx, (2,5): vmap_data_transform_cvx,
                                            (3,5): vmap_data_transform_cvx, (4,5): vmap_data_transform_cvx},
                       'convex_estimator': {(0,5): vmap_data_transform_cvx, (1,5): vmap_data_transform_cvx, (2,5): vmap_data_transform_cvx,
                                            (3,5): vmap_data_transform_cvx, (4,5): vmap_data_transform_cvx},
                        'convex_underestimator': {(0,5): vmap_data_transform_cvx, (1,5): vmap_cs34, (2,5): vmap_cs34,
                                            (3,5): vmap_data_transform_cvx, (4,5): vmap_data_transform_cvx},
                        'affine_study': {(0,2): vmap_data_transform_cvx, (1,2): vmap_data_transform_cvx, (2,3): vmap_cs34, (2,4): vmap_cs35},
                        'cstr': vmap_cstr_edge,
                        'waste_water': vmap_waste_water_edge,
                        'hydrogen_export': vmap_hydrogen_export_edge,
                        'biohydrogen': vmap_biohydrogen_edge}

