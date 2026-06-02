"""Direct method constructor: dispatches to the requested discretisation scheme."""

from mu_F.direct.multiple_shooting import MultipleShooting
from mu_F.direct.single_shooting import SingleShooting
from mu_F.direct.sampling import DirectSampler

def apply_direct_method(cfg, graph, method='sampling'):
    """
    Applies the direct method to the given configuration and graph.
    """
    if method == 'sampling':
        return DirectSampler(cfg, graph).solve()
    elif method == 'single_shooting':
        return SingleShooting(cfg, graph).solve()
    elif method == 'multiple_shooting':
        return MultipleShooting(cfg, graph).solve()
    else:
        raise ValueError(f"Unknown direct method: {method}")