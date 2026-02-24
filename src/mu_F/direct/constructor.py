"""
Direct method constructor
"""

from mu_F.direct.deterministic import DeterministicMonolithic
from mu_F.direct.sampling import DirectSampler

def apply_direct_method(cfg, graph, method='sampling'):
    """
    Applies the direct method to the given configuration and graph.
    """
    if method == 'sampling':
        return DirectSampler(cfg, graph).solve()
    elif method == 'monolithic':
        return DeterministicMonolithic(cfg, graph).solve()
    else:
        raise ValueError(f"Unknown direct method: {method}")