"""Shared jaxtyping array-shape vocabulary for self-checking annotations across muF."""

import os

from jaxtyping import Array, Bool, Float, Int


# ---------------------------------------------------------------------------
# Opt-in runtime checker  (set MU_F_TYPECHECK=1; identity passthrough otherwise)
# ---------------------------------------------------------------------------
if os.environ.get("MU_F_TYPECHECK", "0") == "1":
    from beartype import beartype as _beartype
    from jaxtyping import jaxtyped as _jaxtyped

    def typecheck(fn):
        """Enforce the annotated array shapes/dtypes; wrap array-boundary functions."""
        return _jaxtyped(typechecker=_beartype)(fn)
else:
    def typecheck(fn):
        """No-op when checks are disabled, so production runs pay nothing."""
        return fn


# ---------------------------------------------------------------------------
# Axis legend  (named axes; the same name must match within a checked function)
# ---------------------------------------------------------------------------
#   N   samples / trajectories / live points       (leading batch)
#   S   uncertainty scenarios (theta broadcast)
#   F   coupled state passed between nodes          (sizes.F_SIZE)
#   X   full integrated state                       (sizes.X_SIZE)
#   U   per-node control / design                   (sizes.U_SIZE)
#   I   per-node input (incoming coupling)
#   G   per-node feasibility constraints            (sizes.G_SIZE)
#   Z   uncertain parameters                        (sizes.Z_SIZE)
#   L   stage cost / reward                         (sizes.L_SIZE)
#   D   flat joint decision vector (monolithic)


# ---------------------------------------------------------------------------
# Per-element (one sample)
# ---------------------------------------------------------------------------
State        = Float[Array, "F"]
Design       = Float[Array, "U"]
Inputs       = Float[Array, "I"]
Constraints  = Float[Array, "G"]
Uncertain    = Float[Array, "Z"]
Decision     = Float[Array, "D"]


# ---------------------------------------------------------------------------
# Batched (N samples)
# ---------------------------------------------------------------------------
DesignBatch      = Float[Array, "N U"]
InputBatch       = Float[Array, "N I"]
StateBatch       = Float[Array, "N F"]
ConstraintBatch  = Float[Array, "N G"]
UncertainBatch   = Float[Array, "N Z"]


# ---------------------------------------------------------------------------
# Batched with scenario axis (N samples x S scenarios)
# ---------------------------------------------------------------------------
StateScen        = Float[Array, "N S F"]
ConstraintScen   = Float[Array, "N S G"]
