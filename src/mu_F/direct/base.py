"""Abstract base for direct (full-discretisation) solver implementations."""

from abc import ABC, abstractmethod
from dataclasses import replace

from omegaconf import OmegaConf

from mu_F.solvers.septal import DEFAULT_SQP_CONFIG


class SolveDirect(ABC):
    """Common interface for the monolithic direct solvers.

    Holds the cfg/graph and shared SQP-config plumbing, while leaving the
    model build, solve and solution-extraction steps to each subclass
    (single/multiple shooting, sampling).

    """

    # ---- External Methods ----

    def __init__(self, cfg, G):
        self.cfg = cfg
        self.G = G
        self.pos_feas = (
            True if cfg.samplers.notion_of_feasibility.lower() == "positive" else False
        )

    # ---- Private Methods ----

    def _monolithic_sqp_config(self):
        """
        SQPConfig for the single/multiple shooting paths, overriding the
        default with the monolithic solver block promoted to cfg.solvers.
        """
        s = self.cfg.solvers
        # anything under solvers.<method>.sqp passes straight through to septal
        extra = OmegaConf.to_container(s.sqp, resolve=True) if 'sqp' in s else {}

        return replace(
            DEFAULT_SQP_CONFIG,
            tol_stationarity=float(s.optimality_tol),
            tol_feasibility=float(s.feasibility_tol),
            max_iter=int(s.max_iter),
            use_exact_hessian=bool(s.use_exact_hessian),
            **extra,
        )

    # ---- Base Methods ----

    @abstractmethod
    def solve(self, problem_data, x0=None):
        """
        Solves the problem using the loaded solver and prepared model.
        """
        return None

    @abstractmethod
    def _load_solver(self):
        """
        Loads in the solver object.
        """
        return self._solver

    @abstractmethod
    def _prepare_model(self, graph):
        """
        Prepares the model for solving, building the monolithic NLP.
        """
        pass

    @abstractmethod
    def _get_solution(self, solver_output):
        """
        Extracts the solution from the solver output, post-processing as needed.
        """
        return None