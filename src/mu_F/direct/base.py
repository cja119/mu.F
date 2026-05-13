
from abc import ABC, abstractmethod
from dataclasses import replace

from mu_F.solvers.septal import DEFAULT_SQP_CONFIG


class SolveDirect(ABC):
    def __init__(self, cfg, G):
        self.cfg = cfg
        self.G = G
        self.pos_feas = (
            True if cfg.samplers.notion_of_feasibility.lower() == "positive" else False
        )

    def _monolithic_sqp_config(self):
        """SQPConfig for the single/multiple shooting paths.  Reads the
        monolithic solver block (promoted to `cfg.solvers` by
        `main._select_solver_block`) and overrides the default `SQPConfig`."""
        s = self.cfg.solvers
        return replace(
            DEFAULT_SQP_CONFIG,
            tol_stationarity=float(s.optimality_tol),
            tol_feasibility=float(s.feasibility_tol),
            max_iter=int(s.max_iter),
            use_exact_hessian=bool(s.use_exact_hessian),
        )

    @abstractmethod
    def solve(self, problem_data, x0=None):
        """
        Solves the problem using the loaded solver and prepared model.
        """
        return None

    @abstractmethod
    def _load_solver(self):
        """
        Loads in solver object
        """
        return self._solver
    
    @abstractmethod
    def _prepare_model(self, graph):
        """
        Prepares the model for solving. This is where the monolithic NLP will be built.
        """
        pass

    @abstractmethod
    def _get_solution(self, solver_output):
        """
        Extracts the solution from the solver output. This is where any necessary post-processing of the solution will be done.
        """
        return None