
from abc import ABC, abstractmethod

class SolveDirect(ABC):
    def __init__(self, cfg, G):
        self.cfg = cfg
        self.G = G
        self.pos_feas = (
            True if cfg.samplers.notion_of_feasibility.lower() == "positive" else False
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