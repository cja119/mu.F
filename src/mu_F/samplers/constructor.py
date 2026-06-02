"""Sampler constructors wiring model callbacks into DEUS problems."""
from abc import ABC
from typing import Type
import pickle
import logging
import time
import numpy as np


class ConstructBase(ABC):
    """Abstract base for sampling-based problem constructors.

    Loads a model's callbacks into a sampler instance and exposes the
    solve / solution / regression hooks that concrete subclasses fill
    in for their specific solver.

    """

    # ---- External Methods ----

    def __init__(self, sampler_class, problem_description: dict, model):
        """
        Wire a model and its problem description into a sampler.
        """

        self.problem_description = problem_description
        self.model = model
        self.sampler_class =  sampler_class
        self.load_model_to_sampler()
        self.construct_problem()

    def construct_problem(self):
        """
        Instantiate the sampler from the problem description.
        """
        self.sampler = self.sampler_class(self.problem_description)
        return

    def load_model_to_sampler(self):
        """
        Solver-dependent hook; implemented by the derived class.
        """
        raise NotImplementedError("load_model_to_sampler should be implemented in the derived class")

    def solve(self):
        """
        Solver-dependent hook; implemented by the derived class.
        """
        raise NotImplementedError("solve should be implemented in the derived class")

    def get_solution(self):
        """
        Solver-dependent hook; implemented by the derived class.
        """
        raise NotImplementedError("get_solution should be implemented in the derived class")

    def get_regresssion_data(self):
        """
        Solver-dependent hook; implemented by the derived class.
        """
        raise NotImplementedError("get_regresssion_data should be implemented in the derived class")

    def get_function_evaluations(self):
        """
        Solver-dependent hook; implemented by the derived class.
        """
        raise NotImplementedError("get_function_evaluations should be implemented in the derived class")


class ConstructDeusProblem(ConstructBase):
    """Per-unit DEUS problem constructor.

    Loads the model's score / probability callbacks into the DEUS
    sampler and reads the pickled study back to return feasible and
    infeasible sample sets to the rest of the pipeline.

    """

    # ---- External Methods ----

    def __init__(self, sampler_class, problem_description: dict, model):
        super().__init__(sampler_class, problem_description, model)

    def load_model_to_sampler(self):
        """
        Bind score / EFP constraint callbacks to the DEUS settings.
        """
        settings = self.problem_description['solver']['settings']

        if settings['efp_evaluation'].get('direct_probability', False):
            settings['score_evaluation']['constraints_func_ptr'] = self.model.get_score_constraint
            settings['efp_evaluation']['constraints_func_ptr'] = self.model.get_probability
        else:
            settings['score_evaluation']['constraints_func_ptr'] = self.model.get_constraints
            settings['efp_evaluation']['constraints_func_ptr'] = self.model.get_constraints
        return

    def solve(self):
        """
        Run the sampler and log the wall-clock solve time.
        """
        t0 = time.time()
        self.sampler.solve()
        logging.info(f"Time taken to solve the problem: {time.time() - t0} seconds")
        return

    def get_log_evidence(self):
        """
        Return the log-evidence from the saved study.
        """
        pd = self.problem_description
        output = self.load_study(pd)
        return output["solution"]["log_z"]

    def get_solution(self):
        """
        Return feasible and infeasible samples from the saved study.
        """
        pd = self.problem_description
        output = self.load_study(pd)
        feasible_samples, infeasible_samples = self.return_solution(pd, output)
        return feasible_samples, infeasible_samples

    def get_regresssion_data(self, graph, str_='post_process_lower_'):
        """
        Push the model's regression data onto the graph.
        """
        self.model.desired_regressor_data.load_regression_data_to_graph(graph, str_=str_)
        return graph

    @staticmethod
    def load_study(problem_description):
        """
        Load the pickled DEUS output for this case.
        """
        cs_path = problem_description["activity_settings"]["case_path"]
        cs_name = problem_description["activity_settings"]["case_name"]

        with open(cs_path + '/' + cs_name + '/' + 'output.pkl', 'rb') \
                as file:
            output = pickle.load(file)

        return output

    @staticmethod
    def return_solution(problem_description, output):
        """
        Split saved samples into feasible / infeasible by target reliability.
        """
        samples = output["solution"]["probabilistic_phase"]["samples"]
        if samples["coordinates"] == []:
            samples = output["solution"]["deterministic_phase"]["samples"]
        n_d = len(samples["coordinates"][0])
        inside_samples_coords = np.empty((0, n_d))
        inside_samples_prob = []
        outside_samples_coords = np.empty((0, n_d))
        outside_sample_prob = []
        for i, phi in enumerate(samples["phi"]):
            if phi >= problem_description["problem"]["target_reliability"]:
                inside_samples_coords = np.append(inside_samples_coords,
                                                [samples["coordinates"][i]], axis=0)
                inside_samples_prob.append(samples["phi"][i])
            else:
                outside_samples_coords = np.append(outside_samples_coords,
                                                [samples["coordinates"][i]], axis=0)
                outside_sample_prob.append(samples["phi"][i])

        if inside_samples_coords.shape[0] == 0:
            logging.warning("Warning!! ---- No feasible samples found")

        return (inside_samples_coords, np.array(inside_samples_prob)), (outside_samples_coords, np.array(outside_sample_prob))


class ConstructDeusProblemNetwork(ConstructDeusProblem):
    """Whole-network variant of the DEUS problem constructor.

    Identical to ConstructDeusProblem but binds the network model's
    single direct_evaluate callback to both DEUS evaluation phases.

    """

    # ---- External Methods ----

    def __init__(self, sampler_class, problem_description: dict, model):
        super().__init__(sampler_class, problem_description, model)

    def load_model_to_sampler(self):
        """
        Bind the network direct_evaluate callback to both DEUS phases.
        """
        self.problem_description['solver']['settings']['score_evaluation']['constraints_func_ptr'] = self.model.direct_evaluate
        self.problem_description['solver']['settings']['efp_evaluation']['constraints_func_ptr'] = self.model.direct_evaluate
        return
