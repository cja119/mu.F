"""Gaussian-process Bayesian optimisation used to place backoff design points."""
import torch
import torch.optim as optim
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel
from gpytorch.distributions import MultivariateNormal
import numpy as np
import sobol_seq
from mu_F.utils import savefig_with_svg_limit

import logging


def generate_sobol_points(lower_bound, upper_bound, num_points=10):
    """
    Draw a Sobol sequence over the box and scale it into the bounds.
    """
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)

    dim = len(lower_bound)
    sample = sobol_seq.i4_sobol_generate(dim, num_points)

    points = lower_bound + (upper_bound - lower_bound) * sample
    return points


class GPRegressionModel(ExactGP):
    """Exact GP surrogate with a constant mean and RBF kernel.

    Fitted to the observed objective values and queried by the acquisition
    functions to propose the next design point.

    """

    # ---- External Methods ----

    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(C=train_y.mean())
        self.covar_module = RBFKernel()

    def forward(self, x):
        """
        Return the GP predictive distribution at the inputs.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def bayesian_optimization(f, lower_bound, upper_bound, num_initial_points, num_iterations, acq):
    """
    Run the BO loop: seed with Sobol points, then iteratively fit the GP
    and add the point chosen by the acquisition function.
    """
    logging.info("Starting Bayesian optimization...")
    logging.info(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    logging.info(f"Num initial points: {num_initial_points}, Num iterations: {num_iterations}")
    torch.set_default_dtype(torch.float32)
    # treat each Sobol point as a separate candidate
    train_x = torch.tensor(generate_sobol_points(lower_bound, upper_bound, num_initial_points)).float().T

    logging.info(f"Shape of train_x: {train_x.shape}")
    logging.info(f"x-values: {train_x}")

    if train_x.dim() == 1:
        train_x = train_x.unsqueeze(-1)
    logging.info(f"Shape of train_x: {train_x.shape}")

    # objective at each seed candidate
    train_y = torch.vstack([f(train_x[:, i].reshape(-1,1)).reshape(1,1) for i in range(train_x.shape[1])]).squeeze()

    likelihood = GaussianLikelihood()


    #train_x = (train_x - train_x.mean()) / train_x.std()
    #train_y = (train_y - train_y.mean()) / train_y.std()

    model = GPRegressionModel(train_x.T, train_y, likelihood)

    model, _ = fit_gpytorch_model(likelihood, model, train_x.T, train_y)

    logging.info('Initial min:', train_y.min().item())

    for _ in range(num_iterations):
        candidate_x = select_next_point(model, lower_bound, upper_bound, acq)
        candidate_y = f(candidate_x.reshape(-1,1)).reshape(-1,)

        train_x = torch.cat([train_x, candidate_x.T], dim=1)
        train_y = torch.cat([train_y, candidate_y])

        model.set_train_data(train_x.T, train_y, strict=False)

        # refit the GP on the augmented data
        likelihood = GaussianLikelihood()
        model = GPRegressionModel(train_x.T, train_y, likelihood)
        model, _ = fit_gpytorch_model(likelihood, model, train_x.T, train_y)
        print(model.mean_module.constant.item(), train_y.mean().item())

    # best observed point
    best_index = torch.argmin(train_y)
    best_candidate = train_x[:, best_index]

    logging.info(f'Final min: {train_y.min().item()}')
    logging.info(f'Best candidate: {best_candidate}')

    torch.save(train_y, 'opt_y.pt')
    torch.save(train_x, 'opt_x.pt')

    if train_x.shape[0] ==2 : plot(train_x, train_y, model, likelihood, upper_bound, lower_bound)

    return best_candidate, best_index


def fit_gpytorch_model(likelihood, model, train_x, train_y):
    """
    Fit GP hyperparameters by maximising the marginal log-likelihood
    with an LBFGS multi-start scheme, keeping the best run.
    """
    optimizer = optim.LBFGS(model.parameters(), lr=0.0001, line_search_fn='strong_wolfe')
    model.train()
    likelihood.train()

    lml = ExactMarginalLogLikelihood(likelihood, model)

    tolerance = 1e-6
    best_loss = float('inf')
    best_model = None
    best_likelihood = None

    # 10 random restarts
    for start in range(10):
        model = GPRegressionModel(train_x, train_y, likelihood)
        previous_loss = None

        for k in range(1000):
            def closure():
                optimizer.zero_grad()
                output = model(train_x)
                loss = -lml(output, train_y.squeeze())
                loss.backward()
                return loss

            optimizer.step(closure)

            loss = -lml(model(train_x), train_y).item()
            logging.info(f"GP lml, start {start}, iteration {k}, loss: {loss}")

            if previous_loss is not None:
                change_in_loss = abs(loss - previous_loss)
                if change_in_loss <= tolerance:
                    logging.info(f"Convergence achieved after {k+1} iterations for start {start}.")
                    break

            previous_loss = loss

        # keep the best restart
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
            best_likelihood = likelihood.state_dict()

    model.load_state_dict(best_model)
    likelihood.load_state_dict(best_likelihood)
    likelihood.eval()
    model.eval()

    return model, likelihood


def plot(train_x, train_y, model, likelihood, ub, lb):
    """
    Contour-plot the fitted GP mean over a 2D design space with the
    training points overlaid; only used when the design is two-dimensional.
    """
    import matplotlib.pyplot as plt
    def plotting_format():
        font = {"family": "serif", "weight": "bold", "size": 20}
        plt.rc("font", **font)  # pass in the font dict as kwargs
        plt.rc("axes", labelsize=15)  # fontsize of the x and y label
        plt.rc("axes", linewidth=3)
        plt.rc("axes", labelpad=20)
        plt.rc("xtick", labelsize=10)
        plt.rc("ytick", labelsize=10)

        return
    
    plotting_format()

    grid_size = 100
    x1 = torch.linspace(lb[0], ub[0], grid_size)
    x2 = torch.linspace(lb[1], ub[1], grid_size)

    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    grid_points = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1)  # (grid_size^2, 2)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(grid_points))
        mean = predictions.mean.reshape(grid_size, grid_size)

    plt.figure(figsize=(12, 10))
    contour = plt.contourf(X1.numpy(), X2.numpy(), mean.numpy(), levels=20, cmap='viridis')
    plt.colorbar(contour, label='Predicted Mean')

    plt.scatter(train_x[0,:].numpy(), train_x[1,:].numpy(),
                c='red', marker='x', label='Training Data')

    plt.title('Contour Plot of GP Model Mean')
    plt.xlabel('Unit 1 Backoff, $\epsilon_1$')
    plt.ylabel('Unit 2 Backoff , $\epsilon_2$')
    plt.legend()
    savefig_with_svg_limit(plt.savefig, 'Bayesian optimization iterations.svg')


def select_next_point(model, lower_bound, upper_bound, acq:str='ei'):
    """
    Dispatch to the requested acquisition function and return the point
    it proposes.
    """
    if acq == 'ucb':
        return original_ucb(model, lower_bound, upper_bound)
    elif acq == 'ei':
        return ei(model, lower_bound, upper_bound)
    else:
        raise ValueError(f"Invalid acquisition function: {acq}")


def ei(model, lower_bound, upper_bound):
    """
    Expected-improvement acquisition: minimise -EI from several Sobol
    starts with LBFGS and return the best feasible candidate.
    """
    bounds = torch.stack([torch.tensor(lower_bound), torch.tensor(upper_bound)], dim=1)  # (d, 2)

    def acquisition(x):
        model.eval()
        with gpytorch.settings.fast_pred_var():
            if x.dim() == 1:
                x = x.unsqueeze(0)

            pred = model(x)
            mean = pred.mean
            variance = pred.variance
            stddev = torch.sqrt(variance)

            # expected improvement below the best observed value
            f_best = torch.min(model.train_targets)
            z = (f_best - mean) / stddev
            ei = (f_best - mean) * torch.distributions.Normal(0, 1).cdf(z) + stddev * torch.distributions.Normal(0, 1).log_prob(z).exp()
            return -ei

    n_multi_start = 10
    n_grad_steps = 50
    points = generate_sobol_points(lower_bound, upper_bound, n_multi_start)
    minima = []
    sol_x  = []
    tolerance = torch.tensor([1e-8])
    previous_acquisition_value = None

    for i in range(n_multi_start):
        next_point = torch.tensor(points[i].reshape(1,-1)).float().requires_grad_(True)
        optimizer = optim.LBFGS([next_point], line_search_fn='strong_wolfe', lr=0.01)

        for j in range(n_grad_steps):
            def closure():
                optimizer.zero_grad()
                acquisition_value = acquisition(next_point)
                acquisition_value.backward()
                return acquisition_value

            optimizer.step(closure)
            with torch.no_grad(): next_point[:] = next_point.clamp(torch.tensor(lower_bound), torch.tensor(upper_bound))

            current_acquisition_value = acquisition(next_point).float().detach().clone().requires_grad_(False)

            if previous_acquisition_value is not None:
                change_in_value = torch.absolute(current_acquisition_value - previous_acquisition_value)
                if change_in_value <= tolerance:
                    logging.info(f"Multistart {i}: Convergence achieved after {j+1} iterations; EI: {current_acquisition_value}.")
                    break

            previous_acquisition_value = current_acquisition_value

        minima.append(current_acquisition_value)
        sol_x.append(next_point.clone().detach())

    minima = torch.tensor(minima)
    min_idx = torch.argmin(minima)
    next_point = sol_x[min_idx]

    # fall back to a non-NaN candidate if the best one is invalid
    if torch.isnan(next_point).any():
        logging.warning("Selected point contains NaN values. Selecting a different point.")
        valid_points = [(point, minima[i]) for i, point in enumerate(sol_x) if not torch.isnan(point).any()]
        if valid_points:
            next_point, _ = min(valid_points, key=lambda x: x[1])
        else:
            raise ValueError("All candidate points contain NaN values.")
    return next_point


def original_ucb(model, lower_bound, upper_bound):
    """
    Upper-confidence-bound acquisition: minimise mean - beta*stddev from
    several Sobol starts with LBFGS and return the best feasible candidate.
    """
    def ucb_acquisition(x):
        # optimistic lower bound, mean - beta*stddev, to select minima
        pred  = model(x)
        beta = 2
        return pred.mean - beta * pred.stddev

    n_multi_start = 10
    n_grad_steps = 50
    points = generate_sobol_points(lower_bound, upper_bound, n_multi_start)
    minima = []
    sol_x  = []
    tolerance = torch.tensor([1e-8])
    previous_acquisition_value = None

    for i in range(n_multi_start):
        next_point = torch.tensor(points[i].reshape(1,-1)).float().requires_grad_(True)
        optimizer = optim.LBFGS([next_point], line_search_fn='strong_wolfe', lr=0.01)

        for j in range(n_grad_steps):
            def closure():
                optimizer.zero_grad()
                acquisition_value = ucb_acquisition(next_point)
                acquisition_value.backward()
                return acquisition_value

            optimizer.step(closure)
            with torch.no_grad(): next_point[:] = next_point.clamp(torch.tensor(lower_bound), torch.tensor(upper_bound))

            current_acquisition_value = ucb_acquisition(next_point).float().detach().clone().requires_grad_(False)

            if previous_acquisition_value is not None:
                change_in_value = torch.absolute(current_acquisition_value - previous_acquisition_value)
                if change_in_value <= tolerance:
                    logging.info(f"Multistart {i}: Convergence achieved after {j+1} iterations; UCB: {current_acquisition_value}.")
                    break

            previous_acquisition_value = current_acquisition_value

        minima.append(current_acquisition_value)
        sol_x.append(next_point.clone().detach())

    minima = torch.tensor(minima)
    min_idx = torch.argmin(minima)
    next_point = sol_x[min_idx]

    # fall back to a non-NaN candidate if the best one is invalid
    if torch.isnan(next_point).any():
        logging.warning("Selected point contains NaN values. Selecting a different point.")
        valid_points = [(point, minima[i]) for i, point in enumerate(sol_x) if not torch.isnan(point).any()]
        if valid_points:
            next_point, _ = min(valid_points, key=lambda x: x[1])
        else:
            raise ValueError("All candidate points contain NaN values.")
    return next_point




if __name__ == '__main__':
    import unittest

    class TestBayesianOptimization(unittest.TestCase):

        def test_bayesian_optimization(self):
            def objective(x):
                return torch.sum(x**2) + x[0]

            lower_bound = [-1, -1]
            upper_bound = [1, 1]

            num_initial_points = 3
            num_iterations = 10

            best_candidate, best_index = bayesian_optimization(objective, lower_bound, upper_bound, num_initial_points, num_iterations, acq='ei')

            self.assertTrue((np.array(lower_bound) <= best_candidate.numpy()).all() and (best_candidate.numpy() <= np.array(upper_bound)).all())

            self.assertIsNotNone(best_candidate)

            self.assertAlmostEqual(best_candidate.numpy()[0], 0.0, places=1)


    def objective(x):
        return torch.sum(x**2) + x[0]

    lower_bound = [-2, -2]
    upper_bound = [1, 1]

    num_initial_points = 2
    num_iterations = 10

    best_candidate, best_index = bayesian_optimization(objective, lower_bound, upper_bound, num_initial_points, num_iterations, acq='ei')

    print(f"Best candidate: {best_candidate}")
    print(f"Best function value: {objective(best_candidate)}")

