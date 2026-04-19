import jax.numpy as jnp
import numpy as np
from scipy.stats import beta
from jax import jit, pmap, devices, lax
from functools import partial

from mu_F.solvers.utilities import generate_initial_guess
from mu_F.solvers.functions import multi_start_solve_bounds_nonlinear_program
from mu_F.constraints.utils import standardise_inputs, standardise_model_decisions, mask_classifier, get_successor_inputs
from mu_F.solvers.utilities import determine_batches, create_batches
       
def assess_feasibility(feasibility, input):
    """
    Assesses the feasibility of the input
    """
    if feasibility == 'positive':
        return input >= 0
    elif feasibility == 'negative':
        return input <= 0
    else:
        raise ValueError("Invalid notion of feasibility.")
    

""" ---- JaxOpt solver evaluation methods (written as pure functions not classes) --- """


def shaping_function(x, cfg):
    """
    Shaping function
    """
    if cfg.samplers.notion_of_feasibility == 'positive':
        return -x
    elif cfg.samplers.notion_of_feasibility == 'negative':
        return x


def construct_solver(objective_func, bounds, tol, sobol_pts=None):
    solver = partial(multi_start_solve_bounds_nonlinear_program, objective_func=objective_func, bounds_=(bounds[0], bounds[1]), tol=tol, sobol_pts=sobol_pts)
    return solver

def initial_guess(cfg, bounds):
    n_d = len(bounds[0])
    return generate_initial_guess(cfg.n_starts, n_d, bounds)

def solve(solver, initial_guesses):
    obj_r, e, params_r = [], [], []

    for solve, init in zip(solver, initial_guesses):
        objective, error, params = solve(init)
        obj_r.append(objective)
        e.append(error)
        params_r.append(params)

    return {'objective': jnp.array(obj_r), 'error': jnp.array(e), 'params': jnp.array(params_r)}
    

def load_solver(objective_func, bounds):
    """
    Loads the solver
    """
    return construct_solver(objective_func, bounds)


def get_backward_bounds(graph, node, cfg):
    """
    Extracts decision bounds per successor from the graph structure.
    Does not depend on outputs — safe to call outside pmap.
    """
    if node is None:
        return None
    backward_bounds = {}
    for succ in graph.successors(node):
        n_d = graph.nodes[succ]['n_design_args']
        input_indices = np.copy(np.array([n_d + input_ for input_ in graph.edges[node, succ]['input_indices']]))
        aux_indices = np.copy(np.array([input_ for input_ in graph.edges[node, succ]['auxiliary_indices']]))
        decision_bounds = graph.nodes[succ]["extendedDS_bounds"].copy()
        if cfg.solvers.standardised:
            decision_bounds = standardise_model_decisions(graph, decision_bounds, succ)
        decision_bounds = [jnp.delete(bound, np.hstack([input_indices, aux_indices]).astype(int), axis=1) for bound in decision_bounds]
        backward_bounds[succ] = decision_bounds
    return backward_bounds


def prepare_backward_problem(outputs, graph, node, cfg):
    """
    Prepares the forward constraints surrogates and decision variables
    - ouptuts from a nodes unit functions are inputs to the next unit

    """
    if node is None:
        return None, None
    else: 
        # TODO make sure that this is not going to throw errors in tracing.
        backward_bounds = {succ: None for succ in graph.successors(node)}
        backward_objective = {succ: None for succ in graph.successors(node)}

        # get the outputs from the successors of the node
        succ_inputs = get_successor_inputs(graph, node, outputs)

        for succ in graph.successors(node):

            n_d  = graph.nodes[succ]['n_design_args']
            input_indices = np.copy(np.array([n_d + input_ for input_ in graph.edges[node, succ]['input_indices']]))
            aux_indices = np.copy(np.array([input_ for input_ in graph.edges[node, succ]['auxiliary_indices']]))
        
            
            # standardisation of outputs if required
            if cfg.solvers.standardised: succ_inputs[succ] = succ_inputs[succ].at[:].set(standardise_inputs(graph, succ_inputs[succ], succ, jnp.hstack([input_indices, aux_indices]).astype(int)))
            
            # load the standardised bounds
            decision_bounds = graph.nodes[succ]["extendedDS_bounds"].copy()
            ndim = graph.nodes[succ]['n_design_args'] + graph.nodes[succ]['n_input_args'] + graph.graph['n_aux_args']
            decision_indices = jnp.delete(jnp.arange(ndim), np.hstack([input_indices, aux_indices]).astype(int))  # indices of the decision variables
            # get the decision bounds
            if cfg.solvers.standardised: decision_bounds = standardise_model_decisions(graph, decision_bounds, succ)
            
            decision_bounds = [jnp.delete(bound, np.hstack([input_indices,aux_indices]).astype(int), axis=1) for bound in decision_bounds]
            backward_bounds[succ] = [decision_bounds.copy() for i in range(succ_inputs[succ].shape[0])]

            # load the forward objective
            classifier = graph.nodes[succ]["classifier"]
            wrapper_classifier = mask_classifier(classifier, ndim, input_indices, aux_indices)
            backward_objective[succ] = [jit(partial(lambda x,y: wrapper_classifier(x,y).squeeze(), y=succ_inputs[succ][i].reshape(1,-1))) for i in range(succ_inputs[succ].shape[0])]

        # return the forward surrogates and decision bounds
        return backward_objective, backward_bounds

def prepare_global_problem(inputs, aux, graph, cfg):
    """
    Prepares the global problem defined for handling nuisance parameters in the reconstruction 
        - loads the objective function and bounds from the graph by
            1: loads the classifier from the graph 
            2: loads the bounds from the graph
            3: loads the fixed indices from the graph
            4: standardises the inputs and decisions if required
            5: masks the classifier to only use the decision variables
            6: prepares the global problem for the solver
    """
    n_d     = graph.graph['n_design_args'] # number of design variables in the successors of the root node
    n_aux   = graph.graph['n_aux_args']

    # get the fixed indices and auxiliary indices
    dec_ind = np.array(graph.graph['post_process_decision_indices'])
    total_ind = np.arange(n_d + n_aux)
    fix_ind = np.delete(total_ind, dec_ind).astype(int)  # indices of the fixed decision variables

    # introduce bounds 
    lb =     jnp.hstack([jnp.array(bound[0]).reshape(-1,) for bound in graph.graph['bounds'] if bound[0] != 'None'])
    ub =     jnp.hstack([jnp.array(bound[1]).reshape(-1,) for bound in graph.graph['bounds'] if bound[1] != 'None'])
    bounds = [lb, ub]
    
    # standardise the inputs and decisions if required
    if cfg.solvers.standardised:
        inputs = standardise_inputs(graph, inputs, None, jnp.hstack([fix_ind]).astype(int))
        bounds = standardise_model_decisions(graph, bounds, None)

    # mask the classifier to only use the decision variables
    classifier = mask_classifier(graph.graph['post_process_lower_classifier'], n_d + n_aux, fix_ind, np.empty((0,)).astype(int))

    # prepare the objective function # NOTE this should be a maximization problem -> therefore negative values of the objective indicate constraint violations.
    objective_func = partial(lambda x, y: -classifier(x, y).squeeze(), y=inputs.reshape(1,-1))

    # prepare the bounds
    bounds = [jnp.delete(bounds[0], fix_ind), jnp.delete(bounds[1], fix_ind)]
    

    return objective_func, bounds



def evaluate(outputs, aux, graph, node, cfg, sobol_pts_dict=None):
    """
    Evaluates the constraints.
    Handles both graph-wide and node-local (backward) problems.
    sobol_pts_dict: dict mapping successor node -> pre-generated sobol points array, or None
    """

    evaluate_method = solve

    def graph_wide_branch(args):
        (outputs, aux) = args
        if outputs.ndim < 2: outputs = outputs.reshape(-1, 1)
        if aux.ndim < 2: aux = aux.reshape(-1, 1)
        objective, bounds = prepare_global_problem(outputs, aux, graph, cfg)
        solver = construct_solver(objective, bounds, tol=cfg.solvers.post.jax_opt_options.error_tol)
        initial_guesses = initial_guess(cfg.solvers.backward_coupling, bounds)
        result = evaluate_method([solver], [initial_guesses])
        fn_evaluations = result['objective'].reshape(-1, 1)
        return -shaping_function(fn_evaluations, cfg) # maximisation problem

    def node_local_branch(args):
        (outputs, aux) = args
        # note that auxiliary variables are assumed global and propagated through the graph constituent functions
        objective, bounds = prepare_backward_problem(outputs, graph, node, cfg)
        # enabling tracing
        if objective is None or bounds is None:
            return jnp.zeros((outputs.shape[0], 1)), None
        # function body
        else:
            succ_fn_evaluations = {}
            for succ in graph.successors(node):
                succ_sobol = sobol_pts_dict.get(succ) if sobol_pts_dict is not None else None
                backward_solver = [
                    construct_solver(objective[succ][i], bounds[succ][i], tol=cfg.solvers.backward_coupling.jax_opt_options.error_tol, sobol_pts=succ_sobol)
                    for i in range(outputs.shape[0])
                ]
                initial_guesses = [
                    initial_guess(cfg.solvers.backward_coupling, bounds[succ][i])
                    for i in range(outputs.shape[0])
                ]
                succ_fn_evaluations[succ] = evaluate_method(backward_solver, initial_guesses)
            fn_evaluations = [
                succ_fn_evaluations[succ]['objective'].reshape(-1, 1)
                for succ in graph.successors(node)
            ]
            warmstart_params = {succ: succ_fn_evaluations[succ]['params'] for succ in graph.successors(node)}
            return shaping_function(jnp.hstack(fn_evaluations), cfg), warmstart_params

    is_graph_wide = bool(graph.graph["solve_post_processing_problem"])
    if is_graph_wide:
        return graph_wide_branch((outputs, aux)), None
    else:
        return node_local_branch((outputs, aux))


def jax_pmap_evaluator(outputs, aux, sobol_pts_tuple, cfg, graph, node, successor_order=None):
    """
    p-map constraint evaluation call - called by backward_surrogate_pmap_batch_evaluator
    sobol_pts_tuple: tuple of pre-generated sobol arrays (one per successor), passed as pmap arg
    """
    # reconstruct the dict from the ordered tuple
    sobol_pts_dict = None
    if successor_order is not None and sobol_pts_tuple is not None:
        sobol_pts_dict = {succ: pts for succ, pts in zip(successor_order, sobol_pts_tuple)}

    constraint_evaluator = partial(evaluate, graph=graph, node=node, cfg=cfg, sobol_pts_dict=sobol_pts_dict)

    return constraint_evaluator(outputs, aux)


def backward_surrogate_pmap_batch_evaluator(outputs, aux, cfg, graph, node):
    """
    Evaluates the constraints on a batch using jax-pmap - called by the backward_constraint_evaluator
    """
    # Pre-generate sobol points per successor outside pmap — bounds depend only on graph structure
    n_sobol_screen = getattr(cfg.solvers.backward_coupling, 'n_sobol_screen', 16_384)
    backward_bounds = get_backward_bounds(graph, node, cfg)
    sobol_pts_tuple = ()
    successor_order = None
    if backward_bounds is not None:
        successor_order = list(backward_bounds.keys())
        sobol_pts_tuple = tuple(
            generate_initial_guess(n_sobol_screen, None, bounds)
            for bounds in backward_bounds.values()
        )

    feasibility_call = partial(jax_pmap_evaluator, cfg=cfg, graph=graph, node=node, successor_order=successor_order)

    # sobol_pts_tuple passed as pmap arg with in_axes=None so it's a dynamic input, not a traced constant
    return pmap(feasibility_call, in_axes=(0, 0, None), out_axes=0, devices=[device for i, device in enumerate(devices('cpu')) if i<outputs.shape[0]])(outputs, aux, sobol_pts_tuple)
   
 

def backward_constraint_evaluator(outputs, aux, cfg, graph, node, pool):
    """
    Evaluates the constraints using jax-pmap - this is what should be called

    Syntax:
        call: method_(outputs, cfg, graph, node, pool)

    """
    max_devices = cfg.max_devices
    batch_sizes, remainder = determine_batches(outputs.shape[0], max_devices)
    # get batches of outputs
    output_batches = create_batches(batch_sizes, outputs)
    aux_batches = create_batches(batch_sizes, jnp.repeat(jnp.expand_dims(aux, axis=1), outputs.shape[1], axis=1))
    # evaluate the constraints
    results, warmstarts = [], []
    for i, (output_batch, aux_batch) in enumerate(zip(output_batches, aux_batches)):
        evals, params = backward_surrogate_pmap_batch_evaluator(output_batch, aux_batch, cfg, graph, node)
        results.append(evals)
        if params is not None:
            warmstarts.append(params)
    # concatenate the results, keeping warmstarts keyed by successor node
    warmstarts = {succ: jnp.vstack([w[succ] for w in warmstarts]) for succ in warmstarts[0]} if warmstarts else None

    del output_batches, aux_batches, batch_sizes

    return jnp.vstack(results), warmstarts


""" ---- Constrained (septal) solver interface ---------------------------------

The evaluators below drive a general NLP of the shape

    min  f(x)
    s.t. lhs <= g(x) <= rhs
         lb  <=   x  <= ub

via septal's `ParametricSQPFactory`. The interface is deliberately symmetric with
`construct_solver` / `solve` above so every evaluator in this module reads the
same: build a solver, hand it an initial-guess batch, get back an objective and
a converged flag.

`construct_constrained_solver` is a frozen stub — Phase 2 of the septal
integration fills the body. Freezing the signature here keeps downstream
evaluator code stable across the swap.
"""

def construct_constrained_solver(objective_func, constraint_func, bounds, tol,
                                 sobol_pts=None, constraint_lhs=None, constraint_rhs=None):
    """
    Construct a constrained NLP solver.

    Parameters
    ----------
    objective_func : Callable
        Scalar JAX objective `f(x) -> ()`.
    constraint_func : Callable
        Vector JAX constraint `g(x) -> (n_g,)`. Pass `None` for box-only
        problems (use `construct_solver` instead in that case).
    bounds : list[jnp.ndarray]
        `[lb, ub]` box bounds, each shape `(n_d,)`.
    tol : float
        KKT tolerance (maps to septal's `SQPConfig.tol_stationarity`
        and `tol_feasibility`).
    sobol_pts : jnp.ndarray, optional
        Pre-generated Sobol screen points, shape `(n_screen, n_d)`. If `None`,
        generated internally.
    constraint_lhs, constraint_rhs : jnp.ndarray, optional
        Two-sided bounds on `g(x)`, each shape `(n_g,)`. Default: one-sided
        inequality `g(x) <= 0` (`lhs = -inf`, `rhs = 0`).
    """
    raise NotImplementedError(
        "Constrained solver wiring lands in Phase 2 of the septal integration; "
        "signature is frozen so call sites don't churn at swap time."
    )


def solve_constrained(solver, initial_guesses):
    """
    Driver symmetric with `solve` for the constrained path.

    Returns a dict keyed `objective`, `error`, `converged`, `params` so the
    evaluator layer can reuse the same unpacking logic it uses today.
    """
    obj_r, err_r, conv_r, params_r = [], [], [], []

    for s, init in zip(solver, initial_guesses):
        objective, error, converged, params = s(init)
        obj_r.append(objective)
        err_r.append(error)
        conv_r.append(converged)
        params_r.append(params)

    return {
        'objective': jnp.array(obj_r),
        'error': jnp.array(err_r),
        'converged': jnp.array(conv_r),
        'params': jnp.array(params_r),
    }


""" ---- Cost-to-go problem preparation + pmap evaluator ----------------------
"""

def prepare_ctg_problem(outputs, graph, node, cfg):
    """
    Prepares the cost-to-go sub-problem for each successor of `node`.

    For each successor:
      - decision variables live in the successor's reduced space
        (design + non-current-edge inputs + aux)
      - the inputs coming from `node` are held fixed; `mask_classifier`
        scatters them into the full NLP space at call time
      - objective  = successor's live CTG surrogate
      - constraint = successor's live classifier, enforced as `g(x) <= 0`

    Mirrors `prepare_backward_problem` — same standardisation path, same
    bound-masking — with the CTG surrogate added as objective and the
    classifier reused as a feasibility constraint instead of the objective.
    """
    if node is None:
        return None, None, None

    ctg_objective  = {succ: None for succ in graph.successors(node)}
    ctg_constraint = {succ: None for succ in graph.successors(node)}
    ctg_bounds     = {succ: None for succ in graph.successors(node)}

    succ_inputs = get_successor_inputs(graph, node, outputs)

    for succ in graph.successors(node):

        n_d           = graph.nodes[succ]['n_design_args']
        input_indices = np.copy(np.array([n_d + inp for inp in graph.edges[node, succ]['input_indices']]))
        aux_indices   = np.copy(np.array([inp for inp in graph.edges[node, succ]['auxiliary_indices']]))
        fix_indices   = jnp.hstack([input_indices, aux_indices]).astype(int)

        # standardise fixed inputs if requested
        if cfg.solvers.standardised:
            succ_inputs[succ] = succ_inputs[succ].at[:].set(
                standardise_inputs(graph, succ_inputs[succ], succ, fix_indices)
            )

        # reduced-space bounds (drop the indices held fixed)
        decision_bounds = graph.nodes[succ]["extendedDS_bounds"].copy()
        ndim = (graph.nodes[succ]['n_design_args']
                + graph.nodes[succ]['n_input_args']
                + graph.graph['n_aux_args'])
        if cfg.solvers.standardised:
            decision_bounds = standardise_model_decisions(graph, decision_bounds, succ)
        decision_bounds = [jnp.delete(bound, fix_indices, axis=1) for bound in decision_bounds]
        ctg_bounds[succ] = [decision_bounds for _ in range(succ_inputs[succ].shape[0])]

        # live JAX callables straight off the graph — no reconstruction
        ctg_surrogate = graph.nodes[succ]["ctg_surrogate"]
        classifier    = graph.nodes[succ]["classifier"]

        wrapped_ctg        = mask_classifier(ctg_surrogate, ndim, input_indices, aux_indices)
        wrapped_classifier = mask_classifier(classifier,    ndim, input_indices, aux_indices)

        ctg_objective[succ] = [
            jit(partial(lambda x, y: wrapped_ctg(x, y).squeeze(),
                        y=succ_inputs[succ][i].reshape(1, -1)))
            for i in range(succ_inputs[succ].shape[0])
        ]
        ctg_constraint[succ] = [
            jit(partial(lambda x, y: wrapped_classifier(x, y).reshape(-1),
                        y=succ_inputs[succ][i].reshape(1, -1)))
            for i in range(succ_inputs[succ].shape[0])
        ]

    return ctg_objective, ctg_constraint, ctg_bounds


def evaluate_ctg(outputs, aux, graph, node, cfg, sobol_pts_dict=None, warmstarts=None):
    """
    Evaluates the cost-to-go surrogate for each successor of `node`.

    Mirrors `evaluate`'s node-local branch but swaps the objective for the CTG
    surrogate and promotes the classifier to a feasibility constraint. Consumes
    `warmstarts` produced by `backward_constraint_evaluator` when provided.
    """
    objective, constraint, bounds = prepare_ctg_problem(outputs, graph, node, cfg)
    if objective is None or bounds is None:
        return jnp.zeros((outputs.shape[0], 1)), None

    succ_fn_evaluations = {}
    for succ in graph.successors(node):
        succ_sobol = sobol_pts_dict.get(succ) if sobol_pts_dict is not None else None

        solvers = [
            construct_constrained_solver(
                objective[succ][i], constraint[succ][i], bounds[succ][i],
                tol=cfg.solvers.ctg.jax_opt_options.error_tol,
                sobol_pts=succ_sobol,
            )
            for i in range(outputs.shape[0])
        ]

        # warmstart per sample if provided; otherwise sobol-sampled initial guess
        if warmstarts is not None and succ in warmstarts:
            initial_guesses = [warmstarts[succ][i].reshape(1, -1)
                               for i in range(outputs.shape[0])]
        else:
            initial_guesses = [initial_guess(cfg.solvers.ctg, bounds[succ][i])
                               for i in range(outputs.shape[0])]

        succ_fn_evaluations[succ] = solve_constrained(solvers, initial_guesses)

    fn_evaluations = [
        succ_fn_evaluations[succ]['objective'].reshape(-1, 1)
        for succ in graph.successors(node)
    ]
    success_flags = [
        succ_fn_evaluations[succ]['converged'].reshape(-1, 1)
        for succ in graph.successors(node)
    ]
    return jnp.hstack(fn_evaluations), jnp.hstack(success_flags)


def ctg_pmap_batch_evaluator(outputs, aux, cfg, graph, node, warmstarts=None):
    """
    pmap wrapper for `evaluate_ctg` — mirrors `backward_surrogate_pmap_batch_evaluator`.

    Sobol points are precomputed outside pmap (bounds are static per-successor)
    and passed with `in_axes=None`.
    """
    n_sobol_screen  = getattr(cfg.solvers.ctg, 'n_sobol_screen', 16_384)
    backward_bounds = get_backward_bounds(graph, node, cfg)
    sobol_pts_tuple = ()
    successor_order = None
    if backward_bounds is not None:
        successor_order = list(backward_bounds.keys())
        sobol_pts_tuple = tuple(
            generate_initial_guess(n_sobol_screen, None, bounds)
            for bounds in backward_bounds.values()
        )

    def shard_call(outputs_s, aux_s, sobol_tuple, warmstarts_s):
        sobol_dict = None
        if successor_order is not None and sobol_tuple is not None:
            sobol_dict = {succ: pts for succ, pts in zip(successor_order, sobol_tuple)}
        return evaluate_ctg(outputs_s, aux_s, graph, node, cfg,
                            sobol_pts_dict=sobol_dict, warmstarts=warmstarts_s)

    devs = [d for i, d in enumerate(devices('cpu')) if i < outputs.shape[0]]
    return pmap(shard_call, in_axes=(0, 0, None, 0), out_axes=0, devices=devs)(
        outputs, aux, sobol_pts_tuple, warmstarts
    )


def cost_to_go_evaluator(outputs, aux, cfg, graph, node, warmstarts=None):
    """
    Top-level CTG evaluator — mirrors `backward_constraint_evaluator`.

    Shards samples across CPU devices via pmap. Returns
    `(ctg_evaluations, success_flags)` per successor.
    """
    max_devices = cfg.max_devices
    batch_sizes, _ = determine_batches(outputs.shape[0], max_devices)

    output_batches = create_batches(batch_sizes, outputs)
    aux_batches    = create_batches(batch_sizes,
                                    jnp.repeat(jnp.expand_dims(aux, axis=1),
                                               outputs.shape[1], axis=1))
    warmstart_batches = None
    if warmstarts is not None:
        per_succ_batches = {succ: create_batches(batch_sizes, warmstarts[succ])
                            for succ in warmstarts}
        warmstart_batches = [
            {succ: per_succ_batches[succ][i] for succ in per_succ_batches}
            for i in range(len(batch_sizes))
        ]

    evals, flags = [], []
    for i, (output_batch, aux_batch) in enumerate(zip(output_batches, aux_batches)):
        ws_i = warmstart_batches[i] if warmstart_batches is not None else None
        evals_i, flags_i = ctg_pmap_batch_evaluator(output_batch, aux_batch, cfg, graph, node,
                                                    warmstarts=ws_i)
        evals.append(evals_i)
        flags.append(flags_i)

    del output_batches, aux_batches, batch_sizes

    return jnp.vstack(evals), jnp.vstack(flags)


""" ---- Current-node surrogate problem preparation + pmap evaluators ---------
"""

def prepare_current_constraint_problem(inputs, graph, node, cfg):
    """
    Current-node feasibility check: find `(design, aux)` satisfying the
    current node's live classifier given fixed `inputs`.

    Decision space excludes the input indices (those are fixed by `inputs`).
    Objective = classifier(assemble(x, inputs)); no general constraints.
    """
    if node is None:
        return None, None

    n_design = graph.nodes[node]['n_design_args']
    n_input  = graph.nodes[node]['n_input_args']
    n_aux    = graph.graph['n_aux_args']

    ndim = n_design + n_input + n_aux
    input_indices = np.arange(n_design, n_design + n_input).astype(int)
    aux_indices   = np.arange(ndim - n_aux, ndim).astype(int)
    fix_indices   = jnp.hstack([input_indices, aux_indices]).astype(int)

    if cfg.solvers.standardised:
        inputs = inputs.at[:].set(standardise_inputs(graph, inputs, node, fix_indices))
        decision_bounds = standardise_model_decisions(graph,
                                                      graph.nodes[node]["extendedDS_bounds"],
                                                      node)
    else:
        decision_bounds = graph.nodes[node]["extendedDS_bounds"]

    decision_bounds = [jnp.delete(bound, fix_indices, axis=1) for bound in decision_bounds]

    classifier         = graph.nodes[node]["classifier"]
    wrapped_classifier = mask_classifier(classifier, ndim, input_indices, aux_indices)

    objective_func = jit(partial(lambda x, y: wrapped_classifier(x, y).squeeze(),
                                 y=inputs.reshape(1, -1)))
    return objective_func, decision_bounds


def prepare_current_cost_problem(inputs, graph, node, cfg):
    """
    Current-node CTG: minimise the current node's CTG surrogate subject to
    the current node's live classifier as a feasibility constraint.

    Same reduced decision space as `prepare_current_constraint_problem`;
    objective is swapped for `ctg_surrogate`, the classifier becomes `g(x) <= 0`.
    """
    if node is None:
        return None, None, None

    n_design = graph.nodes[node]['n_design_args']
    n_input  = graph.nodes[node]['n_input_args']
    n_aux    = graph.graph['n_aux_args']

    ndim = n_design + n_input + n_aux
    input_indices = np.arange(n_design, n_design + n_input).astype(int)
    aux_indices   = np.arange(ndim - n_aux, ndim).astype(int)
    fix_indices   = jnp.hstack([input_indices, aux_indices]).astype(int)

    if inputs is None:
        inputs = jnp.empty((1, 0))

    if cfg.solvers.standardised and inputs.size > 0:
        inputs = inputs.at[:].set(standardise_inputs(graph, inputs, node, fix_indices))
        decision_bounds = standardise_model_decisions(graph,
                                                      graph.nodes[node]["extendedDS_bounds"],
                                                      node)
    else:
        decision_bounds = graph.nodes[node]["extendedDS_bounds"]

    decision_bounds = [jnp.delete(bound, fix_indices, axis=1) for bound in decision_bounds]

    ctg_surrogate      = graph.nodes[node]["ctg_surrogate"]
    classifier         = graph.nodes[node]["classifier"]
    wrapped_ctg        = mask_classifier(ctg_surrogate, ndim, input_indices, aux_indices)
    wrapped_classifier = mask_classifier(classifier,    ndim, input_indices, aux_indices)

    objective_func  = jit(partial(lambda x, y: wrapped_ctg(x, y).squeeze(),
                                  y=inputs.reshape(1, -1)))
    constraint_func = jit(partial(lambda x, y: wrapped_classifier(x, y).reshape(-1),
                                  y=inputs.reshape(1, -1)))

    return objective_func, constraint_func, decision_bounds


def current_constraint_surrogate(inputs, aux, cfg, graph, node):
    """
    Top-level current-node feasibility evaluator.

    Box-only — uses the existing `construct_solver` path (jaxopt today,
    septal in Phase 2 once the constrained path is wired).
    """
    objective_func, bounds = prepare_current_constraint_problem(inputs, graph, node, cfg)
    if objective_func is None or bounds is None:
        return jnp.zeros((inputs.shape[0], 1))

    solver = construct_solver(objective_func, bounds,
                              tol=cfg.solvers.backward_coupling.jax_opt_options.error_tol)
    guesses = initial_guess(cfg.solvers.backward_coupling, bounds)
    result  = solve([solver], [guesses])
    return result['objective'].reshape(-1, 1)


def current_cost_surrogate(inputs, aux, cfg, graph, node):
    """
    Top-level current-node CTG evaluator.

    Constrained — routes through `construct_constrained_solver`. Single-problem
    (no successor fan-out), so pmap is unnecessary; batching over samples is
    handled via vmap inside the septal solver.
    """
    objective_func, constraint_func, bounds = prepare_current_cost_problem(inputs, graph, node, cfg)
    if objective_func is None or bounds is None:
        return jnp.zeros((inputs.shape[0], 1)), None

    solver = construct_constrained_solver(
        objective_func, constraint_func, bounds,
        tol=cfg.solvers.ctg.jax_opt_options.error_tol,
    )
    guesses = initial_guess(cfg.solvers.ctg, bounds)
    result  = solve_constrained([solver], [guesses])
    return result['objective'].reshape(-1, 1), result['converged'].reshape(-1, 1)
