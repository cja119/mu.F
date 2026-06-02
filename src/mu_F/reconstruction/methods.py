"""Graph (de)serialisation helpers for reconstruction."""
import pickle
import numpy as np


def construct_cartesian_product_of_live_sets(graph):
    """
    Build the cartesian product of per-node live sets, shuffled
    so designs are decorrelated across nodes.
    """
    unit_ls = {}
    n_aux = graph.graph["n_aux_args"]
    for node in graph.nodes:
        n_d = graph.nodes[node]['n_design_args']
        rng = np.random.default_rng()
        try:
            lset = np.copy(graph.nodes[node]["live_set_inner"][:,:n_d]).reshape(-1, n_d)
        except:
            lset = graph.nodes[node]["live_set_inner"][:,-n_aux:].reshape(-1, n_aux)
        rng.shuffle(lset, axis = 0)
        unit_ls[node] = np.copy(lset)

    return unit_ls


def save_graph(G, filename):
    """
    Pickle the graph after dropping non-serialisable fields
    (evaluators, edge functions, surrogates, constraints, classifiers).
    """
    for node in G.nodes:
        G.nodes[node]["forward_evaluator"] = None
        for predec in G.predecessors(node):
            G.edges[predec, node]["edge_fn"] = None
            G.edges[predec, node]["forward_surrogate"] = None
        G.nodes[node]["constraints"] = None
        G.nodes[node]['classifier'] = None

    with open(filename, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_graph(filename):
    """
    Unpickle a graph previously saved by save_graph.
    """
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    return G

def overwrite_graph(G, blank_graph_object):
    """
    Restore the dropped non-serialisable fields on G from a freshly
    built blank graph object.
    """
    for node in G.nodes:
        G.nodes[node]["forward_evaluator"] = blank_graph_object.nodes[node]["forward_evaluator"]
        for predec in G.predecessors(node):
            G.edges[predec, node]["edge_fn"] = blank_graph_object.edges[predec, node]["edge_fn"]
        G.nodes[node]["constraints"] = blank_graph_object.nodes[node]["constraints"]

    return G
