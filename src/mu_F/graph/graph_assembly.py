"""DAG construction from adjacency matrices and per-graph argument wiring."""
from abc import ABC
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mu_F.utils import check_requires


def _normalise_aux_spec(spec):
    """
    Coerce an aux spec (int K or list of IDs) to a (count, ids_tuple) pair.
    The list form lets different edges into the same successor route distinct
    aux IDs to distinct slots; int K is the back-compat form for IDs (0,…,K-1).
    """
    if isinstance(spec, (list, tuple)):
        ids = tuple(int(x) for x in spec)
        return len(ids), ids
    count = int(spec)
    return count, tuple(range(count))


class GraphConstructorBase(ABC):
    """Base builder wrapping a networkx DAG.

    Holds the configuration and adjacency-derived DiGraph, exposing helpers to
    attach arbitrary arguments and objects to the graph, its nodes and edges.

    """

    # ---- External Methods ----

    def __init__(self, cfg, adjacency_matrix):
        self.cfg = cfg
        self.adjacency_matrix = adjacency_matrix
        self.G = load_dag_from_adjacency_matrix(adjacency_matrix)
        self.num_nodes = len(adjacency_matrix)

    def load_graph(self, G):
        self.G = G

    def get_graph(self):
        return self.G.copy()

    def add_arg_to_graph(self,  arg_name, arg_value):
        self.G.graph[arg_name] = arg_value
        return

    
    def add_arg_to_nodes(self, arg_name, arg_value):
        for node in self.G.nodes:
            self.G.nodes[node][arg_name] = arg_value[node]
        return
    
    def add_arg_to_edges(self, arg_name, arg_value):
        for (i, j) in arg_value.keys():
            self.G.edges[i,j][arg_name] = arg_value[(i, j)]
        return
    
    def add_node_object(self, node, node_object, node_object_name):
        self.G.nodes[node][node_object_name] = node_object
        return
    
    def add_edge_object(self, i, j, edge_object, edge_object_name):
        self.G.edges[i, j][edge_object_name] = edge_object
        return

class GraphConstructor(GraphConstructorBase):
    """General DAG constructor for explicit adjacency case studies.

    Extends the base with input/aux argument counting and the per-edge index
    bookkeeping that maps each predecessor's outputs into a successor's
    decision vector.

    """

    # ---- External Methods ----

    def __init__(self, cfg, adjacency_matrix):
        super().__init__(cfg, adjacency_matrix)

    def add_n_input_args(self, n_input_args):
        for (i, j) in self.G.edges:
            self.G.edges[i,j]["n_input_args"] = n_input_args[f'({i},{j})']
        for node in self.G.nodes:
            self.G.nodes[node]["n_input_args"] = sum([self.G.edges[predec, node]["n_input_args"] for predec in self.G.predecessors(node)])
        return
    
    def add_n_aux_args(self, n_aux_args):
        """
        Set per-edge and per-node aux counts ('n_auxiliary_args') and 'aux_ids'.
        Each entry may be an int K (IDs 0,…,K-1) or an explicit list of aux IDs.
        """
        for (i, j) in self.G.edges:
            count, ids = _normalise_aux_spec(n_aux_args[f'({i},{j})'])
            self.G.edges[i, j]["n_auxiliary_args"] = count
            self.G.edges[i, j]["aux_ids"]          = ids

        for node in self.G.nodes:
            count, ids = _normalise_aux_spec(n_aux_args[f'node_{node}'])
            self.G.nodes[node]["n_auxiliary_args"] = count
            self.G.nodes[node]["aux_ids"]          = ids

        return

    def add_input_aux_indices(self):
        """
        Compute per-edge input and aux slot positions in each successor's
        decision vector. In 'global' mode aux_ids index the shared aux slots;
        in 'local' mode each edge's aux block is appended after its inputs.
        """
        for node in self.G.nodes:
            n_d = 0
            for predec in self.G.predecessors(node):
                self.G.edges[predec, node]["input_indices"] = [
                    n_d + i for i in range(self.G.edges[predec, node]["n_input_args"])
                ]
                n_d += self.G.edges[predec, node]["n_input_args"]

            mode = self.cfg.model.constraint.auxiliary
            if mode == 'global':
                for pred in self.G.predecessors(node):
                    aux_ids = self.G.edges[pred, node]["aux_ids"]
                    self.G.edges[pred, node]["auxiliary_indices"] = [n_d + a for a in aux_ids]
            elif mode == 'local':
                for pred in self.G.predecessors(node):
                    aux_ids = self.G.edges[pred, node]["aux_ids"]
                    self.G.edges[pred, node]["auxiliary_indices"] = [n_d + a for a in aux_ids]
                    n_d += self.G.edges[pred, node]["n_auxiliary_args"]
            elif mode == 'None':
                for pred in self.G.predecessors(node):
                    self.G.edges[pred, node]["auxiliary_indices"] = []
            else:
                raise ValueError(f"Invalid auxiliary variable structure: {mode!r}")
        return
    
    def add_arg_to_edges(self, arg_name, arg_value):
        for (i, j) in self.G.edges:
            self.G.edges[i,j][arg_name] = arg_value
        return

class MarkovGraphConstructor(GraphConstructor):
    """Constructor for the linear Markov-chain graph topology.

    Builds a chain adjacency matrix from the node count and broadcasts a single
    shared argument spec across every node and edge, so every stage of the chain
    is wired identically.

    """

    # ---- External Methods ----

    def __init__(self, cfg):
        adjacency_matrix = self._build_adjacency_matrix(cfg)
        super().__init__(cfg, adjacency_matrix)

    def get_graph(self):
        return self.G.copy()

    def add_n_input_args(self, n_input_args):
        for (i, j) in self.G.edges:
            self.G.edges[i,j]["n_input_args"] = n_input_args
        for node in self.G.nodes:
            self.G.nodes[node]["n_input_args"] = sum([self.G.edges[predec, node]["n_input_args"] for predec in self.G.predecessors(node)])
        return
    
    def add_n_aux_args(self, n_aux_args):
        """
        Markov variant: pull one edge spec and one node spec, then broadcast
        them across every edge and node in the chain. Each spec may be int K
        (aux_ids=(0,…,K-1)) or an explicit list of aux IDs.
        """
        if hasattr(n_aux_args, 'keys'):
            edge_spec = n_aux_args.get('(0,1)', 0)
            node_spec = n_aux_args.get('node_0', 0)
        else:
            edge_spec = n_aux_args
            node_spec = n_aux_args

        edge_count, edge_ids = _normalise_aux_spec(edge_spec)
        node_count, node_ids = _normalise_aux_spec(node_spec)

        for (i, j) in self.G.edges:
            self.G.edges[i, j]["n_auxiliary_args"] = edge_count
            self.G.edges[i, j]["aux_ids"]          = edge_ids
        for node in self.G.nodes:
            self.G.nodes[node]["n_auxiliary_args"] = node_count
            self.G.nodes[node]["aux_ids"]          = node_ids
        return


    def add_arg_to_nodes(self, arg_name, arg_value):
        for node in self.G.nodes:
            if callable(arg_value) and check_requires(arg_value, 'cfg'):
                self.G.nodes[node][arg_name] = arg_value(self.cfg)
            else:
                self.G.nodes[node][arg_name] = arg_value
        return None

    def add_arg_to_edges(self, arg_name, arg_value):
        for (i, j) in self.G.edges:
            v = arg_value[(i, j)] if isinstance(arg_value, dict) else arg_value
            if callable(v) and check_requires(v, 'cfg'):
                self.G.edges[i, j][arg_name] = v(self.cfg)
            else:
                self.G.edges[i, j][arg_name] = v
        return None

    def add_arg_to_edge(self, i, j, arg_name, arg_value):
        if callable(arg_value) and check_requires(arg_value, 'cfg'):
            self.G.edges[i,j][arg_name] = arg_value(self.cfg)
        elif callable(arg_value):
            self.G.edges[i,j][arg_name] = arg_value
        else:
            self.G.edges[i,j][arg_name] = arg_value
        return None

    # ---- Private Methods ----

    def _build_adjacency_matrix(self, cfg):
        N = cfg.case_study.num_nodes
        adjacency_matrix = np.zeros((N, N), dtype=int)
        adjacency_matrix[np.arange(N-1), np.arange(1, N)] = 1
        return adjacency_matrix

def case_study_uncertain_parameters_dummy(G):
    """
    Attach dummy best-estimate and sampled uncertain parameters to each node.
    """
    parameter_be = 1.0  # best estimate
    p_sdev = 0. #p.sqrt(0.3)
    np.random.seed(1)
    n_samples_p = 1
    p_samples = np.random.normal(parameter_be, p_sdev, n_samples_p)
    p_samples = [{'c': [p], 'w': 1.0/n_samples_p} for p in p_samples]
    P_BE, P_S = {}, {}

    for node in G.nodes:
        P_BE[node] = [parameter_be]
        P_S[node] = p_samples

    return P_BE, P_S


def load_dag_from_adjacency_matrix(adj_matrix):
    """
    Build a networkx DiGraph from a binary adjacency matrix.
    """
    G = nx.DiGraph()
    num_nodes = len(adj_matrix)
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)
    return G



