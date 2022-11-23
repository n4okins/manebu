import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
import networkx as nx
import matplotlib.pyplot as plt


class GrowGraph:
    def __init__(self, nodes, edges, senders, receivers,
                 n_node, n_edge, globals, **kwargs):
        self.graph = jraph.GraphsTuple(
            nodes=nodes, edges=edges, senders=senders, receivers=receivers,
            n_node=n_node, n_edge=n_edge, globals=globals
        )


def convert_networkx(graph: jraph.GraphsTuple):
    nodes, edges, receivers, senders, _, _, _ = graph
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(graph.n_node[0]):
            nx_graph.add_node(n)
    else:
        for n in range(graph.n_node[0]):
            nx_graph.add_node(n, node_feature=nodes[n])
    if edges is None:
        for e in range(graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(graph.n_edge[0]):
            nx_graph.add_edge(
                int(senders[e]), int(receivers[e]), edge_feature=edges[e])
    return nx_graph


def draw_structure(graph: jraph.GraphsTuple) -> None:
    nx_graph = convert_networkx(graph)
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos=pos, with_labels=True, node_size=500, font_color='yellow')
    plt.show()


def simplified_gcn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    nodes, _, receivers, senders, _, _, _ = graph
    update_node_fn = lambda nodes: nodes
    nodes = update_node_fn(nodes)

    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
    aggregate_nodes_fn = jax.ops.segment_sum

    nodes = tree.tree_map(lambda x: aggregate_nodes_fn(x[senders], receivers, total_num_nodes), nodes)
    out_graph = graph._replace(nodes=nodes)
    return out_graph


if __name__ == "__main__":
    g = jraph.GraphsTuple(
        nodes=jnp.array([[0], [2], [4], [6]]), edges=jnp.array([[5], [6], [7], [8], [8]]),
        senders=jnp.array([0, 1, 2, 3, 0]), receivers=jnp.array([1, 2, 0, 0, 3]),
        n_node=jnp.array([4]), n_edge=jnp.array([5]), globals=jnp.array([[1]])
    )
    draw_structure(g)
    print(g.nodes)

    g = simplified_gcn(g)
    draw_structure(g)
    print(g.nodes)
