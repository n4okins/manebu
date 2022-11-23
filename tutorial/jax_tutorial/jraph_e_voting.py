
import collections
import logging
import random

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax


Problem = collections.namedtuple("Problem", ("graph", "labels"))


def get_voting_problem(min_n_voters: int, max_n_voters: int) -> Problem:
    n_candidates = 20
    n_voters = random.randint(min_n_voters, max_n_voters)
    votes = np.random.randint(0, n_candidates, size=(n_voters,))
    one_hot_votes = np.eye(n_candidates)[votes]
    winner = np.argmax(np.sum(one_hot_votes, axis=0))
    one_hot_winner = np.eye(n_candidates)[winner]

    graph = jraph.GraphsTuple(
        n_node=np.asarray([n_voters]),
        n_edge=np.asarray([0]),
        nodes=one_hot_votes,
        edges=None,
        globals=np.zeros((1, n_candidates)),
        senders=np.array([], dtype=np.int32),
        receivers=np.array([], dtype=np.int32))

    graph = jraph.pad_with_graphs(graph, max_n_voters+1, 0)

    return Problem(graph=graph, labels=one_hot_winner)


def network_definition(
        graph: jraph.GraphsTuple,
        num_message_passing_steps: int = 1) -> jraph.ArrayTree:
    @jax.vmap
    def update_fn(*args):
        size = args[0].shape[-1]
        return hk.nets.MLP([size, size])(jnp.concatenate(args, axis=-1))

    for _ in range(num_message_passing_steps):
        gn = jraph.DeepSets(
            update_node_fn=update_fn,
            update_global_fn=update_fn,
            aggregate_nodes_for_globals_fn=jraph.segment_mean,
        )
        graph = gn(graph)

    return hk.Linear(graph.globals.shape[-1])(graph.globals)


def train(num_steps: int):
    """Trains a graph neural network on an electronic voting problem."""
    train_dataset = (2, 15)
    test_dataset = (16, 20)
    random.seed(42)

    network = hk.without_apply_rng(hk.transform(network_definition))
    problem = get_voting_problem(*train_dataset)
    params = network.init(jax.random.PRNGKey(42), problem.graph)

    @jax.jit
    def prediction_loss(params, problem):
        globals_ = network.apply(params, problem.graph)
        # We interpret the globals as logits for the winner.
        # Only the first graph is real, the second graph is for padding.
        log_prob = jax.nn.log_softmax(globals_[0]) * problem.labels
        return -jnp.sum(log_prob)

    @jax.jit
    def accuracy_loss(params, problem):
        globals_ = network.apply(params, problem.graph)
        # We interpret the globals as logits for the winner.
        # Only the first graph is real, the second graph is for padding.
        equal = jnp.argmax(globals_[0]) == jnp.argmax(problem.labels)
        return equal.astype(np.int32)

    opt_init, opt_update = optax.adam(2e-4)
    opt_state = opt_init(params)

    @jax.jit
    def update(params, opt_state, problem):
        g = jax.grad(prediction_loss)(params, problem)
        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    for step in range(num_steps):
        problem = get_voting_problem(*train_dataset)
        params, opt_state = update(params, opt_state, problem)
        if step % 1000 == 0:
            train_loss = jnp.mean(
                jnp.asarray([
                    accuracy_loss(params, get_voting_problem(*train_dataset))
                    for _ in range(100)
                ])).item()
            test_loss = jnp.mean(
                jnp.asarray([
                    accuracy_loss(params, get_voting_problem(*test_dataset))
                    for _ in range(100)
                ])).item()
            logging.info("step %r loss train %r test %r", step, train_loss, test_loss)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    train(num_steps=100000)


if __name__ == "__main__":
    app.run(main)