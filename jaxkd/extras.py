import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import Partial
from collections import namedtuple

from .tree import build_tree, query_neighbors


@Partial(jax.jit, static_argnums=(2,))
def k_means(points, initial_means, iterations):
    """
    Simple k-means clustering using k-nearest neighbor search.

    Args:
        points: (N, d) Points to cluster.
        initial_means: (k, d) Initial cluster means.
        iterations: Number of iterations to run.

    Returns:
        means: (k, d) Final cluster means.
        cluster: (N,) Cluster assignment for each point.
    """
    def step(carry, _):
        means, _ = carry
        tree = build_tree(means)
        cluster = query_neighbors(tree, points, 1)[0].squeeze(-1)
        means = jax.ops.segment_sum(points, cluster, len(means)) / jax.ops.segment_sum(jnp.ones_like(points), cluster, len(means))
        return (means, cluster), None
    
    (means, cluster), _ = jax.lax.scan(step, (initial_means, jnp.zeros(len(points), dtype=int)), length=iterations)
    return means, cluster