from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.tree_util import Partial

from .tree import build_tree, query_neighbors, query_neighbors_pairwise


__all__ = ['k_means', 'k_means_optimize']


KeyArray = Any


def _k_means_plus_plus_init(key: KeyArray, points: jax.Array, k: int) -> jax.Array:
    """
    Initialize means for k-means clustering using the k-means plus plus algorithm.

    Args:
        key: A random key.
        points: (n_samples, n_features) Points to cluster.
        k: The number of means to produce.

    Returns:
        means: (k, n_features) Final cluster means.
    """
    # Initialize the centroid array.
    n_samples, _ = points.shape
    cluster = -1 * jnp.ones(k, dtype=int)
    keys = jr.split(key, k)

    # Choose the first centroid randomly.
    first_idx = jr.randint(keys[0], shape=(), minval=0, maxval=n_samples)
    cluster = cluster.at[0].set(first_idx)

    def step(cluster: jax.Array, key_i_i: tuple[jax.Array, KeyArray]) -> tuple[jax.Array, None]:
        key_i, i = key_i_i
        masked_means = jnp.where(cluster[:, jnp.newaxis] >= 0, points[cluster], jnp.inf)
        distances = query_neighbors_pairwise(masked_means, points, 1)[1].squeeze(-1)
        square_distances = jnp.square(distances)
        probability = square_distances / jnp.sum(square_distances)
        next_mean = jr.choice(key_i, a=n_samples, p=probability)
        cluster = lax.dynamic_update_slice(cluster, next_mean[jnp.newaxis], (i,))
        return cluster, None

    # for i in range(1, k): cluster = step(cluster, key[i], i)[0]
    cluster = lax.scan(step, cluster, (keys[1:], jnp.arange(1, k)))[0]
    return points[cluster]


@Partial(jax.jit, static_argnames=('iterations',))
def k_means_optimize(points: jax.Array,
                     initial_means: jax.Array,
                     *,
                     iterations: int
                     ) -> tuple[jax.Array, jax.Array]:
    """
    Optimize k-means clusters using k-nearest neighbor search.

    Args:
        points: (n_samples, n_features) Points to cluster.
        initial_means: (k, n_features) Initial cluster means.
        iterations: Number of iterations to run.

    Returns:
        means: (k, n_features) Final cluster means.
        cluster: (n_samples,) Cluster assignment for each point.
    """
    n_samples, _ = points.shape
    k, _ = initial_means.shape

    def step(carry: tuple[jax.Array, jax.Array], _: None
             ) -> tuple[tuple[jax.Array, jax.Array], None]:
        means, _cluster = carry
        tree = build_tree(means)
        cluster = query_neighbors(tree, points, 1)[0].squeeze(-1)
        total = jax.ops.segment_sum(points, cluster, k)
        count = jax.ops.segment_sum(jnp.ones_like(points), cluster, k)
        means = total / count
        return (means, cluster), None

    (means, cluster), _ = lax.scan(step, (initial_means, jnp.zeros(n_samples, dtype=int)),
                                   length=iterations)
    return means, cluster


@Partial(jax.jit, static_argnames=('k', 'iterations'))
def k_means(key: KeyArray, points: jax.Array, *, k: int, iterations: int) -> jax.Array:
    """
    Cluster with k-means.

    Args:
        key: A random key.
        points: (n_samples, n_features) Points to cluster.
        k: The number of clusters to produce.
        iterations: The number of iterations to do.
    """
    initial_means = _k_means_plus_plus_init(key, points, k)
    _, cluster_assignments = k_means_optimize(points, initial_means, iterations=iterations)
    return cluster_assignments
