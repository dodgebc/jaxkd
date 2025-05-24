from typing import Any
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.tree_util import Partial
from .tree import build_tree, query_neighbors

__all__ = [
    "query_neighbors_pairwise",
    "count_neighbors_pairwise",
    "k_means",
    "k_means_optimize",
    "k_means_plus_plus_init",
]

KeyArray = Any


@Partial(jax.jit, static_argnums=(2,))
def query_neighbors_pairwise(points, query, k):
    """
    Find the k nearest neighbors by forming a pairwise distance matrix.
    This will not scale for large problems, but may be faster for small problems.

    Args:
        points: (N, d) Points to search.
        query: (d,) or (Q, d) Query point(s).
        k (int): Number of neighbors to return.

    Returns:
        neighbors: (k,) or (Q, k) Indices of the k nearest neighbors of query point(s).
        distances: (k,) or (Q, k) Distances to the k nearest neighbors of query point(s).
    """
    query_shaped = jnp.atleast_2d(query)
    pairwise_distances = jnp.linalg.norm(points - query_shaped[:, None], axis=-1)
    distances, indices = lax.top_k(-1 * pairwise_distances, k)
    if query.ndim == 1:
        return indices.squeeze(0), -1 * distances.squeeze(0)
    return indices, -1 * distances


@jax.jit
def count_neighbors_pairwise(points: jax.Array, query: jax.Array, *, r: float | jax.Array):
    """
    Count the neighbors within a given radius in by forming a pairwise distance matrix.
    This will not scale for large problems, but may be faster for small problems.

    Args:
        points: (N, d) Points to search.
        query: (d,) or (Q, d) Query point(s).
        r: (float) (R,) or (Q, R) Radius or radii to count neighbors within, multiple radii are done in a single tree traversal.

    Returns:
        counts: (1,) (Q,) (R,) or (Q, R) Number of neighbors within the given radius(i) of query point(s).
    """
    query_shaped = jnp.atleast_2d(query)
    r_shaped = jnp.atleast_2d(r)
    r_shaped = jnp.broadcast_to(r_shaped, (query_shaped.shape[0], r_shaped.shape[-1]))
    pairwise_distances = jnp.linalg.norm(points - query_shaped[:, None], axis=-1)
    # (Q, N) < (Q, R) -> (Q, N, R) -> (Q, R)
    counts = jnp.sum(pairwise_distances[:, :, None] <= r_shaped[:, None], axis=1)
    if query.ndim == 1 and r.ndim == 0:
        return counts.squeeze((0, 1))
    if query.ndim == 1 and r.ndim == 1:
        return counts.squeeze(0)
    if query.ndim == 2 and r.ndim == 0:
        return counts.squeeze(1)
    return counts


@Partial(jax.jit, static_argnames=("k", "iter", "pairwise"))
def k_means(
    key: KeyArray, points: jax.Array, *, k: int, iter: int, pairwise: bool = True
) -> jax.Array:
    """
    Cluster with k-means, using k-means++ initialization.

    Args:
        key: A random key.
        points: (N, d) Points to cluster.
        k: The number of clusters to produce.
        iter: The number of iterations to run.
        pairwise: If True, use pairwise distance rather than tree search. Recommended for small k.

    Returns:
        means: (k, d) Final cluster means.
        clusters: (N,) Cluster assignment for each point.
    """
    initial_means = k_means_plus_plus_init(key, points, k=k, pairwise=pairwise)
    means, clusters = k_means_optimize(points, initial_means, iter=iter, pairwise=pairwise)
    return means, clusters


@Partial(jax.jit, static_argnames=("iter", "pairwise"))
def k_means_optimize(
    points: jax.Array, initial_means: jax.Array, *, iter: int, pairwise: bool = True
) -> tuple[jax.Array, jax.Array]:
    """
    Optimize k-means clusters.

    Args:
        points: (N, d) Points to cluster.
        initial_means: (k, d) Initial cluster means.
        iter: Number of iterations to run.
        pairwise: If True, use pairwise distance rather than tree search. Recommended for small k.

    Returns:
        means: (k, d) Final cluster means.
        clusters: (N,) Cluster assignment for each point.
    """
    n_points, _ = points.shape
    k, _ = initial_means.shape

    def step(carry, _):
        means, _ = carry
        if pairwise:
            clusters = query_neighbors_pairwise(means, points, k=1)[0].squeeze(-1)
        else:
            tree = build_tree(means)
            clusters = query_neighbors(tree, points, k=1)[0].squeeze(-1)
        total = jax.ops.segment_sum(points, clusters, k)
        count = jax.ops.segment_sum(jnp.ones_like(points), clusters, k)
        means = total / count
        return (means, clusters), None

    (means, clusters), _ = lax.scan(
        step, (initial_means, jnp.zeros(n_points, dtype=int)), length=iter
    )
    return means, clusters


@Partial(jax.jit, static_argnames=("k", "pairwise"))
def k_means_plus_plus_init(
    key: KeyArray, points: jax.Array, *, k: int, pairwise: bool = True
) -> jax.Array:
    """
    Initialize means for k-means clustering using the k-means plus plus algorithm.

    Args:
        key: A random key.
        points: (N, d) Points to cluster.
        k: The number of means to produce.
        pairwise: If True, use pairwise distance rather than tree search. Recommended for small k.

    Returns:
        means: (k, d) Initial cluster means.
    """
    # Initialize the centroid array.
    n_points, _ = points.shape
    indices = -1 * jnp.ones(k, dtype=int)
    keys = jr.split(key, k)

    # Choose the first centroid randomly.
    first_idx = jr.randint(keys[0], shape=(), minval=0, maxval=n_points)
    indices = indices.at[0].set(first_idx)

    def step(indices, key_i):
        key, i = key_i
        masked_means = jnp.where(indices[:, jnp.newaxis] >= 0, points[indices], jnp.inf)
        if pairwise:
            distances = query_neighbors_pairwise(masked_means, points, k=1)[1].squeeze(-1)
        else:
            tree = build_tree(masked_means)
            distances = query_neighbors(tree, points, k=1)[1].squeeze(-1)
        square_distances = jnp.square(distances)
        probability = square_distances / jnp.sum(square_distances)
        next_mean = jr.choice(key, a=n_points, p=probability)
        indices = lax.dynamic_update_slice(indices, next_mean[jnp.newaxis], (i,))
        return indices, None

    indices = lax.scan(step, indices, (keys[1:], jnp.arange(1, k)))[0]
    return points[indices]
