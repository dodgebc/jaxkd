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


@Partial(jax.jit, static_argnames=("k",))
def query_neighbors_pairwise(
    points: jax.Array, query: jax.Array, *, k: int
) -> tuple[jax.Array, jax.Array]:
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
    if len(points) > 2**32 - 1:
        raise ValueError("Pairwise neighbors does not support more than 2**32 - 1 points.")
    query_shaped = jnp.atleast_2d(query)

    pairwise_distances = jnp.linalg.norm(points - query_shaped[:, None], axis=-1)
    distances, indices = lax.top_k(-1 * pairwise_distances, k)
    indices = jnp.asarray(indices, dtype=int)  # top_k returns int32 even in x64 mode

    if query.ndim == 1:
        return jnp.squeeze(indices, axis=0), -1 * jnp.squeeze(distances, axis=0)
    return indices, -1 * distances


@jax.jit
def count_neighbors_pairwise(
    points: jax.Array, query: jax.Array, *, r: float | jax.Array
) -> jax.Array:
    """
    Count the neighbors inside a given radius by forming a pairwise distance matrix.
    This will not scale for large problems, but may be faster for small problems.

    Args:
        points: (N, d) Points to search.
        query: (d,) or (Q, d) Query point(s).
        r: (float) (R,) or (Q, R) Radius or radii to count neighbors within, multiple radii are done in a single tree traversal.

    Returns:
        counts: (int) (Q,) (R,) or (Q, R) Number of neighbors within the given radius(i) of query point(s).
    """
    r = jnp.asarray(r)
    query_shaped = jnp.atleast_2d(query)
    r_shaped = jnp.atleast_2d(r)
    r_shaped = jnp.broadcast_to(r_shaped, (query_shaped.shape[0], r_shaped.shape[-1]))
    pairwise_distances = jnp.linalg.norm(points - query_shaped[:, None], axis=-1)
    counts = jnp.sum(pairwise_distances[:, :, None] <= r_shaped[:, None], axis=1)
    # (Q, N) < (Q, R) -> (Q, N, R) -> (Q, R)

    if r.ndim == 0:
        counts = jnp.squeeze(counts, axis=1)
    if query.ndim == 1:
        counts = jnp.squeeze(counts, axis=0)
    return counts


@Partial(jax.jit, static_argnames=("k", "steps", "pairwise"))
def k_means(
    key: KeyArray, points: jax.Array, *, k: int, steps: int, pairwise: bool = True
) -> tuple[jax.Array, jax.Array]:
    """
    Cluster with k-means, using k-means++ initialization.

    Args:
        key: A random key.
        points: (N, d) Points to cluster.
        k: The number of clusters to produce.
        steps: The number of optimization steps to run.
        pairwise: If True, use pairwise distance rather than tree search. Recommended for small k.

    Returns:
        means: (k, d) Final cluster means.
        labels: (N,) Cluster assignment for each point. If unconverged, may not be closest mean.
    """
    initial_means = k_means_plus_plus_init(key, points, k=k, pairwise=pairwise)
    means, labels = k_means_optimize(points, initial_means, steps=steps, pairwise=pairwise)
    return means, labels


@Partial(jax.jit, static_argnames=("steps", "pairwise"))
def k_means_optimize(
    points: jax.Array, initial_means: jax.Array, *, steps: int, pairwise: bool = True
) -> tuple[jax.Array, jax.Array]:
    """
    Optimize k-means clusters.

    Args:
        points: (N, d) Points to cluster.
        initial_means: (k, d) Initial cluster means.
        steps: Number of optimization steps to run.
        pairwise: If True, use pairwise distance rather than tree search. Recommended for small k.

    Returns:
        means: (k, d) Final cluster means.
        labels: (N,) Cluster assignment for each point. If unconverged, may not be closest mean.
    """
    n_points, _ = points.shape
    k, _ = initial_means.shape

    def step(carry, _):
        means, _ = carry
        if pairwise:
            labels, _ = query_neighbors_pairwise(means, points, k=1)
        else:
            tree = build_tree(means)
            labels, _ = query_neighbors(tree, points, k=1)
        labels = jnp.squeeze(labels, axis=-1)
        total = jax.ops.segment_sum(points, labels, k)
        count = jax.ops.segment_sum(jnp.ones_like(points), labels, k)
        means = total / count
        return (means, labels), None

    (means, labels), _ = lax.scan(
        step, (initial_means, jnp.zeros(n_points, dtype=int)), length=steps
    )
    return means, labels


@Partial(jax.jit, static_argnames=("k", "pairwise"))
def k_means_plus_plus_init(
    key: KeyArray, points: jax.Array, *, k: int, pairwise: bool = True
) -> jax.Array:
    """
    Initialize means for k-means clustering using the k-means++ algorithm.

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
            _, distances = query_neighbors_pairwise(masked_means, points, k=1)
        else:
            tree = build_tree(masked_means)
            _, distances = query_neighbors(tree, points, k=1)
        distances = jnp.squeeze(distances, axis=-1)
        square_distances = jnp.square(distances)
        probability = square_distances / jnp.sum(square_distances)
        next_mean = jr.choice(key, a=n_points, p=probability)
        indices = lax.dynamic_update_slice(indices, next_mean[jnp.newaxis], (i,))
        return indices, None

    indices, _ = lax.scan(step, indices, (keys[1:], jnp.arange(1, k)))
    return points[indices]
