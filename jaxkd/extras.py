import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import Partial
from collections import namedtuple

from .tree import build_tree, query_neighbors

@Partial(jax.jit, static_argnums=(2,))
def query_neighbors_pairwise(points, query, k):
    """
    Find the k-nearest neighbors in a k-d tree by forming a pairwise distance matrix.
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
    pairwise_distances = jnp.linalg.norm(points - query_shaped[:,None], axis=-1)
    distances, indices = lax.top_k(-1 * pairwise_distances, k)
    if query.ndim == 1:
        return indices.squeeze(0), -1 * distances.squeeze(0)
    return indices, -1 * distances

@jax.jit
def count_neighbors_pairwise(points, query, radius):
    """
    Count the neighbors within a given radius in a k-d tree by forming a pairwise distance matrix.

    Args:
        points: (N, d) Points to search.
        query: (d,) or (Q, d) Query point(s).
        radius: (float) (R,) or (Q, R) Radius or radii to count neighbors within, multiple radii are done in a single tree traversal.

    Returns:
        counts: (1,) (Q,) (R,) or (Q, R) Number of neighbors within the given radius(i) of query point(s).
    """
    query_shaped = jnp.atleast_2d(query)
    radius_shaped = jnp.atleast_2d(radius)
    radius_shaped = jnp.broadcast_to(radius_shaped, (query_shaped.shape[0], radius_shaped.shape[-1]))
    pairwise_distances = jnp.linalg.norm(points - query_shaped[:,None], axis=-1)
    counts = jnp.sum(pairwise_distances[:,:,None] <= radius_shaped[:,None], axis=1) # (Q, N) < (Q, R) -> (Q, N, R) -> (Q, R)
    if query.ndim == 1 and radius.ndim == 0: return counts.squeeze((0, 1))
    if query.ndim == 1 and radius.ndim == 1: return counts.squeeze(0)
    if query.ndim == 2 and radius.ndim == 0: return counts.squeeze(1)
    return counts


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