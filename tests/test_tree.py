import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import enable_x64

import jaxkd as jk
from jaxkd.extras import count_neighbors_pairwise, query_neighbors_pairwise


def test_shapes():
    points = jnp.zeros((100, 3))
    queries = jnp.zeros((50, 3))
    radius = jnp.zeros((50, 20))

    # Tree construction
    tree = jk.build_tree(points)
    assert tree.points.shape == points.shape
    assert tree.indices.shape == points.shape[:-1]
    assert tree.split_dims.shape == points.shape[:-1]

    # Tree construction, not optimized
    tree = jk.build_tree(points, optimize=False)
    assert tree.points.shape == points.shape
    assert tree.indices.shape == points.shape[:-1]
    assert tree.split_dims is None

    # Queries
    neighbors, distances = jk.query_neighbors(tree, queries[0], k=10)
    assert neighbors.shape == (10,)
    assert distances.shape == (10,)
    neighbors, distances = jk.query_neighbors(tree, queries, k=10)
    assert neighbors.shape == (queries.shape[0], 10)
    assert distances.shape == (queries.shape[0], 10)

    # Five possible cases for count
    assert jk.count_neighbors(tree, queries[0], r=radius[0, 0]).shape == ()
    assert jk.count_neighbors(tree, queries[0], r=radius[0]).shape == (radius.shape[1],)
    assert jk.count_neighbors(tree, queries, r=radius[0, 0]).shape == (queries.shape[0],)
    assert jk.count_neighbors(tree, queries, r=radius[0]).shape == (
        queries.shape[0],
        radius.shape[1],
    )
    assert jk.count_neighbors(tree, queries, r=radius).shape == radius.shape

    # Pairwise queries
    neighbors, distances = query_neighbors_pairwise(points, queries[0], k=10)
    assert neighbors.shape == (10,)
    assert distances.shape == (10,)
    neighbors, distances = query_neighbors_pairwise(points, queries, k=10)
    assert neighbors.shape == (queries.shape[0], 10)
    assert distances.shape == (queries.shape[0], 10)

    # Pairwise count
    assert count_neighbors_pairwise(points, queries[0], r=radius[0, 0]).shape == ()
    assert count_neighbors_pairwise(points, queries[0], r=radius[0]).shape == (radius.shape[1],)
    assert count_neighbors_pairwise(points, queries, r=radius[0, 0]).shape == (queries.shape[0],)
    assert count_neighbors_pairwise(points, queries, r=radius[0]).shape == (
        queries.shape[0],
        radius.shape[1],
    )
    assert count_neighbors_pairwise(points, queries, r=radius).shape == radius.shape


def test_small_case():
    points = jnp.array(
        [
            [10, 46, 68, 40, 25, 15, 44, 45, 62, 53],
            [15, 63, 21, 33, 54, 43, 58, 40, 69, 67],
        ]
    ).T

    tree = jk.build_tree(points)
    assert jnp.all(tree.points == points)
    assert jnp.all(tree.indices == jnp.array([1, 5, 9, 3, 6, 2, 8, 0, 7, 4]))
    assert jnp.all(tree.split_dims == jnp.array([0, 1, 1, 0, 0, -1, -1, -1, -1, -1]))


def test_random():
    kp, kq = jr.split(jr.key(83))
    points = jr.normal(kp, shape=(1_000, 3))
    queries = jr.normal(kq, shape=(1_000, 3))

    tree = jk.build_tree(points)
    neighbors, distances = jk.query_neighbors(tree, queries, k=100)
    counts = jk.count_neighbors(tree, queries, r=0.3)

    tree_no = jk.build_tree(points, optimize=False)
    neighbors_no, distances_no = jk.query_neighbors(tree_no, queries, k=100)
    counts_no = jk.count_neighbors(tree_no, queries, r=0.3)

    neighbors_pair, distances_pair = query_neighbors_pairwise(points, queries, k=100)
    counts_pair = count_neighbors_pairwise(points, queries, r=0.3)

    assert jnp.all(neighbors == neighbors_no) & jnp.all(neighbors == neighbors_pair)
    assert jnp.all(distances == distances_no) & jnp.all(distances == distances_pair)
    assert jnp.all(counts == counts_no) & jnp.all(counts == counts_pair)


def test_random_all():
    kp, kq = jr.split(jr.key(83))
    points = jr.normal(kp, shape=(1_000, 3))
    queries = jr.normal(kq, shape=(1_000, 3))

    tree = jk.build_tree(points)
    neighbors, distances = jk.query_neighbors(tree, queries, k=1_000)
    counts = jk.count_neighbors(tree, queries, r=jnp.inf)

    tree_no = jk.build_tree(points, optimize=False)
    neighbors_no, distances_no = jk.query_neighbors(tree_no, queries, k=1_000)
    counts_no = jk.count_neighbors(tree_no, queries, r=jnp.inf)

    neighbors_pair, distances_pair = query_neighbors_pairwise(points, queries, k=1_000)
    counts_pair = count_neighbors_pairwise(points, queries, r=jnp.inf)

    assert jnp.all(neighbors == neighbors_no) & jnp.all(neighbors == neighbors_pair)
    assert jnp.all(distances == distances_no) & jnp.all(distances == distances_pair)
    assert jnp.all(counts == counts_no) & jnp.all(counts == counts_pair)


def test_random_batched():
    kp, kq = jr.split(jr.key(83))
    points = jr.normal(kp, shape=(1_000, 3))
    queries = jr.normal(kq, shape=(1_000, 3))

    tree = jk.build_tree(points)
    counts = jk.count_neighbors(tree, queries, r=jnp.array([0.3, 0.4, 0.5]))

    tree_no = jk.build_tree(points, optimize=False)
    counts_no = jk.count_neighbors(tree_no, queries, r=jnp.array([0.3, 0.4, 0.5]))

    counts_pair = count_neighbors_pairwise(points, queries, r=jnp.array([0.3, 0.4, 0.5]))
    assert jnp.all(counts == counts_pair)
    assert jnp.all(counts == counts_no)


def test_grad():
    def loss_func(points):
        tree = jk.build_tree(points)
        neighbors, _ = jk.query_neighbors(tree, points, k=5)
        distances = jnp.linalg.norm(points[:, None] - points[neighbors][:, 1:], axis=-1)
        return jnp.sum(distances**2)

    x = jnp.reshape(jnp.arange(30, dtype=jnp.float32), (10, 3))
    grad = jax.grad(loss_func)(x)

    def loss_func_pair(points):
        neighbors, _ = query_neighbors_pairwise(points, points, k=5)
        distances = jnp.linalg.norm(points[:, None] - points[neighbors][:, 1:], axis=-1)
        return jnp.sum(distances**2)

    grad_pair = jax.grad(loss_func_pair)(x)

    assert jnp.allclose(grad, grad_pair, atol=1e-5)


def test_random_64():
    # NOTE: 64 bit results will not match 32 bit results due to precission differences in calculated squared distances

    with enable_x64():  # type: ignore
        kp, kq = jr.split(jr.key(83))
        points = jr.normal(kp, shape=(1_000, 3))
        queries = jr.normal(kq, shape=(1_000, 3))

        tree = jk.build_tree(points)
        neighbors, distances = jk.query_neighbors(tree, queries, k=100)
        counts = jk.count_neighbors(tree, queries, r=0.3)

        tree_no = jk.build_tree(points, optimize=False)
        neighbors_no, distances_no = jk.query_neighbors(tree_no, queries, k=100)
        counts_no = jk.count_neighbors(tree_no, queries, r=0.3)

        neighbors_pair, distances_pair = query_neighbors_pairwise(points, queries, k=100)
        counts_pair = count_neighbors_pairwise(points, queries, r=0.3)

        assert jnp.all(neighbors == neighbors_no)
        assert jnp.all(distances == distances_no)
        assert jnp.all(counts == counts_no)
        assert jnp.all(neighbors == neighbors_pair)
        assert jnp.all(distances == distances_pair)
        assert jnp.all(counts == counts_pair)
