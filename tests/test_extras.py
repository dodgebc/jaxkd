import jax.numpy as jnp
import jax.random as jr
import jaxkd as jk


def test_k_means_optimize() -> None:
    kp, _ = jr.split(jr.key(83))
    points = jr.normal(kp, shape=(100, 2))
    means = points[:3]
    means, clusters = jk.extras.k_means_optimize(points, means, iter=100)
    means_tree, clusters_tree = jk.extras.k_means_optimize(points, means, iter=100, pairwise=False)

    assert jnp.allclose(
        means,
        jnp.array([[-1.1099241, -0.4269332], [0.5121479, -0.57635087], [0.04202678, 1.1571903]]),
        atol=1e-5,
    )
    assert jnp.all(clusters[:10] == jnp.array([0, 1, 0, 1, 1, 2, 0, 2, 2, 2]))

    assert jnp.allclose(
        means_tree,
        jnp.array([[-1.1099241, -0.4269332], [0.5121479, -0.57635087], [0.04202678, 1.1571903]]),
        atol=1e-5,
    )
    assert jnp.all(clusters_tree[:10] == jnp.array([0, 1, 0, 1, 1, 2, 0, 2, 2, 2]))


def test_k_means() -> None:
    kp, kc = jr.split(jr.key(83))
    points = jr.normal(kp, shape=(100, 2))
    _, clusters = jk.extras.k_means(kc, points, k=100, iter=50)

    assert jnp.all(clusters[:10] == jnp.array([37, 14, 68, 93, 42, 3, 4, 38, 26, 77]))
