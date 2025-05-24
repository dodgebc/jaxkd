import jax.numpy as jnp
import jax.random as jr
import jaxkd as jk


def test_k_means_optimize() -> None:
    kp, _ = jr.split(jr.key(83))
    points = jr.normal(kp, shape=(100, 2))
    means = points[:3]
    means, clusters = jk.extras.k_means_optimize(points, means, iterations=100)

    assert jnp.allclose(means, jnp.array([[-1.1099241, -0.4269332],
                                          [0.5121479, -0.57635087],
                                          [0.04202678, 1.1571903]]), atol=1e-5)
    assert jnp.all(clusters[:10] == jnp.array([0, 1, 0, 1, 1, 2, 0, 2, 2, 2]))


def test_k_means() -> None:
    kp, _ = jr.split(jr.key(83))
    points = jr.normal(kp, shape=(100, 2))
    clusters = jk.extras.k_means(jr.key(102), points, k=100, iterations=50)

    assert jnp.all(clusters[:10] == jnp.array([99, 82, 18, 94, 5, 9, 33, 43, 19, 97]))
