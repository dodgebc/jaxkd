import jax.numpy as jnp
import jax.random as jr
from jax.experimental import enable_x64

import jaxkd as jk


def test_k_means_optimize() -> None:
    kp, _ = jr.split(jr.key(83))
    points = jr.normal(kp, shape=(100, 2))
    means = points[:3]
    means, labels = jk.extras.k_means_optimize(points, means, steps=100)
    means_tree, labels_tree = jk.extras.k_means_optimize(points, means, steps=100, pairwise=False)

    assert jnp.allclose(
        means,
        jnp.array([[-1.1099241, -0.4269332], [0.5121479, -0.57635087], [0.04202678, 1.1571903]]),
        atol=1e-5,
    )
    assert jnp.all(labels[:10] == jnp.array([0, 1, 0, 1, 1, 2, 0, 2, 2, 2]))

    assert jnp.allclose(
        means_tree,
        jnp.array([[-1.1099241, -0.4269332], [0.5121479, -0.57635087], [0.04202678, 1.1571903]]),
        atol=1e-5,
    )
    assert jnp.all(labels_tree[:10] == jnp.array([0, 1, 0, 1, 1, 2, 0, 2, 2, 2]))


def test_k_means() -> None:
    kp, kc = jr.split(jr.key(83))
    points = jr.normal(kp, shape=(100, 2))
    _, labels = jk.extras.k_means(kc, points, k=100, steps=50)

    assert jnp.all(labels[:10] == jnp.array([37, 14, 68, 93, 42, 3, 4, 38, 26, 77]))


def test_k_means_64() -> None:
    with enable_x64():  # type: ignore
        kp, kc = jr.split(jr.key(83))
        points = jr.normal(kp, shape=(100, 2))
        _, labels = jk.extras.k_means(kc, points, k=100, steps=50)

    assert jnp.all(labels[:10] == jnp.array([97, 98, 12, 99, 94, 90, 73, 84, 0, 93]))
