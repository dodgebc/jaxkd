import jax
import jax.numpy as jnp
import jaxkd as jk

def test_k_means():
    kp, _ = jax.random.split(jax.random.key(83))
    points = jax.random.normal(kp, shape=(100, 2))
    means = points[:3]
    means, clusters = jk.extras.k_means(points, means, 100)

    assert jnp.allclose(means, jnp.array([[-1.1099241, -0.4269332],[0.5121479, -0.57635087],[0.04202678, 1.1571903]]), atol=1e-5)
    assert jnp.all(clusters[:10] == jnp.array([0, 1, 0, 1, 1, 2, 0, 2, 2, 2]))
