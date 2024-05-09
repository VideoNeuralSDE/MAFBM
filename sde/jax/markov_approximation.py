import jax
import jax.numpy as jnp
from functools import partial
import jax.scipy.special as sp
from jax.scipy.optimize import minimize
import diffrax


def br_fraction(sequence):
    g = 0
    for u in reversed(sequence):
        g = u / (1 + g)
    return g

# def gammaincc_ez(a, z):
#     return sp.gammaincc(a, z) * jnp.exp(z)

@jax.jit
def gammaincc_ez_fractions(a, z, num_fractions=5):
    # numerically stable alternative of sp.gammaincc(a, z) * jnp.exp(z)
    # num_fractions should be uneven!
    # accuracy is good for num_fractions >= 5, but not for z < 10.
    # switch to sp.gammaincc(a, z) * jnp.exp(z) for z < 10.
    a_m = [1] + [i // 2 - a if i % 2 == 0 else i // 2 for i in range(2, num_fractions + 1)]
    g = 0.
    for u in reversed(a_m):
        g = u / z / (1 + g)
    return z ** a * g / sp.gamma(a)

@jax.jit
def gammaincc_ez(a, z):
    return jnp.where(
        z < 10.,
        sp.gammaincc(a, z) * jnp.exp(z),
        gammaincc_ez_fractions(a, z),
    )

def gamma_by_r(num_k, r, offset=0.):
    n = (num_k + 1) / 2 + offset
    k = jnp.arange(1, num_k + 1)
    gamma = r ** (k - n)
    return gamma

def gamma_by_gamma_max(num_k, gamma_max, offset=0.):
    r = gamma_max ** (2 / (num_k - 1 - 2 * offset))
    return gamma_by_r(num_k, r, offset)

def omega_riemann(gamma, hurst):
    # based on Riemann sum approximation of gamma integrals
    alpha = hurst + .5

    d1 = gamma[1:] - gamma[:-1]
    d1ma = gamma[1:] ** (1 - alpha) - gamma[:-1] ** (1 - alpha)
    d2ma = gamma[1:] ** (2 - alpha) - gamma[:-1] ** (2 - alpha)

    # pad everything with a zero, so that we can easily handle k = 1 and k = K
    g = jnp.pad(gamma, 1)
    d1 = jnp.pad(d1, 1) + 1e-11
    d1ma = jnp.pad(d1ma, 1)
    d2ma = jnp.pad(d2ma, 1)

    omegas = jnp.where(
        hurst < .5,
        1 / sp.gamma(alpha) / sp.gamma(1 - alpha) * ((d2ma[:-1] / (2 - alpha) - g[:-2] * d1ma[:-1] / (1 - alpha)) / d1[:-1] + (g[2:] * d1ma[1:] / (1 - alpha) - d2ma[1:] / (2 - alpha)) / d1[1:]),
        - 1 / sp.gamma(alpha) / sp.gamma(2 - alpha) / (2 - alpha) * (d2ma[:-1] / d1[:-1] - d2ma[1:] / d1[1:])
    )
    return omegas

def omega_optimized_1(gamma, hurst, time_horizon, return_cost=False):
    # based on Variance of approximation error with type I fBM
    gamma_i, gamma_j = gamma[None, :], gamma[:, None]

    A = (2 * time_horizon + (jnp.exp(- gamma_i * time_horizon) - 1) / gamma_i + (jnp.exp(- gamma_j * time_horizon) - 1) / gamma_j) / (gamma_i + gamma_j)
    b = 2 * time_horizon / gamma ** (hurst + .5) - time_horizon ** (hurst + .5) / (gamma * sp.gamma(hurst + 1.5)) + (jnp.exp(- gamma * time_horizon) - gammaincc_ez(hurst + .5, gamma * time_horizon)) / gamma ** (hurst + 1.5)
    omega = jax.scipy.linalg.solve(A, b, assume_a='sym')
    if return_cost:
        c = time_horizon ** (2 * hurst + 1) / (2 * hurst + 1)
        cost = (omega.T @ A @ omega - b.T @ omega + c) / time_horizon
        return omega, cost
    else:
        return omega

def omega_optimized_2(gamma, hurst, time_horizon, return_cost=False):
    # based on Variance of approximation error with type II fBM
    gamma_i, gamma_j = gamma[None, :], gamma[:, None]

    A = (time_horizon + (jnp.exp(- (gamma_i + gamma_j) * time_horizon) - 1) / (gamma_i + gamma_j)) / (gamma_i + gamma_j)
    b = time_horizon / gamma ** (hurst + .5) * sp.gammainc(hurst + .5, gamma * time_horizon) - (hurst + .5) / gamma ** (hurst + 1.5) * sp.gammainc(hurst + 1.5, gamma * time_horizon)

    # A += jnp.eye(len(gamma)) * 1e-5   # some regularization

    omega = jax.scipy.linalg.solve(A, b, assume_a='sym')
    if return_cost:
        c = time_horizon ** (2 * hurst + 1) / (2 * hurst) / (2 * hurst + 1) / sp.gamma(hurst + .5) ** 2
        cost = (omega.T @ A @ omega - 2 * b.T @ omega + c) / time_horizon
        return omega, cost
    else:
        return omega

@partial(jax.jit, static_argnames=('model', 'num_steps'))
def solve(params, model, omega, x0, y0, t0, num_steps, dt, key):
    # assert model.gamma.max() * dt < .5, 'dt too large for stable integration, please reduce dt or decrease largest gamma'
    ts = t0 + jnp.arange(num_steps + 1) * dt
    dWs = jax.random.normal(key, (num_steps,)) * jnp.sqrt(dt)

    def step(z, i):
        x, y = z[:1], z[1:]
        t = ts[i]
        dW = dWs[i]

        u = model.u(params, t, x, y)
        dy = - model.gamma * y * dt + u * dt + dW
        dx = model.b(params, t, x) * dt + model.s(params, t, x) * (omega * dy).sum()

        x += dx
        y += dy
        z = jnp.concatenate([x, y])
        return z, (x, u)

    z0 = jnp.concatenate([x0, y0])
    _, (xs, us) = jax.lax.scan(step, z0, jnp.arange(num_steps))
    log_path = .5 * (us ** 2).sum() * dt

    xs = jnp.concatenate([x0, xs[:, 0]])
    return ts, xs, log_path

@partial(jax.jit, static_argnames=('model', 'num_steps'))
def solve_vector(params, model, omega, x0, y0, t0, num_steps, dt, key, args):
    num_latents = len(x0)
    ts = t0 + jnp.arange(num_steps + 1) * dt
    dWs = jax.random.normal(key, (num_steps, num_latents)) * jnp.sqrt(dt)

    def step(z, i):
        x, y = z
        t = ts[i]
        dW = dWs[i]

        u = model.u(params, t, x, y, args)
        dy = - model.gamma[None, :] * y * dt + u[:, None] * dt + dW[:, None]
        dx = model.b(params, t, x, args) * dt + model.s(params, t, x, args) * (omega[None, :] * dy).sum(axis=-1)

        x += dx
        y += dy
        # z = jnp.concatenate([x, y], axis=-1)
        return (x, y), (x, u)

    # z0 = jnp.concatenate([x0, y0], axis=-1)
    _, (xs, us) = jax.lax.scan(step, (x0, y0), jnp.arange(num_steps))
    log_path = .5 * (us ** 2).sum(axis=0) * dt

    xs = jnp.concatenate([x0[None, :], xs], axis=0)
    return ts, xs, log_path

class CustomPath(diffrax.AbstractPath):
    num_latents: int
    num_k: int
    brownian_path: diffrax.VirtualBrownianTree

    def __init__(self, t0, t1, tol, num_latents, num_k, key):
        self.num_latents = num_latents
        self.num_k = num_k
        self.brownian_path = diffrax.VirtualBrownianTree(t0=t0, t1=t1, tol=tol, shape=(num_latents,), key=key)

    def evaluate(self, t0, t1=None, left: bool = True):
        dW = self.brownian_path.evaluate(t0=t0, t1=t1, left=left)
        return (dW, dW, jnp.zeros(self.num_latents))

def solve_diffrax(params, model, omega, x0, y0, ts, dt, key, solver=diffrax.Euler(), args=None):
    def drift(t, state, args):
        x, y, _ = state
        u = model.u(params, t, x, y, args)
        dy = - model.gamma[:, None] * y + u[None, :]
        dx = model.b(params, t, x, args) + model.s(params, t, x, args) * (omega[:, None] * dy).sum(axis=0)
        return (dx, dy, .5 * u ** 2)

    def diffusion(t, state, args):
        x, _, _ = state
        return (model.s(params, t, x, args) * omega.sum(), jnp.ones((model.num_k, model.num_latents)), jnp.zeros(model.num_latents))

    state_init = (x0, y0.T, jnp.zeros(model.num_latents))
    brownian_motion = CustomPath(ts[0], ts[-1], dt / 10, model.num_latents, model.num_k, key)
    terms = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.WeaklyDiagonalControlTerm(diffusion, brownian_motion))
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        ts[0],
        ts[-1],
        dt0=dt,
        y0=state_init,
        saveat=saveat,
        # stepsize_controller=diffrax.PIDController(rtol=1e-2, atol=1e-2),
        max_steps=4096,
        args=args,
    )
    xs, ys, log_path_int = sol.ys
    return xs, log_path_int[-1]
