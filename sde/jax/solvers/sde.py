import jax
import jax.numpy as jnp
import math
import numpy as np
from stochastic.processes.continuous import BrownianMotion, FractionalBrownianMotion


def generate_dmotion(shape, dt, num_steps):
    bm = BrownianMotion(t=dt*num_steps)
    num_paths = math.prod(shape)
    motion = np.stack([bm.sample(num_steps) for _ in range(num_paths)], axis=-1)
    motion = motion.astype(np.float32)
    motion = motion.reshape((num_steps + 1, *shape))
    dmotion = motion[1:] - motion[:-1]
    return dmotion


def sdeint_fixed_ts(sde, dmotion, t0, y0, step, num_steps, num_substeps=1, logqp=False, prior=False):
    assert not (prior and logqp), 'Impossible to calculate logqp when integrating the prior.'
    if prior:
        drift_fn = sde.h
        diffusion_fn = sde.g
    else:
        drift_fn = sde.f
        diffusion_fn = sde.g
        if logqp:
            prior_drift_fn = sde.h

    dt = step / num_substeps
    t = t0
    y = y0
    dm = iter(dmotion)
    log_ratio = []
    ys = [y]
    for _ in range(num_steps):
        logqp_sub = 0.
        for _ in range(num_substeps):
            drift, diffusion = drift_fn(t, y), diffusion_fn(t, y)
            y = y + drift * dt + diffusion * next(dm)
            if logqp:
                u = (drift - prior_drift_fn(t, y)) / diffusion
                logqp_sub += .5 * (u ** 2).sum(axis=-1) * dt
            t += dt
        ys.append(y)
        if logqp:
            log_ratio.append(logqp_sub)

    if logqp:
        log_ratio = jnp.stack(log_ratio)
        return jnp.stack(ys), log_ratio
    else:
        return jnp.stack(ys)
