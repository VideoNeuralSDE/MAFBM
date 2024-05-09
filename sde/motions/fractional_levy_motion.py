# https://github.com/davidkraljic/fractional_levy_motion

import numpy as np
import scipy.fft


def levy_stable(alpha: float, beta: float, size: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """
    Generate random values sampled from an alpha-stable distribution.
    Notice that this method is "exact", in the sense that is derived
    directly from the definition of stable variable.

    :param alpha: stability parameter in (0, 2]
    :param beta: skewness parameter in [-1, -1]
    :param mu: location parameter in (-inf, inf)
    :param sigma: scale parameter in (0, inf)
    :param size: size of resulting array
    """

    if alpha == 2:
        return mu + np.random.standard_normal(size) * np.sqrt(2.0) * sigma  # variance is 2*sigma**2 when alpha=2 (Gaussian)

    # Fails for alpha exactly equal to 1.0
    # but works fine for alpha infinitesimally greater or lower than 1.0
    radius = 1e-15  # <<< this number is *very* small
    if np.absolute(alpha - 1.0) < radius:
        # So doing this will make almost exactly no difference at all
        alpha = 1.0 + radius

    r1 = np.random.random(size)
    r2 = np.random.random(size)
    pi = np.pi

    a = 1.0 - alpha
    b = r1 - 0.5
    c = a * b * pi
    e = beta * np.tan(np.pi * alpha / 2.0)
    f = (-(np.cos(c) + e * np.sin(c)) / (np.log(r2) * np.cos(b * pi))) ** (a / alpha)
    g = np.tan(pi * b / 2.0)
    h = np.tan(c / 2.0)
    i = 1.0 - g ** 2.0
    j = f * (2.0 * (g - h) * (g * h + 1.0) - (h * i - 2.0 * g) * e * 2.0 * h)
    k = j / (i * (h ** 2.0 + 1.0)) + e * (f - 1.0)

    return mu + sigma * k


def truncated_levy_stable(trunc: float, alpha: float, beta: float, size: int, mu: float = 0.0,
                          sigma: float = 1.0) -> np.ndarray:
    """
    Create the empirical levy stable distribution with extremes truncated.

    :param trunc: absolute value at which to truncate distribution. (truncation is symmetric)
    :param alpha: stability parameter in (0, 2]
    :param beta: skewness parameter in [-1, -1]
    :param mu: location parameter in (-inf, inf)
    :param sigma: scale parameter in (0, inf)
    :param size: size of resulting array
    """

    z = levy_stable(alpha=alpha, beta=beta, mu=mu, sigma=sigma, size=size)

    too_big = np.where(np.abs(z) > trunc)[0]

    while too_big.size > 0:
        z[too_big] = levy_stable(alpha=alpha, beta=beta, mu=mu, sigma=sigma, size=too_big.size)
        too_big_remaining = np.where(np.abs(z[too_big]) > trunc)[0]
        too_big = too_big[too_big_remaining]

    return z


def memory_efficient_truncated_levy_stable(trunc: float, alpha: float, beta: float, size: int,
                                           mu: float = 0.0, sigma: float = 1.0, steps: int = 256) -> np.ndarray:
    """
    Create the empirical levy stable distribution with extremes truncated.
    To prevent large inefficient allocations of memory the distribution is generated in chunks.

    :param trunc: absolute value at which to truncate distribution. (truncation is symmetric)
    :param alpha: stability parameter in (0, 2]
    :param beta: skewness parameter in [-1, -1]
    :param mu: location parameter in (-inf, inf)
    :param sigma: scale parameter in (0, inf)
    :param size: size of resulting array
    :param steps: number of chunks to generated the final array
    """
    step_length = size // steps
    remaining = size % steps

    out = np.zeros(size)
    for i in range(steps):
        out[i * step_length:(i + 1) * step_length] = truncated_levy_stable(trunc=trunc, alpha=alpha, beta=beta,
                                                                           size=step_length, mu=mu, sigma=sigma)

    if remaining > 0:
        out[-remaining:] = truncated_levy_stable(trunc=trunc, alpha=alpha, beta=beta,
                                                 size=remaining, mu=mu, sigma=sigma)

    return out


def flm(H: float, alpha: float, N: int, trunc: float, scale: float = 1, C: float = 1, m: int = 256, M: int = 6000,
        steps: int = 256) -> np.ndarray:
    """
    Generate realizations of fractional levy motion, also know as linear fractional stable motion.
    Please ensure that m * ( M + N ) is a power of 2 because ffts are most efficient then.

    :param H: Hurst parameter. Also known as the self-similarity parameter
    :param alpha: the tail-exponent of the stable distribution (between 0 and 2). Lower alpha = heavier tails
    :param m: 1/m is the mesh size
    :param M: kernel cut-off parameter
    :param C: normalization parameter
    :param N: size of sample
    :param scale: scale parameter of Levy distribution
    :param trunc: truncate levy distr at +/-trunc
    :param steps: break down generation of levy stable samples into steps number of batches
    """

    Na = m * (M + N)

    if alpha < 0 or alpha > 2:
        raise ValueError('Alpha must be greater than 0 and less than or equal to 2.')

    mh = 1 / m
    d = H - 1 / alpha
    t0 = np.linspace(mh, 1, m) ** d
    t1 = np.linspace(1 + mh, M, int((M - (1 + mh)) / mh) + 1)
    t1 = t1 ** d - (t1 - 1) ** d
    A = mh ** (1 / alpha) * np.concatenate((t0, t1))
    C = C * (np.abs(A) ** alpha).sum() ** (-1 / alpha)
    A *= C

    A = scipy.fft.fft(A, n=Na)
    Z = memory_efficient_truncated_levy_stable(trunc=trunc, alpha=alpha, beta=0, size=Na, mu=0, sigma=scale,
                                               steps=steps)
    Z = scipy.fft.fft(Z, Na)
    w = np.real(scipy.fft.ifft(Z * A, Na))

    return w[0:N * m:m]