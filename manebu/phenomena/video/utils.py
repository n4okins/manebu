import cv2.cv2 as cv2
import jax
from functools import partial
from jax import jit
import jax.numpy as jnp
from tqdm import tqdm
from collections import deque


def video_robust_pca(M, max_iter=100):
    M = jnp.array(M)
    @jit
    def shrinkage_operator(x, tau):
        return jnp.sign(x) * jnp.maximum((jnp.abs(x) - tau), jnp.zeros_like(x))
    @jit
    def svd_thresholding_operator(X, tau):
        U, S, Vh = jnp.linalg.svd(X, full_matrices=False)
        return U @ jnp.diag(shrinkage_operator(S, tau)) @ Vh


    S = jnp.zeros_like(M)
    Y = jnp.zeros_like(M)
    toi = 1e-7 * jnp.linalg.norm(M, ord="fro")
    mu = M.shape[0] * M.shape[1] / (4 * jnp.linalg.norm(M, ord=1))
    mu_inv = 1 / mu
    lam = 1 / jnp.sqrt(jnp.max(jnp.asarray(M.shape)))

    @jit
    def step(S, Y):
        L = svd_thresholding_operator(M - S + mu_inv * Y, mu_inv)
        S = shrinkage_operator(M - L + mu_inv * Y, lam * mu_inv)
        Y = Y + mu * (M - L - S)
        return L, S, Y

    for i in tqdm(range(max_iter)):
        L, S, Y = step(S, Y)
        E = jnp.linalg.norm(M - L - S, ord='fro')
        if i % 1000 == 0:
            tqdm.write(f"step {i} | error {E}")
        if E <= toi: break
    return L, S


if __name__ == "__main__":
    L, S = video_robust_pca(jnp.arange(6).reshape((2, 3)))
    print(L, S)
