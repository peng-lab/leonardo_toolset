from typing import List
import numpy as np
import scipy
import copy
import math
import torch

try:
    import haiku as hk
    import jax
    import jax.numpy as jnp

    from leonardo_toolset.destripe.utils_jax import generate_mapping_coordinates
except Exception as e:
    print(f"Error: {e}. Proceed without jax")
    pass


def transform_cmplx_model(
    model,
    backend,
    device,
    **model_kwargs,
):
    def forward_pass(**x):
        net = model(**model_kwargs)
        return net(**x)

    if backend == "jax":
        network = hk.without_apply_rng(hk.transform(forward_pass))
    else:
        network = model(**model_kwargs).to(device)
    return network


def crop_center(
    img,
    cropy,
    cropx,
):
    y, x = img.shape[-2:]
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[..., starty : starty + cropy, startx : startx + cropx]


def global_correction(
    mean,
    result,
):
    means = scipy.signal.savgol_filter(mean, min(21, len(mean)), 1)
    MIN, MAX = result.min(), result.max()
    result = result - mean[:, None, None] + means[:, None, None]
    result = (result - result.min()) / (result.max() - result.min()) * (MAX - MIN) + MIN
    return np.clip(result, 0, 65535).astype(np.uint16)


def destripe_train_params(
    resample_ratio: int = 3,
    gf_kernel_size: int = 49,
    hessian_kernel_sigma: float = 0.5,
    lambda_masking_mse: int = 2,
    lambda_tv: float = 1,
    lambda_hessian: float = 1,
    inc: int = 16,
    n_epochs: int = 300,
    wedge_degree: float = 29,
    n_neighbors: int = 16,
    gf_mode: int = 1,
    backend: str = "jax",
):
    kwargs = locals()
    return kwargs


def NeighborSampling(
    m,
    n,
    backend,
    k_neighbor=16,
):
    """
    Do neighbor sampling

    Parameters:
    ---------------------
    m: int
        size of neighbor along X dim
    n: int
        size of neighbor along Y dim
    k_neigher: int, data range [1, 32], 16 by default
        number of neighboring points
    """
    if backend == "jax":
        dep_package = jnp
        key = jax.random.key(0)
    else:
        dep_package = np
    width = 11
    NI = dep_package.zeros((m * n, k_neighbor), dtype=dep_package.int32)
    grid_x, grid_y = dep_package.meshgrid(
        dep_package.linspace(1, m, m), dep_package.linspace(1, n, n), indexing="ij"
    )
    grid_x, grid_y = grid_x - math.floor(m / 2) - 1, grid_y - math.floor(n / 2) - 1
    grid_x, grid_y = grid_x.reshape(-1) ** 2, grid_y.reshape(-1) ** 2

    iter_num = dep_package.sqrt((grid_x + grid_y).max()) // width + 1

    mask_outer = (grid_x + grid_y) < (
        width * dep_package.arange(1, iter_num + 1)[:, None]
    ) ** 2
    mask_inner = (grid_x + grid_y) >= (
        width * dep_package.arange(0, iter_num)[:, None]
    ) ** 2
    mask = mask_outer * mask_inner
    ind = dep_package.where(mask)
    _, counts = dep_package.unique(ind[0], return_counts=True)
    counts_cumsum = dep_package.cumsum(counts)

    low = dep_package.concatenate(
        (dep_package.array([0]), counts_cumsum[:-1]),
    )

    low = low.repeat(counts)
    high = counts_cumsum
    high = high.repeat(counts)
    if backend == "jax":
        indc = jax.random.randint(key, (k_neighbor, len(low)), low, high).T
    else:
        indc = np.random.randint(low, high, (k_neighbor, len(low))).T
    if backend == "jax":
        NI = NI.at[ind[1]].set(ind[1][indc])
    else:
        NI[ind[1]] = ind[1][indc]

    zero_freq = (m * n) // 2
    NI = NI[:zero_freq, :]
    if backend == "jax":
        NI = NI.at[NI > zero_freq].set(2 * zero_freq - NI[NI > zero_freq])
        NI = NI.at[NI == zero_freq].set(zero_freq - 1)
    else:
        NI[NI > zero_freq] = 2 * zero_freq - NI[NI > zero_freq]
        NI[NI == zero_freq] = zero_freq - 1
    return dep_package.concatenate(
        (
            dep_package.linspace(0, NI.shape[0] - 1, NI.shape[0])[
                :, dep_package.newaxis
            ],
            NI,
        ),
        axis=1,
    ).astype(dep_package.int32)


def WedgeMask(
    md,
    nd,
    Angle,
    deg,
    backend,
):
    """
    Add docstring here
    """
    if backend == "jax":
        dep_package = jnp
    else:
        dep_package = np
    md_o, nd_o = copy.deepcopy(md), copy.deepcopy(nd)
    md = max(md_o, nd_o)
    nd = max(md_o, nd_o)

    Xv, Yv = dep_package.meshgrid(
        dep_package.linspace(0, nd, nd + 1), dep_package.linspace(0, md, md + 1)
    )
    tmp = dep_package.arctan2(Xv, Yv)
    tmp = dep_package.hstack((dep_package.flip(tmp[:, 1:], 1), tmp))
    tmp = dep_package.vstack((dep_package.flip(tmp[1:, :], 0), tmp))
    if Angle != 0:
        if backend == "jax":
            rotate_mask = generate_mapping_coordinates(
                -Angle,
                tmp.shape[0],
                tmp.shape[1],
                False,
            )
            tmp = jax.scipy.ndimage.map_coordinates(
                tmp[None, None],
                rotate_mask,
                0,
                mode="nearest",
            )[0, 0]
        else:
            tmp = scipy.ndimage.rotate(
                tmp,
                Angle,
                reshape=False,
                mode="nearest",
                order=1,
            )

    a = crop_center(tmp, md, nd)

    tmp = Xv**2 + Yv**2
    tmp = dep_package.hstack((dep_package.flip(tmp[:, 1:], 1), tmp))
    tmp = dep_package.vstack((dep_package.flip(tmp[1:, :], 0), tmp))
    b = tmp[md - md // 2 : md + md // 2 + 1, nd - nd // 2 : nd + nd // 2 + 1]
    return crop_center(
        (
            ((a < math.pi / 180 * (90 - deg)).astype(dep_package.int32))
            * (b > 1024).astype(dep_package.int32)
        )
        != 0,
        md_o,
        nd_o,
    )


def prepare_aux(
    md: int,
    nd: int,
    is_vertical: bool,
    angleOffset: List[float] = None,
    deg: float = 29,
    Nneighbors: int = 16,
    backend="jax",
):
    if not is_vertical:
        (nd, md) = (md, nd)

    if backend == "jax":
        dep_package = jnp
    else:
        dep_package = np
    angleMask = dep_package.ones((md, nd), dtype=np.int32)
    for angle in angleOffset:
        angleMask = angleMask * WedgeMask(
            md,
            nd,
            Angle=angle,
            deg=deg,
            backend=backend,
        )

    angleMask = angleMask[None]
    angleMask = angleMask.reshape(angleMask.shape[0], -1)[:, : md * nd // 2]
    hier_mask = dep_package.where(angleMask == 1)[1]  # (3, N)

    hier_ind = dep_package.argsort(
        dep_package.concatenate(
            [dep_package.where(angleMask.reshape(-1) == index)[0] for index in range(2)]
        )
    )
    NI_all = NeighborSampling(md, nd, k_neighbor=Nneighbors, backend=backend)
    NI = dep_package.concatenate(
        [NI_all[angle_mask == 0, :].T for angle_mask in angleMask], 1
    )  # 1 : Nneighbors + 1
    if backend == "jax":
        return hier_mask, hier_ind, NI
    else:
        return (
            torch.from_numpy(hier_mask),
            torch.from_numpy(hier_ind),
            torch.from_numpy(NI),
        )
