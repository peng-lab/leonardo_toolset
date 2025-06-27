import math
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import SimpleITK as sitk
from jax import jit, value_and_grad
from jax.example_libraries import optimizers


def generate_mask_dict_jax(
    y,
    hy,
    fusion_mask,
    Dx,
    Dy,
    DGaussxx,
    DGaussyy,
    p_tv,
    p_hessian,
    train_params,
    sample_params,
):
    r = train_params["max_pool_kernel_size"]
    md, nd = y.shape[-2:]

    targetd_bilinear = jax.image.resize(hy, y.shape, method="bilinear")
    targets_f = jax.image.resize(
        targetd_bilinear, (1, 1, md, nd // sample_params["r"]), "bilinear"
    )

    mask_tv = (
        jnp.arctan2(
            jnp.abs(
                jax.lax.conv_general_dilated(
                    jnp.pad(y, p_tv, "reflect"), Dx, (1, 1), "VALID"
                )
            ),
            jnp.abs(
                jax.lax.conv_general_dilated(
                    jnp.pad(y, p_tv, "reflect"), Dy, (1, 1), "VALID"
                )
            ),
        )
        ** 10
    )

    mask_hessian = (
        jnp.arctan2(
            jnp.abs(
                jax.lax.conv_general_dilated(
                    jnp.pad(y, p_hessian, "reflect"),
                    DGaussxx,
                    (1, 1),
                    "VALID",
                )
            ),
            jnp.abs(
                jax.lax.conv_general_dilated(
                    jnp.pad(y, p_hessian, "reflect"),
                    DGaussyy,
                    (1, 1),
                    "VALID",
                )
            ),
        )
        ** 10
    )

    mask_valid = jnp.zeros_like(mask_hessian)
    for i, ao in enumerate(sample_params["angle_offset_individual"]):
        mask_xi_f = np.isin(sample_params["angle_offset"], ao)
        mask_xi = mask_xi_f[:, None].reshape(-1)
        mask_valid += mask_xi[None, :, None, None] * fusion_mask[:, i : i + 1, :, :]
    mask_valid = mask_valid > 0
    mask_tv = mask_tv * mask_valid
    mask_hessian = mask_hessian * mask_valid

    mask_tv_f = -hk.max_pool(
        -mask_tv,
        [1, sample_params["r"]],
        [1, sample_params["r"]],
        "VALID",
        channel_axis=1,
    )

    ind_tv_f = jnp.argmax(mask_tv_f, axis=1, keepdims=True)
    mask_tv_f = jnp.max(mask_tv_f, axis=1, keepdims=True)

    ind_tv = jnp.argmax(mask_tv, axis=1, keepdims=True)
    mask_tv = jnp.max(mask_tv, axis=1, keepdims=True)

    mask_hessian_f = -hk.max_pool(
        -mask_hessian,
        [1, sample_params["r"]],
        [1, sample_params["r"]],
        "VALID",
        channel_axis=1,
    )

    ind_hessian_f = jnp.argmax(mask_hessian_f, axis=1, keepdims=True)
    mask_hessian_f = jnp.max(mask_hessian_f, axis=1, keepdims=True)

    ind_hessian = jnp.argmax(mask_hessian, axis=1, keepdims=True)
    mask_hessian = jnp.max(mask_hessian, axis=1, keepdims=True)

    mask_tv = hk.max_pool(mask_tv, [Dx.shape[-2], Dx.shape[-1]], [1, 1], "SAME")
    mask_hessian = hk.max_pool(
        mask_hessian, [DGaussxx.shape[-2], DGaussxx.shape[-2]], [1, 1], "SAME"
    )

    y_pad = jnp.pad(targets_f, ((0, 0), (0, 0), (0, 0), (r // 2, r // 2)), "constant")
    inds = jax.lax.conv_general_dilated_patches(y_pad, (1, r), (1, 1), "VALID").argmax(
        axis=1, keepdims=True
    )
    inds = inds + jnp.arange(targets_f.shape[-1])[None, None, None, :] - r // 2

    y_pad = jnp.pad(y, ((0, 0), (0, 0), (0, 0), (r // 2, r // 2)), "constant")
    ind = jax.lax.conv_general_dilated_patches(y_pad, (1, r), (1, 1), "VALID").argmax(
        axis=1, keepdims=True
    )
    ind = ind + jnp.arange(0, y.shape[-1])[None, None, None, :] - r // 2

    mask_tv_f = hk.max_pool(
        mask_tv_f,
        [Dx.shape[-2], Dx.shape[-1]],
        [1, 1],
        "SAME",
    )
    mask_hessian_f = hk.max_pool(
        mask_hessian_f,
        [DGaussxx.shape[-2], DGaussxx.shape[-2]],
        [1, 1],
        "SAME",
    )

    mask_tv_f = mask_tv_f.at[
        :, :, jnp.arange(y.shape[-2])[None, None, :, None], inds
    ].set(0)
    mask_hessian_f = mask_hessian_f.at[
        :, :, jnp.arange(y.shape[-2])[None, None, :, None], inds
    ].set(0)

    t = jnp.linspace(0, y.shape[-2] - 1, (y.shape[-2] - 1) * sample_params["r"] + 1)
    t = jnp.concatenate((t, t[1 : sample_params["r"]] + t[-1]))
    coor = jnp.zeros((4, hy.shape[-2], hy.shape[-1]))
    coor = coor.at[2, :, :].set(t[:, None])
    coor = coor.at[3, :, :].set(jnp.arange(hy.shape[-1])[None, :])

    mask = jnp.zeros_like(y)
    mask = mask.at[
        0,
        0,
        jnp.arange(y.shape[-2])[None, None, ::2, None],
        ind[:, :, ::2, :],
    ].set(1)

    mask_dict = {
        "mask_tv": mask_tv,
        "mask_hessian": mask_hessian,
        "mask_hessian_f": mask_hessian_f,
        "mask_tv_f": mask_tv_f,
        "ind_tv": ind_tv,
        "ind_hessian": ind_hessian,
        "ind_hessian_f": ind_hessian_f,
        "ind_tv_f": ind_tv_f,
        "coor": coor,
        "non_positive_mask": mask,
    }

    return mask_dict, targets_f, targetd_bilinear


def generate_mapping_matrix(
    angle,
    m,
    n,
):
    affine = sitk.Euler2DTransform()
    affine.SetCenter([m / 2, n / 2])
    affine.SetAngle(angle / 180 * math.pi)
    A = np.array(affine.GetMatrix()).reshape(2, 2)
    c = np.array(affine.GetCenter())
    t = np.array(affine.GetTranslation())
    T = np.eye(3, dtype=np.float32)
    T[0:2, 0:2] = A
    T[0:2, 2] = -np.dot(A, c) + t + c
    return T


def generate_mapping_coordinates(
    angle,
    m,
    n,
    reshape=True,
):
    T = generate_mapping_matrix(angle, m, n)
    id = np.array([[0, 0], [0, n], [m, 0], [m, n]]).T
    if reshape:
        out_bounds = T[:2, :2] @ id
        out_plane_shape = (np.ptp(out_bounds, axis=1) + 0.5).astype(int)
    else:
        out_plane_shape = np.array([m, n])

    out_center = T[:2, :2] @ ((out_plane_shape - 1) / 2)
    in_center = (np.array([m, n]) - 1) / 2
    offset = in_center - out_center
    xx, yy = jnp.meshgrid(
        jnp.linspace(0, out_plane_shape[0] - 1, out_plane_shape[0]),
        jnp.linspace(0, out_plane_shape[1] - 1, out_plane_shape[1]),
    )
    T = jnp.array(T)
    z = (
        jnp.dot(T[:2, :2], jnp.stack((xx, yy)).reshape(2, -1)).reshape(
            2, out_plane_shape[1], out_plane_shape[0]
        )
        + offset[:, None, None]
    )
    z = z.transpose(0, 2, 1)
    z = jnp.concatenate((jnp.zeros_like(z), z))[:, None, None]
    return z


@optimizers.optimizer
def cADAM(
    step_size,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
):
    step_size = optimizers.make_schedule(step_size)

    def init(x0):
        return x0, jnp.zeros_like(x0), jnp.zeros_like(x0)

    def update(i, g, state):
        x, m, v = state
        m = (1 - b1) * g + b1 * m
        v = (1 - b2) * jnp.array(g) * jnp.conjugate(g) + b2 * v
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params


def initialize_cmplx_model_jax(
    network,
    key,
    dummy_input,
):
    net_params = network.init(key, **dummy_input)
    return net_params


class update_jax:
    def __init__(
        self,
        network,
        Loss,
        learning_rate,
    ):
        self.opt_init, self.opt_update, self.get_params = cADAM(learning_rate)
        self.loss = Loss
        self._network = network

    @partial(jit, static_argnums=(0))
    def __call__(
        self,
        step,
        params,
        opt_state,
        aver,
        xf,
        y,
        mask_dict,
        hy,
        targets_f,
        targetd_bilinear,
    ):
        (l, A), grads = value_and_grad(self.loss, has_aux=True)(
            params,
            self._network,
            {
                "aver": aver,
                "Xf": xf,
                "target": y,
                "target_hr": hy,
                "coor": mask_dict["coor"],
            },
            targetd_bilinear,
            mask_dict,
            hy,
            targets_f,
        )
        grads = jax.tree_util.tree_map(jnp.conjugate, grads)
        opt_state = self.opt_update(step, grads, opt_state)
        return l, self.get_params(opt_state), opt_state, A
