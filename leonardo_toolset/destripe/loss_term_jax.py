import math
from functools import partial

import jax
import jax.numpy as jnp
import jaxwt
import numpy as np
from jax import jit


class GuidedFilterLoss:
    def __init__(self, r, eps=1e-9):
        self.r, self.eps = r, eps

    def diff_x(self, input, r):
        return input[:, :, 2 * r + 1 :, :] - input[:, :, : -2 * r - 1, :]

    def diff_y(self, input, r):
        return input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]

    @partial(jit, static_argnums=(0,))
    def boxfilter(self, input):
        return self.diff_x(
            self.diff_y(
                jnp.pad(
                    input,
                    ((0, 0), (0, 0), (self.r + 1, self.r), (self.r + 1, self.r)),
                    mode="reflect",
                ).cumsum(3),
                self.r,
            ).cumsum(2),
            self.r,
        )

    @partial(jit, static_argnums=(0,))
    def __call__(self, x, y):
        N = self.boxfilter(jnp.ones_like(x))
        mean_x_y = self.boxfilter(x) / N
        mean_x2 = self.boxfilter(x * y) / N
        cov_xy = mean_x2 - mean_x_y * mean_x_y
        var_x = mean_x2 - mean_x_y * mean_x_y
        A = cov_xy / (var_x + self.eps)  # jnp.clip(var_x, self.eps, None)
        b = mean_x_y - A * mean_x_y
        A, b = self.boxfilter(A) / N, self.boxfilter(b) / N
        return A * x + b


def non_pos_unit(
    outputGNNraw_original,
    outputGNNraw,
    mask_dict,
    targets,
    r,
):
    m, n = outputGNNraw_original.shape[-2:]
    outputGNNraw2 = (
        outputGNNraw * (1 - mask_dict["non_positive_mask"])
        + targets * mask_dict["non_positive_mask"]
    )
    outputGNNraw_original = jnp.concatenate(
        (outputGNNraw_original, outputGNNraw, outputGNNraw2), 0
    )
    outputGNNraw_original_f = jax.image.resize(
        outputGNNraw_original,
        (
            3,
            1,
            m,
            n // r,
        ),
        method="bilinear",
    )
    return outputGNNraw_original, outputGNNraw_original_f


def identical_unit(
    outputGNNraw_original,
    outputGNNraw,
    mask_dict,
    targets,
    r,
):
    m, n = outputGNNraw_original.shape[-2:]
    outputGNNraw_original = jnp.concatenate(
        (
            outputGNNraw_original,
            outputGNNraw,
        ),
        0,
    )
    outputGNNraw_original_f = jax.image.resize(
        outputGNNraw_original,
        (
            2,
            1,
            m,
            n // r,
        ),
        method="bilinear",
    )
    return outputGNNraw_original, outputGNNraw_original_f


class Loss_jax:
    def __init__(
        self,
        train_params,
        shape_params,
    ):
        super().__init__()
        self.lambda_tv = train_params["lambda_tv"]
        self.lambda_hessian = train_params["lambda_hessian"]
        self.lambda_masking_mse = train_params["lambda_masking_mse"]
        self.angleOffset = shape_params["angle_offset"]

        self.DGaussxx_f, self.DGaussyy_f, self.DGaussxy_f = [], [], []
        for i in range(len(self.angleOffset)):
            xx, yy, xy = self.generateHessianKernel(
                train_params["hessian_kernel_sigma"],
                shape_params,
                [self.angleOffset[i] + 0.0],
            )
            self.DGaussxx_f.append(xx)
            self.DGaussyy_f.append(yy)
            self.DGaussxy_f.append(xy)
        self.DGaussxx_f = jnp.concatenate(self.DGaussxx_f, 0)
        self.DGaussyy_f = jnp.concatenate(self.DGaussyy_f, 0)
        self.DGaussxy_f = jnp.concatenate(self.DGaussxy_f, 0)

        self.DGaussxx, self.DGaussyy, self.DGaussxy = [], [], []
        for i in range(len(self.angleOffset)):
            xx, yy, xy = self.generateHessianKernel(
                train_params["hessian_kernel_sigma"],
                shape_params,
                np.rad2deg(
                    np.arctan(
                        shape_params["r"]
                        * np.tan(np.deg2rad([self.angleOffset[i] + +0.0]))
                    )
                ).tolist(),
            )
            self.DGaussxx.append(xx)
            self.DGaussyy.append(yy)
            self.DGaussxy.append(xy)

        self.DGaussxx = jnp.concatenate(self.DGaussxx, 0)
        self.DGaussyy = jnp.concatenate(self.DGaussyy, 0)
        self.DGaussxy = jnp.concatenate(self.DGaussxy, 0)

        self.Dx_f, self.Dy_f = [], []
        self.Dx, self.Dy = [], []
        for i in range(len(self.angleOffset)):
            Dx_f, Dy_f = self.total_variation_kernel([self.angleOffset[i] + 0.0])
            Dx, Dy = self.total_variation_kernel(
                [
                    np.rad2deg(
                        np.arctan(
                            shape_params["r"]
                            * np.tan(np.deg2rad(self.angleOffset[i] + 0.0))
                        )
                    )
                ]
            )
            self.Dx_f.append(Dx_f)
            self.Dy_f.append(Dy_f)
            self.Dx.append(Dx)
            self.Dy.append(Dy)

        self.Dx_f = jnp.concatenate(self.Dx_f, 0)
        self.Dy_f = jnp.concatenate(self.Dy_f, 0)
        self.Dx = jnp.concatenate(self.Dx, 0)
        self.Dy = jnp.concatenate(self.Dy, 0)

        self.DGaussxy = (
            self.DGaussxy
            / jnp.abs(self.DGaussxy).sum(axis=(-2, -1), keepdims=True)
            * jnp.abs(self.Dx).sum(axis=(-2, -1), keepdims=True)
        )
        self.DGaussyy = (
            self.DGaussyy
            / jnp.abs(self.DGaussyy).sum(axis=(-2, -1), keepdims=True)
            * jnp.abs(self.Dx).sum(axis=(-2, -1), keepdims=True)
        )
        self.DGaussxx = (
            self.DGaussxx
            / jnp.abs(self.DGaussxx).sum(axis=(-2, -1), keepdims=True)
            * jnp.abs(self.Dx).sum(axis=(-2, -1), keepdims=True)
        )

        self.DGaussxy_f = (
            self.DGaussxy_f
            / jnp.abs(self.DGaussxy_f).sum(axis=(-2, -1), keepdims=True)
            * jnp.abs(self.Dx_f).sum(axis=(-2, -1), keepdims=True)
        )
        self.DGaussyy_f = (
            self.DGaussyy_f
            / jnp.abs(self.DGaussyy_f).sum(axis=(-2, -1), keepdims=True)
            * jnp.abs(self.Dx_f).sum(axis=(-2, -1), keepdims=True)
        )
        self.DGaussxx_f = (
            self.DGaussxx_f
            / jnp.abs(self.DGaussxx_f).sum(axis=(-2, -1), keepdims=True)
            * jnp.abs(self.Dx_f).sum(axis=(-2, -1), keepdims=True)
        )

        self.r = int(shape_params["r"])

        self.p_hessian = (
            (0, 0),
            (0, 0),
            (self.DGaussxx.shape[2] // 2, self.DGaussxx.shape[2] // 2),
            (self.DGaussxx.shape[3] // 2, self.DGaussxx.shape[3] // 2),
        )
        self.p_tv = (
            (0, 0),
            (0, 0),
            (self.Dx.shape[2] // 2, self.Dx.shape[2] // 2),
            (self.Dx.shape[3] // 2, self.Dx.shape[3] // 2),
        )

        p_hessian = (
            int(
                (
                    self.DGaussxx.shape[2]
                    + (self.DGaussxx.shape[2] - 1) * (self.r - 1)
                    + 1
                )
                / 2
            )
            - 1
        )
        p_tv = (
            int((self.Dx.shape[2] + (self.Dx.shape[2] - 1) * (self.r - 1) + 1) / 2) - 1
        )

        self.p_hessian_f = ((0, 0), (0, 0), self.p_hessian[2], (p_hessian, p_hessian))
        self.p_tv_f = ((0, 0), (0, 0), self.p_tv[2], (p_tv, p_tv))

        self.GF_pad = train_params["max_pool_kernel_size"]

        self.total_variation_cal = (
            self.TotalVariationLoss
            if len(self.angleOffset) > 1
            else self.TotalVariationLoss_plain
        )

        self.hessian_cal = (
            self.HessianRegularizationLoss
            if len(self.angleOffset) > 1
            else self.HessianRegularizationLoss_plain
        )

        self.GuidedFilterLoss = GuidedFilterLoss(
            r=train_params["max_pool_kernel_size"],
            eps=1,
        )
        if shape_params["non_positive"]:
            self.non_postive_uint = non_pos_unit
            self.main_loss = lambda a, b: 0
        else:
            self.non_postive_uint = identical_unit
            self.main_loss = self.gf_loss

    def gf_loss(self, targets, outputGNNraw_original):
        return 10 * jnp.sum(
            jnp.abs(
                self.GuidedFilterLoss(targets, targets)
                - self.GuidedFilterLoss(outputGNNraw_original, outputGNNraw_original)
            )
        )

    def total_variation_kernel(
        self,
        angle_list,
    ):
        gx, gy = self.rotatableKernel(Wsize=3, sigma=1)
        Dx_ = []
        Dy_ = []
        for i in angle_list:
            Dx = math.cos(-i / 180 * math.pi) * gx + math.sin(-i / 180 * math.pi) * gy
            Dy = (
                math.cos(-i / 180 * math.pi + math.pi / 2) * gx
                + math.sin(-i / 180 * math.pi + math.pi / 2) * gy
            )
            Dx_.append(Dx)
            Dy_.append(Dy)
        Dx_ = jnp.stack(Dx_, 0)[:, None]
        Dy_ = jnp.stack(Dy_, 0)[:, None]
        Dx_ = Dx_ - Dx_.mean(axis=(-2, -1), keepdims=True)
        Dy_ = Dy_ - Dy_.mean(axis=(-2, -1), keepdims=True)
        return Dx_, Dy_

    def generateHessianKernel(
        self,
        Sigma,
        shape_params,
        angleOffset,
    ):
        Wsize = math.ceil(3 * Sigma)
        KernelSize = 2 * (2 * Wsize + 1) - 1
        gx, gy = self.rotatableKernel(Wsize, Sigma)

        md = shape_params["md"] if shape_params["is_vertical"] else shape_params["nd"]
        nd = shape_params["nd"] if shape_params["is_vertical"] else shape_params["md"]
        gxFFT2, gyFFT2 = jnp.fft.fft2(gx, (md, nd)), jnp.fft.fft2(gy, (md, nd))
        DGaussxx = jnp.zeros((len(angleOffset), 1, KernelSize, KernelSize))
        DGaussxy = jnp.zeros((len(angleOffset), 1, KernelSize, KernelSize))
        DGaussyy = jnp.zeros((len(angleOffset), 1, KernelSize, KernelSize))
        for i, A in enumerate(angleOffset):
            a, b, c, d = (
                math.cos(-A / 180 * math.pi),
                math.sin(-A / 180 * math.pi),
                math.cos(-A / 180 * math.pi + math.pi / 2),
                math.sin(-A / 180 * math.pi + math.pi / 2),
            )
            DGaussxx = DGaussxx.at[i : i + 1].set(
                jnp.fft.ifft2(
                    (a * gxFFT2 + b * gyFFT2) * (a * gxFFT2 + b * gyFFT2), axes=(-2, -1)
                ).real[None, None, :KernelSize, :KernelSize]
            )
            DGaussyy = DGaussyy.at[i : i + 1].set(
                jnp.fft.ifft2(
                    (c * gxFFT2 + d * gyFFT2) * (c * gxFFT2 + d * gyFFT2), axes=(-2, -1)
                ).real[None, None, :KernelSize, :KernelSize]
            )
            DGaussxy = DGaussxy.at[i : i + 1].set(
                jnp.fft.ifft2(
                    (c * gxFFT2 + d * gyFFT2) * (a * gxFFT2 + b * gyFFT2), axes=(-2, -1)
                ).real[None, None, :KernelSize, :KernelSize]
            )
        DGaussxx = DGaussxx - DGaussxx.mean(axis=(-2, -1), keepdims=True)
        DGaussxy = DGaussxy - DGaussxy.mean(axis=(-2, -1), keepdims=True)
        DGaussyy = DGaussyy - DGaussyy.mean(axis=(-2, -1), keepdims=True)
        return DGaussxx, DGaussyy, DGaussxy

    def rotatableKernel(
        self,
        Wsize,
        sigma,
    ):
        k = jnp.linspace(-Wsize, Wsize, 2 * Wsize + 1)[None, :]
        g = jnp.exp(-(k**2) / (2 * sigma**2))
        gp = -(k / sigma) * jnp.exp(-(k**2) / (2 * sigma**2))
        return g.T * gp, gp.T * g

    @partial(jit, static_argnums=(0,))
    def TotalVariationLoss(
        self,
        x,
        target,
        Dx,
        Dy,
        mask,
        ind,
    ):
        return (
            jnp.abs(
                jnp.take_along_axis(
                    jax.lax.conv_general_dilated(
                        jnp.pad(x, self.p_tv, "reflect"),
                        Dx,
                        (1, 1),
                        "VALID",
                        feature_group_count=x.shape[1],
                    ),
                    ind,
                    axis=1,
                )
            )
            * mask
        ).sum() + mask.sum() / mask.size * jnp.abs(
            jnp.take_along_axis(
                jax.lax.conv_general_dilated(
                    jnp.pad(x - target, self.p_tv, "reflect"), Dy, (1, 1), "VALID"
                ),
                ind,
                axis=1,
            )
        ).sum()

    @partial(jit, static_argnums=(0,))
    def TotalVariationLoss_plain(
        self,
        x,
        target,
        Dx,
        Dy,
        mask,
        ind,
    ):
        return (
            jnp.abs(
                jax.lax.conv_general_dilated(
                    jnp.pad(x, self.p_tv, "reflect"),
                    Dx,
                    (1, 1),
                    "VALID",
                    feature_group_count=x.shape[1],
                ),
            )
            * mask
        ).sum() + mask.sum() / mask.size * jnp.abs(
            jax.lax.conv_general_dilated(
                jnp.pad(x - target, self.p_tv, "reflect"), Dy, (1, 1), "VALID"
            ),
        ).sum()

    @partial(jit, static_argnums=0)
    def HessianRegularizationLoss(
        self,
        x,
        target,
        DGaussxx,
        DGaussyy,
        DGaussxy,
        mask,
        ind,
    ):
        return (
            (
                jnp.abs(
                    jnp.take_along_axis(
                        jax.lax.conv_general_dilated(
                            jnp.pad(x, self.p_hessian, "reflect"),
                            DGaussxx,
                            (1, 1),
                            "VALID",
                            feature_group_count=x.shape[1],
                        ),
                        ind,
                        axis=1,
                    )
                )
                * mask
            ).sum()
            + mask.sum()
            / mask.size
            * jnp.abs(
                jnp.take_along_axis(
                    jax.lax.conv_general_dilated(
                        jnp.pad(x - target, self.p_hessian, "reflect"),
                        DGaussyy,
                        (1, 1),
                        "VALID",
                    ),
                    ind,
                    axis=1,
                )
            ).sum()
            + 2
            * mask.sum()
            / mask.size
            * jnp.abs(
                jnp.take_along_axis(
                    jax.lax.conv_general_dilated(
                        jnp.pad(x - target, self.p_hessian, "reflect"),
                        DGaussxy,
                        (1, 1),
                        "VALID",
                    ),
                    ind,
                    axis=1,
                )
            ).sum()
        )

    @partial(jit, static_argnums=0)
    def HessianRegularizationLoss_plain(
        self,
        x,
        target,
        DGaussxx,
        DGaussyy,
        DGaussxy,
        mask,
        ind,
    ):
        return (
            (
                jnp.abs(
                    jax.lax.conv_general_dilated(
                        jnp.pad(x, self.p_hessian, "reflect"),
                        DGaussxx,
                        (1, 1),
                        "VALID",
                        feature_group_count=x.shape[1],
                    ),
                )
                * mask
            ).sum()
            + mask.sum()
            / mask.size
            * jnp.abs(
                jax.lax.conv_general_dilated(
                    jnp.pad(x - target, self.p_hessian, "reflect"),
                    DGaussyy,
                    (1, 1),
                    "VALID",
                ),
            ).sum()
            + 2
            * mask.sum()
            / mask.size
            * jnp.abs(
                jax.lax.conv_general_dilated(
                    jnp.pad(x - target, self.p_hessian, "reflect"),
                    DGaussxy,
                    (1, 1),
                    "VALID",
                ),
            ).sum()
        )

    def __call__(
        self,
        params,
        network,
        inputs,
        targets,
        mask_dict,
        hy,
        targets_f,
    ):
        outputGNNraw_original, output_att = network.apply(params, **inputs)
        outputGNNraw_full = (
            jax.scipy.ndimage.map_coordinates(
                outputGNNraw_original - targets,
                mask_dict["coor"],
                order=1,
                mode="reflect",
            )[None, None]
            + hy
        )
        outputGNNraw = jax.image.resize(
            outputGNNraw_full, targets.shape, method="bilinear"
        )

        mse = self.main_loss(targets, outputGNNraw_original)
        output_att_dict = [jaxwt.wavedec2(output_att, "db4", level=1, mode="reflect")]
        lconst = 6
        for _ in range(lconst - 1):
            output_att_dict.append(
                jaxwt.wavedec2(output_att_dict[-1][0], "db4", level=1, mode="reflect")
            )

        output_gnn_hr_dict = [
            jaxwt.wavedec2(outputGNNraw_full, "db4", level=1, mode="reflect")
        ]
        for _ in range(lconst - 1):
            output_gnn_hr_dict.append(
                jaxwt.wavedec2(
                    output_gnn_hr_dict[-1][0], "db4", level=1, mode="reflect"
                )
            )

        for i in range(lconst):
            mse = mse + 20 * jnp.sum(
                jnp.abs(output_att_dict[i][1][0] - output_gnn_hr_dict[i][1][0])
            )
            mse = mse + 20 * jnp.sum(
                jnp.abs(output_att_dict[i][1][1] - output_gnn_hr_dict[i][1][1])
            )
            mse = mse + 40 * jnp.sum(
                jnp.abs(output_att_dict[i][1][2] - output_gnn_hr_dict[i][1][2])
            )

        outputGNNraw_original, outputGNNraw_original_f = self.non_postive_uint(
            outputGNNraw_original,
            outputGNNraw,
            mask_dict,
            targets,
            self.r,
        )

        tv = self.total_variation_cal(
            outputGNNraw_original[1:],
            targets,
            self.Dx,
            self.Dy,
            mask_dict["mask_tv"],
            mask_dict["ind_tv"],
        )

        tv = tv + self.total_variation_cal(
            outputGNNraw_original_f[1:],
            targets_f,
            self.Dx_f,
            self.Dy_f,
            mask_dict["mask_tv_f"],
            mask_dict["ind_tv_f"],
        )

        hessian = self.hessian_cal(
            outputGNNraw_original[1:],
            targets,
            self.DGaussxx,
            self.DGaussyy,
            self.DGaussxy,
            mask_dict["mask_hessian"],
            mask_dict["ind_hessian"],
        )

        hessian = hessian + self.hessian_cal(
            outputGNNraw_original_f[1:],
            targets_f,
            self.DGaussxx_f,
            self.DGaussyy_f,
            self.DGaussxy_f,
            mask_dict["mask_hessian_f"],
            mask_dict["ind_hessian_f"],
        )

        return (
            mse + self.lambda_tv * tv + self.lambda_hessian * hessian,
            outputGNNraw_original[0:1],
        )
