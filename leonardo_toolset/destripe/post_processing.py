import torch.nn as nn
from torch.optim import Adam
import scipy
import numpy as np
import torch
from leonardo_toolset.destripe.utils import crop_center
import copy
from skimage.filters import threshold_otsu
import tqdm
import torch.nn.functional as F


def generate_seg_mask(Y_raw_full, target):
    seg_mask = (10**target > threshold_otsu(10**target)) * (
        10**Y_raw_full < threshold_otsu(10**Y_raw_full)
    )

    seg_mask = np.asarray(seg_mask)[0, 0]

    return seg_mask[None, None]


class stripe_post(nn.Module):
    def __init__(self, m, n):
        super().__init__()
        self.w = nn.Parameter(torch.ones(m, n))
        self.w_negative = nn.Parameter(torch.ones(m, n))
        self.softplus = nn.Softplus()

        self.m = m
        self.n = n
        self.r = 89

        self.guidedfilterloss = GuidedFilterLoss(49, 1)
        self.alpha = nn.Parameter(torch.ones(1, 1))
        self.relu = nn.ELU()

        self.ww = nn.Parameter(torch.ones(m, n))
        self.decay_col = nn.Parameter(torch.ones(1, 1, torch.arange(m)[::3].numel(), n))
        self.seg_mask = nn.Parameter(torch.ones(1, 2, m, n))

        self.recon_guided = None

        self.weight = nn.Parameter(0.5 * torch.ones(1, n))
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, b, b_negative, foreground_b, hX, recon_old, fusion_mask, b_old, r
    ):

        if self.recon_guided is None:
            self.recon_old_max, self.ind_max = F.max_pool2d_with_indices(
                recon_old,
                (1, 2 * self.r + 1),
                padding=(0, self.r),
                stride=1,
            )
            self.recon_guided, self.A_guided, self.b_guided = self.guidedfilterloss(
                recon_old
            )
            self.p = (0, hX.shape[-1] % 2, 0, hX.shape[-2] % 2)
            self.l_row = None if hX.shape[-2] % 2 == 0 else -1
            self.l_col = None if hX.shape[-1] % 2 == 0 else -1

        b = b * self.softplus(self.w)
        decay_col = F.interpolate(
            torch.cumsum(b_negative * self.softplus(self.decay_col), -2),
            (self.m, self.n),
            mode="bilinear",
        )
        b = torch.cumsum(b, -2)

        b = b + decay_col * (1 - foreground_b)
        b_adpt = b * fusion_mask + b_old * (1 - fusion_mask)

        with torch.no_grad():
            diff = self.guidedfilterloss(hX[:, :, ::r, :] + b_adpt - recon_old)[0]

        diff_adpt = torch.clip(
            torch.diff(diff, dim=-2, prepend=diff[:, :, 0:1, :]), 0.0, None
        )
        diff_adpt = diff_adpt * self.softplus(self.ww)
        diff_adpt = torch.cumsum(diff_adpt, -2)
        b_adpt = b_adpt - diff_adpt

        return b_adpt, diff, diff_adpt, b


class compose_post(nn.Module):
    def __init__(self, m, n):
        super().__init__()
        self.w = nn.Parameter(0.5 * torch.ones(1, n))
        self.sigmoid = nn.Sigmoid()
        self.alpha_up = nn.Parameter(torch.ones(1))
        self.alpha_bottom = nn.Parameter(torch.ones(1))

    def forward(self, b_up, b_bottom, hX):
        w = self.sigmoid(self.w)
        b = w * (b_up + 0 * self.alpha_up) + (1 - w) * (b_bottom + self.alpha_bottom)
        return hX + b, b


class GuidedFilterLoss:
    def __init__(self, r, eps=1e-9):
        self.r, self.eps = r, eps

    def diff_x(self, input, r):
        return input[:, :, 2 * r :, :] - input[:, :, : -2 * r, :]

    def diff_y(self, input, r):
        return input[:, :, :, 2 * r :] - input[:, :, :, : -2 * r]

    def boxfilter(self, input):
        return self.diff_x(
            self.diff_y(
                edge_padding(input, self.r).cumsum(3),
                self.r,
            ).cumsum(2),
            self.r,
        )

    def __call__(self, x, A=None, b=None):
        if A is None:
            N = self.boxfilter(torch.ones_like(x))
            mean_x_y = self.boxfilter(x) / N
            mean_x2 = self.boxfilter(x * x) / N
            cov_xy = mean_x2 - mean_x_y * mean_x_y
            var_x = mean_x2 - mean_x_y * mean_x_y
            A = cov_xy / (var_x + self.eps)  # jnp.clip(var_x, self.eps, None)
            b = mean_x_y - A * mean_x_y
            A, b = self.boxfilter(A) / N, self.boxfilter(b) / N
            return A * x + b, A, b
        else:
            return A * x + b


def edge_padding(x, r):
    x = torch.cat(
        (x[:, :, 0:1, :].repeat(1, 1, r, 1), x, x[:, :, -1:, :].repeat(1, 1, r, 1)),
        -2,
    )
    return torch.cat(
        (x[:, :, :, 0:1].repeat(1, 1, 1, r), x, x[:, :, :, -1:].repeat(1, 1, 1, r)),
        -1,
    )


class loss_post(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        kernel_x, kernel_y = self.rotatableKernel(3, 1)
        kernel_x = kernel_x - kernel_x.mean()
        kernel_y = kernel_y - kernel_y.mean()
        self.register_buffer(
            "kernel_x",
            torch.from_numpy(np.asarray(kernel_x))[None, None].to(torch.float),
        )
        self.register_buffer(
            "kernel_y",
            torch.from_numpy(np.asarray(kernel_y))[None, None].to(torch.float),
        )

        self.GuidedFilterLoss = GuidedFilterLoss(49, 1)
        self.ptv = 3

    def rotatableKernel(
        self,
        Wsize,
        sigma,
    ):
        k = np.linspace(-Wsize, Wsize, 2 * Wsize + 1)[None, :]
        g = np.exp(-(k**2) / (2 * sigma**2))
        gp = -(k / sigma) * np.exp(-(k**2) / (2 * sigma**2))
        return g.T * gp, gp.T * g

    def forward(
        self,
        decay,
        weight,
        y,
        hX,
        weight_tvx,
        weight_tvy,
        weight_tvx_f,
        weight_tvy_f,
        b0,
        r,
    ):

        e1 = torch.conv2d(edge_padding(y, self.ptv), self.kernel_x).abs()
        e2 = torch.conv2d(edge_padding(y, self.ptv), self.kernel_y)

        mask = torch.where(
            torch.conv2d(edge_padding(hX + b0, self.ptv), self.kernel_y).abs()
            < torch.conv2d(edge_padding(hX, self.ptv), self.kernel_y).abs(),
            torch.conv2d(edge_padding(hX, self.ptv), self.kernel_y),
            torch.conv2d(edge_padding(hX + b0, self.ptv), self.kernel_y),
        )

        e22 = torch.diff(y, dim=-1, prepend=y[..., 0:1]).abs()

        e3 = torch.conv2d(edge_padding(y[:, :, :, ::r], self.ptv), self.kernel_x).abs()
        return (
            1 * (decay - weight).abs().sum()
            + (weight_tvx * e22).sum()
            + (weight_tvx * e1).sum()
            + 1 * (weight_tvy * (e2 - mask).abs())[:, :, ::1, :].sum()
            + 1 * (weight_tvx_f * e3).sum()
        )


class loss_compose_post(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        kernel_x, kernel_y = self.rotatableKernel(3, 1)
        self.register_buffer(
            "kernel_x", torch.from_numpy(kernel_x)[None, None].to(torch.float)
        )
        self.register_buffer(
            "kernel_y", torch.from_numpy(kernel_y)[None, None].to(torch.float)
        )

    def rotatableKernel(
        self,
        Wsize,
        sigma,
    ):
        k = np.linspace(-Wsize, Wsize, 2 * Wsize + 1)[None, :]
        g = np.exp(-(k**2) / (2 * sigma**2))
        gp = -(k / sigma) * np.exp(-(k**2) / (2 * sigma**2))
        return g.T * gp, gp.T * g

    def forward(self, y, hX, foreground, valid_mask, fidelity_mask):
        e1 = (
            torch.conv2d(F.pad(y, (3, 3, 3, 3), mode="reflect"), self.kernel_x).abs()
            * valid_mask
        )
        return e1.sum()  # + e2.sum()


def train_post_process_module(
    hX,
    b,
    valid_mask,
    seg_mask,
    fusion_mask,
    foreground,
    n_epochs,
    r,
    device,
):
    m, n = hX[:, :, ::r, :].shape[-2:]
    model = stripe_post(m, n).to(device)
    loss = loss_post().to(device)
    opt = Adam(model.parameters(), lr=1)
    b_sparse = (
        torch.clip(
            torch.diff(b[:, :, ::r, :], dim=-2, prepend=0 * b[:, :, 0:1, :]), 0.0, None
        )
        * valid_mask[:, :, ::r, :]
        * (1 - seg_mask)[:, :, ::r, :]
    )
    b_sparse_negative = (
        torch.clip(
            torch.diff(b[:, :, :: r * 3, :], dim=-2, prepend=b[:, :, 0:1, :]), None, 0
        )
        * valid_mask[:, :, :: r * 3, :]
        * (1 - seg_mask)[:, :, :: r * 3, :]
    )
    rr = 10
    weight_tvx = (1 - seg_mask)[:, :, ::rr, :] * valid_mask[:, :, ::rr, :]
    weight_tvy = (
        valid_mask[:, :, ::rr, :]
        * foreground[:, :, ::rr, :]
        * (1 - seg_mask)[:, :, ::rr, :]
    )
    weight_tvx_f = (1 - seg_mask)[:, :, ::rr, ::rr] * valid_mask[:, :, ::rr, ::rr]
    weight_tvy_f = (
        valid_mask[:, :, ::rr, ::rr]
        * foreground[:, :, ::rr, ::rr]
        * (1 - seg_mask)[:, :, ::rr, ::rr]
    )

    for e in tqdm.tqdm(range(n_epochs)):
        b_new, decay, weight, b_new2 = model(
            b_sparse,
            b_sparse_negative,
            seg_mask[:, :, ::r, :],
            hX,
            torch.maximum((hX + b), hX)[:, :, ::r, :],
            fusion_mask[::r, :],
            b[:, :, ::r, :],
            r,
        )

        b_new = (F.interpolate(b_new, hX.shape[-2:], mode="bilinear") + hX)[
            :, :, ::rr, :
        ]
        ll = loss(
            decay,
            weight,
            b_new,
            hX[:, :, ::rr, :],
            weight_tvx,
            weight_tvy,
            weight_tvx_f,
            weight_tvy_f,
            b[:, :, ::rr, :],
            rr,
        )
        opt.zero_grad()
        ll.backward()
        opt.step()
    return b_new2  # -hX[:, :, ::r, :]


def last_nonzero(arr, mask, axis, invalid_val=np.nan):
    if mask is None:
        mask = arr != 0
    if type(mask) is not np.ndarray:
        mask = mask.cpu().detach().numpy()
    val = mask.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def linear_propagation(
    b,
    hX,
    result_network,
    foreground,
    angle_offset,
    illu_orient="top",
    n_epochs=1000,
    fusion_mask=None,
):
    print(illu_orient)
    r = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_mask = generate_seg_mask(result_network, hX)

    m0, n0 = hX.shape[-2:]
    # hX0 = copy.deepcopy(hX)

    foreground = scipy.ndimage.rotate(
        foreground, -angle_offset, order=0, axes=(-2, -1), mode="constant"
    )
    if fusion_mask is not None:
        fusion_mask = scipy.ndimage.rotate(
            fusion_mask, -angle_offset, order=1, axes=(-2, -1), mode="constant"
        )
        fusion_mask = torch.from_numpy(fusion_mask).to(device)

    foreground_b = scipy.ndimage.binary_erosion(
        foreground,
        np.ones((1, 1, 9, 9), dtype=bool),
    )
    foreground_b = (foreground_b == 0) * foreground

    valid_mask = np.ones_like(b)
    valid_mask = scipy.ndimage.rotate(
        valid_mask, -angle_offset, order=1, axes=(-2, -1), mode="constant"
    )

    seg_mask = (
        scipy.ndimage.rotate(
            seg_mask, -angle_offset, order=1, axes=(-2, -1), mode="constant"
        )
        > 0
    )

    # seg_mask = scipy.ndimage.binary_dilation(
    #     seg_mask,
    #     np.ones((1, 1, 1, 59), dtype=bool),
    # )

    b = scipy.ndimage.rotate(b, -angle_offset, order=1, axes=(-2, -1), mode="nearest")
    hX = scipy.ndimage.rotate(hX, -angle_offset, order=1, axes=(-2, -1), mode="nearest")

    m, n = b[:, :, ::r, :].shape[-2:]

    fidelity_mask = valid_mask * (np.abs(b) < 1e-3)

    b = torch.from_numpy(b).to(device)
    hX = torch.from_numpy(hX).to(device)

    fidelity_mask = torch.from_numpy(fidelity_mask).to(device) + 0.0
    valid_mask = torch.from_numpy(valid_mask).to(device)
    seg_mask = torch.from_numpy(seg_mask).to(device) + 0.0
    # seg_mask_l = torch.from_numpy(seg_mask_l).to(device) + 0.0
    foreground = torch.from_numpy(foreground).to(device) + 0.0
    foreground_b = torch.from_numpy(foreground_b).to(device) + 0.0

    if fusion_mask is None:
        fusion_mask = torch.ones_like(hX)

    if "top" in illu_orient:
        b_new = train_post_process_module(
            hX,
            b,
            valid_mask,
            seg_mask,
            fusion_mask,
            foreground,
            n_epochs,
            r,
            device,
        )

        # b_new = b_new - hX
        # b_new[w == 0] = 0

        recon_up = (
            (
                hX
                + F.interpolate(
                    b_new,
                    (hX.shape[-2], hX.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            .cpu()
            .data.numpy()
        )

    if "bottom" in illu_orient:
        b = torch.flip(b, [-2])
        valid_mask = torch.flip(valid_mask, [-2])
        hX = torch.flip(hX, [-2])
        seg_mask = torch.flip(seg_mask, [-2])
        fidelity_mask = torch.flip(fidelity_mask, [-2])
        foreground = torch.flip(foreground, [-2])
        foreground_b = torch.flip(foreground_b, [-2])
        fusion_mask = torch.flip(fusion_mask, [-2])

        b_new_flip = train_post_process_module(
            hX,
            b,
            valid_mask,
            seg_mask,
            fusion_mask,
            foreground,
            n_epochs,
            r,
            device,
        )

        b_new = torch.flip(b_new_flip, [-2])
        hX = torch.flip(hX, [-2])
        fusion_mask = torch.flip(fusion_mask, [-2])

        # b_new = b_new - hX

        recon_bottom = (
            (
                hX
                + F.interpolate(
                    b_new,
                    (hX.shape[-2], hX.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            .cpu()
            .data.numpy()
        )
    if illu_orient == "top-bottom":
        recon_up = torch.from_numpy(recon_up).to(device)
        recon_bottom = torch.from_numpy(recon_bottom).to(device)

        foreground = torch.flip(foreground, [-2])
        valid_mask = torch.flip(valid_mask, [-2])
        fidelity_mask = torch.flip(fidelity_mask, [-2])

        model = compose_post(m, n).to(device)
        opt = Adam(model.parameters(), lr=1)
        loss = loss_compose_post().to(device)
        for e in tqdm.tqdm(range(1000)):
            recon, b = model(recon_up - hX, recon_bottom - hX, hX)
            ll = loss(recon, hX, foreground, valid_mask, fidelity_mask)
            opt.zero_grad()
            ll.backward()
            opt.step()
        recon = recon.cpu().data.numpy()
    if illu_orient == "top":
        recon = recon_up
    if illu_orient == "bottom":
        recon = recon_bottom
    recon = crop_center(
        scipy.ndimage.rotate(
            recon,
            angle_offset,
            order=1,
            axes=(-2, -1),
            mode="nearest",
        ),
        m0,
        n0,
    )
    return recon


def post_process_module(
    hX,
    result_0,
    result_gnn,
    angle_offset_individual,
    illu_orient,
    allow_stripe_deviation=False,
    fusion_mask=None,
    device=None,
    non_positive=False,
    r=10,
    n_epochs=1000,
):
    if hX.shape[1] > 1:
        assert fusion_mask is not None, print("fusion_mask is missing.")
        assert len(angle_offset_individual) > 1, print(
            "angle_offset_individual must be of length 2."
        )
        fusion_mask = fusion_mask[:, :, : hX.shape[-2], : hX.shape[-1]]

        fusion_mask = np.pad(fusion_mask, ((0, 0), (0, 0), (259, 259), (0, 0)), "edge")
    hX = np.pad(hX, ((0, 0), (0, 0), (259, 259), (0, 0)), "reflect")
    result_0 = np.pad(result_0, ((0, 0), (0, 0), (259, 259), (0, 0)), "reflect")
    hX0 = copy.deepcopy(hX)
    # target = (10**hX * fusion_mask).sum(1, keepdims=True)

    if fusion_mask is None:
        foreground = (np.log10(hX0 + 1) > threshold_otsu(np.log10(hX0 + 1))) + (
            result_0 > threshold_otsu(result_0)
        )
        for angle, illu in zip(angle_offset_individual[0], illu_orient):
            hX = linear_propagation(
                result_0 - hX,
                hX,
                result_0,
                foreground,
                angle_offset=angle,
                illu_orient=illu,
            )
        return 10 ** np.clip(hX[..., 259:-259, :], result_0.min(), result_0.max())
    else:
        recon = []
        for ind, (angle_list, illu) in enumerate(
            zip(angle_offset_individual, illu_orient)
        ):
            hX = hX0[:, ind : ind + 1].astype(np.float32)
            foreground = (
                np.log10(hX0[:, ind : ind + 1] + 1)
                > threshold_otsu(np.log10(hX0[:, ind : ind + 1] + 1))
            ) + (result_0 > threshold_otsu(result_0))
            for angle in angle_list:
                hX = linear_propagation(
                    result_0 - hX,
                    hX,
                    result_0,
                    foreground,
                    angle_offset=angle,
                    illu_orient=illu,
                    fusion_mask=fusion_mask[0, ind],
                )
            hX = np.clip(hX[..., 259:-259, :], result_0.min(), result_0.max())
            recon.append(hX)

        recon = 10 ** np.concatenate(recon, 1)[0]
        recon = np.clip(recon, 0, 65535)
        fusion_mask = fusion_mask[0, :, 259:-259, :]
        recon = (recon * fusion_mask).sum(
            0, keepdims=True
        )  # * fusion_mask[:, :recon.shape[1], :recon.shape[2]]

        return recon[None]
