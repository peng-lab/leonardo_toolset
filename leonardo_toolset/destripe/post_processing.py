import scipy
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import torch.nn as nn
from torch.optim import Adam
from leonardo_toolset.destripe.utils import crop_center
import tqdm
from skimage.filters import threshold_otsu
from leonardo_toolset.destripe.wave_rec import wave_rec
import copy


def rotate(
    x,
    angle,
    mode="constant",
    expand=True,
):

    x = scipy.ndimage.rotate(
        x.cpu().data.numpy(),
        angle,
        axes=(-2, -1),
        reshape=True,
        mode=mode,
    )
    return torch.from_numpy(x).cuda()


def last_nonzero(
    arr,
    mask,
    axis,
    invalid_val=np.nan,
):
    if mask is None:
        mask = arr != 0
    if type(mask) is not np.ndarray:
        mask = mask.cpu().detach().numpy()
    val = mask.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def first_nonzero(
    arr,
    mask,
    axis,
    invalid_val=np.nan,
):
    if mask is None:
        mask = arr != 0
    if type(mask) is not np.ndarray:
        mask = mask.cpu().detach().numpy()
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def edge_padding_xy(x, rx, ry):
    x = torch.cat(
        (x[:, :, 0:1, :].repeat(1, 1, rx, 1), x, x[:, :, -1:, :].repeat(1, 1, rx, 1)),
        -2,
    )
    return torch.cat(
        (x[:, :, :, 0:1].repeat(1, 1, 1, ry), x, x[:, :, :, -1:].repeat(1, 1, 1, ry)),
        -1,
    )


def mask_with_lower_intensity(
    Y_raw_full,
    target,
    thresh_target_exp,
    thresh_target,
    thresh_result_0_exp,
    thresh_result_0,
):
    seg_mask = (10**target > thresh_target_exp) * (10**Y_raw_full < thresh_result_0_exp)

    seg = (10**target > thresh_target_exp) + (10**Y_raw_full > thresh_result_0_exp)

    seg_mask_large = F.max_pool2d(
        seg_mask + 0.0, (1, 49), padding=(0, 24), stride=(1, 1)
    )

    diff = (seg_mask_large == 1) * (seg_mask == 0)
    diff = diff * (seg == 0)

    seg_mask_0 = seg_mask + diff

    seg_mask = (target > thresh_target) * (Y_raw_full < thresh_result_0)

    seg = (target > thresh_target) + (Y_raw_full > thresh_result_0)

    seg_mask_large = F.max_pool2d(
        seg_mask + 0.0, (1, 49), padding=(0, 24), stride=(1, 1)
    )
    diff = (seg_mask_large == 1) * (seg_mask == 0)
    diff = diff * (seg == 0)

    seg_mask_1 = seg_mask + diff

    seg_mask = seg_mask_0 + seg_mask_1

    return seg_mask


def fillHole(segMask):
    h, w = segMask.shape
    h += 2
    w += 2
    _mask = np.pad(segMask, ((1, 1), (1, 1)))
    im_floodfill = 255 * (_mask.astype(np.uint8)).copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(im_floodfill, mask, seedPoint=(0, 0), newVal=255)
    result = (segMask + (~im_floodfill)[1:-1, 1:-1]).astype(bool)
    return result


def extract_boundary(
    Y_raw_full,
    target,
    thresh_target_exp,
    thresh_target,
    thresh_result_0_exp,
    thresh_result_0,
    device,
):
    seg_mask = (10**target > thresh_target_exp) + (10**Y_raw_full > thresh_result_0_exp)
    seg_mask = fillHole(seg_mask[0, 0])[None, None]
    seg_mask = torch.from_numpy(seg_mask).to(device).to(torch.float)
    seg_mask_large = F.max_pool2d(seg_mask, (1, 49), padding=(0, 24), stride=(1, 1))
    seg_mask_small = -F.max_pool2d(-seg_mask, (1, 49), padding=(0, 24), stride=(1, 1))
    mask = (seg_mask_large + seg_mask_small) == 1
    t = mask.sum(-2, keepdim=True) / torch.clip(seg_mask.sum(-2, keepdim=True), 1) > 0.5
    mask = mask * t

    seg_mask = (target > thresh_target) + (Y_raw_full > thresh_result_0)
    seg_mask = fillHole(seg_mask[0, 0])[None, None]
    seg_mask = torch.from_numpy(seg_mask).to(device).to(torch.float)
    seg_mask_large = F.max_pool2d(seg_mask, (1, 49), padding=(0, 24), stride=(1, 1))
    seg_mask_small = -F.max_pool2d(-seg_mask, (1, 49), padding=(0, 24), stride=(1, 1))
    mask1 = (seg_mask_large + seg_mask_small) == 1
    t = (
        mask1.sum(-2, keepdim=True) / torch.clip(seg_mask.sum(-2, keepdim=True), 1)
        > 0.5
    )
    mask1 = mask1 * t

    return mask1 + mask


def mask_with_higher_intensity(
    Y_raw_full,
    target,
    thresh_target_exp,
    thresh_target,
    thresh_result_0_exp,
    thresh_result_0,
):
    mask1 = (target < thresh_target) * (Y_raw_full > thresh_result_0)

    mask2 = (10**target < thresh_target_exp) * (10**Y_raw_full > thresh_result_0_exp)
    return mask1 + mask2


class stripe_post(nn.Module):
    def __init__(self, m, n):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, 1, m, n))
        self.softplus = nn.Softplus()

    def forward(self, b):
        b = b * self.softplus(self.w)
        b_adpt = torch.cumsum(b, -2)
        return b_adpt


class compose_post(nn.Module):
    def __init__(self, m, n):
        super().__init__()
        self.w = nn.Parameter(0.5 * torch.ones(1, 1, 1, n))
        self.sigmoid = nn.Sigmoid()

    def forward(self, b_up, b_bottom, hX):
        w = self.sigmoid(self.w)
        b = w * (b_up) + (1 - w) * (b_bottom)
        return hX + b


class GuidedFilterLoss:
    def __init__(self, seg_mask, r, downsample_ratio, eps=1e-9):
        self.r, self.eps = r, eps
        self.downsample_ratio = downsample_ratio
        self.N = self.boxfilter(1 - seg_mask)

    def diff_x(self, input, r):
        return input[:, :, 2 * r :, :] - input[:, :, : -2 * r, :]

    def diff_y(self, input, r):
        return input[:, :, :, 2 * r :] - input[:, :, :, : -2 * r]

    def boxfilter(self, input):
        return self.diff_x(
            self.diff_y(
                edge_padding_xy(input, self.r, self.r * self.downsample_ratio).cumsum(
                    3
                ),
                self.r * self.downsample_ratio,
            ).cumsum(2),
            self.r,
        )

    def __call__(self, x):
        mean_x_y = self.boxfilter(x) / self.N
        mean_x2 = self.boxfilter(x * x) / self.N
        cov_xy = mean_x2 - mean_x_y * mean_x_y
        var_x = mean_x2 - mean_x_y * mean_x_y
        A = cov_xy / (var_x + self.eps)
        b = mean_x_y - A * mean_x_y
        A, b = self.boxfilter(A) / self.N, self.boxfilter(b) / self.N
        return A * x + b


class loss_post(nn.Module):
    def __init__(
        self,
        weight_tvx,
        weight_tvy,
        weight_tvx_f,
        weight_tvy_f,
        weight_tvx_hr,
        allow_stripe_deviation=False,
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
        self.ptv = 3

        self.register_buffer(
            "weight_tvx",
            weight_tvx,
        )
        self.register_buffer(
            "weight_tvy",
            weight_tvy,
        )
        self.register_buffer(
            "weight_tvx_f",
            weight_tvx_f,
        )
        self.register_buffer(
            "weight_tvy_f",
            weight_tvy_f,
        )
        self.register_buffer(
            "weight_tvx_hr",
            weight_tvx_hr,
        )
        if allow_stripe_deviation:
            self.tv_hr = self.tv_hr_func
        else:
            self.tv_hr = lambda x, y: 0

    def tv_hr_func(self, y, weight_tvx_hr):
        return (
            weight_tvx_hr * torch.conv2d(y, self.kernel_x, stride=(1, 1)).abs()
        ).sum()

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
        y,
        hX,
        h_mask,
        r,
    ):

        e1 = torch.conv2d(y[:, :, ::r, :], self.kernel_x).abs()
        e2 = torch.conv2d(y[:, :, ::r, :], self.kernel_y)  # , stride = (r, 1)

        e121 = (y[..., :-1, :-1] - y[..., :-1, 1:]).abs()
        e3 = torch.conv2d(F.avg_pool2d(y, (r, r), stride=(r, r)), self.kernel_x).abs()
        return (
            (self.weight_tvx_hr * e121[..., 3:-2, 3:-2]).sum()
            + (self.weight_tvx * e1).sum()
            + (self.weight_tvy * (e2 - h_mask).abs()).sum()
            + (self.weight_tvx_f * e3).sum()
            + self.tv_hr(y, self.weight_tvx_hr)
        )


class loss_compose_post(nn.Module):
    def __init__(
        self,
        mask,
    ):
        super().__init__()
        kernel_x, kernel_y = self.rotatableKernel(3, 1)
        self.register_buffer(
            "kernel_x", torch.from_numpy(kernel_x)[None, None].to(torch.float)
        )
        self.register_buffer(
            "kernel_y", torch.from_numpy(kernel_y)[None, None].to(torch.float)
        )
        self.register_buffer(
            "mask",
            mask,
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

    def forward(
        self,
        y,
        hX,
        r,
    ):
        e1 = (
            torch.conv2d(F.pad(y, (3, 3, 3, 3), mode="reflect"), self.kernel_x).abs()
            * self.mask
        )

        e3 = (
            torch.conv2d(
                F.pad(y[..., ::r, ::r], (3, 3, 3, 3), mode="reflect"), self.kernel_x
            ).abs()
            * self.mask[..., ::r, ::r]
        )
        e4 = (
            torch.conv2d(
                F.pad((y - hX)[..., ::r, ::r], (3, 3, 3, 3), mode="reflect"),
                self.kernel_y,
            ).abs()
            * self.mask[..., ::r, ::r]
        )

        e8 = (
            torch.conv2d(
                F.pad((y - hX)[..., ::r, :], (3, 3, 3, 3), mode="reflect"),
                self.kernel_y,
            ).abs()
            * self.mask[..., ::r, :]
        )

        e5 = (y[:, :, :, :-1] - y[:, :, :, 1:]).abs() * self.mask[..., :, :-1]

        return e1.sum() + e3.sum() + e5.sum() + r * e4.sum() + r * e8.sum()


def train_post_process_module(
    hX,
    b,
    valid_mask,
    missing_mask,
    fusion_mask,
    foreground,
    boundary_mask,
    filled_mask,
    n_epochs,
    r,
    device,
    non_positive,
    allow_stripe_deviation,
    desc="",
):
    m, n = hX[:, :, ::r, :].shape[-2:]

    b_sparse_0 = torch.clip(
        torch.diff(b[:, :, ::r, :], dim=-2, prepend=0 * b[:, :, 0:1, :]),
        0.0,
        None,
    ) * (1 - missing_mask[:, :, ::r, :])
    if hX.shape[-2] % r == 0:
        p = (0, 0, 0, 0)
    else:
        p = (0, 0, 0, r - hX.shape[-2] % r)

    model_0 = stripe_post(m, n).to(device)

    if not non_positive:
        b_sparse_1 = torch.clip(
            torch.diff(b[:, :, ::r, :], dim=-2, prepend=0 * b[:, :, 0:1, :]),
            None,
            0.0,
        )
        model_1 = stripe_post(m, n).to(device)

    weight_tvx = valid_mask[:, :, ::r, :]
    valid_mask_for_preserve = valid_mask * foreground
    weight_tvy = valid_mask_for_preserve[:, :, ::r, :]
    weight_tvx_f = F.avg_pool2d(valid_mask, (r, r), stride=(r, r)) >= 1
    weight_tvx_f = weight_tvx_f[:, :, 3:-3, 3:-3]
    weight_tvy_f = valid_mask_for_preserve[:, :, ::r, ::r]
    weight_tvx_hr = valid_mask
    weight_tvx = weight_tvx[..., 3:-3, 3:-3]
    weight_tvy = weight_tvy[..., 3:-3, 3:-3]
    weight_tvx_hr = weight_tvx_hr[..., 3:-3, 3:-3]

    loss = loss_post(
        weight_tvx,
        weight_tvy,
        weight_tvx_f,
        weight_tvy_f,
        weight_tvx_hr,
        allow_stripe_deviation=allow_stripe_deviation,
    ).to(device)

    h_mask = torch.where(
        torch.conv2d((hX + b)[:, :, ::r, :], loss.kernel_y).abs()
        > torch.conv2d(hX[:, :, ::r, :], loss.kernel_y).abs(),
        torch.conv2d(hX[:, :, ::r, :], loss.kernel_y),
        torch.conv2d((hX + b)[:, :, ::r, :], loss.kernel_y),
    )
    h_mask = torch.where(
        boundary_mask[:, :, ::r, :][..., 3:-3, 3:-3] == 1,
        torch.conv2d(hX[:, :, ::r, :], loss.kernel_y),
        h_mask,
    )

    peusdo_recon = torch.maximum((hX + b), hX)
    peusdo_recon = peusdo_recon * fusion_mask + hX * (1 - fusion_mask)
    peusdo_recon = peusdo_recon[:, :, ::r, :]

    if non_positive:
        opt = Adam(model_0.parameters(), lr=1)
    else:
        opt = Adam([*model_0.parameters(), *model_1.parameters()], lr=1)

    for e in tqdm.tqdm(
        range(n_epochs), leave=False, desc="post-process stripes {}: ".format(desc)
    ):
        b_new_0 = model_0(
            b_sparse_0,
        )
        l = loss(
            F.interpolate(
                b_new_0,
                hX.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            + hX,
            hX,
            h_mask,
            r,
        )
        if not non_positive:
            b_new_1 = model_1(
                b_sparse_1,
            )
            l = l + loss(
                F.interpolate(
                    b_new_1,
                    hX.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                + hX,
                hX,
                h_mask,
                r,
            )
        opt.zero_grad()
        l.backward()
        opt.step()

    if non_positive:
        b_new = b_new_0
    else:
        b_new = torch.cat((b_new_0, b_new_1), 0)

    guidedfilterloss = GuidedFilterLoss(
        torch.zeros_like(filled_mask)[:, :, ::r, :],
        49,
        r,
        10,
    )
    b_new = edge_padding_xy(b_new[..., 3:-3, 3:-3], 3, 3)
    b_new = b_new * fusion_mask[:, :, ::r, :] + torch.zeros_like(b_new) * (
        1 - fusion_mask[:, :, ::r, :]
    )
    b_new = b_new.detach()

    diff = guidedfilterloss(
        (hX[:, :, ::r, :] + b_new.detach() - peusdo_recon)
        * (1 - torch.zeros_like(filled_mask)[:, :, ::r, :])
    )

    b_new = b_new - diff.detach()

    b_new = F.interpolate(
        b_new,
        (hX.shape[-2], hX.shape[-1]),
        mode="bilinear",
        align_corners=True,
    )

    recon = (hX + b_new).detach()

    if not non_positive:
        recon_dark, recon_bright = recon[:, None]
        model = compose_post(hX.shape[-2], hX.shape[-1]).to(device)
        opt = Adam(model.parameters(), lr=1)
        mask = valid_mask * (1 - missing_mask)
        loss = loss_compose_post(mask).to(device)
        for e in tqdm.tqdm(
            range(1000),
            leave=False,
            desc="merge positive and non-positive stripe {}: ".format(desc),
        ):
            recon = model(recon_dark - hX, recon_bright - hX, hX)
            l = loss(recon, hX, r)
            opt.zero_grad()
            l.backward()
            opt.step()
    else:
        recon = (
            (1 - boundary_mask) * (recon - hX)
            + boundary_mask * torch.maximum(*(recon - hX), torch.zeros_like(hX))
            + hX
        )
        pass

    return recon


def uniform_fusion_mask(fusion_mask, angle_list, illu_orient, device):
    if isinstance(fusion_mask, np.ndarray):
        fusion_mask = torch.from_numpy(fusion_mask.copy()).to(device)
    m, n = fusion_mask.shape[-2:]
    for angle in angle_list:
        fusion_mask = rotate(fusion_mask, -angle, expand=True, mode="constant")
        if illu_orient == "top":
            fusion_mask = (
                torch.flip(torch.cumsum(torch.flip(fusion_mask > 0, [-2]), -2), [-2])
                > 0
            )
            fusion_mask = (
                torch.flip(torch.cumsum(torch.flip(fusion_mask > 0, [-2]), -2), [-2])
                > 0
            )
            fusion_mask = fusion_mask.to(torch.float)
        if illu_orient == "bottom":
            fusion_mask = torch.cumsum(fusion_mask > 0, -2) > 0
            fusion_mask = torch.cumsum(fusion_mask > 0, -2) > 0
            fusion_mask = fusion_mask.to(torch.float)
        fusion_mask = crop_center(
            rotate(fusion_mask, angle, expand=True, mode="constant"), m, n
        )
    return fusion_mask.cpu().data.numpy()


def padding_size(H, W, angle):
    angle = np.deg2rad(angle)
    H_new = np.cos(angle) * H + np.sin(angle) * W
    W_new = np.sin(angle) * H + np.cos(angle) * W
    return H_new, W_new


def linear_propagation(
    b,
    hX,
    foreground,
    missing_mask,
    boundary_mask,
    filled_mask,
    angle_offset,
    allow_stripe_deviation=False,
    illu_orient="top",
    n_epochs=1000,
    fusion_mask=None,
    device=None,
    non_positive=False,
    r=10,
    gf_kernel_size=49,
    desc="",
):

    m0, n0 = hX.shape[-2:]

    hX0 = copy.deepcopy(hX)

    foreground = torch.from_numpy(foreground).to(device)
    fusion_mask = torch.from_numpy(fusion_mask.copy()).to(device)
    b = torch.from_numpy(b).to(device)
    hX = torch.from_numpy(hX).to(device)

    foreground = rotate(foreground, -angle_offset, mode="constant") > 0
    fusion_mask = rotate(fusion_mask, -angle_offset, mode="constant")
    valid_mask = rotate(torch.ones_like(hX), -angle_offset, mode="constant") > 0
    missing_mask = rotate(missing_mask, -angle_offset, mode="constant") > 0
    boundary_mask = rotate(boundary_mask, -angle_offset, mode="constant") > 0
    filled_mask = rotate(filled_mask, -angle_offset, mode="constant") > 0

    hX = rotate(hX, -angle_offset, mode="nearest")
    b = rotate(b, -angle_offset, mode="nearest")

    b = F.pad(b, (0, 0, gf_kernel_size // 2, gf_kernel_size // 2), "reflect")

    chunks = torch.split(b, 64, dim=-1)
    b = []
    for _, chunk in enumerate(chunks):
        b.append(chunk.unfold(-2, gf_kernel_size, 1).median(dim=-1)[0])

    b = torch.cat(b, -1)
    m, n = b[:, :, ::r, :].shape[-2:]

    foreground = torch.where(foreground.sum(-2, keepdim=True) == 0, 1, foreground)

    foreground = foreground + 0.0
    valid_mask = valid_mask + 0.0
    missing_mask = missing_mask + 0.0
    boundary_mask = boundary_mask + 0.0
    filled_mask = filled_mask + 0.0

    if fusion_mask.sum() == 0:
        return np.zeros(
            (
                1,
                1,
                m0,
                n0,
            ),
            dtype=np.float32,
        )

    if "top" in illu_orient:
        s = min(last_nonzero(fusion_mask, None, -2, 0).max() + 3, hX.shape[-2])
        c0 = max(
            first_nonzero(fusion_mask * valid_mask, None, -1, hX.shape[-1]).min() - 3, 0
        )
        c1 = min(
            last_nonzero(fusion_mask * valid_mask, None, -1, 0).max() + 3, hX.shape[-1]
        )
        fusion_mask_adpt = copy.deepcopy(fusion_mask[..., :s, c0:c1])
        # fusion_mask_adpt[fusion_mask_adpt == 0] = 0.1
        recon_up = train_post_process_module(
            hX[..., :s, c0:c1],
            b[..., :s, c0:c1],
            valid_mask[..., :s, c0:c1] * fusion_mask_adpt,
            missing_mask[..., :s, c0:c1],
            fusion_mask[..., :s, c0:c1],
            foreground[..., :s, c0:c1],
            boundary_mask[..., :s, c0:c1],
            filled_mask[..., :s, c0:c1],
            n_epochs,
            r,
            device,
            non_positive=non_positive,
            allow_stripe_deviation=allow_stripe_deviation,
            desc=desc,
        )
        recon_up = F.pad(recon_up, (c0, n - (c1 - c0) - c0, 0, hX.shape[-2] - s))

    if "bottom" in illu_orient:
        b = torch.flip(b, [-2])
        valid_mask = torch.flip(valid_mask, [-2])
        hX = torch.flip(hX, [-2])
        missing_mask = torch.flip(missing_mask, [-2])
        foreground = torch.flip(foreground, [-2])
        fusion_mask = torch.flip(fusion_mask, [-2])
        boundary_mask = torch.flip(boundary_mask, [-2])
        filled_mask = torch.flip(filled_mask, [-2])

        s = min(last_nonzero(fusion_mask, None, -2, 0).max() + 3, hX.shape[-2])
        c0 = max(
            first_nonzero(fusion_mask * valid_mask, None, -1, hX.shape[-1]).min() - 3, 0
        )
        c1 = min(
            last_nonzero(fusion_mask * valid_mask, None, -1, 0).max() + 3, hX.shape[-1]
        )

        fusion_mask_adpt = copy.deepcopy(fusion_mask[..., :s, c0:c1])
        # fusion_mask_adpt[fusion_mask_adpt == 0] = 0.1

        recon_bottom = train_post_process_module(
            hX[..., :s, c0:c1],
            b[..., :s, c0:c1],
            valid_mask[..., :s, c0:c1] * fusion_mask_adpt,
            missing_mask[..., :s, c0:c1],
            fusion_mask[..., :s, c0:c1],
            foreground[..., :s, c0:c1],
            boundary_mask[..., :s, c0:c1],
            filled_mask[..., :s, c0:c1],
            n_epochs,
            r,
            device,
            non_positive=non_positive,
            allow_stripe_deviation=allow_stripe_deviation,
            desc=desc,
        )

        recon_bottom = F.pad(
            recon_bottom, (c0, n - (c1 - c0) - c0, 0, hX.shape[-2] - s)
        )

        hX = torch.flip(hX, [-2])
        fusion_mask = torch.flip(fusion_mask, [-2])
        valid_mask = torch.flip(valid_mask, [-2])
        missing_mask = torch.flip(missing_mask, [-2])
        foreground = torch.flip(foreground, [-2])
        recon_bottom = torch.flip(recon_bottom, [-2])
        boundary_mask = torch.flip(boundary_mask, [-2])

    if illu_orient == "top-bottom":
        recon_up = recon_up.detach()
        recon_bottom = recon_bottom.detach()
        model = compose_post(m, n).to(device)
        opt = Adam(model.parameters(), lr=1)
        mask = valid_mask * fusion_mask
        loss = loss_compose_post(mask).to(device)

        for e in tqdm.tqdm(
            range(1000), leave=False, desc="merge top-bottom ill. {}: ".format(desc)
        ):
            recon = model(recon_up - hX, recon_bottom - hX, hX)
            l = loss(recon, hX, r)
            opt.zero_grad()
            l.backward()
            opt.step()

    if illu_orient == "top":
        recon = recon_up
    if illu_orient == "bottom":
        recon = recon_bottom

    recon = (
        crop_center(
            rotate(
                recon,
                angle_offset,
                expand=True,
                mode="nearest",
            ),
            m0,
            n0,
        )
        .cpu()
        .data.numpy()
    )

    return recon


def simple_rotate(x, angle, device):
    x = torch.from_numpy(x).to(device)
    H, W = x.shape[-2:]
    x = crop_center(
        rotate(rotate(x, -angle, mode="nearest"), angle=angle, mode="nearest"), H, W
    )
    return x.cpu().data.numpy()


def post_process_module(
    hX,
    result_gu,
    result_gnn,
    angle_offset_individual,
    illu_orient,
    allow_stripe_deviation=False,
    fusion_mask=None,
    device=None,
    non_positive=False,
    r=10,
    n_epochs=1000,
    gf_kernel_size=49,
):
    if illu_orient is not None:
        if device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if hX.shape[1] > 1:
            assert fusion_mask is not None, print("fusion_mask is missing.")
            assert len(angle_offset_individual) > 1, print(
                "angle_offset_individual must be of length 2."
            )
            fusion_mask = fusion_mask[:, :, : hX.shape[-2], : hX.shape[-1]]

        hX0 = copy.deepcopy(hX)
        hX0 = np.where(hX0 == 0, hX0.max(1, keepdims=True), hX0)
        target = np.log10(
            np.clip(((10**hX0) * fusion_mask).sum(1, keepdims=True), 1, None)
        )
        recon = []
        target_wavelet_list = []
        recon_gu_list = []

        iex = 1
        count = lambda lst: sum(count(x) if isinstance(x, list) else 1 for x in lst)
        iex_total = count(angle_offset_individual)
        for ind, (angle_list, illu) in enumerate(
            zip(angle_offset_individual, illu_orient)
        ):
            fusion_mask_ind = uniform_fusion_mask(
                fusion_mask[:, ind : ind + 1],
                angle_list,
                illu,
                device=device,
            )
            hX = hX0[:, ind : ind + 1, ...]

            thresh_target_exp = threshold_otsu(10**target)
            thresh_target = threshold_otsu(target)
            thresh_result_gu_exp = threshold_otsu(10**result_gu)
            thresh_result_gu = threshold_otsu(result_gu)

            foreground = (10**target > thresh_target_exp) + (
                10**result_gu > thresh_result_gu_exp
            )
            result_gu_torch = torch.from_numpy(result_gu.copy()).to(device)
            target_torch = torch.from_numpy(target.copy()).to(device)
            missing_mask = mask_with_lower_intensity(
                result_gu_torch,
                target_torch,
                thresh_target_exp,
                thresh_target,
                thresh_result_gu_exp,
                thresh_result_gu,
            )
            boundary_mask = extract_boundary(
                result_gu,
                target,
                thresh_target_exp,
                thresh_target,
                thresh_result_gu_exp,
                thresh_result_gu,
                device,
            )
            filled_mask = mask_with_higher_intensity(
                result_gu_torch,
                target_torch,
                thresh_target_exp,
                thresh_target,
                thresh_result_gu_exp,
                thresh_result_gu,
            )
            target_wavelet = hX0[:, ind : ind + 1, ...]
            recon_gu_wavelet = result_gu
            for i, angle in enumerate(angle_list):
                hX = linear_propagation(
                    result_gnn - hX,
                    hX,
                    foreground,
                    missing_mask,
                    boundary_mask,
                    filled_mask,
                    angle_offset=angle,
                    illu_orient=illu,
                    fusion_mask=fusion_mask_ind,
                    device=device,
                    non_positive=non_positive,
                    allow_stripe_deviation=allow_stripe_deviation,
                    r=r,
                    n_epochs=n_epochs,
                    gf_kernel_size=gf_kernel_size,
                    desc="(No. {} out of {} angles)".format(iex, iex_total),
                )
                target_wavelet = simple_rotate(target_wavelet, angle, device)
                recon_gu_wavelet = simple_rotate(recon_gu_wavelet, angle, device)
                iex += 1
            recon.append(hX)
            target_wavelet_list.append(target_wavelet)
            recon_gu_list.append(recon_gu_wavelet)

        recon = np.concatenate(recon, 1)
        recon = (recon * fusion_mask).sum(
            1,
            keepdims=True,
        )
        target_wavelet_list = np.concatenate(target_wavelet_list, 1)
        recon_gu_list = np.concatenate(recon_gu_list, 1)
        target_wavelet = (target_wavelet_list * fusion_mask).sum(
            1,
            keepdims=True,
        )
        recon_gu_wavelet = (recon_gu_list * fusion_mask).sum(
            1,
            keepdims=True,
        )
    else:
        recon = copy.deepcopy(result_gu)
        target_wavelet = np.log10(
            np.clip(((10**hX) * fusion_mask).sum(1, keepdims=True), 1, None)
        )
        recon_gu_wavelet = copy.deepcopy(result_gu)
    recon_gu_wavelet = wave_rec(
        10**recon_gu_wavelet,
        10**target_wavelet,
        None,
        kernel="db2",
        mode=2,
        device=device,
    )
    recon = wave_rec(
        10**recon,
        10**target_wavelet,
        recon_gu_wavelet,
        kernel="db2",
        mode=2,
        device=device,
    )
    return recon
