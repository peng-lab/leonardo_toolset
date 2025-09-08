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
    """
    Rotate the input x.

    Args:
        x (torch.Tensor): Input tensor.
        angle (float): Rotation angle in degrees.
        mode (str): Points outside the boundaries are filled according to the given mode.
        expand (bool): Whether to expand the output array to fit the entire rotated input.

    Returns:
        torch.Tensor: rotated tensor.
    """
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
    """
    Find the last nonzero index along a given axis.

    Args:
        arr (np.ndarray or torch.Tensor): Input array.
        mask (np.ndarray or torch.Tensor or None): Mask to apply (nonzero locations).
        axis (int): Axis to search for last nonzero.
        invalid_val: Value to use if no nonzero found.

    Returns:
        np.ndarray: Indices of last nonzero elements.
    """
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
    """
    Find the first nonzero index along a given axis.

    Args:
        arr (np.ndarray or torch.Tensor): Input array.
        mask (np.ndarray or torch.Tensor or None): Mask to apply (nonzero locations).
        axis (int): Axis to search for first nonzero.
        invalid_val: Value to use if no nonzero found.

    Returns:
        np.ndarray: Indices of first nonzero elements.
    """
    if mask is None:
        mask = arr != 0
    if type(mask) is not np.ndarray:
        mask = mask.cpu().detach().numpy()
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def edge_padding_xy(x, rx, ry):
    """
    Pads with the edge values of the tensor.

    Args:
        x (torch.Tensor): Input array.
        rx (int): Padding size for second last axis.
        ry (int): Padding size for last axis.

    Returns:
        torch.tensor: padded tensor.
    """
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
    """
    Create a mask of pixels that are detected as background via segmentation after guided upsampling,
    but are foreground in the original stripy image.

    Args:
        Y_raw_full (torch.Tensor): result of guided upsampling.
        target (torch.Tensor): original stripy image.
        thresh_target_exp (float): OTSU threshold for target in the original space.
        thresh_target (float): OTSU threshold for target in the log space.
        thresh_result_0_exp (float): OTSU threshold for Y_raw_full in the original space.
        thresh_result_0 (float): OTSU threshold for Y_raw_full in the log space.

    Returns:
        torch.tensor: mask.
    """
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
    """
    Fill holes in a binary segmentation mask using flood fill.

    Args:
        segMask (np.ndarray): Binary segmentation mask. Nonzero pixels are treated as foreground.

    Returns:
        np.ndarray: Binary mask with holes filled.
    """
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
    """
    Find the boudnary regions of the specimen.

    Args:
        Y_raw_full (torch.Tensor): result of guided upsampling.
        target (torch.Tensor): original stripy image.
        thresh_target_exp (float): OTSU threshold for target in the original space.
        thresh_target (float): OTSU threshold for target in the log space.
        thresh_result_0_exp (float): OTSU threshold for Y_raw_full in the original space.
        thresh_result_0 (float): OTSU threshold for Y_raw_full in the log space.
        device (torch.device): Target device for returning the final mask.

    Returns:
        torch.Tensor: Binary mask, with 1 marking boundary regions.
    """
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
    """
    Create a mask of pixels that are detected as foreground via segmentation after guided upsampling,
    but are background in the original stripy image.

    Args:
        Y_raw_full (torch.Tensor): result of guided upsampling.
        target (torch.Tensor): original stripy image.
        thresh_target_exp (float): OTSU threshold for target in the original space.
        thresh_target (float): OTSU threshold for target in the log space.
        thresh_result_0_exp (float): OTSU threshold for Y_raw_full in the original space.
        thresh_result_0 (float): OTSU threshold for Y_raw_full in the log space.

    Returns:
        torch.tensor: bidnary mask.
    """
    mask1 = (target < thresh_target) * (Y_raw_full > thresh_result_0)

    mask2 = (10**target < thresh_target_exp) * (10**Y_raw_full > thresh_result_0_exp)
    return mask1 + mask2


class stripe_post(nn.Module):
    """
    Global post-processing using illumination prior (ill. prior).

    This module models stripe propagation along one image axis by:
    (1) applying a positive, learnable per-pixel gain; then
    (2) cumulative summation along the stripe direction.
    """

    def __init__(self, m, n):
        """
        Initialize the stripe_post module.

        Args:
            m (int): image height H.
            n (int): image width W.
        """
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, 1, m, n))
        self.softplus = nn.Softplus()

    def forward(self, b):
        """
        Learnable linear propagation along the stripe direction.

        Args:
            b (torch.Tensor): horizontal directional gradient of the stripe residual map.

        Returns:
            torch.Tensor: propagated residual.
        """
        b = b * self.softplus(self.w)
        b_adpt = torch.cumsum(b, -2)
        return b_adpt


class compose_post(nn.Module):
    """
    This module models the composition of either:
    (1) bright and dark stripes; or
    (2) stripes coming from opposite orientations for simultaneous dual-sided illumination.
    """

    def __init__(self, m, n):
        """
        Initialize the compose_post module.

        Args:
            m (int): image height H.
            n (int): image width W.
        """
        super().__init__()
        self.w = nn.Parameter(0.5 * torch.ones(1, 1, 1, n))
        self.sigmoid = nn.Sigmoid()

    def forward(self, b_up, b_bottom, hX):
        """
        Learnable combination of two residual maps.

        Args:
            b_up (torch.Tensor): first input for the composition.
            b_bottom (torch.Tensor): second input for the composition.
            hX (torch.Tensor): the original stripy image.

        Returns:
            torch.Tensor: combined residual.
        """
        w = self.sigmoid(self.w)
        b = w * (b_up) + (1 - w) * (b_bottom)
        return hX + b


class GuidedFilterLoss:
    """
    correct global intensity drift
    """

    def __init__(self, seg_mask, r, downsample_ratio, eps=1e-9):
        """
        Initialize the GuidedFilterLoss module.

        Args:
            seg_mask (torch.Tensor): binary segmentation mask. Nonzero pixels will not be included to estimate the drift.
            r (int): radius of the guided filter.
            downsample_ratio (int): initial downsample ratio along stripe direction for the whole post-processing workflow.
            eps (float): regularization parameter in guided filter.
        """
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
        """
        Normal workflow of guided filtering with guidance and input being the same.

        Args:
            input (torch.Tensor): destriped image either by guided upsampling or by the post-processing workflow.
        """
        mean_x_y = self.boxfilter(x) / self.N
        mean_x2 = self.boxfilter(x * x) / self.N
        cov_xy = mean_x2 - mean_x_y * mean_x_y
        var_x = mean_x2 - mean_x_y * mean_x_y
        A = cov_xy / (var_x + self.eps)
        b = mean_x_y - A * mean_x_y
        A, b = self.boxfilter(A) / self.N, self.boxfilter(b) / self.N
        return A * x + b


class loss_post(nn.Module):
    """
    Post-processing loss for the ill. prior module.
    """

    def __init__(
        self,
        weight_tvx,
        weight_tvy,
        weight_tvx_f,
        weight_tvx_hr,
        allow_stripe_deviation=False,
    ):
        """
        Args:
            weight_tvx (torch.Tensor): Mask for gradient penalty across the stripe direction.
            weight_tvy (torch.Tensor): Mask for gradient fidelity along the stripe direction.
            weight_tvx_f (torch.Tensor): Mask for gradient penalty across the stripe direction
                in a donwsampled space.
            weight_tvx_hr (torch.Tensor): Mask for gradient penalty across the stripe direction
                at full resolution.
            allow_stripe_deviation (bool): Whether to enable an extra penalty on stripes during
                post-processing.
        """
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
        """
        Compute the post-processing loss.

        Args:
            y (torch.Tensor): current reconstruction/correction.
            hX (torch.Tensor): Original stripy image, shape (B, C, H, W)
            h_mask (torch.Tensor): Reference along-stripe-direction edge strength.
            r (int):initial downsample ratio along stripe direction for the whole post-processing workflow.

        Returns:
            torch.Tensor: scalar loss.
        """
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
    """
    Composition loss.
    """

    def __init__(
        self,
        mask,
    ):
        """
        Args:
            mask (torch.Tensor): Mask for applying the loss.
        """
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
        """
        Compute the loss.

        Args:
            y (torch.Tensor): current reconstruction/correction.
            hX (torch.Tensor): Original stripy image, shape (B, C, H, W)
            r (int):initial downsample ratio along stripe direction for the whole post-processing workflow.

        Returns:
            torch.Tensor: scalar loss.
        """
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
    weight_tvx_hr = valid_mask
    weight_tvx = weight_tvx[..., 3:-3, 3:-3]
    weight_tvy = weight_tvy[..., 3:-3, 3:-3]
    weight_tvx_hr = weight_tvx_hr[..., 3:-3, 3:-3]

    loss = loss_post(
        weight_tvx,
        weight_tvy,
        weight_tvx_f,
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
        l_loss = loss(
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
            l_loss = l_loss + loss(
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
        l_loss.backward()
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
            l_loss = loss(recon, hX, r)
            opt.zero_grad()
            l_loss.backward()
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
    """
    Generate a uniform fusion mask for sequential dual-sided illumination.

    Args:
        fusion_mask (np.ndarray or torch.Tensor): Initial fusion mask of shape (..., H, W) from Leonardo-Fuse.
        angle_list (list of float): List of illumination angles (in degrees) to process sequentially.
        illu_orient (str): Illumination orientation. Must be either:
            - "top": propagate downward from the top edge.
            - "bottom": propagate upward from the bottom edge.
        device (torch.device): Device on which computations are performed.

    Returns:
        np.ndarray: Processed unified fusion mask, ensuring consistent
            coverage along the specified illumination direction.
    """
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
    """
    Compute the new height and width required to fit an image after rotation.

    Args:
        H (int): Original image height.
        W (int): Original image width.
        angle (float): Rotation angle in degrees.

    Returns:
        tuple of floats: (H_new, W_new), the height and width of the rotated
        bounding box.
    """
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
    """
    Perform post-processing via learnable linear propagation.

    This function applies the `ill. prior` module on residual predictions rotated into the
    illumination axis. It simulates monotonic stripe propagation along the
    stripe direction (top or bottom illumination), suppresses residual
    artefacts, and preserves true structures. Optionally, results from top
    and bottom illumination can be fused.

    Args:
        b (np.ndarray): stripe residual map.
        hX (np.ndarray): Original log-transformed stripy image.
        foreground (np.ndarray): Binary mask marking foreground regions.
        missing_mask (np.ndarray or torch.Tensor):
            Binary mask of pixels detected as missing (foreground in raw but
            background after guided upsampling). Used to protect structures
            during stripe suppression.
        boundary_mask (torch.Tensor): Binary mask of specimen boundary regions.
        filled_mask (torch.Tensor): Binary mask of pixels detected as filled (background in raw
            but foreground after guided upsampling).
        angle_offset (float): Illumination orientation in degrees (counterclockwise).
        allow_stripe_deviation (bool): If True, enables additional penalty to the stripes in the loss
            to suppress wavy stripes.
        illu_orient (str): Illumination orientation. Must be either:
            - "top": propagate downward stripes.
            - "bottom": propagate upward stripes.
            - "top-bottom": use both.
        n_epochs (int): Number of optimization iterations for the post-processing model.
        fusion_mask (np.ndarray): fusion mask generated by Leonardo-Fuse (save_separate_result=True).
        device (torch.device): Device for computation ("cpu" or "cuda")
        non_positive (bool): If True, only negative (dark) stripe residuals are modeled, otherwise,
            both positive and negative stripe residuals are modeled.
        r (int): Downsampling stride along stripe direction for computation. Default.
        gf_kernel_size (int): Kernel size of the guided filter used for drift correction.
        desc (str): Text description for tqdm progress bar.

    Returns:
        np.ndarray:
            Reconstructed stripe-corrected image.
    """
    m0, n0 = hX.shape[-2:]

    # Rotate everything into the illumination axis (angle_offset)
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

    # Propagation from top illumination
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

    # Propagation from bottom illumination (mirrored)
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

    # If both top and bottom are enabled, learn to merge them
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
            l_loss = loss(recon, hX, r)
            opt.zero_grad()
            l_loss.backward()
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
    """
    Rotate an image by a given angle and crop back to its original size.

    Args:
        x (np.ndarray): Input image.
        angle (float): Rotation angle in degrees.
        device (torch.device): Device on which to perform the rotation

    Returns:
        np.ndarray: Rotated and cropped image, same dtype as input.
    """
    x = torch.from_numpy(x).to(device)
    H, W = x.shape[-2:]
    x = crop_center(
        rotate(rotate(x, -angle, mode="nearest"), angle=angle, mode="nearest"), H, W
    )
    return x.cpu().data.numpy()


def count(lst):
    return sum(count(x) if isinstance(x, list) else 1 for x in lst)


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
    """
    Global post-processing using illumination prior knowledge.

    This routine entires the illumination-prior post-processing workflow:
    it builds foreground/missing/boundary/filled masks from the raw image
    and the guided-upsampling result, applies learnable directional propagation
    (per illumination angle and orientation), optionally merges top/bottom reconstructions,
    and finally performs wavelet-based reconstruction to prevent creating artifacts.

    Args:
        hX (np.ndarray): Raw log-transformed stripy images of shape (1, V, H, W), where V is the number
            of illumination/detection sources.
        result_gu (np.ndarray): Guided upsampling result in log space, shape (1, 1, H, W).
        result_gnn (np.ndarray): GNN's stripe removal result, shape (1, 1, H, W).
        angle_offset_individual (list[list[float]]: lists of illumination angles (degrees) for everything slice in hX.
            Length must equal V. Example: [[θ1_top, θ2_top], [θ1_bot]].
        illu_orient (list[str]): illumination orientation for everything slice in hX, length V.
            Each entry in {"top", "bottom", "top-bottom"}.
        allow_stripe_deviation (bool): If True, enables an additional penalty to the stripes to remove wavy stripes.
        fusion_mask (np.ndarray, optional): FUsion mask(s) generated by Leonardo-Fuse (used in Leonardo-DeStripe-Fuse mode),
            shape (1, V, H, W).
        device (torch.device or str): Compute device ("cuda" or "cpu"). If None, auto-detects CUDA if available.
        non_positive (bool, optional): If True, only model negative (dark) stripe residuals in propagation;
            otherwise use both dark/bright channels.
        r (int): Downsampling stride along stripe direction for the post-processing.
        n_epochs (int): Number of optimization iterations for each propagation stage.
        gf_kernel_size (int, optional): Guided-filter kernel size used for drift correction after propagation.

    Returns:
        np.ndarray:
            Final reconstructed, stripe-corrected image in log space, shape (1, 1, H, W).

    Notes:
        - Inputs/outputs are in log space.
    """
    if illu_orient is not None:
        if device is None:
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

        # Iterate over each illumination-detection set
        iex = 1
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

            # compute masks for missing pixels, filled pixels, and sample boundary regions
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

            # Process each angle for this illumination orientation
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

        # Fuse across all views using fusion_mask
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
        # no ill_orint provided, simply do wavelet reconstruction
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
