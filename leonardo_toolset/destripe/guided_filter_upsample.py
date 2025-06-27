import copy

import numpy as np
import ptwt
import pywt
import torch
import torch.nn.functional as F

from leonardo_toolset.destripe.constant import WaveletDetailTuple2d

try:
    import jax
except Exception as e:
    print(f"Error: {e}. Proceed without jax")
    pass


def wave_rec(
    recon,
    hX,
    kernel,
    mode,
):

    y_dict = ptwt.wavedec2(
        recon[:, :, :-1, :-1], pywt.Wavelet(kernel), level=6, mode="constant"
    )
    X_dict = ptwt.wavedec2(
        hX[:, :, :-1, :-1], pywt.Wavelet(kernel), level=6, mode="constant"
    )
    x_base_dict = [y_dict[0]]

    mask_dict = []
    for ll, (detail, target) in enumerate(zip(y_dict[1:], X_dict[1:])):
        mask_dict.append(
            [
                torch.abs(detail[0]) < torch.abs(target[0]),
                torch.abs(detail[1]) < torch.abs(target[1]),
                torch.abs(detail[2]) < torch.abs(target[2]),
            ]
        )

    for ll, (detail, target, mask) in enumerate(zip(y_dict[1:], X_dict[1:], mask_dict)):
        if mode == 1:
            x_base_dict.append(
                WaveletDetailTuple2d(
                    torch.where(
                        ~mask[0],
                        detail[0],
                        target[0],
                    ),
                    torch.where(
                        mask[1],
                        detail[1],
                        target[1],
                    ),
                    torch.where(
                        ~mask[2],
                        detail[2],
                        target[2],
                    ),
                )
            )  # torch.sign(detail[1])*target[1].abs()
        else:
            x_base_dict.append(
                WaveletDetailTuple2d(
                    torch.where(
                        mask[0],
                        detail[0],
                        torch.sign(detail[0]) * target[0].abs(),
                    ),
                    torch.where(
                        mask[1],
                        detail[1],
                        torch.sign(detail[1]) * target[1].abs(),
                    ),
                    torch.where(
                        mask[2],
                        detail[2],
                        torch.sign(detail[2]) * target[2].abs(),
                    ),
                )
            )  # torch.sign(detail[1])*target[1].abs()
    x_base_dict = tuple(x_base_dict)
    recon = ptwt.waverec2(x_base_dict, pywt.Wavelet(kernel))
    recon = F.pad(recon, (0, 1, 0, 1), "reflect")
    return recon


class GuidedUpsample:
    def __init__(
        self,
        rx,
        device,
    ):
        self.rx = rx
        self.device = device

    def __call__(
        self,
        yy,
        hX,
        targetd,
        target,
        coor,
        fusion_mask,
        angle_offset_individual,
        backend,
    ):
        if backend == "jax":
            recon = (
                jax.scipy.ndimage.map_coordinates(
                    yy - targetd, coor, order=1, mode="reflect"
                )[None, None]
                + target
            )
            recon = torch.from_numpy(np.array(recon, copy=True)).to(self.device)
            hX = torch.from_numpy(np.array(hX)).to(self.device)
            fusion_mask = np.asarray(fusion_mask)
        else:
            recon = (
                F.grid_sample(
                    yy - targetd,
                    coor,
                    mode="bilinear",
                    padding_mode="reflection",
                    align_corners=True,
                )
                + target
            )
            fusion_mask = fusion_mask.cpu().data.numpy()

        m, n = hX.shape[-2:]

        y = np.ones_like(fusion_mask)

        for i, angle_list in enumerate(angle_offset_individual):
            hX_slice = hX[:, i : i + 1, :, :]

            y[:, i : i + 1, :, :] = (
                self.GF(
                    recon,
                    hX_slice,
                    angle_list,
                )
                .cpu()
                .data.numpy()
            )
        y = (10**y) * fusion_mask
        return np.log10(y.sum(1, keepdims=True))

    def GF(
        self,
        yy,
        hX,
        angle_list,
    ):
        hX_original = copy.deepcopy(hX)
        _, _, m, n = hX.shape
        for i, Angle in enumerate((-1 * np.array(angle_list)).tolist()):
            b = yy - hX
            rx = self.rx  # // 3 // 2 * 2 + 1
            lval = np.arange(rx) - rx // 2
            lval = np.round(lval * np.tan(np.deg2rad(-Angle))).astype(np.int32)
            b_batch = torch.zeros(rx, 1, 1, m, n)
            for ind, r in enumerate(range(rx)):
                data = F.pad(b, (lval.max(), lval.max(), rx // 2, rx // 2), "reflect")
                b_batch[ind] = data[
                    :, :, r : r + m, lval[ind] - lval.min() : lval[ind] - lval.min() + n
                ].cpu()
            b = torch.median(b_batch, 0)[0]

            b = b.to(self.device)
            hX = hX + b

        hX_base = F.avg_pool2d(F.pad(hX, (4, 4, 4, 4), "reflect"), 9, 1, 0)
        hX_original_base = F.avg_pool2d(
            F.pad(hX_original, (4, 4, 4, 4), "reflect"), 9, 1, 0
        )
        hX_detail = hX - hX_base
        hX_original_detail = hX_original - hX_original_base

        hX = wave_rec(hX_detail, hX_original_detail, "db2", mode=2) + wave_rec(
            hX_base, hX_original_base, "db2", mode=2
        )

        return hX
