import copy

import numpy as np
import torch
import torch.nn.functional as F

try:
    import jax
except Exception as e:
    print(f"Error: {e}. Proceed without jax")
    pass


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
        return np.log10(np.clip(y.sum(1, keepdims=True), 1, None))

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

        return hX
