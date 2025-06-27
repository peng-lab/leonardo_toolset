import numpy as np
import torch
import torch.nn.functional as F


def initialize_cmplx_model_torch(
    network,
    key,
    dummy_input,
):
    net_params = network.parameters()
    return net_params


class update_torch:
    def __init__(
        self,
        network,
        Loss,
        learning_rate,
    ):
        self.learning_rate = learning_rate
        self.loss = Loss
        self._network = network

    def opt_init(self, net_params):
        return torch.optim.Adam(net_params, lr=self.learning_rate)

    def __call__(
        self,
        step,
        params,
        optimizer,
        aver,
        xf,
        y,
        mask_dict,
        hy,
        targets_f,
        targetd_bilinear,
    ):
        optimizer.zero_grad()
        l, A = self.loss(
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
        l.backward()
        optimizer.step()
        return l, self._network.parameters(), optimizer, A


def generate_mask_dict_torch(
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
    md, nd = y.shape[-2:]
    r = train_params["max_pool_kernel_size"]

    targetd_bilinear = F.interpolate(
        hy, y.shape[-2:], mode="bilinear", align_corners=True
    )
    targets_f = F.interpolate(
        targetd_bilinear,
        (md, nd // sample_params["r"]),
        mode="bilinear",
        align_corners=True,
    )

    mask_tv = (
        torch.atan2(
            torch.abs(
                torch.conv2d(
                    F.pad(y, p_tv, "reflect"), Dx, stride=(1, 1), padding=(0, 0)
                )
            ),
            torch.abs(
                torch.conv2d(
                    F.pad(y, p_tv, "reflect"), Dy, stride=(1, 1), padding=(0, 0)
                )
            ),
        )
        ** 10
    )

    mask_hessian = (
        torch.atan2(
            torch.abs(
                torch.conv2d(
                    F.pad(y, p_hessian, "reflect"),
                    DGaussxx,
                    stride=(1, 1),
                    padding=(0, 0),
                )
            ),
            torch.abs(
                torch.conv2d(
                    F.pad(y, p_hessian, "reflect"),
                    DGaussyy,
                    stride=(1, 1),
                    padding=(0, 0),
                )
            ),
        )
        ** 10
    )

    mask_valid = torch.zeros_like(mask_hessian)
    for i, ao in enumerate(sample_params["angle_offset_individual"]):
        mask_xi_f = torch.from_numpy(np.isin(sample_params["angle_offset"], ao)).to(
            hy.device
        )
        mask_xi = mask_xi_f[:, None].reshape(-1)
        mask_valid += mask_xi[None, :, None, None] * fusion_mask[:, i : i + 1, :, :]
    mask_valid = mask_valid > 0
    mask_tv = mask_tv * mask_valid
    mask_hessian = mask_hessian * mask_valid

    mask_tv_f = -F.max_pool2d(
        -mask_tv,
        [1, sample_params["r"]],
        stride=(1, sample_params["r"]),
        padding=(0, 0),
    )

    ind_tv_f = torch.argmax(mask_tv_f, dim=1, keepdim=True)
    mask_tv_f = torch.max(mask_tv_f, dim=1, keepdim=True)[0]

    ind_tv = torch.argmax(mask_tv, dim=1, keepdim=True)
    mask_tv = torch.max(mask_tv, dim=1, keepdim=True)[0]

    mask_hessian_f = -F.max_pool2d(
        -mask_hessian,
        [1, sample_params["r"]],
        stride=(1, sample_params["r"]),
        padding=(0, 0),
    )

    ind_hessian_f = torch.argmax(mask_hessian_f, dim=1, keepdim=True)
    mask_hessian_f = torch.max(mask_hessian_f, dim=1, keepdim=True)[0]

    ind_hessian = torch.argmax(mask_hessian, dim=1, keepdim=True)
    mask_hessian = torch.max(mask_hessian, dim=1, keepdim=True)[0]

    mask_tv = F.max_pool2d(
        mask_tv,
        (Dx.shape[-2], Dx.shape[-1]),
        stride=(1, 1),
        padding=Dx.shape[-1] // 2,
    )
    mask_hessian = F.max_pool2d(
        mask_hessian,
        (DGaussxx.shape[-2], DGaussxx.shape[-2]),
        stride=(1, 1),
        padding=DGaussxx.shape[-1] // 2,
    )

    y_pad = F.pad(targets_f, (r // 2, r // 2, 0, 0), "constant").cpu()
    _, inds = y_pad.unfold(-1, r, 1).max(-1)
    inds = inds + torch.arange(targets_f.shape[-1])[None, None, None, :] - r // 2
    inds = inds.to(hy.device)

    y_pad = F.pad(y, (r // 2, r // 2, 0, 0), "constant").cpu()
    _, ind = y_pad.unfold(-1, r, 1).max(-1)
    ind = ind + torch.arange(y.shape[-1])[None, None, None, :] - r // 2
    ind = ind.to(hy.device)

    mask_tv_f = F.max_pool2d(
        mask_tv_f,
        (Dx.shape[-2], Dx.shape[-1]),
        stride=(1, 1),
        padding=Dx.shape[-1] // 2,
    )
    mask_hessian_f = F.max_pool2d(
        mask_hessian_f,
        (DGaussxx.shape[-2], DGaussxx.shape[-2]),
        stride=(1, 1),
        padding=DGaussxx.shape[-1] // 2,
    )

    mask_tv_f[
        :,
        :,
        torch.arange(y.shape[-2])[None, None, :, None].to(hy.device),
        inds,
    ] = 0

    mask_hessian_f[
        :,
        :,
        torch.arange(y.shape[-2])[None, None, :, None].to(hy.device),
        inds,
    ] = 0

    t = torch.linspace(0, y.shape[-2] - 1, (y.shape[-2] - 1) * sample_params["r"] + 1)
    t_max = t.max()
    t = torch.cat((t, t[1 : sample_params["r"]] + t[-1]))

    t = t / t_max
    t = (t - 0.5) * 2

    t2 = torch.arange(hy.shape[-1])[None, :].to(hy.device)
    t2 = 2 * (t2 / t2.max() - 0.5)

    coor = torch.zeros((1, hy.shape[-2], hy.shape[-1], 2)).to(hy.device)
    coor[0, :, :, 1] = t[:, None]
    coor[0, :, :, 0] = t2

    mask = torch.zeros_like(y)
    mask[
        0,
        0,
        torch.arange(y.shape[-2])[None, None, ::2, None].to(hy.device),
        ind[:, :, ::2, :],
    ] = 1
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
