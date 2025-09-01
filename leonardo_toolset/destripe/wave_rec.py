import ptwt
import pywt
import torch
import torch.nn.functional as F

from leonardo_toolset.destripe.constant import WaveletDetailTuple2d


def wave_rec(
    recon,
    hX,
    result_gu,
    kernel,
    mode,
    device,
):
    recon = torch.from_numpy(recon).to(device)
    hX = torch.from_numpy(hX).to(device)
    y_dict = ptwt.wavedec2(
        recon[:, :, :-1, :-1],
        pywt.Wavelet(kernel),
        level=6,
        mode="constant",
    )
    X_dict = ptwt.wavedec2(
        hX[:, :, :-1, :-1],
        pywt.Wavelet(kernel),
        level=6,
        mode="constant",
    )
    if result_gu is not None:
        result_gu = torch.from_numpy(result_gu).to(device)
        result_gu_dict = ptwt.wavedec2(
            result_gu[:, :, :-1, :-1],
            pywt.Wavelet(kernel),
            level=6,
            mode="constant",
        )
    else:
        result_gu_dict = y_dict
    x_base_dict = [y_dict[0]]

    mask_dict = []
    for ll, (detail, target) in enumerate(zip(result_gu_dict[1:], X_dict[1:])):
        mask_dict.append(
            [
                torch.abs(detail[0]) < torch.abs(target[0]),
                torch.abs(detail[1]) < torch.abs(target[1]),
                torch.abs(detail[2]) < torch.abs(target[2]),
            ]
        )

    for ll, (detail, target, mask) in enumerate(
        zip(y_dict[1:], result_gu_dict[1:], mask_dict)
    ):
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
    return recon.cpu().data.numpy()
