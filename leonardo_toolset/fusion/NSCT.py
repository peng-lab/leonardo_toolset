import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NSCTdec(nn.Module):
    def __init__(self, levels, device):
        super().__init__()
        self.device = device
        self.levels = levels
        self.max_filter = nn.MaxPool2d((9, 9), stride=(1, 1), padding=(4, 4))
        self.dKernel = torch.ones(1, 1, 3, 3).to(self.device) / 9
        self.stdpadding = nn.ReflectionPad2d((1, 1, 1, 1))

        h1, h2 = self.dfilters()
        self.register_buffer(
            "level_0_0",
            torch.from_numpy(self.modulate2(h1, "c")[None, None, :, :]).float(),
        )
        self.register_buffer(
            "level_0_1",
            torch.from_numpy(self.modulate2(h2, "c")[None, None, :, :]).float(),
        )

        level_1_0, level_1_1 = self.modulate_kernel(
            self.level_0_0,
            self.level_0_1,
            np.array([[1, -1], [1, 1]]),
        )

        self.register_buffer("level_1_0", level_1_0)
        self.register_buffer("level_1_1", level_1_1)

        f1, f2 = self.parafilters(h1, h2)

        for l in range(3, max(levels) + 1):  # noqa: E741
            level_0, level_1 = [], []
            for k in range(1, 2 ** (l - 2) + 1):
                slk = 2 * math.floor((k - 1) / 2) - 2 ** (l - 3) + 1
                mkl = 2 * np.matmul(
                    np.array([[2 ** (l - 3), 0], [0, 1]]),
                    np.array([[1, 0], [-slk, 1]]),
                )
                i = (k - 1) % 2 + 1
                kernel_1, kernel_2 = self.modulate_kernel(
                    f1["{}".format(i - 1)],
                    f2["{}".format(i - 1)],
                    mkl,
                )
                level_0.append(kernel_1)
                level_1.append(kernel_2)
            for k in range(2 ** (l - 2) + 1, 2 ** (l - 1) + 1):
                slk = 2 * math.floor((k - 2 ** (l - 2) - 1) / 2) - 2 ** (l - 3) + 1
                mkl = 2 * np.matmul(
                    np.array([[1, 0], [0, 2 ** (l - 3)]]),
                    np.array([[1, -slk], [0, 1]]),
                )
                i = (k - 1) % 2 + 3
                kernel_1, kernel_2 = self.modulate_kernel(
                    f1["{}".format(i - 1)],
                    f2["{}".format(i - 1)],
                    mkl,
                )
                level_0.append(kernel_1)
                level_1.append(kernel_2)

            level_0 = torch.cat(level_0, 0)
            # level_0 = level_0.repeat(1, level_0.shape[0], 1, 1)
            level_1 = torch.cat(level_1, 0)
            # level_1 = level_1.repeat(1, level_1.shape[0], 1, 1)
            self.register_buffer("level_{}_0".format(l - 1), level_0)
            self.register_buffer("level_{}_1".format(l - 1), level_1)

        self.h1, self.h2 = self.atrousfilters()

    def atrousfilters(self):
        A = np.array(
            [
                [-0.00323604, -0.01294417, -0.01941626, -0.01294417, -0.00323604],
                [-0.01294417, 0.0625, 0.15088835, 0.0625, -0.01294417],
                [-0.01941626, 0.15088835, 0.34060922, 0.15088835, -0.01941626],
                [-0.01294417, 0.0625, 0.15088835, 0.0625, -0.01294417],
                [-0.00323604, -0.01294417, -0.01941626, -0.01294417, -0.00323604],
            ]
        )
        B = np.array(
            [
                [-0.00323604, -0.01294417, -0.01941626, -0.01294417, -0.00323604],
                [-0.01294417, -0.0625, -0.09911165, -0.0625, -0.01294417],
                [-0.01941626, -0.09911165, 0.84060922, -0.09911165, -0.01941626],
                [-0.01294417, -0.0625, -0.09911165, -0.0625, -0.01294417],
                [-0.00323604, -0.01294417, -0.01941626, -0.01294417, -0.00323604],
            ]
        )
        return torch.from_numpy(A)[None, None, :, :].float().to(
            self.device
        ), torch.from_numpy(B)[None, None, :, :].float().to(self.device)

    def parafilters(self, f1, f2):
        y1, y2 = {}, {}
        y1["0"], y2["0"] = self.modulate2(f1, "r"), self.modulate2(f2, "r")
        y1["1"], y2["1"] = self.modulate2(f1, "c"), self.modulate2(f2, "c")
        y1["2"], y2["2"] = y1["0"].T, y2["0"].T
        y1["3"], y2["3"] = y1["1"].T, y2["1"].T
        for i in range(4):
            y1["{}".format(i)] = (
                torch.from_numpy(
                    self.resampz(y1["{}".format(i)], i + 1)[None, None, :, :]
                )
                .float()
                .to(self.device)
            )
            y2["{}".format(i)] = (
                torch.from_numpy(
                    self.resampz(y2["{}".format(i)], i + 1)[None, None, :, :]
                )
                .float()
                .to(self.device)
            )
        return y1, y2

    def resampz(self, x, sampleType):
        shift, sx = 1, x.shape  # noqa: F841
        if (sampleType == 1) or (sampleType == 2):
            y = np.zeros((sx[0] + sx[1] - 1, sx[1]))
            shift1 = (
                -1 * np.arange(0, sx[1], 1, dtype=int)
                if sampleType == 1
                else np.arange(0, sx[1], 1, dtype=int)
            )
            if shift1[-1] < 0:
                shift1 = shift1 - shift1[-1]
            for n in range(sx[1]):
                y[shift1[n] + np.arange(0, sx[0], 1, dtype=int), n] = x[:, n]
            start, finish = 0, y.shape[0] - 1
            while np.sum(np.abs(y[start, :])) == 0:
                start = start + 1
            while np.sum(np.abs(y[finish, :])) == 0:
                finish = finish - 1
            y = y[start : finish + 1, :]
        else:
            y = np.zeros((sx[0], sx[1] + sx[0] - 1))
            shift2 = (
                -1 * np.arange(0, sx[0], 1, dtype=int)
                if sampleType == 3
                else np.arange(0, sx[0], 1, dtype=int)
            )
            if shift2[-1] < 0:
                shift2 = shift2 - shift2[-1]
            for m in range(sx[0]):
                y[m, shift2[m] + np.arange(0, sx[1], 1, dtype=int)] = x[m, :]
            start, finish = 0, y.shape[1] - 1
            while np.sum(np.abs(y[:, start])) == 0:
                start = start + 1
            while np.sum(np.abs(y[:, finish])) == 0:
                finish = finish - 1
            y = y[:, start : finish + 1]
        return y

    def modulate2(self, x, modulateType):
        o = np.floor(np.array(x.shape) / 2) + 1
        n1, n2 = (
            np.arange(1, x.shape[0] + 1, 1) - o[0],
            np.arange(1, x.shape[1] + 1, 1) - o[1],
        )
        if modulateType == "c":
            m2 = (-1) ** n2
            return x * np.repeat(m2[None, :], x.shape[0], axis=0)
        elif modulateType == "r":
            m1 = (-1) ** n1
            return x * np.repeat(m1[:, None], x.shape[1], axis=1)

    def dfilters(self):
        A = np.array([[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]])
        B = np.array(
            [
                [-0.0, -0.0, -0.0625, -0.0, -0.0],
                [-0.0, -0.125, -0.25, -0.125, -0.0],
                [-0.0625, -0.25, 1.75, -0.25, -0.0625],
                [-0.0, -0.125, -0.25, -0.125, -0.0],
                [-0.0, -0.0, -0.0625, -0.0, -0.0],
            ]
        )
        return A / math.sqrt(2), B / math.sqrt(2)

    def nsfbdec(self, x, h0, h1, lev):
        if lev != 0:
            y0 = torch.conv2d(
                self.symext(
                    x,
                    (2 ** (lev - 1)) * (h0.size(-2) - 1),
                    (2 ** (lev - 1)) * (h0.size(-1) - 1),
                ),
                h0,
                dilation=2**lev,
            )
            y1 = torch.conv2d(
                self.symext(
                    x,
                    (2 ** (lev - 1)) * (h1.size(-2) - 1),
                    (2 ** (lev - 1)) * (h1.size(-1) - 1),
                ),
                h1,
                dilation=2**lev,
            )
        else:
            y0, y1 = torch.conv2d(
                self.symext(x, h0.size(-2) // 2, h0.size(-1) // 2), h0
            ), torch.conv2d(self.symext(x, h1.size(-2) // 2, h1.size(-1) // 2), h1)
        return y0, y1

    def symext(self, x, er, ec):
        x = torch.cat(
            (torch.flip(x[:, :, :er, :], [-2]), x, torch.flip(x[:, :, -er:, :], [-2])),
            -2,
        )
        return torch.cat(
            (torch.flip(x[:, :, :, :ec], [-1]), x, torch.flip(x[:, :, :, -ec:], [-1])),
            -1,
        )

    def nsdfbdec(self, x, clevels):
        H, W = x.shape[-2:]
        if clevels == 1:
            y = torch.cat((self.nssfbdec(x, self.level_0_0, self.level_0_1)), 1)
        else:
            x1, x2 = self.nssfbdec(x, self.level_0_0, self.level_0_1)
            y = torch.cat(
                (
                    *self.nssfbdec(x1, self.level_1_0, self.level_1_1),
                    *self.nssfbdec(x2, self.level_1_0, self.level_1_1),
                ),
                1,
            )
            for ll in range(3, clevels + 1):
                y = torch.cat(
                    (
                        self.conv_perext(y, getattr(self, f"level_{ll-1}_0")),
                        self.conv_perext(y, getattr(self, f"level_{ll-1}_1")),
                    ),
                    1,
                )
        return y

    def modulate_kernel_fft(self, h1, h2, H, W, m="None"):
        if isinstance(m, np.ndarray) and (sum(sum(m == np.eye(2))) != 4):
            h1 = (
                self.rot45(h1)
                if sum(sum(m == np.array([[1, -1], [1, 1]]))) == 4
                else self.my_upsamp2df(h1, m)
            )
            h2 = (
                self.rot45(h2)
                if sum(sum(m == np.array([[1, -1], [1, 1]]))) == 4
                else self.my_upsamp2df(h2, m)
            )
        h1_padded = torch.zeros((1, 1, H, W), dtype=h1.dtype, device=h1.device)
        h1_padded[..., : h1.shape[-2], : h1.shape[-1]] = h1

        h2_padded = torch.zeros((1, 1, H, W), dtype=h2.dtype, device=h2.device)
        h2_padded[..., : h2.shape[-2], : h2.shape[-1]] = h2

        return torch.cat(
            (
                torch.fft.rfft2(h1_padded, dim=(-2, -1)),
                torch.fft.rfft2(h2_padded, dim=(-2, -1)),
            ),
            1,
        )

    def nssfbdec(self, x, f1, f2):
        # f1 = self.modulate_kernel(f1, mup)
        # f2 = self.modulate_kernel(f2, mup)
        return self.conv_perext(x, f1), self.conv_perext(x, f2)

    def conv_perext(self, x, f):
        return torch.conv2d(
            self.perext(x, f.size(-2) // 2, f.size(-1) // 2),
            f,
            groups=x.shape[1],
        )

    def modulate_kernel(self, h1, h2, m="None"):
        if isinstance(m, np.ndarray) and (sum(sum(m == np.eye(2))) != 4):
            h1 = (
                self.rot45(h1)
                if sum(sum(m == np.array([[1, -1], [1, 1]]))) == 4
                else self.my_upsamp2df(h1, m)
            )

            h2 = (
                self.rot45(h2)
                if sum(sum(m == np.array([[1, -1], [1, 1]]))) == 4
                else self.my_upsamp2df(h2, m)
            )
        return h1, h2

    def rot45(self, h0):
        h = torch.zeros(1, 1, 2 * h0.size()[-2] - 1, 2 * h0.size()[-1] - 1).to(
            self.device
        )
        sz1, sz2 = h0.size()[-2:]
        for i in range(1, sz1 + 1):
            r, c = i + np.arange(0, sz2, 1), sz2 - i + np.arange(1, sz2 + 1, 1)
            for j in range(1, sz2 + 1):
                h[:, :, r[j - 1] - 1, c[j - 1] - 1] = h0[:, :, i - 1, j - 1]
        return h

    def my_upsamp2df(self, h0, mup):
        m, n = h0.size()[-2:]
        power = math.log2(mup[0, 0])
        R1, R2 = torch.zeros((1, 1, int(2**power * (m - 1) + 1), m)).to(
            self.device
        ), torch.zeros((1, 1, n, int(2**power * (n - 1) + 1))).to(self.device)
        for i in range(1, m + 1):
            R1[:, :, int((i - 1) * 2 ** (power)), i - 1] = 1
        for i in range(1, n + 1):
            R2[:, :, i - 1, int((i - 1) * 2 ** (power))] = 1
        return torch.matmul(torch.matmul(R1, h0), R2)

    def perext(self, x, er, ec):
        return F.pad(x, (ec, ec, er, er), "circular")

    def extractFeatures(self, x):
        b, _, m, n = x[0].size()
        f = torch.zeros(b, 1, m, n).to(self.device)
        L = sum([2**ll for ll in self.levels])
        for d in x:
            f += torch.sum(d.abs(), dim=1, keepdim=True)
        return f / L

    @torch.no_grad()
    def nsctDec(self, x, stride=None, _forFeatures=False):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        if x.ndim == 3:
            x = x[:, None, ...]
        clevels, nIndex = len(self.levels), len(self.levels) + 1
        y = []
        for i in range(1, clevels + 1):
            xlo, xhi = self.nsfbdec(x, self.h1, self.h2, i - 1)
            if self.levels[nIndex - 2] > 0:
                xhi_dir = self.nsdfbdec(xhi, self.levels[nIndex - 2])
                y.append(xhi_dir)

            else:
                y.append(xhi)

            nIndex = nIndex - 1
            x = xlo
        if _forFeatures:
            f = self.extractFeatures(y)
            df, dfbase = torch.conv2d(
                f, self.dKernel, stride=stride, padding=self.dKernel.shape[-1] // 2
            ), torch.conv2d(
                x, self.dKernel, stride=stride, padding=self.dKernel.shape[-1] // 2
            )
            dfstd = (
                self.stdpadding(df).unfold(2, 3, 1).unfold(3, 3, 1).std(dim=(-2, -1))
            )
            del f, x
            return (
                df[:, 0, :, :].cpu().data.numpy(),
                dfbase[:, 0, :, :].cpu().data.numpy(),
                dfstd[:, 0, :, :].cpu().data.numpy(),
            )
        else:
            return y, x


# import time

# a = torch.from_numpy(np.random.rand(10, 1, 2048, 2048)).cuda().to(torch.float)
# model = NSCTdec(
#     levels=[3, 3, 3],
#     device="cuda",
# ).cuda()
# aa = time.time()
# for i in range(10):
#     y = model.nsctDec(a[i : i + 1], stride=2, _forFeatures=True)
# bb = time.time()
# print(bb - aa)

# # model = NSCTrec(levels=[3, 3, 3], device="cuda")
# # z = model.nsctRec(y, x)

# # print((z - a).abs().sum(), a.abs().sum())
