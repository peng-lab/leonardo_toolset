import copy

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from leonardo_toolset.destripe.utils_jax import generate_mapping_coordinates


class Cmplx_Xavier_Init(hk.initializers.Initializer):
    def __init__(
        self,
        input_units,
        output_units,
    ):
        self.n_in, self.n_out = input_units, output_units

    def __call__(
        self,
        shape,
        dtype,
    ):
        magnitudes = jnp.ones(shape) / self.n_in
        phases = jax.random.uniform(
            hk.next_rng_key(),
            shape,
            dtype="float32",
            minval=-np.pi,
            maxval=np.pi,
        )
        return magnitudes * jnp.exp(1.0j * phases)


class CmplxRndUniform(hk.initializers.Initializer):
    def __init__(
        self,
        minval=0,
        maxval=1.0,
    ):
        self.minval, self.maxval = minval, maxval

    def __call__(
        self,
        shape,
        dtype,
    ):
        real_part = jax.random.uniform(
            hk.next_rng_key(),
            shape,
            dtype="float32",
            minval=self.minval,
            maxval=self.maxval,
        )
        imag_part = jax.random.uniform(
            hk.next_rng_key(),
            shape,
            dtype="float32",
            minval=self.minval,
            maxval=self.maxval,
        )
        return jax.lax.complex(real_part, imag_part)


class CLinear(hk.Module):
    def __init__(
        self,
        output_size,
        name=None,
    ):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(
        self,
        inputs,
    ):
        input_size = inputs.shape[-1]
        dtype = inputs.dtype
        w = hk.get_parameter(
            "w",
            [input_size, self.output_size],
            dtype,
            init=Cmplx_Xavier_Init(input_size, self.output_size),
        )
        b = hk.get_parameter(
            "b",
            [self.output_size],
            dtype,
            init=CmplxRndUniform(minval=-0.001, maxval=0.001),
        )
        return jnp.dot(inputs, w) + b


def complex_relu(x):
    return jax.lax.complex(jax.nn.elu(x.real), jax.nn.elu(x.imag))


class ResLearning(hk.Module):
    def __init__(
        self,
        outc,
    ):
        super().__init__()
        self.outc = outc

    def __call__(
        self,
        x,
    ):
        inx = CLinear(self.outc)(complex_relu(CLinear(self.outc)(x)))
        return complex_relu(inx + CLinear(self.outc)(x))


class gnn(hk.Module):
    def __init__(
        self,
        NI,
        hier_mask,
        hier_ind,
        inc,
    ):
        super().__init__()
        self.NI = NI
        self.hier_mask = hier_mask
        self.hier_ind = hier_ind
        self.inc = inc

    def __call__(
        self,
        Xf,
    ):
        w = hk.get_parameter(
            "w",
            (self.NI.shape[0], self.NI.shape[1]),
            jnp.complex64,
            init=Cmplx_Xavier_Init(self.NI.shape[0], self.NI.shape[1]),
        )
        Xfcx = jnp.sum(jnp.einsum("knbc,kn->knbc", Xf[self.NI, :], w), 0)  # (M*N, c)
        Xf_tvx = jnp.concatenate((Xfcx, Xf[self.hier_mask, :]), 0)[
            self.hier_ind, :
        ].reshape(
            1, -1, 1, self.inc
        )  # (A, M*N, 2, c)
        return Xf_tvx


class tv_uint(hk.Module):
    def __init__(
        self,
        TVfftx,
        TVffty,
        inverseTVfftx,
        inverseTVffty,
        eigDtD,
        edgeProcess,
        latentProcess,
    ):
        super().__init__()
        self.TVfftx, self.TVffty = TVfftx, TVffty
        self.inverseTVfftx, self.inverseTVffty = inverseTVfftx, inverseTVffty
        self.eigDtD = eigDtD
        self.edgeProcess, self.latentProcess = edgeProcess, latentProcess

    def __call__(self, Xf_tvx, Xf):
        Xf_tvx, Xf_tvy = Xf_tvx * self.TVfftx, Xf * self.TVffty
        X_fourier = []
        for (
            x,
            y,
            inverseTVfftx,
            inverseTVffty,
            eigDtD,
            edgeProcess,
            latentProcess,
        ) in zip(
            Xf_tvx,
            Xf_tvy,
            self.inverseTVfftx,
            self.inverseTVffty,
            self.eigDtD,
            self.edgeProcess,
            self.latentProcess,
        ):
            X_fourier.append(
                complex_relu(
                    latentProcess(
                        complex_relu(edgeProcess(x) * inverseTVfftx)
                        + complex_relu(edgeProcess(y) * inverseTVffty)
                    )
                    / eigDtD
                )
            )
        return jnp.concatenate(X_fourier, -1)


class GuidedFilterJAX(hk.Module):
    def __init__(self, rx, ry, r, Angle, m=None, n=None, eps=1e-9):
        super().__init__()
        Angle = np.rad2deg(np.arctan(r * np.tan(np.deg2rad(Angle))))
        self.kernelL = []
        for A in Angle:
            lval = np.arange(rx) - rx // 2
            lval = np.round(lval * np.tan(np.deg2rad(A))).astype(np.int32)
            ry = (lval.max() - lval.min()) // 2 * 2 + 1
            i, j = jnp.meshgrid(jnp.arange(ry), jnp.arange(rx))
            i = i - ry // 2
            j = j - rx // 2
            kernel = (i == lval[:, None]).astype(jnp.float32)[None, None]

            self.kernelL.append(kernel)
        self.AngleNum = len(Angle)
        self.Angle = Angle

        self.pr = [self.kernelL[i].shape[-1] // 2 for i in range(self.AngleNum)]
        self.pc = [self.kernelL[i].shape[-2] // 2 for i in range(self.AngleNum)]

        XN = jnp.ones((1, 1, m, n))
        self.N = [
            self.boxfilter(XN, self.kernelL[i], self.pc[i], self.pr[i])
            for i in range(self.AngleNum)
        ]
        self.kernel_fft = []
        for k in self.kernelL:
            self.kernel_fft.append(
                jnp.fft.fftshift(
                    jnp.fft.fft2(
                        k / k.sum(),
                        (m, n),
                    )
                ).reshape(-1)[
                    : (m * n) // 2
                ][..., None]
            )
        self.m_l = m
        self.n_l = n

    def fourierResult(
        self,
        z,
        aver,
    ):
        return jnp.abs(
            jnp.fft.ifft2(
                jnp.fft.ifftshift(
                    jnp.concatenate(
                        (z, aver, jnp.conj(jnp.flip(z, -2))),
                        -2,
                    )
                    .reshape(1, self.m_l, -1, 1)
                    .transpose(0, 3, 1, 2),
                    axes=(-2, -1),
                )
            )
        )

    def boxfilter(self, x, k, pc, pr):
        return jax.lax.conv_general_dilated(
            jnp.pad(
                x,
                ((0, 0), (0, 0), (pc, pc), (pr, pr)),
                mode="constant",
                constant_values=0,
            ),
            k,
            (1, 1),
            "VALID",
            feature_group_count=x.shape[1],
        )

    def __call__(self, X, X0, y, aver, hX, coor):
        for i in range(self.AngleNum):
            b = self.kernel_fft[i] * self.kernel_fft[i] * (y - X)
            X = X + b
        X = self.fourierResult(X, aver)

        hX = (
            jax.scipy.ndimage.map_coordinates(X - X0, coor, order=1, mode="reflect")[
                None, None
            ]
            + hX
        )
        return hX


class DeStripeModel_jax(hk.Module):
    def __init__(
        self,
        Angle,
        hier_mask,
        hier_ind,
        NI,
        m_l,
        n_l,
        r,
        inc=16,
    ):
        super().__init__()
        self.NI, self.hier_mask, self.hier_ind, self.inc = NI, hier_mask, hier_ind, inc

        self.m_l, self.n_l = m_l, n_l
        self.Angle = Angle

        gx_0 = jnp.fft.fftshift(
            jnp.fft.fft2(jnp.array([[1, -1]], dtype=jnp.float32), (self.m_l, self.n_l))
        )
        gy_0 = jnp.fft.fftshift(
            jnp.fft.fft2(
                jnp.array([[1], [-1]], dtype=jnp.float32), (self.m_l, self.n_l)
            )
        )

        self.TVfftx = []
        self.inverseTVfftx = []
        self.TVffty = []
        self.inverseTVffty = []
        for i, A in enumerate(Angle):
            trans_matrix = generate_mapping_coordinates(
                np.rad2deg(np.arctan(r * np.tan(np.deg2rad(-A)))),
                gx_0.shape[0],
                gx_0.shape[1],
                reshape=False,
            )
            self.fftnt(
                jax.scipy.ndimage.map_coordinates(
                    gx_0[None, None], trans_matrix, 1, mode="nearest"
                )[0, 0],
                self.m_l,
                self.n_l,
            )
            self.fftn(
                jax.scipy.ndimage.map_coordinates(
                    gy_0[None, None], trans_matrix, 1, mode="nearest"
                )[0, 0],
                self.m_l,
                self.n_l,
            )
        self.TVfftx = jnp.concatenate(self.TVfftx, 0)
        self.inverseTVfftx = jnp.concatenate(self.inverseTVfftx, 0)
        self.TVffty = jnp.concatenate(self.TVffty, 0)
        self.inverseTVffty = jnp.concatenate(self.inverseTVffty, 0)

        self.eigDtD = jnp.power(jnp.abs(self.TVfftx), 2) + jnp.power(
            jnp.abs(self.TVffty), 2
        )
        self.TVfftx, self.inverseTVfftx = (
            self.TVfftx[..., None],
            self.inverseTVfftx[..., None],
        )
        self.TVffty, self.inverseTVffty = (
            self.TVffty[..., None],
            self.inverseTVffty[..., None],
        )
        self.eigDtD = self.eigDtD[..., None]

        self.p = ResLearning(inc)
        self.edgeProcess = []
        for _ in Angle:
            self.edgeProcess.append(
                hk.Sequential([CLinear(inc), complex_relu, CLinear(inc)])
            )
        self.latentProcess = []
        for _ in Angle:
            self.latentProcess.append(
                hk.Sequential([CLinear(inc), complex_relu, CLinear(inc)])
            )
        self.ainput = jnp.ones((1, 1))
        self.merge = hk.Sequential(
            [
                CLinear(inc),
                complex_relu,
                CLinear(inc),
                complex_relu,
                CLinear(1),
            ]
        )

        self.gnn = gnn(self.NI, self.hier_mask, self.hier_ind, inc)

        self.tv_uint = tv_uint(
            self.TVfftx,
            self.TVffty,
            self.inverseTVfftx,
            self.inverseTVffty,
            self.eigDtD,
            self.edgeProcess,
            self.latentProcess,
        )
        self.GuidedFilter = GuidedFilterJAX(
            rx=49,
            ry=3,
            r=r,
            m=self.m_l,
            n=self.n_l,
            Angle=Angle,
        )
        self.basep = hk.Sequential(
            [
                hk.Linear(inc),
                jax.nn.elu,
                hk.Linear(inc),
                jax.nn.elu,
                hk.Linear(1),
            ]
        )

    def fftnt(
        self,
        x,
        row,
        col,
    ):
        y = x.reshape(-1)[: col * row // 2][None, :, None]
        self.TVfftx.append(y)
        self.inverseTVfftx.append(jnp.conj(y))

    def fftn(
        self,
        x,
        row,
        col,
    ):
        y = x.reshape(-1)[: col * row // 2][None, :, None]
        self.TVffty.append(y)
        self.inverseTVffty.append(jnp.conj(y))

    def fourierResult(
        self,
        z,
        aver,
    ):
        return jnp.abs(
            jnp.fft.ifft2(
                jnp.fft.ifftshift(
                    jnp.concatenate(
                        (z, aver, jnp.conj(jnp.flip(z, -2))),
                        -2,
                    )
                    .reshape(1, self.m_l, -1, 1)
                    .transpose(0, 3, 1, 2),
                    axes=(-2, -1),
                )
            )
        )

    def __call__(
        self,
        aver,
        Xf,
        target,
        target_hr,
        coor,
    ):
        Xf = self.p(Xf)  # (M*N, 2,)
        Xf_tvx = self.gnn(Xf)

        # X_fourier = self.merge(Xf_tvx)[0]
        X_fourier = self.merge(self.tv_uint(Xf_tvx, Xf))

        outputGNNraw = self.fourierResult(X_fourier[..., 0], aver)
        alpha = hk.get_parameter(
            "alpha",
            (1,),
            jnp.float32,
            init=jnp.ones,
        )
        outputGNNraw = outputGNNraw + alpha
        outputLR = self.GuidedFilter(
            Xf[..., 0],
            target,
            X_fourier[..., 0],
            aver + alpha * self.m_l * self.n_l,
            target_hr,
            coor,
        )
        return outputGNNraw, outputLR
