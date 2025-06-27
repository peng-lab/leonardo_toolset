import warnings
from typing import Dict, Union
import numpy as np
import copy
import dask.array as da
import matplotlib.pyplot as plt
import torch
import tqdm
from bioio import BioImage

from leonardo_toolset.destripe.guided_filter_upsample import GuidedUpsample
from leonardo_toolset.destripe.loss_term_torch import Loss_torch
from leonardo_toolset.destripe.network_torch import DeStripeModel_torch
from leonardo_toolset.destripe.utils import (
    destripe_train_params,
    global_correction,
    prepare_aux,
    transform_cmplx_model,
)
from leonardo_toolset.destripe.utils_torch import (
    generate_mask_dict_torch,
    initialize_cmplx_model_torch,
    update_torch,
)
from leonardo_toolset.destripe.post_processing import post_process_module

warnings.filterwarnings("ignore", message="ignoring keyword argument 'read_only'")

try:
    # import haiku as hk
    import jax
    import jax.numpy as jnp

    from leonardo_toolset.destripe.loss_term_jax import Loss_jax
    from leonardo_toolset.destripe.network_jax import DeStripeModel_jax
    from leonardo_toolset.destripe.utils_jax import (
        generate_mask_dict_jax,
        initialize_cmplx_model_jax,
        update_jax,
    )

    # from jax import jit
    # import jaxwt

    jax_flag = 1
except Exception as e:
    print(f"Error: {e}. process without jax")
    jax_flag = 0


class DeStripe:
    def __init__(
        self,
        resample_ratio: int = 3,
        guided_upsample_kernel: int = 49,
        hessian_kernel_sigma: float = 1,
        lambda_masking_mse: int = 1,
        lambda_tv: float = 1,
        lambda_hessian: float = 1,
        inc: int = 16,
        n_epochs: int = 300,
        wedge_degree: float = 29,
        n_neighbors: int = 16,
        backend: str = "jax",
        device: str = None,
    ):
        self.train_params = {
            "gf_kernel_size": guided_upsample_kernel,
            "n_neighbors": n_neighbors,
            "inc": inc,
            "hessian_kernel_sigma": hessian_kernel_sigma,
            "lambda_tv": lambda_tv,
            "lambda_hessian": lambda_hessian,
            "lambda_masking_mse": lambda_masking_mse,
            "resample_ratio": resample_ratio,
            "n_epochs": n_epochs,
            "wedge_degree": wedge_degree,
        }
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.backend = backend
        if jax_flag == 0:
            if self.backend == "jax":
                print("jax not available on the current env, use torch instead.")
            self.backend = "torch"

    # ## INTERFACE ###
    # # THIS FUNCTION IS CALLED FROM NAPARI. DO NOT USE IT DIRECTLY.
    @staticmethod
    def process(params: dict) -> None:
        """
        Interface function for napari plugin.
        """
        model = DeStripe(
            resample_ratio=params["resample_ratio"],
            guided_upsample_kernel=params["guided_upsample_kernel"],
            hessian_kernel_sigma=params["hessian_kernel_sigma"],
            lambda_masking_mse=params["lambda_masking_mse"],
            lambda_tv=params["lambda_tv"],
            lambda_hessian=params["lambda_hessian"],
            inc=params["angular_size"],
            n_epochs=params["n_epochs"],
            wedge_degree=params["latent_dimension"],
            n_neighbors=params["n_neighbors"],
            backend=params["backend"],
        )

        return model.train(
            is_vertical=params["is_vertical"],
            x=params["input_image"],
            mask=params["mask"],
            angle_offset=params["angle_offset"],
            display=False,
            non_positive=params["non_positive"],
            display_angle_orientation=False,
        )

    @staticmethod
    def train_on_one_slice(
        GuidedFilterHRModel,
        update_method,
        sample_params: Dict,
        train_params: Dict,
        X: np.ndarray,
        mask: np.ndarray = None,
        fusion_mask: np.ndarray = None,
        s_: int = 1,
        z: int = 1,
        backend: str = "jax",
    ):
        rng_seq = jax.random.PRNGKey(0) if backend == "jax" else None
        md = (
            sample_params["md"]
            if sample_params["is_vertical"]
            else sample_params["nd"]  # noqa: E501
        )
        nd = (
            sample_params["nd"]
            if sample_params["is_vertical"]
            else sample_params["md"]  # noqa: E501
        )
        target = (X * fusion_mask).sum(1, keepdims=True)
        targetd = target[:, :, :: sample_params["r"], :]

        # Xd = X[:, :, :: sample_params["r"], :]
        fusion_maskd = fusion_mask[:, :, :: sample_params["r"], :]

        # to Fourier
        if backend == "jax":
            targetf = (
                jnp.fft.fftshift(jnp.fft.fft2(targetd), axes=(-2, -1))
                .reshape(1, targetd.shape[1], -1)[0]
                .transpose(1, 0)[: md * nd // 2, :][..., None]
            )
        else:
            targetf = (
                torch.fft.fftshift(torch.fft.fft2(targetd), dim=(-2, -1))
                .reshape(1, targetd.shape[1], -1)[0]
                .transpose(1, 0)[: md * nd // 2, :][..., None]
            )

        # initialize
        generate_mask_dict_func = (
            generate_mask_dict_jax if backend == "jax" else generate_mask_dict_torch
        )
        mask_dict, targets_f, targetd_bilinear = generate_mask_dict_func(
            targetd,
            target,
            fusion_maskd,
            update_method.loss.Dx,
            update_method.loss.Dy,
            update_method.loss.DGaussxx,
            update_method.loss.DGaussyy,
            update_method.loss.p_tv,
            update_method.loss.p_hessian,
            train_params,
            sample_params,
        )

        aver = targetd.sum((2, 3))

        initialize_cmplx_model = (
            initialize_cmplx_model_jax
            if backend == "jax"
            else initialize_cmplx_model_torch
        )

        net_params = initialize_cmplx_model(
            update_method._network,
            rng_seq,
            {
                "aver": aver,
                "Xf": targetf,
                "target": targetd,
                "target_hr": target,
                "coor": mask_dict["coor"],
            },
        )

        opt_state = update_method.opt_init(net_params)

        mask_dict.update(
            {
                "mse_mask": mask[:, :, :: sample_params["r"], :],
            }
        )

        for epoch in tqdm.tqdm(
            range(train_params["n_epochs"]),
            leave=False,
            desc="for {} ({} slices in total): ".format(s_, z),
        ):
            l, net_params, opt_state, Y_raw = update_method(
                epoch,
                net_params,
                opt_state,
                aver,
                targetf,
                targetd,
                mask_dict,
                target,
                targets_f,
                targetd_bilinear,
            )

        Y_GU = GuidedFilterHRModel(
            Y_raw,
            X,
            targetd,
            target,
            mask_dict["coor"],
            fusion_mask,
            sample_params["angle_offset_individual"],
            backend=backend,
        )

        if len(sample_params["illu_orient"]) > 0:
            Y = post_process_module(
                np.asarray(X) if backend == "jax" else X.cpu().data.numpy(),
                Y_GU,
                angle_offset_individual=sample_params["angle_offset_individual"],
                fusion_mask=(
                    np.asarray(fusion_mask)
                    if backend == "jax"
                    else fusion_mask.cpu().data.numpy()
                ),
                illu_orient=sample_params["illu_orient"],
            )
        else:
            Y = 10**Y_GU
        return Y[0, 0], (
            10 ** np.asarray(target[0, 0])
            if backend == "jax"
            else 10 ** target[0, 0].cpu().data.numpy()
        )

    @staticmethod
    def train_on_full_arr(
        X: Union[np.ndarray, da.core.Array],
        is_vertical: bool,
        angle_offset_dict: Dict,
        mask: Union[np.ndarray, da.core.Array],
        train_params: Dict = None,
        fusion_mask: Union[np.ndarray, da.core.Array] = None,
        display: bool = False,
        device: str = "cpu",
        non_positive: bool = False,
        backend: str = "jax",
        flag_compose: bool = False,
        display_angle_orientation: bool = True,
        illu_orient: str = None,
    ):
        if train_params is None:
            train_params = destripe_train_params()
        else:
            train_params = destripe_train_params(**train_params)
        angle_offset = []
        for key, item in angle_offset_dict.items():
            angle_offset = angle_offset + item
        angle_offset = list(set(angle_offset))
        angle_offset_individual = []
        if flag_compose:
            for i in range(len(angle_offset_dict)):
                angle_offset_individual.append(
                    angle_offset_dict["angle_offset_{}".format(i)]
                )
        else:
            angle_offset_individual.append(angle_offset_dict["angle_offset"])

        r = copy.deepcopy(train_params["resample_ratio"])

        illu_orient_new = []
        for illu in illu_orient:
            if illu in ["top", "left"]:
                illu_orient_new.append("top")
            elif illu in ["bottom", "right"]:
                illu_orient_new.append("bottom")
            elif illu in ["left-right", "top-bottom"]:
                illu_orient_new.append("top-bottom")
            else:
                pass
        sample_params = {
            "is_vertical": is_vertical,
            "angle_offset": angle_offset,
            "angle_offset_individual": angle_offset_individual,
            "r": r,
            "non_positive": non_positive,
            "illu_orient": illu_orient_new,
        }
        z, _, m, n = X.shape
        result = copy.deepcopy(X[:, 0, :, :])
        mean = np.zeros(z)
        if sample_params["is_vertical"]:
            n = n if n % 2 == 1 else n - 1
            m = m // train_params["resample_ratio"]
            if m % 2 == 0:
                m = m - 1
            m = m * train_params["resample_ratio"]
        else:
            m = m if m % 2 == 1 else m - 1
            n = n // train_params["resample_ratio"]
            if n % 2 == 0:
                n = n - 1
            n = n * train_params["resample_ratio"]
        sample_params["m"], sample_params["n"] = m, n
        if sample_params["is_vertical"]:
            sample_params["md"], sample_params["nd"] = (
                m // train_params["resample_ratio"],
                n,
            )
        else:
            sample_params["md"], sample_params["nd"] = (
                m,
                n // train_params["resample_ratio"],
            )

        hier_mask_arr, hier_ind_arr, NI_arr = prepare_aux(
            sample_params["md"],
            sample_params["nd"],
            sample_params["is_vertical"],
            np.rad2deg(
                np.arctan(r * np.tan(np.deg2rad(sample_params["angle_offset"])))
            ),
            train_params["wedge_degree"],
            train_params["n_neighbors"],
            backend=backend,
        )
        if display_angle_orientation:
            print("Please check the orientation of the stripes...")
            fig, ax = plt.subplots(
                1, 2 if not flag_compose else len(angle_offset_individual), dpi=200
            )
            if not flag_compose:
                ax[1].set_visible(False)
            for i in range(len(angle_offset_individual)):
                demo_img = X[z // 2, :, :, :]
                demo_m, demo_n = demo_img.shape[-2:]
                if not sample_params["is_vertical"]:
                    (demo_m, demo_n) = (demo_n, demo_m)
                ax[i].imshow(demo_img[i, :].compute() + 1.0)
                for deg in sample_params["angle_offset_individual"][i]:
                    d = np.tan(np.deg2rad(deg)) * demo_m
                    p0 = [0 + demo_n // 2 - d // 2, d + demo_n // 2 - d // 2]
                    p1 = [0, demo_m - 1]
                    if not sample_params["is_vertical"]:
                        (p0, p1) = (p1, p0)
                    ax[i].plot(p0, p1, "r")
                ax[i].axis("off")
            plt.show()

        GuidedFilterHRModel = GuidedUpsample(
            rx=train_params["gf_kernel_size"],
            device=device,
        )

        network = transform_cmplx_model(
            model=DeStripeModel_jax if backend == "jax" else DeStripeModel_torch,
            inc=train_params["inc"],
            m_l=(
                sample_params["md"]
                if sample_params["is_vertical"]
                else sample_params["nd"]
            ),
            n_l=(
                sample_params["nd"]
                if sample_params["is_vertical"]
                else sample_params["md"]
            ),
            Angle=sample_params["angle_offset"],
            NI=NI_arr,
            hier_mask=hier_mask_arr,
            hier_ind=hier_ind_arr,
            r=sample_params["r"],
            backend=backend,
            device=device,
        )

        train_params.update(
            {
                "max_pool_kernel_size": (
                    n // 20 * 2 + 1 if sample_params["is_vertical"] else m // 20 * 2 + 1
                )
            }
        )
        if backend == "jax":
            update_method = update_jax(
                network,
                Loss_jax(train_params, sample_params),
                0.01,
            )
        else:
            update_method = update_torch(
                network,
                Loss_torch(train_params, sample_params).to(device),
                0.01,
            )

        for i in range(z):
            input = np.log10(np.clip(np.asarray(X[i : i + 1])[:, :, :m, :n], 1, None))
            mask_slice = np.asarray(mask[i : i + 1, :m, :n])[None]
            if flag_compose:
                fusion_mask_slice = np.asarray(fusion_mask[i : i + 1])[:, :, :m, :n]
            else:
                fusion_mask_slice = np.ones(input.shape, dtype=np.float32)

            if not sample_params["is_vertical"]:
                input = input.transpose(0, 1, 3, 2)
                mask_slice = mask_slice.transpose(0, 1, 3, 2)
                fusion_mask_slice = fusion_mask_slice.transpose(0, 1, 3, 2)
            if backend == "jax":
                input = jnp.asarray(input)
                mask_slice = jnp.asarray(mask_slice)
                fusion_mask_slice = jnp.asarray(fusion_mask_slice)
            else:
                input = torch.from_numpy(input).to(device)
                mask_slice = torch.from_numpy(mask_slice).to(device)
                fusion_mask_slice = torch.from_numpy(fusion_mask_slice).to(device)

            Y, target = DeStripe.train_on_one_slice(
                GuidedFilterHRModel,
                update_method,
                sample_params,
                train_params,
                input,
                mask_slice,
                fusion_mask_slice,
                i + 1,
                z,
                backend=backend,
            )

            if not sample_params["is_vertical"]:
                Y = Y.T
                target = target.T

            if display:
                plt.figure(dpi=300)
                ax = plt.subplot(1, 2, 2)
                plt.imshow(Y, vmin=Y.min(), vmax=Y.max(), cmap="gray")
                ax.set_title("output", fontsize=8, pad=1)
                plt.axis("off")
                ax = plt.subplot(1, 2, 1)
                plt.imshow(target, vmin=Y.min(), vmax=Y.max(), cmap="gray")
                ax.set_title("input", fontsize=8, pad=1)
                plt.axis("off")
                plt.show()
            result[i:, : Y.shape[0], : Y.shape[1]] = np.clip(Y, 0, 65535).astype(
                np.uint16
            )
            mean[i] = np.mean(result[i] + 0.1)

        if (z != 1) and (not sample_params["non_positive"]):
            print("global correcting...")
            result = global_correction(mean, result)
        print("Done")
        return result

    def train(
        self,
        is_vertical: bool = None,
        x: Union[str, np.ndarray, da.core.Array] = None,
        mask: Union[str, np.ndarray, da.core.Array] = None,
        fusion_mask: Union[da.core.Array, np.ndarray] = None,
        illu_orient: str = None,
        display: bool = False,
        display_angle_orientation: bool = False,
        non_positive: bool = False,
        **kwargs,
    ):

        if x is not None:
            if (illu_orient is None) and (is_vertical is None):
                print("is_vertical and illu_orient cannot be missing at the same time.")
                return
            elif illu_orient is None:
                print(
                    "warning: illumination orientation is not given. post-processing will be ignored."
                )
            else:
                pass
            if illu_orient is not None:
                assert illu_orient in [
                    "top",
                    "bottom",
                    "left",
                    "right",
                    "left-right",
                    "top-bottom",
                ], print(
                    "illu_orient should be only top, bottom, left, right, left-right, or top-bottom."
                )
                if illu_orient in ["top", "bottom", "top-bottom"]:
                    is_vertical_illu = True
                else:
                    is_vertical_illu = False
                if is_vertical is not None:
                    assert is_vertical == is_vertical_illu, print(
                        "is_vertical should align with illu_orient."
                    )
                else:
                    is_vertical = is_vertical_illu
                illu_orient = [illu_orient]
            else:
                illu_orient = []
            print("Start DeStripe...\n")
            flag_compose = False
            X_handle = BioImage(x)
            X = X_handle.get_image_dask_data("ZYX", T=0, C=0)[:, None, ...]
        else:
            print("Start DeStripe-Fuse...\n")
            if fusion_mask is None:
                print("fusion_mask cannot be missing.")
                return
            flag_compose = True
            X_data = []
            for key, item in kwargs.items():
                if key.startswith("x_"):
                    X_handle = BioImage(item)
                    X_data.append(X_handle.get_image_dask_data("ZYX", T=0, C=0))
            X = da.stack(X_data, 1)

        angle_offset_dict = {}
        for key, item in kwargs.items():
            if key.startswith("angle_offset"):
                angle_offset_dict.update({key: item})

        z, _, m, n = X.shape

        # read in mask
        if mask is None:
            mask_data = np.zeros((z, m, n), dtype=bool)
        else:
            mask_handle = BioImage(mask)
            mask_data = mask_handle.get_image_dask_data("ZYX", T=0, C=0)
            assert mask_data.shape == (z, m, n), print(
                "mask should be of same shape as input volume(s)."
            )
        # read in dual-result, if applicable
        if flag_compose:
            assert not isinstance(fusion_mask, type(None)), print(
                "fusion mask is missing."
            )
            if fusion_mask.ndim == 3:
                fusion_mask = fusion_mask[None]
            assert (
                (fusion_mask.shape[0] == z)
                and (fusion_mask.shape[2] == m)
                and (fusion_mask.shape[3] == n)
            ), print(
                "fusion mask should be of shape [z_slices, ..., m rows, n columns]."
            )
            assert X.shape[1] == fusion_mask.shape[1], print(
                "inputs should be {} in total.".format(fusion_mask.shape[1])
            )
            assert len(angle_offset_dict) == fusion_mask.shape[1], print(
                "angle offsets should be {} in total.".format(fusion_mask.shape[1])
            )
            illu_orient = []
            for key, item in kwargs.items():
                if key.startswith("illu_orient_"):
                    illu_orient.append(item)
            if len(illu_orient) == 0:
                print(
                    "warning: illumination orientation is not given. post-processing will be ignored."
                )
            else:
                assert len(illu_orient) == fusion_mask.shape[1], print(
                    "illu_orient_ should be {} in total.".format(fusion_mask.shape[1])
                )
                for illu in illu_orient:
                    if illu in ["top", "bottom", "top-bottom"]:
                        is_vertical_illu = True
                    else:
                        is_vertical_illu = False
                    if is_vertical is not None:
                        assert is_vertical == is_vertical_illu, print(
                            "is_vertical should align with illu_orient."
                        )
                    else:
                        is_vertical = is_vertical_illu

        # training
        out = self.train_on_full_arr(
            X,
            is_vertical,
            angle_offset_dict,
            mask_data,
            self.train_params,
            fusion_mask,
            display=display,
            device=self.device,
            non_positive=non_positive,
            backend=self.backend,
            flag_compose=flag_compose,
            display_angle_orientation=display_angle_orientation,
            illu_orient=illu_orient,
        )
        return out
