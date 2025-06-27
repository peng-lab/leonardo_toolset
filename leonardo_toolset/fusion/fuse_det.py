from datetime import datetime
import os
from bioio.writers import OmeTiffWriter
from typing import Union
import numpy as np
import traceback
import dask
import open3d as o3d
import torch
from bioio import BioImage
import copy
import gc
import shutil
import matplotlib.pyplot as plt
import tqdm
import matplotlib.patches as patches
import tifffile
import torch.nn.functional as F
from skimage import morphology

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

from leonardo_toolset.fusion.blobs_dog import DoG
from leonardo_toolset.fusion.fuse_illu import FUSE_illu
from leonardo_toolset.fusion.NSCT import NSCTdec
from leonardo_toolset.fusion.utils import (
    EM2DPlus,
    extendBoundary2,
    refineShape,
    sgolay2dkernel,
    waterShed,
    parse_yaml_det,
    extract_leaf_file_paths_from_file,
    read_with_bioio,
    strip_ext,
    fusionResult_VD,
    fusionResultFour,
    volumeTranslate_compose,
    boundaryInclude,
    fineReg,
    coarseRegistrationXY,
    coarseRegistrationZX,
)

import pandas as pd

pd.set_option("display.width", 10000)


def make_Ramp(ramp_colors):
    from colour import Color
    from matplotlib.colors import LinearSegmentedColormap

    color_ramp = LinearSegmentedColormap.from_list(
        "my_list", [Color(c1).rgb for c1 in ramp_colors]
    )
    return color_ramp


custom_ramp = make_Ramp(["#000000", "#D62728"])
red = custom_ramp(range(256))[:, :-1] * 255

custom_ramp = make_Ramp(["#000000", "#FF7F0E"])
orange = custom_ramp(range(256))[:, :-1] * 255

custom_ramp = make_Ramp(["#000000", "#17BECF"])
blue = custom_ramp(range(256))[:, :-1] * 255

custom_ramp = make_Ramp(["#000000", "#2CA02C"])
green = custom_ramp(range(256))[:, :-1] * 255

custom_ramp = make_Ramp(["#000000", "#9467BD"])
purple = custom_ramp(range(256))[:, :-1] * 255

red = red.astype(np.uint8).T
orange = orange.astype(np.uint8).T
blue = blue.astype(np.uint8).T
purple = purple.astype(np.uint8).T
green = green.astype(np.uint8).T


def define_registration_params(
    use_exist_reg: bool = False,
    require_reg_finetune: bool = True,
    axial_downsample: int = 1,
    lateral_downsample: int = 2,
    skip_refine_registration: bool = False,
):
    kwargs = locals()
    return kwargs


class FUSE_det:
    def __init__(
        self,
        require_precropping: bool = True,
        precropping_params: list[int, int, int, int] = [],
        resample_ratio: int = 2,
        window_size: list[int, int] = [5, 59],
        poly_order: list[int, int] = [2, 2],
        n_epochs: int = 50,
        require_segmentation: bool = True,
        skip_illuFusion: bool = True,
        device: str = None,
        registration_params=None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_params = {
            "require_precropping": require_precropping,
            "precropping_params": precropping_params,
            "resample_ratio": resample_ratio,
            "window_size": window_size,
            "poly_order": poly_order,
            "n_epochs": n_epochs,
            "require_segmentation": require_segmentation,
            "device": device,
        }
        self.modelFront = FUSE_illu(**self.train_params)
        self.modelBack = FUSE_illu(**self.train_params)
        self.train_params.update(
            {
                "skip_illuFusion": skip_illuFusion,
            }
        )
        self.train_params["kernel2d"] = (
            torch.from_numpy(
                sgolay2dkernel(
                    np.array([window_size[1], window_size[1]]),
                    np.array(
                        [
                            self.train_params["poly_order"][1],
                            self.train_params["poly_order"][1],
                        ]
                    ),
                )
            )
            .to(torch.float)
            .to(self.train_params["device"])
        )
        if registration_params is None:
            self.registration_params = define_registration_params()
        else:
            self.registration_params = define_registration_params(**registration_params)

    def train_from_params(
        self,
        params: dict,
    ):
        """Parses training parameters from dictionary"""
        if params["method"] != "detection":
            raise ValueError(f"Invalid method: {params['method']}")
        if params["amount"] not in [2, 4]:
            raise ValueError("Only 2 or 4 images are supported for detection")
        image1 = params["image1"]
        image3 = params["image3"]
        direction1 = params["direction1"]
        direction3 = params["direction3"]

        top_ventral_data = None
        bottom_ventral_data = None
        top_dorsal_data = None
        bottom_dorsal_data = None
        left_ventral_data = None
        right_ventral_data = None
        left_dorsal_data = None
        right_dorsal_data = None

        ventral_data = None
        dorsal_data = None
        left_right = None

        if params["amount"] == 4:
            image2 = params["image2"]
            image4 = params["image4"]
            direction2 = params["direction2"]
            direction4 = params["direction4"]
            if direction1 == "Top" and direction2 == "Bottom":
                top_ventral_data = image1
                bottom_ventral_data = image2
            elif direction1 == "Bottom" and direction2 == "Top":
                top_ventral_data = image2
                bottom_ventral_data = image1
            elif direction1 == "Left" and direction2 == "Right":
                left_ventral_data = image1
                right_ventral_data = image2
            elif direction1 == "Right" and direction2 == "Left":
                left_ventral_data = image2
                right_ventral_data = image1
            else:
                raise ValueError(
                    f"Invalid directions for ventral detection: {direction1}, "
                    f"{direction2}"
                )

            if (
                direction3 not in [direction1, direction2]
                or direction4 not in [direction1, direction2]
                or direction3 == direction4
            ):
                raise ValueError(
                    f"Invalid directions for dorsal detection: {direction3}, "
                    f"{direction4}"
                )

            if direction3 == "Top" and direction4 == "Bottom":
                top_dorsal_data = image3
                bottom_dorsal_data = image4
            elif direction3 == "Bottom" and direction4 == "Top":
                top_dorsal_data = image4
                bottom_dorsal_data = image3
            elif direction3 == "Left" and direction4 == "Right":
                left_dorsal_data = image3
                right_dorsal_data = image4
            elif direction3 == "Right" and direction4 == "Left":
                left_dorsal_data = image4
                right_dorsal_data = image3

        else:
            ventral_data = image1
            dorsal_data = image3
            if (
                direction1 in ["Top", "Bottom"]
                and direction3 in ["Top", "Bottom"]
                and direction1 != direction3
            ):
                left_right = False
            elif (
                direction1 in ["Left", "Right"]
                and direction3 in ["Left", "Right"]
                and direction1 != direction3
            ):
                left_right = True
            else:
                raise ValueError(
                    f"Invalid directions for detection: {direction1}, {direction3}"
                )

        require_registration = params["require_registration"]
        xy_spacing, z_spacing = None, None
        if require_registration:
            z_spacing = params["axial_resolution"]
            xy_spacing = params["lateral_resolution"]
        tmp_path = params["tmp_path"]
        # Create a directory under the intermediate_path
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir_path = os.path.join(tmp_path, current_time)
        os.makedirs(new_dir_path, exist_ok=True)

        xy_downsample_ratio = params["xy_downsample_ratio"]
        z_downsample_ratio = params["z_downsample_ratio"]

        output_image = self.train(
            require_registration=require_registration,
            require_flipping_along_illu_for_dorsaldet=params["require_flip_illu"],
            require_flipping_along_det_for_dorsaldet=params["require_flip_det"],
            top_illu_ventral_det_data=top_ventral_data,
            bottom_illu_ventral_det_data=bottom_ventral_data,
            top_illu_dorsal_det_data=top_dorsal_data,
            bottom_illu_dorsal_det_data=bottom_dorsal_data,
            left_illu_ventral_det_data=left_ventral_data,
            right_illu_ventral_det_data=right_ventral_data,
            left_illu_dorsal_det_data=left_dorsal_data,
            right_illu_dorsal_det_data=right_dorsal_data,
            ventral_det_data=ventral_data,
            dorsal_det_data=dorsal_data,
            save_path=new_dir_path,
            z_spacing=z_spacing,
            xy_spacing=xy_spacing,
            left_right=left_right,
            display=False,
            sparse_sample=params["sparse_sample"],
            save_separate_results=params["save_separate_results"],
            xy_downsample_ratio=xy_downsample_ratio,
            z_downsample_ratio=z_downsample_ratio,
            # TODO: more parameters?
        )
        if not params["keep_intermediates"]:
            # Clean up the intermediate directory
            shutil.rmtree(new_dir_path)
        return output_image

    def train(
        self,
        require_registration: bool,
        require_flipping_along_illu_for_dorsaldet: bool,
        require_flipping_along_det_for_dorsaldet: bool,
        data_path: str = "",
        sparse_sample=False,
        top_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        bottom_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        top_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        bottom_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        left_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        right_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        left_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        right_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        ventral_det_data: Union[dask.array.core.Array, str] = None,
        dorsal_det_data: Union[dask.array.core.Array, str] = None,
        save_path: str = "",
        save_folder: str = "",
        save_separate_results: bool = False,
        z_spacing: float = None,  # axial
        xy_spacing: float = None,  # lateral
        left_right: bool = None,
        xy_downsample_ratio: int = None,
        z_downsample_ratio: int = None,
        display: bool = True,
    ):
        if not os.path.exists(save_path):
            print("saving path does not exist.")
            return

        if not os.path.exists(os.path.join(save_path, save_folder)):
            os.makedirs(os.path.join(save_path, save_folder))

        if xy_downsample_ratio == 1:
            xy_downsample_ratio = None
        if z_downsample_ratio == 1:
            z_downsample_ratio = None

        allowed_keys = parse_yaml_det.__code__.co_varnames  # or手动列出
        args_dict = {k: v for k, v in locals().items() if k in allowed_keys}
        args_dict.update({"train_params": self.train_params})
        args_dict.update({"registration_params": self.registration_params})
        args_dict.update(
            {"file_name": os.path.join(save_path, save_folder, "det_info.yaml")}
        )
        yaml_path = parse_yaml_det(**args_dict)

        if (xy_downsample_ratio is None) and (z_downsample_ratio is None):
            result = self.train_down_sample(
                require_registration,
                require_flipping_along_illu_for_dorsaldet,
                require_flipping_along_det_for_dorsaldet,
                data_path,
                sparse_sample,
                top_illu_ventral_det_data,
                bottom_illu_ventral_det_data,
                top_illu_dorsal_det_data,
                bottom_illu_dorsal_det_data,
                left_illu_ventral_det_data,
                right_illu_ventral_det_data,
                left_illu_dorsal_det_data,
                right_illu_dorsal_det_data,
                ventral_det_data,
                dorsal_det_data,
                save_path,
                save_folder,
                save_separate_results,
                z_spacing,
                xy_spacing,
                left_right,
                display,
                yaml_path,
            )
        else:
            try:
                if xy_downsample_ratio is None:
                    xy_downsample_ratio = 1
                if z_downsample_ratio is None:
                    z_downsample_ratio = 1
                print("Save inputs as .dat files temporarily...")

                ventral_det_data = self.save_memmap_from_images(
                    ventral_det_data,
                    data_path=data_path,
                    save_path=os.path.join(save_path, save_folder, "ventral.dat"),
                )

                dorsal_det_data = self.save_memmap_from_images(
                    dorsal_det_data,
                    data_path=data_path,
                    save_path=os.path.join(save_path, save_folder, "dorsal.dat"),
                )

                print("Down-sample the inputs...")
                ventral_det_data_lr = self.downsample_h5_files(
                    ventral_det_data,
                    xy_downsample_ratio,
                    z_downsample_ratio,
                    self.train_params["device"],
                )
                dorsal_det_data_lr = self.downsample_h5_files(
                    dorsal_det_data,
                    xy_downsample_ratio,
                    z_downsample_ratio,
                    self.train_params["device"],
                )

                result = self.train_down_sample(
                    require_registration,
                    require_flipping_along_illu_for_dorsaldet,
                    require_flipping_along_det_for_dorsaldet,
                    data_path,
                    sparse_sample,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    ventral_det_data_lr,
                    dorsal_det_data_lr,
                    save_path,
                    save_folder,
                    save_separate_results,
                    z_spacing * z_downsample_ratio,
                    xy_spacing * xy_downsample_ratio,
                    left_right,
                    display,
                    yaml_path,
                )

                del ventral_det_data_lr, dorsal_det_data_lr

                flip_axes = []
                if require_flipping_along_det_for_dorsaldet:
                    flip_axes.append(0)
                if require_flipping_along_illu_for_dorsaldet:
                    if left_right:
                        flip_axes.append(2)
                    else:
                        flip_axes.append(1)

                files_to_be_removed = self.apply(
                    require_registration,
                    flip_axes,
                    save_path,
                    save_folder,
                    ventral_det_data,
                    dorsal_det_data,
                    True if left_right is False else False,
                    yaml_path,
                    z_downsample_ratio,
                    xy_downsample_ratio,
                    save_separate_results,
                    z_spacing,
                    xy_spacing,
                    self.registration_params["skip_refine_registration"],
                    self.train_params["window_size"],
                )

                del ventral_det_data
                del dorsal_det_data
                return

            except Exception as e:
                traceback.print_exc()
                print(e)
            finally:
                torch.cuda.empty_cache()
                try:
                    os.remove(os.path.join(save_path, save_folder, "ventral.dat"))
                    os.remove(os.path.join(save_path, save_folder, "dorsal.dat"))
                    for f in files_to_be_removed:
                        os.remove(os.path.splitext(f)[0] + ".dat")
                except Exception as e:
                    traceback.print_exc()
                    print(e)

        return result

    def save_memmap_from_images(
        self,
        all_images,
        data_path,
        save_path,
    ):
        if isinstance(all_images, list):
            sample_slice = tifffile.imread(all_images[0])
            Z, Y, X = len(all_images), sample_slice.shape[0], sample_slice.shape[1]
            dtype = sample_slice.dtype

            if os.path.exists(save_path):
                os.remove(save_path)

            mm = np.memmap(save_path, dtype=dtype, mode="w+", shape=(Z, Y, X))

            for i in tqdm.tqdm(range(Z), desc="saving to memmap: ", leave=False):
                mm[i] = tifffile.imread(all_images[i])
            mm.flush()
        else:
            if isinstance(all_images, str):
                all_images = os.path.join(data_path, all_images)
            data = read_with_bioio(all_images)
            Z, Y, X = data.shape
            dtype = data.dtype

            if os.path.exists(save_path):
                os.remove(save_path)

            mm = np.memmap(save_path, dtype=dtype, mode="w+", shape=(Z, Y, X))

            mm[:] = data
            mm.flush()

        return mm

    def train_down_sample(
        self,
        require_registration: bool,
        require_flipping_along_illu_for_dorsaldet: bool,
        require_flipping_along_det_for_dorsaldet: bool,
        data_path: str = "",
        sparse_sample=False,
        top_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        bottom_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        top_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        bottom_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        left_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        right_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        left_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        right_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        ventral_det_data: Union[dask.array.core.Array, str] = None,
        dorsal_det_data: Union[dask.array.core.Array, str] = None,
        save_path: str = "",
        save_folder: str = "",
        save_separate_results: bool = False,
        z_spacing: float = None,  # axial
        xy_spacing: float = None,  # lateral
        left_right: bool = None,
        display: bool = True,
        yaml_path: str = "",
    ):
        args = locals()

        for k in args:
            if k.endswith("_data") and isinstance(args[k], str):
                args[k] = os.path.join(data_path, args[k])

        top_illu_ventral_det_data = args["top_illu_ventral_det_data"]
        bottom_illu_ventral_det_data = args["bottom_illu_ventral_det_data"]
        top_illu_dorsal_det_data = args["top_illu_dorsal_det_data"]
        bottom_illu_dorsal_det_data = args["bottom_illu_dorsal_det_data"]
        left_illu_ventral_det_data = args["left_illu_ventral_det_data"]
        right_illu_ventral_det_data = args["right_illu_ventral_det_data"]
        left_illu_dorsal_det_data = args["left_illu_dorsal_det_data"]
        right_illu_dorsal_det_data = args["right_illu_dorsal_det_data"]
        ventral_det_data = args["ventral_det_data"]
        dorsal_det_data = args["dorsal_det_data"]

        flip_det = require_flipping_along_det_for_dorsaldet
        flip_ill = require_flipping_along_illu_for_dorsaldet

        if require_registration:
            if (z_spacing is None) or (xy_spacing is None):
                print("spacing information is missing.")
                return
        if self.train_params["require_segmentation"]:
            _suffix = ""
        else:
            _suffix = "_without_segmentation"

        illu_name = f"illuFusionResult{_suffix}"
        fb_z = f"fusionBoundary_z{_suffix}"
        fb_xy = f"fusionBoundary_xy{_suffix}"
        det_name = f"quadrupleFusionResult{_suffix}"

        self.sample_params = {
            "require_registration": require_registration,
            "z_spacing": z_spacing,
            "xy_spacing": xy_spacing,
            "flip_det": flip_det,
            "flip_ill": flip_ill,
        }

        _name = {}
        if (
            (top_illu_ventral_det_data is not None)
            and (bottom_illu_ventral_det_data is not None)
            and (top_illu_dorsal_det_data is not None)
            and (bottom_illu_dorsal_det_data is not None)
        ):
            T_flag = 0
            det_only_flag = 0
            if isinstance(top_illu_ventral_det_data, str):
                _name["top_ventral"] = strip_ext(top_illu_ventral_det_data)
            else:
                _name["top_ventral"] = "top_illu+ventral_det"
            if isinstance(bottom_illu_ventral_det_data, str):
                _name["bottom_ventral"] = strip_ext(bottom_illu_ventral_det_data)
            else:
                _name["bottom_ventral"] = "bottom_illu+ventral_det"
            if isinstance(top_illu_dorsal_det_data, str):
                _name["top_dorsal"] = strip_ext(top_illu_dorsal_det_data)
            else:
                _name["top_dorsal"] = "top_illu+dorsal_det"
            if isinstance(bottom_illu_dorsal_det_data, str):
                _name["bottom_dorsal"] = strip_ext(bottom_illu_dorsal_det_data)
            else:
                _name["bottom_dorsal"] = "bottom_illu+dorsal_det"
        elif (
            (left_illu_ventral_det_data is not None)
            and (right_illu_ventral_det_data is not None)
            and (left_illu_dorsal_det_data is not None)
            and (right_illu_dorsal_det_data is not None)
        ):
            T_flag = 1
            det_only_flag = 0
            if isinstance(left_illu_ventral_det_data, str):
                _name["top_ventral"] = strip_ext(left_illu_ventral_det_data)
            else:
                _name["top_ventral"] = "left_illu+ventral_det"
            if isinstance(right_illu_ventral_det_data, str):
                _name["bottom_ventral"] = strip_ext(right_illu_ventral_det_data)
            else:
                _name["bottom_ventral"] = "right_illu+ventral_det"
            if isinstance(left_illu_dorsal_det_data, str):
                _name["top_dorsal"] = strip_ext(left_illu_dorsal_det_data)
            else:
                _name["top_dorsal"] = "left_illu+dorsal_det"
            if isinstance(right_illu_dorsal_det_data, str):
                _name["bottom_dorsal"] = strip_ext(right_illu_dorsal_det_data)
            else:
                _name["bottom_dorsal"] = "right_illu+dorsal_det"
        elif (ventral_det_data is not None) and ((dorsal_det_data is not None)):
            if left_right is None:
                print("left-right marker is missing.")
                return
            if left_right is True:
                T_flag = 1
            else:
                T_flag = 0
            det_only_flag = 1
            if isinstance(ventral_det_data, str):
                _name["top_ventral"] = strip_ext(ventral_det_data)
            else:
                _name["top_ventral"] = "ventral_det"
            if isinstance(dorsal_det_data, str):
                _name["top_dorsal"] = strip_ext(dorsal_det_data)
            else:
                _name["top_dorsal"] = "dorsal_det"
        else:
            print("input(s) missing, please check.")
            return
        self.sample_params.update(_name)

        save_path = os.path.join(save_path, save_folder)

        for k in _name.keys():
            sub_folder = os.path.join(save_path, _name[k])
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)

        leaf_paths = extract_leaf_file_paths_from_file(yaml_path)
        leaf_paths.update(
            {
                "illu_name": illu_name,
                "fb_xy": fb_xy,
                "fb_z": fb_z,
                "det_name": det_name,
            }
        )

        flip_axes = []
        if flip_det:
            flip_axes.append(0)
        if flip_ill:
            flip_axes.append(2 if T_flag else 1)
        flip_axes = tuple(flip_axes)

        if not det_only_flag:
            if self.train_params["skip_illuFusion"]:
                if os.path.exists(leaf_paths[illu_name + ".tif"][0]):
                    print("Skip dual-illu fusion for ventral det...")
                    illu_flag_ventral = 0
                else:
                    print("Cannot skip dual-illu fusion for ventral det...")
                    illu_flag_ventral = 1
                if os.path.exists(leaf_paths[illu_name + ".tif"][1]):
                    print("Skip dual-illu fusion for dorsal det...")
                    illu_flag_dorsal = 0
                else:
                    print("Cannot skip dual-illu fusion for dorsal det...")
                    illu_flag_dorsal = 1
            else:
                illu_flag_dorsal = 1
                illu_flag_ventral = 1
        else:
            illu_flag_dorsal = 0
            illu_flag_ventral = 0

        if illu_flag_ventral:
            print("\nFusion along illumination for ventral camera...")
            self.modelFront.train(
                data_path=data_path,
                top_illu_data=strip_ext(top_illu_ventral_det_data, 1),
                bottom_illu_data=strip_ext(bottom_illu_ventral_det_data, 1),
                left_illu_data=strip_ext(left_illu_ventral_det_data, 1),
                right_illu_data=strip_ext(right_illu_ventral_det_data, 1),
                save_path=save_path,
                save_folder="",
                save_separate_results=False,
                sparse_sample=sparse_sample,
                camera_position="ventral_det",
                display=display,
            )
        if illu_flag_dorsal:
            print("\nFusion along illumination for dorsal camera...")
            self.modelBack.train(
                data_path=data_path,
                top_illu_data=strip_ext(top_illu_dorsal_det_data, 1),
                bottom_illu_data=strip_ext(bottom_illu_dorsal_det_data, 1),
                left_illu_data=strip_ext(left_illu_dorsal_det_data, 1),
                right_illu_data=strip_ext(right_illu_dorsal_det_data, 1),
                save_path=save_path,
                save_folder="",
                save_separate_results=False,
                sparse_sample=sparse_sample,
                cam_pos=("back" if flip_det is False else "front"),
                camera_position="dorsal_det",
                display=display,
            )

        if require_registration:
            self.coarse_to_fine_registration(
                top_illu_dorsal_det_data,
                bottom_illu_dorsal_det_data,
                left_illu_dorsal_det_data,
                right_illu_dorsal_det_data,
                ventral_det_data,
                dorsal_det_data,
                save_path,
                det_only_flag,
                flip_axes,
                T_flag,
                leaf_paths,
            )

        flip_axes = tuple(1 if x == 2 else x for x in flip_axes)

        segMask, xs, xe, ys, ye = self.generate_seg_mask(
            ventral_det_data,
            dorsal_det_data,
            det_only_flag,
            save_path,
            T_flag,
            require_registration,
            flip_axes,
            display,
            leaf_paths,
        )

        boundaryETop = self.process_top_or_left_side(
            top_illu_ventral_det_data,
            left_illu_ventral_det_data,
            top_illu_dorsal_det_data,
            bottom_illu_dorsal_det_data,
            left_illu_dorsal_det_data,
            right_illu_dorsal_det_data,
            ventral_det_data,
            dorsal_det_data,
            segMask,
            det_only_flag,
            T_flag,
            xs,
            xe,
            ys,
            ye,
            require_registration,
            flip_ill,
            save_path,
            flip_axes,
        )
        tifffile.imwrite(leaf_paths[fb_z + ".tif"][0], boundaryETop)

        if not det_only_flag:
            boundaryEBottom = self.process_bottom_or_right_side(
                bottom_illu_ventral_det_data,
                right_illu_ventral_det_data,
                top_illu_dorsal_det_data,
                bottom_illu_dorsal_det_data,
                left_illu_dorsal_det_data,
                right_illu_dorsal_det_data,
                ventral_det_data,
                dorsal_det_data,
                segMask,
                det_only_flag,
                T_flag,
                xs,
                xe,
                ys,
                ye,
                require_registration,
                flip_ill,
                save_path,
                flip_axes,
            )
            tifffile.imwrite(leaf_paths[fb_z + ".tif"][1], boundaryEBottom)

        print("\n\nStitching...")
        print("read in...")

        if not det_only_flag:
            illu_front = read_with_bioio(leaf_paths[f"{illu_name}.tif"][0], T_flag)
        else:
            illu_front = read_with_bioio(ventral_det_data, T_flag)

        if require_registration:
            if self.registration_params["require_reg_finetune"]:
                reg_level = "_reg"
            else:
                reg_level = "_coarse_reg"
            if not det_only_flag:
                illu_back = read_with_bioio(
                    leaf_paths[f"{illu_name}{reg_level}.tif"],
                    T_flag,
                )
            else:
                illu_back = read_with_bioio(
                    leaf_paths[f"{self.sample_params['top_dorsal']}{reg_level}.tif"],
                    T_flag,
                )
        else:
            if not det_only_flag:
                illu_back = read_with_bioio(leaf_paths[f"{illu_name}.tif"][1], T_flag)
            else:
                illu_back = read_with_bioio(dorsal_det_data, T_flag)

        if not require_registration:
            illu_back[:] = np.flip(illu_back, flip_axes)

        s, m0, n0 = illu_back.shape

        if not det_only_flag:
            boundaryEFront = tifffile.imread(leaf_paths[f"{fb_xy}.tif"][0])
            if require_registration:
                boundaryEBack = np.load(leaf_paths[f"{fb_xy}_reg.npy"])
            else:
                boundaryEBack = tifffile.imread(leaf_paths[f"{fb_xy}.tif"][1])
                if flip_ill:
                    boundaryEBack[:] = m0 - boundaryEBack
                if flip_det:
                    boundaryEBack[:] = np.flip(boundaryEBack, 0)

        boundaryTop = tifffile.imread(leaf_paths[f"{fb_z}.tif"][0])

        if not det_only_flag:
            boundaryBottom = tifffile.imread(leaf_paths[f"{fb_z}.tif"][1])

        if T_flag:
            boundaryTop = boundaryTop.T
            if not det_only_flag:
                boundaryBottom = boundaryBottom.T
                if require_registration:
                    boundaryEBack = boundaryEBack.swapaxes(1, 2)

        if not require_registration:
            pass
        else:
            if flip_ill:
                if not det_only_flag:
                    boundaryEBack[:] = ~boundaryEBack

        if require_registration:
            translating_information = np.load(
                leaf_paths["translating_information.npy"],
                allow_pickle=True,
            ).item()
            invalid_region = (
                volumeTranslate_compose(
                    np.ones((s, m0, n0) if not T_flag else (s, n0, m0), bool),
                    translating_information["reg_matrix_inv"],
                    translating_information["T2"],
                    translating_information["padding_z"],
                    None,
                    translating_information["flip_axes"],
                    device=self.train_params["device"],
                    xy_spacing=self.sample_params["xy_spacing"],
                    z_spacing=self.sample_params["z_spacing"],
                )
                < 1
            )
            if T_flag:
                invalid_region = invalid_region.transpose(0, 2, 1)
            invalid_region[:, xs:xe, ys:ye] = 0
        else:
            invalid_region = np.zeros((s, m0, n0), dtype=bool)

        if save_separate_results:
            if os.path.exists(leaf_paths["fuse_det_mask"]):
                shutil.rmtree(leaf_paths["fuse_det_mask"])
            os.makedirs(leaf_paths["fuse_det_mask"])
            p = leaf_paths["fuse_det_mask"]
        else:
            p = None

        if not det_only_flag:
            reconVol = fusionResultFour(
                T_flag,
                boundaryTop,
                boundaryBottom,
                boundaryEFront,
                boundaryEBack,
                illu_front,
                illu_back,
                self.train_params["device"],
                self.sample_params,
                invalid_region,
                save_separate_results,
                path=p,
                GFr=copy.deepcopy(self.train_params["window_size"]),
            )
        else:
            reconVol = fusionResult_VD(
                T_flag,
                illu_front,
                illu_back,
                boundaryTop,
                self.train_params["device"],
                save_separate_results,
                path=p,
                GFr=copy.deepcopy(self.train_params["window_size"]),
            )

        if T_flag:
            result = reconVol.swapaxes(1, 2)
        else:
            result = reconVol
        del reconVol
        if display:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=200)
            ax1.imshow(result.max(0))
            ax1.set_title("result in xy", fontsize=8, pad=1)
            ax1.axis("off")
            ax2.imshow(result.max(1))
            ax2.set_title("result in zy", fontsize=8, pad=1)
            ax2.axis("off")
            ax3.imshow(result.max(2))
            ax3.set_title("result in zx", fontsize=8, pad=1)
            ax3.axis("off")
            plt.show()
        print("Save...")

        tifffile.imwrite(leaf_paths[f"{det_name}.tif"], result)
        del illu_front, illu_back
        gc.collect()
        return result

    def apply(
        self,
        require_registration: bool,
        flip_axes,
        save_path: str,
        save_folder: str,
        ventral_det_data: np.ndarray,
        dorsal_det_data: np.ndarray,
        T_flag: bool,
        yaml_path,
        z_upsample_ratio: int = 1,
        xy_upsample_ratio: int = 1,
        save_separate_results: bool = False,
        z_spacing: int = None,
        xy_spacing: int = None,
        skip_refine_registration: bool = False,
        window_size=[5, 59],
    ):
        if self.train_params["require_segmentation"]:
            _suffix = ""
        else:
            _suffix = "_without_segmentation"
        fb_z = f"fusionBoundary_z{_suffix}.tif"

        save_path = os.path.join(save_path, save_folder)
        leaf_paths = extract_leaf_file_paths_from_file(yaml_path)

        if not os.path.exists(os.path.join(save_path, "high_res")):
            os.makedirs(os.path.join(save_path, "high_res"))

        if require_registration:
            print("Apply registration...")

            translating_information = np.load(
                leaf_paths["translating_information.npy"],
                allow_pickle=True,
            ).item()

            dorsal_det_data_reg = volumeTranslate_compose(
                dorsal_det_data,
                translating_information["reg_matrix_inv"],
                translating_information["T2"],
                translating_information["padding_z"] * z_upsample_ratio,
                leaf_paths["dorsal_det_reg_hr.tif"],
                translating_information["flip_axes"],
                device=self.train_params["device"],
                xy_spacing=xy_spacing,
                z_spacing=z_spacing,
                large_vol=True,
            )
        else:
            print("Skip registration...")

        if require_registration and (not skip_refine_registration):
            trans_path = leaf_paths["dorsal_det_fine_reg_hr.tif"]
            if (not skip_refine_registration) or (not os.path.exists(trans_path)):
                target_points, source_points = DoG(
                    ventral_det_data,
                    dorsal_det_data_reg,
                    z_spacing=z_spacing,
                    xy_spacing=xy_spacing,
                    device=self.train_params["device"],
                    max_p=1e7,
                )

                source_pcd = o3d.geometry.PointCloud()
                source_pcd.points = o3d.utility.Vector3dVector(source_points)
                target_pcd = o3d.geometry.PointCloud()
                target_pcd.points = o3d.utility.Vector3dVector(target_points)
                print("Refine registration...")

                target_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=5.0, max_nn=30
                    )
                )
                source_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=5.0, max_nn=30
                    )
                )

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    target_pcd,
                    source_pcd,
                    (z_spacing + 2 * xy_spacing) / 3 * 10,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=5000
                    ),
                )

                np.save(
                    leaf_paths["regInfo_refine_hr.npy"],
                    {
                        "source_points": source_points,
                        "target_points": target_points,
                        "transformation": reg_p2p.transformation,
                    },
                )

                dorsal_det_data_fine_reg = volumeTranslate_compose(
                    dorsal_det_data_reg,
                    None,
                    reg_p2p.transformation,
                    dorsal_det_data_reg.shape[0],
                    trans_path,
                    tuple([]),
                    device=self.train_params["device"],
                    xy_spacing=xy_spacing,
                    z_spacing=z_spacing,
                    large_vol=True,
                )
            else:
                print("Skip refine registration...")

        boundary = tifffile.imread(leaf_paths[fb_z][0])

        if require_registration:
            dorsal_det_data = dorsal_det_data_reg
            if not skip_refine_registration:
                dorsal_det_data = dorsal_det_data_fine_reg

        z, x, y = dorsal_det_data.shape
        boundary = (
            z_upsample_ratio
            * F.interpolate(
                torch.from_numpy(boundary[None, None].astype(np.float32)),
                size=(x, y),
                mode="bilinear",
                align_corners=True,
            ).data.numpy()[0, 0]
        )

        if save_separate_results:
            if os.path.exists(leaf_paths["fuse_det_mask_hr"]):
                shutil.rmtree(leaf_paths["fuse_det_mask_hr"])
            os.makedirs(leaf_paths["fuse_det_mask_hr"])
            p = leaf_paths["fuse_det_mask_hr"]
        else:
            p = None

        reconVol = fusionResult_VD(
            T_flag,
            ventral_det_data,
            dorsal_det_data,
            boundary,
            self.train_params["device"],
            save_separate_results,
            path=p,
            flip_axes=flip_axes if not require_registration else tuple([]),
            GFr=window_size,
        )

        print("Save...")
        OmeTiffWriter.save(
            reconVol,
            leaf_paths["quadrupleFusionResult_hr.tif"],
            dim_order="ZYX",
        )
        file_to_be_removed = []
        if require_registration:
            file_to_be_removed.append(leaf_paths["dorsal_det_reg_hr.tif"])
            del dorsal_det_data_reg
        if not skip_refine_registration:
            file_to_be_removed.append(leaf_paths["dorsal_det_fine_reg_hr.tif"])
            del dorsal_det_data_fine_reg

        del reconVol
        return file_to_be_removed

    def coarse_to_fine_registration(
        self,
        top_illu_dorsal_det_data,
        bottom_illu_dorsal_det_data,
        left_illu_dorsal_det_data,
        right_illu_dorsal_det_data,
        ventral_det_data,
        dorsal_det_data,
        save_path,
        det_only_flag,
        flip_axes,
        T_flag,
        leaf_paths,
    ):
        illu_name = leaf_paths["illu_name"]
        fb_xy = leaf_paths["fb_xy"]

        if not det_only_flag:
            vol_to_be_reg = leaf_paths["illu_name"]
        else:
            vol_to_be_reg = self.sample_params["top_dorsal"]

        if self.registration_params["use_exist_reg"] is False:
            run_coarse = 1
            if self.registration_params["require_reg_finetune"]:
                run_fine = 1
            else:
                run_fine = 0
        else:
            if os.path.exists(leaf_paths["regInfo.npy"]) and os.path.exists(
                leaf_paths[f"{vol_to_be_reg}_coarse_reg.tif"]
            ):
                run_coarse = 0
                if self.registration_params["require_reg_finetune"]:
                    if self.registration_params["require_reg_finetune"]:
                        if os.path.exists(
                            leaf_paths["regInfo_refine.npy"]
                        ) and os.path.exists(leaf_paths[f"{vol_to_be_reg}_reg.tif"]):
                            run_fine = 0
                        else:
                            run_fine = 1
                    else:
                        run_fine = 0
                else:
                    run_fine = 0

            else:
                print("\nCannot skip registration...")
                run_coarse = 1
                if self.registration_params["require_reg_finetune"]:
                    run_fine = 1
                else:
                    run_fine = 0

        if run_coarse:
            print("\nRegister...")
            print("read in...")

            if not det_only_flag:
                static_view_uint16 = read_with_bioio(leaf_paths[f"{illu_name}.tif"][0])
                moving_view_uint16 = read_with_bioio(leaf_paths[f"{illu_name}.tif"][1])
            else:
                static_view_uint16 = read_with_bioio(ventral_det_data)
                moving_view_uint16 = read_with_bioio(dorsal_det_data)

            moving_view_uint16[:] = np.flip(moving_view_uint16, flip_axes)

            s_r, m, n = static_view_uint16.shape
            s_m, _, _ = moving_view_uint16.shape
            if s_r == s_m:
                moving_view_uint16_pad = copy.deepcopy(moving_view_uint16)
                static_view_uint16_pad = copy.deepcopy(static_view_uint16)
            elif s_r > s_m:
                moving_view_uint16_pad = np.concatenate(
                    (
                        moving_view_uint16,
                        np.zeros((s_r - s_m, m, n), dtype=moving_view_uint16.dtype),
                    ),
                    0,
                )
                static_view_uint16_pad = copy.deepcopy(static_view_uint16)
            else:
                static_view_uint16_pad = np.concatenate(
                    (
                        static_view_uint16,
                        np.zeros((s_m - s_r, m, n), dtype=static_view_uint16.dtype),
                    ),
                    0,
                )
                moving_view_uint16_pad = copy.deepcopy(moving_view_uint16)
            del static_view_uint16

            print("reg in xy...")
            zMP_static = static_view_uint16_pad.max(0)
            zMP_moving = moving_view_uint16_pad.max(0)
            AffineMapZXY, frontMIP, backMIP = coarseRegistrationXY(
                zMP_static.swapaxes(0, 1) if T_flag else zMP_static,
                zMP_moving.swapaxes(0, 1) if T_flag else zMP_moving,
                self.sample_params["z_spacing"],
                self.sample_params["xy_spacing"],
            )
            print("reg in zx...")
            yMP_static = static_view_uint16_pad.max(1 if T_flag else 2)
            yMP_moving = moving_view_uint16_pad.max(1 if T_flag else 2)
            AffineMapZXY = coarseRegistrationZX(
                yMP_static,
                yMP_moving,
                self.sample_params["z_spacing"],
                self.sample_params["xy_spacing"],
                AffineMapZXY,
            )

            if T_flag:
                AffineMapZXY[[1, 2]] = AffineMapZXY[[2, 1]]

            print("rigid registration in 3D...")
            frontMIP = frontMIP.swapaxes(0, 1) if T_flag else frontMIP
            backMIP = backMIP.swapaxes(0, 1) if T_flag else backMIP
            th = filters.threshold_otsu(frontMIP)
            a0, b0, c0, d0 = self.segMIP(frontMIP, th=th)
            a1, b1, c1, d1 = self.segMIP(backMIP, th=th)
            xs, xe, ys, ye = min(c0, c1), max(d0, d1), min(a0, a1), max(b0, b1)

            reg_info = fineReg(
                static_view_uint16_pad,
                moving_view_uint16_pad,
                xs,
                xe,
                ys,
                ye,
                AffineMapZXY,
                z_spacing=self.sample_params["z_spacing"],
                xy_spacing=self.sample_params["xy_spacing"],
                registration_params=self.registration_params,
            )
            del static_view_uint16_pad, moving_view_uint16_pad

            reg_info.update(
                {
                    "zfront": s_r,
                    "zback": s_m,
                    "m": m,
                    "n": n,
                    "z": max(s_r, s_m),
                }
            )
            np.save(leaf_paths["regInfo.npy"], reg_info)

            padding_z = (
                boundaryInclude(
                    reg_info,
                    reg_info["z"] * self.sample_params["z_spacing"],
                    m * self.sample_params["xy_spacing"],
                    n * self.sample_params["xy_spacing"],
                    spacing=self.sample_params["z_spacing"],
                )
                / self.sample_params["z_spacing"]
            )

            volumeTranslate_compose(
                moving_view_uint16,
                reg_info["reg_matrix_inv"],
                None,
                padding_z,
                os.path.join(leaf_paths[f"{vol_to_be_reg}_coarse_reg.tif"]),
                tuple([]),
                self.train_params["device"],
                self.sample_params["xy_spacing"],
                self.sample_params["z_spacing"],
            )
            del moving_view_uint16

            np.save(
                leaf_paths["translating_information.npy"],
                {
                    **{
                        "T2": None,
                        "padding_z": padding_z,
                        "flip_axes": flip_axes,
                    },
                    **reg_info,
                },
            )
        else:
            print("\nSkip coarse registration...")

        if run_fine:
            print("refine registration...")

            if not det_only_flag:
                static_view_uint16 = read_with_bioio(leaf_paths[f"{illu_name}.tif"][0])
            else:
                static_view_uint16 = read_with_bioio(ventral_det_data)
            moving_view_uint16 = read_with_bioio(
                leaf_paths[f"{vol_to_be_reg}_coarse_reg.tif"]
            )

            target_points, source_points = DoG(
                static_view_uint16,
                moving_view_uint16,
                z_spacing=self.sample_params["z_spacing"],
                xy_spacing=self.sample_params["xy_spacing"],
                device=self.train_params["device"],
                max_p=1e6,
            )

            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_points)

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_points)
            # target = respctive

            target_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
            )
            source_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
            )

            th = (
                self.sample_params["z_spacing"] + 2 * self.sample_params["xy_spacing"]
            ) * 3.3

            reg_p2p = o3d.pipelines.registration.registration_icp(
                target_pcd,
                source_pcd,
                th,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000),
            )

            np.save(
                leaf_paths["regInfo_refine.npy"],
                {
                    "source_points": source_points,
                    "target_points": target_points,
                    "transformation": reg_p2p.transformation,
                },
            )

            volumeTranslate_compose(
                moving_view_uint16,
                None,
                reg_p2p.transformation,
                moving_view_uint16.shape[0],
                leaf_paths[f"{vol_to_be_reg}_reg.tif"],
                tuple([]),
                self.train_params["device"],
                self.sample_params["xy_spacing"],
                self.sample_params["z_spacing"],
            )

            regInfo = np.load(
                leaf_paths["translating_information.npy"],
                allow_pickle=True,
            ).item()
            regInfo["T2"] = reg_p2p.transformation

            np.save(leaf_paths["translating_information.npy"], regInfo)
        else:
            print("\nSkip fine registration...\n")

        if not det_only_flag:
            regInfo = np.load(
                leaf_paths["translating_information.npy"],
                allow_pickle=True,
            ).item()
            padding_z = regInfo["padding_z"]
            T2 = regInfo["T2"]
            for f, f_name in zip(
                ["top", "bottom"] if (not T_flag) else ["left", "right"],
                ["top", "bottom"],
            ):
                inputs = read_with_bioio(locals()[f + "_illu_dorsal_det_data"])
                trans_path = os.path.join(
                    save_path,
                    self.sample_params[f_name + "_dorsal"],
                    self.sample_params[f_name + "_dorsal"] + "_reg.tif",
                )

                volumeTranslate_compose(
                    inputs,
                    regInfo["reg_matrix_inv"],
                    T2,
                    regInfo["padding_z"],
                    trans_path,
                    flip_axes,
                    device=self.train_params["device"],
                    xy_spacing=self.sample_params["xy_spacing"],
                    z_spacing=self.sample_params["z_spacing"],
                )
            m = regInfo["m"]
            n = regInfo["n"]
            boundary = tifffile.imread(leaf_paths[f"{fb_xy}.tif"][1]).astype(np.float32)
            mask = np.arange(n if T_flag else m)[None, :, None] > boundary[:, None, :]
            if T_flag:
                mask = mask.transpose(0, 2, 1)
            volumeTranslate_compose(
                mask,
                regInfo["reg_matrix_inv"],
                T2,
                padding_z,
                leaf_paths[f"{fb_xy}_reg.npy"],
                flip_axes,
                device=self.train_params["device"],
                xy_spacing=self.sample_params["xy_spacing"],
                z_spacing=self.sample_params["z_spacing"],
            )

    def generate_seg_mask(
        self,
        ventral_det_data,
        dorsal_det_data,
        det_only_flag,
        save_path,
        T_flag,
        require_registration,
        flip_axes,
        display,
        leaf_paths,
    ):
        print("\nLocalize sample...")
        print("read in...")
        illu_name = leaf_paths["illu_name"]
        if not det_only_flag:
            illu_front = read_with_bioio(leaf_paths[f"{illu_name}.tif"][0], T_flag)
        else:
            illu_front = read_with_bioio(ventral_det_data, T_flag)

        if (not det_only_flag) or require_registration:
            if self.registration_params["require_reg_finetune"]:
                reg_level = "_reg"
            else:
                reg_level = "_coarse_reg"
            f_handle = BioImage(
                os.path.join(
                    save_path,
                    self.sample_params["top_dorsal"],
                    "{}".format(
                        illu_name
                        if not det_only_flag
                        else self.sample_params["top_dorsal"]
                    )
                    + "{}.tif".format(
                        reg_level if require_registration else "",
                    ),
                )
            )
            illu_back = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        else:
            illu_back = read_with_bioio(dorsal_det_data, T_flag)

        if not require_registration:
            illu_back[:] = np.flip(illu_back, flip_axes)

        cropInfo = self.localizingSample(
            illu_front.max(0),
            illu_back.max(0),
            save_path,
            det_only_flag,
            leaf_paths,
        )
        print(cropInfo)

        if self.train_params["require_precropping"]:
            if len(self.train_params["precropping_params"]) == 0:
                xs, xe, ys, ye = cropInfo.loc[
                    "in summary", ["startX", "endX", "startY", "endY"]
                ].astype(int)
            else:
                if T_flag:
                    ys, ye, xs, xe = self.train_params["precropping_params"]
                else:
                    xs, xe, ys, ye = self.train_params["precropping_params"]
        else:
            xs, xe, ys, ye = None, None, None, None
        if display:
            _, (ax1, ax2) = plt.subplots(1, 2, dpi=200)
            ax1.imshow(illu_front.max(0).T if T_flag else illu_front.max(0))
            if self.train_params["require_precropping"]:
                rect = patches.Rectangle(
                    (ys, xs) if (not T_flag) else (xs, ys),
                    (ye - ys) if (not T_flag) else (xe - xs),
                    (xe - xs) if (not T_flag) else (ye - ys),
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax1.add_patch(rect)
            ax1.set_title("ventral det", fontsize=8, pad=1)
            ax1.axis("off")
            ax2.imshow(illu_back.max(0).T if T_flag else illu_back.max(0))
            if self.train_params["require_precropping"]:
                rect = patches.Rectangle(
                    (ys, xs) if (not T_flag) else (xs, ys),
                    (ye - ys) if (not T_flag) else (xe - xs),
                    (xe - xs) if (not T_flag) else (ye - ys),
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax2.add_patch(rect)
            ax2.set_title("dorsal det", fontsize=8, pad=1)
            ax2.axis("off")
            plt.show()

        if self.train_params["require_segmentation"]:
            segMask = self.segmentSample(
                illu_front[:, xs:xe, ys:ye][
                    :,
                    :: self.train_params["resample_ratio"],
                    :: self.train_params["resample_ratio"],
                ],
                illu_back[:, xs:xe, ys:ye][
                    :,
                    :: self.train_params["resample_ratio"],
                    :: self.train_params["resample_ratio"],
                ],
                save_path,
                det_only_flag,
                leaf_paths,
            )
            del illu_front, illu_back
            np.save(
                leaf_paths["segmentation_det.npy"],
                segMask.transpose(1, 2, 0),
            )
        else:
            segMask = np.ones(
                illu_back[:, xs:xe, ys:ye][
                    :,
                    :: self.train_params["resample_ratio"],
                    :: self.train_params["resample_ratio"],
                ].shape,
                dtype=bool,
            )
        return segMask, xs, xe, ys, ye

    def process_top_or_left_side(
        self,
        top_illu_ventral_det_data,
        left_illu_ventral_det_data,
        top_illu_dorsal_det_data,
        bottom_illu_dorsal_det_data,
        left_illu_dorsal_det_data,
        right_illu_dorsal_det_data,
        ventral_det_data,
        dorsal_det_data,
        segMask,
        det_only_flag,
        T_flag,
        xs,
        xe,
        ys,
        ye,
        require_registration,
        flip_ill,
        save_path,
        flip_axes,
    ):
        if not det_only_flag:
            print("\nFor top/left Illu...")
        else:
            print("\nEstimate boundary along detection...")

        input_1 = "{}_illu_ventral_det_data".format("left" if T_flag else "top")
        if not det_only_flag:
            if flip_ill:
                illu_direct = "right" if T_flag else "bottom"
            else:
                illu_direct = "left" if T_flag else "top"
        else:
            illu_direct = "top"
        input_2 = "{}_illu_dorsal_det_data".format(illu_direct)
        print("read in...")
        if not det_only_flag:
            rawPlanesTopO = read_with_bioio(locals()[input_1], T_flag)
        else:
            rawPlanesTopO = read_with_bioio(ventral_det_data, T_flag)

        rawPlanesToptmp = rawPlanesTopO[:, xs:xe, ys:ye]
        _, m_c, n_c = rawPlanesToptmp.shape
        m = len(np.arange(m_c)[:: self.train_params["resample_ratio"]])
        n = len(np.arange(n_c)[:: self.train_params["resample_ratio"]])
        del rawPlanesTopO

        if require_registration:
            if self.registration_params["require_reg_finetune"]:
                reg_level = "_reg"
            else:
                reg_level = "_coarse_reg"
            if not det_only_flag:
                bottom_name = "bottom" if flip_ill else "top"
            else:
                bottom_name = "top"
            bottom_handle = BioImage(
                os.path.join(
                    save_path,
                    self.sample_params["{}_dorsal".format(bottom_name)],
                    self.sample_params["{}_dorsal".format(bottom_name)]
                    + "{}.tif".format(reg_level),
                )
            )
            rawPlanesBottomO = bottom_handle.get_image_data(
                "ZXY" if T_flag else "ZYX", T=0, C=0
            )
        else:
            if not det_only_flag:
                rawPlanesBottomO = read_with_bioio(locals()[input_2], T_flag)
            else:
                rawPlanesBottomO = read_with_bioio(dorsal_det_data, T_flag)

        if not require_registration:
            rawPlanesBottomO[:] = np.flip(rawPlanesBottomO, flip_axes)
        m0, n0 = rawPlanesBottomO.shape[-2:]
        rawPlanesBottom = rawPlanesBottomO[:, xs:xe, ys:ye]
        del rawPlanesBottomO
        s = rawPlanesBottom.shape[0]

        if rawPlanesToptmp.shape[0] < rawPlanesBottom.shape[0]:
            rawPlanesTop = np.concatenate(
                (
                    rawPlanesToptmp,
                    np.zeros(
                        (
                            rawPlanesBottom.shape[0] - rawPlanesToptmp.shape[0],
                            rawPlanesBottom.shape[1],
                            rawPlanesBottom.shape[2],
                        ),
                        dtype=rawPlanesToptmp.dtype,
                    ),
                ),
                0,
            )
        else:
            rawPlanesTop = rawPlanesToptmp
        del rawPlanesToptmp

        fb = self.predict_fusion_boundary(
            rawPlanesTop,
            rawPlanesBottom,
            segMask,
            s,
            m,
            n,
            m_c,
            n_c,
            m0,
            n0,
            xs,
            xe,
            ys,
            ye,
        )
        return fb.T if T_flag else fb

    def process_bottom_or_right_side(
        self,
        bottom_illu_ventral_det_data,
        right_illu_ventral_det_data,
        top_illu_dorsal_det_data,
        bottom_illu_dorsal_det_data,
        left_illu_dorsal_det_data,
        right_illu_dorsal_det_data,
        ventral_det_data,
        dorsal_det_data,
        segMask,
        det_only_flag,
        T_flag,
        xs,
        xe,
        ys,
        ye,
        require_registration,
        flip_ill,
        save_path,
        flip_axes,
    ):
        print("\n\nFor bottom/right Illu...")
        print("read in...")

        input_1 = "{}_illu_ventral_det_data".format("right" if T_flag else "bottom")
        if flip_ill:
            illu_direct = "left" if T_flag else "top"
        else:
            illu_direct = "right" if T_flag else "bottom"
        input_2 = "{}_illu_dorsal_det_data".format(illu_direct)

        rawPlanesTopO = read_with_bioio(locals()[input_1], T_flag)
        rawPlanesToptmp = rawPlanesTopO[:, xs:xe, ys:ye]
        _, m_c, n_c = rawPlanesToptmp.shape
        m = len(np.arange(m_c)[:: self.train_params["resample_ratio"]])
        n = len(np.arange(n_c)[:: self.train_params["resample_ratio"]])
        del rawPlanesTopO
        if require_registration:
            if self.registration_params["require_reg_finetune"]:
                reg_level = "_reg"
            else:
                reg_level = "_coarse_reg"

            bottom_handle = BioImage(
                os.path.join(
                    save_path,
                    self.sample_params[
                        "{}_dorsal".format("top" if flip_ill else "bottom")
                    ],
                    self.sample_params[
                        "{}_dorsal".format("top" if flip_ill else "bottom")
                    ]
                    + "{}.tif".format(reg_level),
                )
            )
            rawPlanesBottomO = bottom_handle.get_image_data(
                "ZXY" if T_flag else "ZYX", T=0, C=0
            )
        else:
            rawPlanesBottomO = read_with_bioio(locals()[input_2], T_flag)
        if not require_registration:
            rawPlanesBottomO[:] = np.flip(rawPlanesBottomO, flip_axes)
        m0, n0 = rawPlanesBottomO.shape[-2:]
        rawPlanesBottom = rawPlanesBottomO[:, xs:xe, ys:ye]
        del rawPlanesBottomO
        s = rawPlanesBottom.shape[0]
        if rawPlanesToptmp.shape[0] < rawPlanesBottom.shape[0]:
            rawPlanesTop = np.concatenate(
                (
                    rawPlanesToptmp,
                    np.zeros(
                        (
                            rawPlanesBottom.shape[0] - rawPlanesToptmp.shape[0],
                            rawPlanesBottom.shape[1],
                            rawPlanesBottom.shape[2],
                        ),
                        dtype=np.uint16,
                    ),
                ),
                0,
            )
        else:
            rawPlanesTop = rawPlanesToptmp
        del rawPlanesToptmp

        fb = self.predict_fusion_boundary(
            rawPlanesTop,
            rawPlanesBottom,
            segMask,
            s,
            m,
            n,
            m_c,
            n_c,
            m0,
            n0,
            xs,
            xe,
            ys,
            ye,
        )

        return fb.T if T_flag else fb

    def predict_fusion_boundary(
        self,
        rawPlanesTop,
        rawPlanesBottom,
        segMask,
        s,
        m,
        n,
        m_c,
        n_c,
        m0,
        n0,
        xs,
        xe,
        ys,
        ye,
    ):
        topF, bottomF = self.extractNSCTF(
            s,
            m,
            n,
            topVol=rawPlanesTop,
            bottomVol=rawPlanesBottom,
            device=self.train_params["device"],
        )

        boundary = self.dualViewFusion(
            topF,
            bottomF,
            segMask,
        )

        boundary = (
            F.interpolate(
                torch.from_numpy(boundary[None, None, :, :]),
                size=(m_c, n_c),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .data.numpy()
        )
        boundaryE = np.zeros((m0, n0))
        boundaryE[xs:xe, ys:ye] = boundary
        boundaryE = boundaryE.T
        if xs is not None:
            boundaryE = extendBoundary2(boundaryE, 11)
        if ys is not None:
            boundaryE = extendBoundary2(boundaryE.T, 11).T
        boundaryE = boundaryE.T
        return np.clip(boundaryE, 0, s).astype(np.uint16)

    def downsample_h5_files(
        self,
        data,
        xy_downsample_ratio,
        z_downsample_ratio,
        device,
    ):
        for r, i in enumerate(
            tqdm.tqdm(
                range(0, data.shape[0], z_downsample_ratio),
                desc="downsample: ",
                leave=False,
            )
        ):
            data_slice = np.array(data[i]).astype(np.float32)
            tmp = F.interpolate(
                torch.from_numpy(data_slice)[None, None].to(device),
                scale_factor=1 / xy_downsample_ratio,
                mode="bilinear",
                align_corners=True,
            )
            if i == 0:
                data_lr = np.zeros(
                    (
                        len(
                            range(
                                0,
                                data.shape[0],
                                z_downsample_ratio,
                            )
                        ),
                        tmp.shape[-2],
                        tmp.shape[-1],
                    ),
                    dtype=data.dtype,
                )
            data_lr[r] = tmp.cpu().data.numpy().squeeze().astype(data_lr.dtype)
        return data_lr

    def dualViewFusion(
        self,
        topF,
        bottomF,
        segMask,
    ):
        print("to GPU...")
        segMaskGPU = torch.from_numpy(segMask).to(self.train_params["device"])
        topFGPU = torch.from_numpy(topF**2).to(self.train_params["device"])
        bottomFGPU = torch.from_numpy(bottomF**2).to(self.train_params["device"])
        boundary = EM2DPlus(
            segMaskGPU,
            topFGPU,
            bottomFGPU,
            [
                self.train_params["kernel2d"].shape[0],
                self.train_params["kernel2d"].shape[1],
            ],
            [self.train_params["poly_order"][1], self.train_params["poly_order"][1]],
            self.train_params["kernel2d"],
            self.train_params["n_epochs"],
            device=self.train_params["device"],
            _xy=False,
        )
        del topFGPU, bottomFGPU, segMaskGPU
        return boundary

    def extractNSCTF(
        self,
        s,
        m,
        n,
        topVol,
        bottomVol,
        device,
    ):
        r = self.train_params["resample_ratio"]
        featureExtrac = NSCTdec(levels=[3, 3, 3], device=device).to(device)
        topF = np.empty((s, m, n), dtype=np.float32)
        bottomF = np.empty((s, m, n), dtype=np.float32)
        tmp0, tmp1 = np.arange(0, s, 1), np.arange(1, s + 1, 1)
        for p, q in tqdm.tqdm(zip(tmp0, tmp1), desc="NSCT: ", total=len(tmp0)):
            topDataFloat = topVol[p:q, :, :].astype(np.float32)
            bottomDataFloat = bottomVol[p:q, :, :].astype(np.float32)
            topDataGPU = torch.from_numpy(topDataFloat[:, None, :, :]).to(device)
            bottomDataGPU = torch.from_numpy(bottomDataFloat[:, None, :, :]).to(device)

            a, b, c = featureExtrac.nsctDec(topDataGPU, r, _forFeatures=True)

            # TODO: check the code below, if no need any more, remove it
            # max_filter = nn.MaxPool2d(
            #     (59, 59), stride=(1, 1), padding=(59 // 2, 59 // 2)
            # )
            # c = max_filter(c[None])[0]
            topF[p:q] = c
            a[:], b[:], c[:] = featureExtrac.nsctDec(
                bottomDataGPU,
                r,
                _forFeatures=True,
            )
            # c = max_filter(c[None])[0]
            bottomF[p:q] = c
            del topDataFloat, bottomDataFloat, topDataGPU, bottomDataGPU, a, b, c
        gc.collect()
        return topF, bottomF

    def segmentSample(
        self,
        topVoltmp,
        bottomVol,
        info_path,
        det_only_flag,
        leaf_paths,
    ):
        if not det_only_flag:
            Min, Max = [], []
            th = 0
            for f in leaf_paths["info.npy"]:
                t = np.load(f, allow_pickle=True).item()
                Min.append(t["minvol"])
                Max.append(t["maxvol"])
                th += t["thvol"]
            Min = max(Min)
            Max = max(Max)
            th = th / 4
        else:
            pass

        m, n = topVoltmp.shape[-2:]
        zfront, zback = topVoltmp.shape[0], bottomVol.shape[0]
        if zfront < zback:
            topVol = np.concatenate(
                (
                    topVoltmp,
                    np.zeros((zback - zfront, m, n), dtype=np.uint16),
                ),
                0,
            )
        else:
            topVol = copy.deepcopy(topVoltmp)
        del topVoltmp
        s = zback
        topSegMask = np.zeros((n, zback, m), dtype=bool)
        bottomSegMask = np.zeros((n, zback, m), dtype=bool)
        for i in tqdm.tqdm(range(zback), desc="watershed: "):
            x_top = topVol[i]
            x_bottom = bottomVol[i]
            if det_only_flag:
                th = filters.threshold_otsu(x_top + 0.0 + x_bottom) / 2
                Min = max(x_top.min(), x_bottom.min())
                Max = max(x_top.max(), x_bottom.max())
            th_top = 255 * (morphology.remove_small_objects(x_top > th, 25))
            th_bottom = 255 * (morphology.remove_small_objects(x_bottom > th, 25))
            th_top = th_top.astype(np.uint8)
            th_bottom = th_bottom.astype(np.uint8)
            topSegMask[:, i, :] = waterShed(x_top, th_top, Max, Min, m, n).T
            bottomSegMask[:, i, :] = waterShed(x_bottom, th_bottom, Max, Min, m, n).T
        segMask = refineShape(
            topSegMask,
            bottomSegMask,
            None,
            None,
            n,
            s,
            m,
            r=self.train_params["window_size"][1],
            _xy=False,
            max_seg=[-1] * m,
        )
        del topSegMask, bottomSegMask, topVol, bottomVol
        return segMask

    def localizingSample(
        self,
        rawPlanes_ventral,
        rawPlanes_dorsal,
        info_path,
        det_only_flag,
        leaf_paths,
    ):
        cropInfo = pd.DataFrame(
            columns=["startX", "endX", "startY", "endY", "maxv"],
            index=["ventral", "dorsal"],
        )
        for f in ["ventral", "dorsal"]:
            maximumProjection = locals()["rawPlanes_" + f].astype(np.float32)
            maximumProjection = np.log(np.clip(maximumProjection, 1, None))
            m, n = maximumProjection.shape
            if not det_only_flag:
                th = 0
                maxv = 0
                for ll in leaf_paths["info.npy"]:
                    t = np.load(ll, allow_pickle=True).item()
                    th += t["MIP_th"]
                    maxv = max(maxv, t["MIP_max"])
                th = th / len(leaf_paths["info.npy"])
                thresh = maximumProjection > th
            else:
                thresh = maximumProjection > filters.threshold_otsu(maximumProjection)
                maxv = np.log(max(rawPlanes_ventral.max(), rawPlanes_dorsal.max()))
            segMask = morphology.remove_small_objects(thresh, min_size=25)
            d1 = np.where(np.sum(segMask, axis=0) != 0)[0]
            d2 = np.where(np.sum(segMask, axis=1) != 0)[0]
            a = max(0, d1[0] - 100)
            b = min(n, d1[-1] + 100)
            c = max(0, d2[0] - 100)
            d = min(m, d2[-1] + 100)
            cropInfo.loc[f, :] = [c, d, a, b, np.exp(maxv)]
        cropInfo.loc["in summary"] = (
            min(cropInfo["startX"]),
            max(cropInfo["endX"]),
            min(cropInfo["startY"]),
            max(cropInfo["endY"]),
            max(cropInfo["maxv"]),
        )
        return cropInfo

    def segMIP(
        self,
        maximumProjection,
        th=None,
    ):
        m, n = maximumProjection.shape
        if th is None:
            th = filters.threshold_otsu(maximumProjection)
        thresh = maximumProjection > th
        segMask = morphology.remove_small_objects(thresh, min_size=25)
        d1 = np.where(np.sum(segMask, axis=0) != 0)[0]
        d2 = np.where(np.sum(segMask, axis=1) != 0)[0]
        a = max(0, d1[0] - 100)
        b = min(n, d1[-1] + 100)
        c = max(0, d2[0] - 100)
        d = min(m, d2[-1] + 100)
        return a, b, c, d


if __name__ == "__main__":
    model = FUSE_det(
        require_segmentation=True,
        registration_params={
            "use_exist_reg": True,
            "lateral_downsample": 2,
            "skip_refine_registration": True,
        },
    )
    # import os
    # path = "C:/Users/yu/Downloads/new_for_revision2/img_90_degree_4x_CH1_500_CH2_630"
    # dir = os.listdir(path)
    # # all_images = ["/content/drive/MyDrive/LSFM/Multi-View-Reconstruction/Data/new_for_revision/img_90_degree_4x_CH1_500_CH2_630/10-47-57_4x 0-6x 4z 500nm 630nm 90 degrees_Blaze[00 x 00]_C00_xyz-Table Z0000.ome.tif"]
    # all_images = [f for f in dir if "C01" in f]
    # all_images_ventral = sorted([os.path.join(path, f) for f in all_images])

    # path = "C:/Users/yu/Downloads/new_for_revision2/stack3"
    # dir = os.listdir(path)
    # # all_images = ["/content/drive/MyDrive/LSFM/Multi-View-Reconstruction/Data/new_for_revision/img_90_degree_4x_CH1_500_CH2_630/10-47-57_4x 0-6x 4z 500nm 630nm 90 degrees_Blaze[00 x 00]_C00_xyz-Table Z0000.ome.tif"]
    # # all_images = [f for f in dir if "C01" in f]
    # all_images = [f for f in dir]
    # all_images_dorsal = sorted([os.path.join(path, f) for f in all_images])

    # out = model.train(
    #     ventral_det_data=all_images_ventral,
    #     dorsal_det_data=all_images_dorsal,
    #     data_path="C:/Users/yu/AppData/Local/Temp",
    #     xy_downsample_ratio=2,  # the downsampling ratio along xy to trainthe low-resolution fusion model
    #     z_downsample_ratio=4,  # the downsampling ratio along z to trainthe low-resolution fusion model
    #     z_spacing=3.9,
    #     xy_spacing=2.71,
    #     require_registration=False,
    #     require_flipping_along_illu_for_dorsaldet=True,
    #     require_flipping_along_det_for_dorsaldet=True,
    #     save_path="C:/Users/yu/Downloads/new_for_revision2",
    #     save_folder="20250613_224758",
    #     left_right=True,
    #     display=False,
    # )

    # # C:/Users/yu/anaconda3/envs/leonardo_toolset_v2/Lib/intermediates/20250529_175243/high_res/tmpnszb38jyh5_reg.h5
    # v = tifffile.imread("C:/Users/yu/Downloads/illuFusionResult-front-1.tif").transpose(
    #     0, 1, 2
    # )
    # d = tifffile.imread("C:/Users/yu/Downloads/illuFusionResult-back-1.tif").transpose(
    #     0, 1, 2
    # )
    # out = model.train(
    #     ventral_det_data=v,
    #     dorsal_det_data=d,
    #     data_path="C:/Users/yu/Downloads",
    #     xy_spacing=0.65,
    #     z_spacing=4,
    #     require_registration=True,
    #     require_flipping_along_illu_for_dorsaldet=True,
    #     require_flipping_along_det_for_dorsaldet=True,
    #     save_path="C:/Users/yu/Downloads",
    #     save_folder="zebrafish_result",
    #     left_right=True,
    #     display=False,
    # )

    # out = model.train(
    #     top_illu_ventral_det_data="S000_t000000_V000_R0000_X000_Y000_C02_I1_D0_P00237.tif",
    #     bottom_illu_ventral_det_data="S000_t000000_V000_R0000_X000_Y000_C02_I0_D0_P00237.tif",
    #     top_illu_dorsal_det_data="S000_t000000_V000_R0000_X000_Y000_C02_I1_D0_P00249.tif",
    #     bottom_illu_dorsal_det_data="S000_t000000_V000_R0000_X000_Y000_C02_I0_D0_P00249.tif",
    #     data_path="D:/H2BGFP",
    #     xy_spacing=0.65,
    #     z_spacing=4,
    #     require_registration=True,
    #     require_flipping_along_illu_for_dorsaldet=True,
    #     require_flipping_along_det_for_dorsaldet=True,
    #     save_path="D:/H2BGFP/0-180-yaml",
    #     save_folder="result",
    #     display=False,
    # )

    out = model.train(
        top_illu_ventral_det_data="S000_t000000_V000_R0000_X000_Y000_C02_I1_D0_P00342.tif",
        bottom_illu_ventral_det_data="S000_t000000_V000_R0000_X000_Y000_C02_I0_D0_P00342.tif",
        top_illu_dorsal_det_data="S000_t000000_V000_R0000_X000_Y000_C02_I0_D1_P00342.tif",
        bottom_illu_dorsal_det_data="S000_t000000_V000_R0000_X000_Y000_C02_I1_D1_P00342.tif",
        data_path="D:/XSPIM-new/R0",
        xy_spacing=0.65,
        z_spacing=4,
        require_registration=False,
        require_flipping_along_illu_for_dorsaldet=True,
        require_flipping_along_det_for_dorsaldet=False,
        save_path="D:/XSPIM-new/R0",
        save_folder="20250614_023005",
        display=False,
    )
