import shutil
from datetime import datetime
import numpy as np
import copy
import os
from typing import Union
import dask.array as da
import scipy
import torch
import torch.nn.functional as F
from bioio import BioImage
from scipy import signal
import gc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tifffile
import tqdm
from skimage import morphology

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

from leonardo_toolset.fusion.NSCT import NSCTdec
from leonardo_toolset.fusion.utils import (
    EM2DPlus,
    extendBoundary2,
    fusion_perslice,
    refineShape,
    sgolay2dkernel,
    waterShed,
    parse_yaml_illu,
    extract_leaf_file_paths_from_file,
    read_with_bioio,
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


class FUSE_illu:
    def __init__(
        self,
        require_precropping: bool = True,
        precropping_params: list[int, int, int, int] = [],
        resample_ratio: int = 2,
        window_size: list[int, int] = [5, 59],
        poly_order: list[int, int] = [2, 2],
        n_epochs: int = 50,
        require_segmentation: bool = True,
        device: str = None,
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
        self.train_params["kernel2d"] = (
            torch.from_numpy(
                sgolay2dkernel(
                    np.array(self.train_params["window_size"]),
                    np.array(self.train_params["poly_order"]),
                )
            )
            .to(torch.float)
            .to(self.train_params["device"])
        )

    def train_from_params(self, params: dict):
        """Parses training parameters from dictionary"""
        if params["method"] != "illumination":
            raise ValueError(f"Invalid method: {params['method']}")
        if params["amount"] != 2:
            raise ValueError("Only 2 images are supported for illumination")
        image1 = params["image1"]
        image2 = params["image2"]
        direction1 = params["direction1"]
        direction2 = params["direction2"]
        top_illu_data = None
        bottom_illu_data = None
        left_illu_data = None
        right_illu_data = None
        if direction1 == "Top" and direction2 == "Bottom":
            top_illu_data = image1
            bottom_illu_data = image2
        elif direction1 == "Bottom" and direction2 == "Top":
            top_illu_data = image2
            bottom_illu_data = image1
        elif direction1 == "Left" and direction2 == "Right":
            left_illu_data = image1
            right_illu_data = image2
        elif direction1 == "Right" and direction2 == "Left":
            left_illu_data = image2
            right_illu_data = image1
        else:
            raise ValueError(
                f"Invalid combination of directions: {direction1}, {direction2}"
            )
        tmp_path = params["tmp_path"]
        # Create a directory under the intermediate_path
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir_path = os.path.join(tmp_path, current_time)
        os.makedirs(new_dir_path, exist_ok=True)

        # TODO: stop opening windows
        output_image = self.train(
            top_illu_data=top_illu_data,
            bottom_illu_data=bottom_illu_data,
            left_illu_data=left_illu_data,
            right_illu_data=right_illu_data,
            save_path=new_dir_path,
            save_separate_results=params["save_separate_results"],
            sparse_sample=params["sparse_sample"],
            cam_pos=params["cam_pos"],
            display=False,
            # TODO: more parameters?
        )

        if not params["keep_intermediates"]:
            # Clean up the intermediate directory
            shutil.rmtree(new_dir_path)
        return output_image

    def train(
        self,
        data_path: str = "",
        top_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        bottom_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        left_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        right_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        save_path: str = "",
        save_folder: str = "",
        save_separate_results: bool = False,
        sparse_sample=False,
        cam_pos: str = "front",
        camera_position: str = "",
        display: bool = True,
    ):
        if not os.path.exists(save_path):
            print("saving path does not exist.")
            return
        if not os.path.exists(os.path.join(save_path, save_folder)):
            os.makedirs(os.path.join(save_path, save_folder))
        allowed_keys = parse_yaml_illu.__code__.co_varnames  # or手动列出
        args_dict = {k: v for k, v in locals().items() if k in allowed_keys}
        args_dict.update({"train_params": self.train_params})
        args_dict.update(
            {
                "file_name": os.path.join(
                    save_path,
                    save_folder,
                    f"{camera_position+'_' if len(camera_position) != 0 else ''}illu_info.yaml",
                )
            }
        )
        yaml_path = parse_yaml_illu(**args_dict)

        illu_name = "illuFusionResult{}.tif".format(
            "" if self.train_params["require_segmentation"] else "_without_segmentation"
        )
        fb_xy = "fusionBoundary_xy{}.tif".format(
            "" if self.train_params["require_segmentation"] else "_without_segmentation"
        )

        leaf_paths = extract_leaf_file_paths_from_file(yaml_path)
        leaf_paths.update(
            {
                "illu_name": illu_name,
                "fb_xy": fb_xy,
            }
        )

        self.sample_params = {}
        _name = {}
        _sfx = "+" + camera_position if len(camera_position) != 0 else ""
        print("Read in...")
        if (left_illu_data is not None) and (right_illu_data is not None):
            T_flag = 1
            if isinstance(left_illu_data, str):
                _name["topillu"] = os.path.splitext(left_illu_data)[0]
                left_illu_data = os.path.join(data_path, left_illu_data)
            else:
                _name["topillu"] = "left_illu{}".format(_sfx)
            rawPlanes_top = read_with_bioio(left_illu_data, T_flag)
            if isinstance(right_illu_data, str):
                _name["bottomillu"] = os.path.splitext(right_illu_data)[0]
                right_illu_data = os.path.join(data_path, right_illu_data)
            else:
                _name["bottomillu"] = "right_illu{}".format(_sfx)
            rawPlanes_bottom = read_with_bioio(right_illu_data, T_flag)

        elif (top_illu_data is not None) and (bottom_illu_data is not None):
            T_flag = 0
            if isinstance(top_illu_data, str):
                _name["topillu"] = os.path.splitext(top_illu_data)[0]
                top_illu_data = os.path.join(data_path, top_illu_data)
            else:
                _name["topillu"] = "top_illu{}".format(_sfx)
            rawPlanes_top = read_with_bioio(top_illu_data, T_flag)

            if isinstance(bottom_illu_data, str):
                _name["bottomillu"] = os.path.splitext(bottom_illu_data)[0]
                bottom_illu_data = os.path.join(data_path, bottom_illu_data)
            else:
                _name["bottomillu"] = "bottom_illu{}".format(_sfx)
            rawPlanes_bottom = read_with_bioio(bottom_illu_data, T_flag)
        else:
            print("input(s) missing, please check.")
            return
        save_path = os.path.join(save_path, save_folder)
        for k in _name.keys():
            sub_folder = os.path.join(save_path, _name[k])
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
        if cam_pos == "back":
            rawPlanes_top = rawPlanes_top[::-1, :, :]
            rawPlanes_bottom = rawPlanes_bottom[::-1, :, :]

        print("\nLocalize sample...")
        cropInfo, MIP_info = self.localizingSample(rawPlanes_top, rawPlanes_bottom)
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

        s_o, m_o, n_o = rawPlanes_top.shape
        if display:
            fig, (ax1, ax2) = plt.subplots(1, 2, dpi=200)
            MIP_top = rawPlanes_top.max(0)
            if T_flag:
                MIP_top = MIP_top.T
            MIP_bottom = rawPlanes_bottom.max(0)
            if self.train_params["require_precropping"]:
                top_left_point = [ys, xs]
                if T_flag:
                    top_left_point = [top_left_point[1], top_left_point[0]]
            if T_flag:
                MIP_bottom = MIP_bottom.T
            ax1.imshow(MIP_top)
            if self.train_params["require_precropping"]:
                rect = patches.Rectangle(
                    tuple(top_left_point),
                    (ye - ys) if (not T_flag) else (xe - xs),
                    (xe - xs) if (not T_flag) else (ye - ys),
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax1.add_patch(rect)
            ax1.set_title(
                "{} illu".format("left" if T_flag else "top"), fontsize=8, pad=1
            )
            ax1.axis("off")
            ax2.imshow(MIP_bottom)
            if self.train_params["require_precropping"]:
                rect = patches.Rectangle(
                    tuple(top_left_point),
                    (ye - ys) if (not T_flag) else (xe - xs),
                    (xe - xs) if (not T_flag) else (ye - ys),
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax2.add_patch(rect)
            ax2.set_title(
                "{} illu".format("right" if T_flag else "bottom"), fontsize=8, pad=1
            )
            ax2.axis("off")
            plt.show()

        print("\nCalculate volumetric measurements...")
        thvol_top, maxvvol_top, minvvol_top = self.measureSample(
            rawPlanes_top[:, xs:xe, ys:ye][::4, ::4, ::4],
            "top",
            leaf_paths["info.npy"][0],
            MIP_info,
        )
        thvol_bottom, maxvvol_bottom, minvvol_bottom = self.measureSample(
            rawPlanes_bottom[:, xs:xe, ys:ye][::4, ::4, ::4],
            "bottom",
            leaf_paths["info.npy"][1],
            MIP_info,
        )

        s, m_c, n_c = rawPlanes_top.shape

        m_c = len(np.arange(m_c)[xs:xe])
        n_c = len(np.arange(n_c)[ys:ye])
        m = len(np.arange(m_c)[:: self.train_params["resample_ratio"]])
        n = len(np.arange(n_c)[:: self.train_params["resample_ratio"]])

        print("\nExtract features...")
        topF, bottomF = self.extractNSCTF(
            s,
            m,
            n,
            topVol=rawPlanes_top[:, xs:xe, ys:ye],
            bottomVol=rawPlanes_bottom[:, xs:xe, ys:ye],
        )

        t_topF = filters.threshold_otsu(topF[::4, ::2, ::2])
        t_bottomF = filters.threshold_otsu(bottomF[::4, ::2, ::2])

        print("\nSegment sample...")
        if self.train_params["require_segmentation"]:
            segMask = self.segmentSample(
                th_top=thvol_top,
                th_bottom=thvol_bottom,
                max_top=maxvvol_top,
                max_bottom=maxvvol_bottom,
                topVol=rawPlanes_top[
                    :,
                    xs : xe : self.train_params["resample_ratio"],
                    ys : ye : self.train_params["resample_ratio"],
                ],
                bottomVol=rawPlanes_bottom[
                    :,
                    xs : xe : self.train_params["resample_ratio"],
                    ys : ye : self.train_params["resample_ratio"],
                ],
                topVol_F=topF.transpose(1, 0, 2),
                bottomVol_F=bottomF.transpose(1, 0, 2),
                th_top_F=t_topF,
                th_bottom_F=t_bottomF,
                sparse_sample=sparse_sample,
            )
            np.save(leaf_paths["segmentation_illu.npy"], segMask)
        else:
            segMask = np.ones((s, m, n), dtype=bool)

        print("\nDual-illumination fusion...")
        boundary = self.dualViewFusion(topF, bottomF, segMask)

        boundary = (
            F.interpolate(
                torch.from_numpy(boundary[None, None, :, :]),
                size=(s, n_c),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .data.numpy()
        )

        boundary = boundary * self.train_params["resample_ratio"]

        boundaryE = np.zeros((s, n_o))
        boundaryE[:, ys:ye] = boundary
        if ys is not None:
            boundaryE = extendBoundary2(boundaryE, 11)
        if xs is not None:
            boundaryE += xs
        boundaryE = np.clip(boundaryE, 0, m_o).astype(np.uint16)
        if cam_pos == "back":
            boundaryE = boundaryE[::-1, :]
        tifffile.imwrite(leaf_paths[fb_xy][0], boundaryE)

        print("\nStitching...")
        boundaryE = tifffile.imread(leaf_paths[fb_xy][0]).astype(np.float32)
        if cam_pos == "back":
            boundaryE = boundaryE[::-1, :]

        if save_separate_results:
            if os.path.exists(leaf_paths["fuse_illu_mask"]):
                shutil.rmtree(leaf_paths["fuse_illu_mask"])
            os.makedirs(leaf_paths["fuse_illu_mask"])
            p = leaf_paths["fuse_illu_mask"]
        else:
            p = None

        recon = fusionResult(
            T_flag,
            rawPlanes_top,
            rawPlanes_bottom,
            copy.deepcopy(boundaryE),
            self.train_params["device"],
            save_separate_results,
            path=p,
            GFr=copy.deepcopy(self.train_params["window_size"]),
        )

        if T_flag:
            result = recon.transpose(0, 2, 1)
        else:
            result = recon
        del recon
        if display:
            fig, (ax1, ax2) = plt.subplots(1, 2, dpi=200)
            xyMIP = result.max(0)
            ax1.imshow(xyMIP)
            ax1.set_title("result", fontsize=8, pad=1)
            ax1.axis("off")
            ax2.imshow(np.zeros_like(xyMIP))
            ax2.axis("off")
            plt.show()
        if cam_pos == "back":
            result = result[::-1, :, :]

        print("Save...")
        tifffile.imwrite(leaf_paths[illu_name][0], result)
        return result

    def train_with_boundary(
        self,
        data_path: str = "",
        sample_name: str = "",
        top_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        bottom_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        left_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        right_illu_data: Union[np.ndarray, da.core.Array, str] = None,
        save_path: str = "",
        save_folder: str = "",
        save_separate_results: bool = False,
        cam_pos: str = "front",
        camera_position: str = "",
        display: bool = True,
        boundary_path: str = "",
    ):
        print("Running illum fusion using precomputed boundary...")
        # from train method start (180-288)
        data_path = os.path.join(data_path, sample_name)
        if not os.path.exists(save_path):
            print("saving path does not exist.")
            return
        # save_path = os.path.join(save_path, save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.sample_params = {}
        print("Read in...")
        if (left_illu_data is not None) and (right_illu_data is not None):
            T_flag = 1
            if isinstance(left_illu_data, str):
                left_illu_path = os.path.join(data_path, left_illu_data)
                left_illu_handle = BioImage(left_illu_path)
                self.sample_params["topillu_saving_name"] = os.path.splitext(
                    left_illu_data
                )[0]
                rawPlanes_top = np.transpose(
                    left_illu_handle.get_image_data("ZYX", T=0, C=0), (0, 2, 1)
                )
            else:
                if left_illu_data.ndim == 2:
                    left_illu_data_axis = left_illu_data[np.newaxis, :, :]
                else:
                    left_illu_data_axis = left_illu_data
                self.sample_params["topillu_saving_name"] = "left_illu{}".format(
                    "+" + camera_position if len(camera_position) != 0 else ""
                )
                rawPlanes_top = np.transpose(left_illu_data_axis, (0, 2, 1))
                if isinstance(rawPlanes_top, da.core.Array):
                    rawPlanes_top = rawPlanes_top.compute()
                del left_illu_data_axis, left_illu_data
            if isinstance(right_illu_data, str):
                right_illu_path = os.path.join(data_path, right_illu_data)
                right_illu_handle = BioImage(right_illu_path)
                self.sample_params["bottomillu_saving_name"] = os.path.splitext(
                    right_illu_data
                )[0]
                rawPlanes_bottom = np.transpose(
                    right_illu_handle.get_image_data("ZYX", T=0, C=0), (0, 2, 1)
                )
            else:
                if right_illu_data.ndim == 2:
                    right_illu_data_axis = right_illu_data[np.newaxis, :, :]
                else:
                    right_illu_data_axis = right_illu_data
                self.sample_params["bottomillu_saving_name"] = "right_illu{}".format(
                    "+" + camera_position if len(camera_position) != 0 else ""
                )
                rawPlanes_bottom = np.transpose(right_illu_data_axis, (0, 2, 1))
                if isinstance(rawPlanes_bottom, da.core.Array):
                    rawPlanes_bottom = rawPlanes_bottom.compute()
                del right_illu_data_axis, right_illu_data

        elif (top_illu_data is not None) and (bottom_illu_data is not None):
            T_flag = 0
            if isinstance(top_illu_data, str):
                top_illu_path = os.path.join(data_path, top_illu_data)
                top_illu_handle = BioImage(top_illu_path)
                self.sample_params["topillu_saving_name"] = os.path.splitext(
                    top_illu_data
                )[0]
                rawPlanes_top = top_illu_handle.get_image_data("ZYX", T=0, C=0)
            else:
                if top_illu_data.ndim == 2:
                    top_illu_data_axis = top_illu_data[np.newaxis, :, :]
                else:
                    top_illu_data_axis = top_illu_data
                self.sample_params["topillu_saving_name"] = "top_illu{}".format(
                    "+" + camera_position if len(camera_position) != 0 else ""
                )
                rawPlanes_top = top_illu_data_axis
                if isinstance(rawPlanes_top, da.core.Array):
                    rawPlanes_top = rawPlanes_top.compute()
                del top_illu_data_axis, top_illu_data

            if isinstance(bottom_illu_data, str):
                bottom_illu_path = os.path.join(data_path, bottom_illu_data)
                bottom_illu_handle = BioImage(bottom_illu_path)
                self.sample_params["bottomillu_saving_name"] = os.path.splitext(
                    bottom_illu_data
                )[0]
                rawPlanes_bottom = bottom_illu_handle.get_image_data("ZYX", T=0, C=0)
            else:
                if bottom_illu_data.ndim == 2:
                    bottom_illu_data_axis = bottom_illu_data[np.newaxis, :, :]
                else:
                    bottom_illu_data_axis = bottom_illu_data
                self.sample_params["bottomillu_saving_name"] = "bottom_illu{}".format(
                    "+" + camera_position if len(camera_position) != 0 else ""
                )
                rawPlanes_bottom = bottom_illu_data_axis
                if isinstance(rawPlanes_bottom, da.core.Array):
                    rawPlanes_bottom = rawPlanes_bottom.compute()
                del bottom_illu_data_axis, bottom_illu_data
        else:
            print("input(s) missing, please check.")
            return

        """for k in self.sample_params.keys():
            if "saving_name" in k:
                sub_folder = os.path.join(save_path, self.sample_params[k])
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)"""
        # sub_folder = save_path

        if cam_pos == "back":
            rawPlanes_top = rawPlanes_top[::-1, :, :]
            rawPlanes_bottom = rawPlanes_bottom[::-1, :, :]
        # finished copy from train method

        # copy from 455 - end of train function
        print("\nStitching...")
        boundaryE = tifffile.imread(
            os.path.join(boundary_path, "fusionBoundary_xy.tif")
        ).astype(np.float32)

        if cam_pos == "back":
            boundaryE = boundaryE[::-1, :]

        if save_separate_results:
            if os.path.exists(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_saving_name"],
                    "fuse_illu_mask",
                )
            ):
                shutil.rmtree(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_saving_name"],
                        "fuse_illu_mask",
                    )
                )
            os.makedirs(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_saving_name"],
                    "fuse_illu_mask",
                )
            )

        recon = fusionResult(
            T_flag,
            rawPlanes_top,
            rawPlanes_bottom,
            copy.deepcopy(boundaryE),
            self.train_params["device"],
            save_separate_results,
            path=os.path.join(
                save_path, "fuse_illu_mask"
            ),  # path=os.path.join(save_path,self.sample_params["topillu_saving_name"],"fuse_illu_mask",)
            GFr=copy.deepcopy(self.train_params["window_size"]),
        )

        if T_flag:
            result = recon.transpose(0, 2, 1)
        else:
            result = recon
        del recon
        if display:
            fig, (ax1, ax2) = plt.subplots(1, 2, dpi=200)
            xyMIP = result.max(0)
            ax1.imshow(xyMIP)
            ax1.set_title("result", fontsize=8, pad=1)
            ax1.axis("off")
            ax2.imshow(np.zeros_like(xyMIP))
            ax2.axis("off")
            plt.show()
        if cam_pos == "back":
            result = result[::-1, :, :]

        print("Save...")
        """tifffile.imwrite(
            os.path.join(save_path, self.sample_params["topillu_saving_name"])
            + "/illuFusionResult{}.tif".format(
                ""
                if self.train_params["require_segmentation"]
                else "_without_segmentation"
            ),
            result,
        )"""
        tifffile.imwrite(
            os.path.join(
                save_path,
                "illuFusionResult{}.tif".format(
                    ""
                    if self.train_params["require_segmentation"]
                    else "_without_segmentation"
                ),
            ),
            result,
        )
        return result

    def dualViewFusion(self, topF, bottomF, segMask):
        print("to GPU...")
        segMask_GPU = torch.from_numpy(segMask.transpose(1, 0, 2)).to(
            self.train_params["device"]
        )
        topFGPU = torch.from_numpy(topF**2).to(self.train_params["device"])
        bottomFGPU = torch.from_numpy(bottomF**2).to(self.train_params["device"])

        boundary = EM2DPlus(
            segMask_GPU,
            topFGPU,
            bottomFGPU,
            self.train_params["window_size"],
            self.train_params["poly_order"],
            self.train_params["kernel2d"],
            self.train_params["n_epochs"],
            device=self.train_params["device"],
            _xy=True,
        )
        del segMask, segMask_GPU, topFGPU, bottomFGPU
        return boundary

    def extractNSCTF(self, s, m, n, topVol, bottomVol):
        r = self.train_params["resample_ratio"]
        device = self.train_params["device"]
        featureExtrac = NSCTdec(levels=[3, 3, 3], device=device).to(device)
        topSTD = np.empty((m, s, n), dtype=np.float32)
        bottomSTD = np.empty((m, s, n), dtype=np.float32)
        tmp0, tmp1 = np.arange(0, s, 1), np.arange(1, s + 1, 1)
        for p, q in tqdm.tqdm(zip(tmp0, tmp1), desc="NSCT: ", total=len(tmp0)):
            topDataFloat = topVol[p:q, :, :].astype(np.float32)
            bottomDataFloat = bottomVol[p:q, :, :].astype(np.float32)

            a, b, c = featureExtrac.nsctDec(
                topDataFloat,
                r,
                _forFeatures=True,
            )

            topSTD[:, p:q, :] = c.transpose(1, 0, 2)

            a[:], b[:], c[:] = featureExtrac.nsctDec(
                bottomDataFloat,
                r,
                _forFeatures=True,
            )
            bottomSTD[:, p:q, :] = c.transpose(1, 0, 2)

            del topDataFloat, bottomDataFloat, a, b, c
        gc.collect()
        return topSTD, bottomSTD

    def segmentSample(
        self,
        th_top,
        th_bottom,
        max_top,
        max_bottom,
        topVol,
        bottomVol,
        topVol_F,
        bottomVol_F,
        th_top_F,
        th_bottom_F,
        sparse_sample,
    ):
        s, m, n = topVol.shape
        topSegMask = np.zeros((s, m, n), dtype=bool)
        bottomSegMask = np.zeros((s, m, n), dtype=bool)
        l_temp = signal.savgol_filter(
            ((topVol + 0.0 + bottomVol) > (th_top + th_bottom)).sum(1),
            11,
            1,
            axis=0,
        )
        l_all = signal.savgol_filter(
            ((topVol + 0.0 + bottomVol) > (th_top + th_bottom)).sum((1, 2)),
            11,
            1,
        )
        l_all = scipy.signal.find_peaks(l_all, height=l_all.max() / 10)[0]
        c_all = min(l_all[0] + 1 if len(l_all) > 0 else s // 2, s // 2)
        c = []
        for i in range(l_temp.shape[1]):
            peaks, _ = scipy.signal.find_peaks(
                l_temp[:, i], height=l_temp[:, i].max() / 10
            )
            if len(peaks) > 0:
                cc = peaks[0] + 1
            else:
                cc = s // 2
            cc = min(cc, s // 2)
            c.append(cc)
        t = np.linspace(0.5, 0.1, s - c_all)
        th_top_result = np.zeros((m, n), dtype=np.uint8)
        th_bottom_result = np.zeros((m, n), dtype=np.uint8)
        for i in tqdm.tqdm(range(s), desc="watershed ({}): ".format(c_all)):
            x_top = topVol[i]
            x_bottom = bottomVol[i]
            x_top_F = topVol_F[i]
            x_bottom_F = bottomVol_F[i]
            if i < c_all:
                th_top_slice = (th_top + th_bottom) / 2
                th_bottom_slice = (th_top + th_bottom) / 2
                th_top_F_slice = (th_top_F + th_bottom_F) / 2
                th_bottom_F_slice = (th_top_F + th_bottom_F) / 2
            else:
                x_top = x_top ** t[0]
                x_bottom = x_bottom ** t[0]
                a = filters.threshold_otsu(x_top)
                b = filters.threshold_otsu(x_bottom)
                th_top_slice = (a + b) / 2
                th_bottom_slice = (a + b) / 2
                th_top_F = filters.threshold_otsu(x_top_F)
                th_bottom_F = filters.threshold_otsu(x_bottom_F)
                th_top_F_slice = (th_top_F + th_bottom_F) / 2
                th_bottom_F_slice = (th_top_F + th_bottom_F) / 2
                t = t[1:]

            th_top_result = 255 * (
                morphology.remove_small_objects(
                    (x_top > th_top_slice) + (x_top_F > th_top_F_slice), 25
                )
            ).astype(np.uint8)
            th_bottom_result = 255 * (
                morphology.remove_small_objects(
                    (x_bottom > th_bottom_slice) + (x_bottom_F > th_bottom_F_slice), 25
                )
            ).astype(np.uint8)

            topSegMask[i, :, :] = waterShed(
                x_top,
                th_top_result,
                max(x_top.max(), x_bottom.max()),
                min(x_top.min(), x_bottom.min()),
                m,
                n,
            )
            bottomSegMask[i, :, :] = waterShed(
                x_bottom,
                th_bottom_result,
                max(x_top.max(), x_bottom.max()),
                min(x_top.min(), x_bottom.min()),
                m,
                n,
            )

        segMask = refineShape(
            topSegMask,
            bottomSegMask,
            topVol_F,
            bottomVol_F,
            s,
            m,
            n,
            self.train_params["window_size"][1],
            _xy=True,
            max_seg=c if sparse_sample is False else [-1] * n,
        )
        del topSegMask, bottomSegMask, topVol, bottomVol
        return segMask

    def measureSample(self, rawPlanes, f, save_path, MIP_info):
        if "top" in f:
            f_name = "top/left"
        if "bottom" in f:
            f_name = "bottom/right"
        print("\r{} view: ".format(f_name), end="")
        rawPlanes_vectored = rawPlanes[rawPlanes != 0]
        maxvvol = rawPlanes_vectored.max()
        minvvol = rawPlanes_vectored.min()
        thvol = filters.threshold_otsu(rawPlanes_vectored)
        print(
            f"minimum intensity = {minvvol:.1f}, "
            f"maximum intensity = {maxvvol:.1f}, "
            f"OTSU threshold = {thvol:.1f}"
        )
        np.save(
            save_path,
            {
                **{
                    "thvol": thvol,
                    "maxvol": maxvvol,
                    "minvol": minvvol,
                },
                **MIP_info,
            },
        )
        return thvol, maxvvol, minvvol

    def localizingSample(self, rawPlanes_top, rawPlanes_bottom):
        cropInfo = pd.DataFrame(
            columns=["startX", "endX", "startY", "endY", "maxv"],
            index=["top", "bottom"],
        )
        for f in ["top", "bottom"]:
            if "top" in f:
                f_name = "top/left"
            if "bottom" in f:
                f_name = "bottom/right"
            maximumProjection = locals()["rawPlanes_" + f].max(0).astype(np.float32)
            maximumProjection = np.log(np.clip(maximumProjection, 1, None))
            m, n = maximumProjection.shape
            maxv, th = (
                maximumProjection.max(),
                filters.threshold_otsu(maximumProjection),
            )
            thresh = maximumProjection > th
            segMask = morphology.remove_small_objects(thresh, min_size=25)
            d1, d2 = (
                np.where(np.sum(segMask, axis=0) != 0)[0],
                np.where(np.sum(segMask, axis=1) != 0)[0],
            )
            a, b, c, d = (
                max(0, d1[0] - 100),
                min(n, d1[-1] + 100),
                max(0, d2[0] - 100),
                min(m, d2[-1] + 100),
            )
            cropInfo.loc[f, :] = [c, d, a, b, np.exp(maxv)]
        cropInfo.loc["in summary"] = (
            min(cropInfo["startX"]),
            max(cropInfo["endX"]),
            min(cropInfo["startY"]),
            max(cropInfo["endY"]),
            max(cropInfo["maxv"]),
        )
        return cropInfo, {"MIP_th": th, "MIP_max": maxv}


def fusionResult(
    T_flag,
    topVol,
    bottomVol,
    boundary,
    device,
    save_separate_results,
    path,
    GFr=[5, 49],
):
    s, m, n = topVol.shape
    boundary = torch.from_numpy(boundary[None, :, None, :]).to(device)

    mask = torch.arange(m, device=device)[None, None, :, None]
    GFr[1] = GFr[1] // 4 * 2 + 1

    l_temp = np.concatenate(
        (
            np.arange(GFr[0] // 2, 0, -1),
            np.arange(s),
            np.arange(s - GFr[0] // 2, s - GFr[0] + 1, -1),
        ),
        0,
    )
    recon = np.zeros(topVol.shape, dtype=np.uint16)

    for ii in tqdm.tqdm(
        range(GFr[0] // 2, len(l_temp) - GFr[0] // 2), desc="fusion: "
    ):  # topVol.shape[0]
        l_s = l_temp[ii - GFr[0] // 2 : ii + GFr[0] // 2 + 1]
        boundary_slice = boundary[:, l_s, :, :]

        bottomMask = (mask > boundary_slice).to(torch.float)
        topMask = (mask <= boundary_slice).to(torch.float)

        ind = ii - GFr[0] // 2

        a, c = fusion_perslice(
            np.stack(
                (
                    topVol[l_s, :, :].astype(np.float32),
                    bottomVol[l_s, :, :].astype(np.float32),
                ),
                0,
            ),
            torch.cat((topMask, bottomMask), 0),
            GFr,
            device,
        )
        if save_separate_results:
            np.savez_compressed(
                os.path.join(
                    path,
                    "{:0>{}}".format(ind, 5) + ".npz",
                ),
                mask=c.transpose(0, 2, 1) if T_flag else c,
            )
        recon[ind] = a
    return recon


if __name__ == "__main__":
    model = FUSE_illu(require_segmentation=False)

    A = tifffile.imread(
        "D:/embryo/R0/V000_R0000_X000_Y000_C02_I1_D0_P00182.tif"
    ).transpose(0, 2, 1)
    B = tifffile.imread(
        "D:/embryo/R0/V000_R0000_X000_Y000_C02_I0_D0_P00182.tif"
    ).transpose(0, 2, 1)

    model.train(
        data_path="D:/embryo/R0",
        left_illu_data=A,
        right_illu_data=B,
        sparse_sample=True,
        save_path="C:/Users/yu/Downloads",
        save_folder="illu_fuse",
        display=False,
    )
