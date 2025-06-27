import copy

import cv2
import numpy as np
import pandas as pd
import scipy
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy import signal
from skimage import measure, morphology

from jinja2 import Template
import yaml
import os
from os.path import splitext
from pathlib import Path
from pathlib import PurePosixPath
import gc
import tifffile
from bioio.writers import OmeTiffWriter
from collections import OrderedDict
from bioio import BioImage
import re
from leonardo_toolset.fusion.NSCT import NSCTdec
import ants
import SimpleITK as sitk
import sys

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
import scipy.io as scipyio


def fusionResult_VD(
    T_flag,
    topVol,
    bottomVol,
    boundary,
    device,
    save_separate_results,
    path,
    flip_axes=tuple([]),
    GFr=[5, 49],
):
    flip_z = 0
    flip_xy = []
    if 0 in flip_axes:
        flip_z = 1
    if 1 in flip_axes:
        flip_xy.append(1)
    if 2 in flip_axes:
        flip_xy.append(2)
    s, m, n = bottomVol.shape
    boundary = boundary[None, None, :, :]
    mask = np.arange(s)[None, :, None, None]
    GFr[1] = GFr[1] // 4 * 2 + 1
    boundary = mask > boundary

    l_temp = np.concatenate(
        (
            np.zeros(GFr[0] // 2, dtype=np.int32),
            np.arange(s) if not flip_z else np.arange(s)[::-1],
            (s - 1) * np.ones(s - GFr[0] // 2 - (s - GFr[0] + 1), dtype=np.int32),
        ),
        0,
    )
    recon = np.zeros(bottomVol.shape, dtype=np.uint16)

    for ii in tqdm.tqdm(
        range(GFr[0] // 2, len(l_temp) - GFr[0] // 2), desc="fusion: "
    ):  # topVol.shape[0]
        l_s = l_temp[ii - GFr[0] // 2 : ii + GFr[0] // 2 + 1]

        bottomMask = torch.from_numpy(boundary[:, l_s, :, :]).to(device).to(torch.float)
        topMask = torch.from_numpy((~boundary[:, l_s, :, :])).to(device).to(torch.float)

        ind = ii - GFr[0] // 2

        topVol_tmp = np.zeros((len(l_s), m, n), dtype=np.float32)
        bottomVol_tmp = np.zeros((len(l_s), m, n), dtype=np.float32)

        indd = np.where(l_s < topVol.shape[0])[0]
        tmp = l_s[indd]
        b, c = np.unique(tmp, return_counts=True)

        topVol_tmp[indd, :, :] = np.array(topVol[b, :, :].repeat(c, 0))
        b, c = np.unique(l_s, return_counts=True)
        bottomVol_tmp[:] = np.array(bottomVol[b, :, :].repeat(c, 0))

        bottomVol_tmp[:] = np.flip(bottomVol_tmp, flip_xy)
        bottomMask[:] = torch.flip(bottomMask, flip_xy)

        # topVol[l_s, :, :].astype(np.float32)[None]

        a, c = fusion_perslice(
            np.stack(
                (
                    topVol_tmp,
                    bottomVol_tmp,
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


def fusionResultFour(
    T_flag,
    boundaryTop,
    boundaryBottom,
    boundaryFront,
    boundaryBack,
    illu_front,
    illu_back,
    device,
    sample_params,
    invalid_region,
    save_separate_results,
    path,
    GFr=49,
):
    s, m, n = illu_back.shape
    zmax = boundaryBack.shape[0]
    decModel = NSCTdec(levels=[3, 3, 3], device=device).to(device)

    mask = np.arange(m)[None, :, None]
    if boundaryFront.shape[0] < boundaryBack.shape[0]:
        boundaryFront = np.concatenate(
            (
                boundaryFront,
                np.zeros(
                    (
                        boundaryBack.shape[0] - boundaryFront.shape[0],
                        boundaryFront.shape[1],
                    )
                ),
            ),
            0,
        )
    mask_front = mask > boundaryFront[:, None, :]  # 1是下面，0是上面
    if boundaryBack.ndim == 2:
        mask_back = mask > boundaryBack[:, None, :]
    else:
        mask_back = boundaryBack
    mask_ztop = (
        np.arange(s)[:, None, None] > boundaryTop[None, :, :]
    )  # ##1是后面，0是前面
    mask_zbottom = (
        np.arange(s)[:, None, None] > boundaryBottom[None, :, :]
    )  # ##1是后面，0是前面

    listPair1 = {"1": "4", "2": "3", "4": "1", "3": "2"}
    reconVol = np.empty(illu_back.shape, dtype=np.uint16)
    allList = [
        value
        for key, value in sample_params.items()
        if ("saving_name" in key) and ("dorsal" in key)
    ]
    boundary_mask = np.zeros((s, m, n), dtype=bool)

    for ii in tqdm.tqdm(range(s), desc="intergrate fusion decision: "):
        if ii < illu_front.shape[0]:
            s1, s2, s3, s4 = (
                copy.deepcopy(illu_front[ii]),
                copy.deepcopy(illu_front[ii]),
                copy.deepcopy(illu_back[ii]),
                copy.deepcopy(illu_back[ii]),
            )
        else:
            s3, s4 = copy.deepcopy(illu_back[ii]), copy.deepcopy(illu_back[ii])
            s1, s2 = np.zeros(s3.shape), np.zeros(s3.shape)

        x = np.zeros((5, 1, m, n), dtype=np.float32)
        x[1, ...] = s1
        x[2, ...] = s2
        x[3, ...] = s3
        x[4, ...] = s4
        xtorch = torch.from_numpy(x).to(device)
        maskList = np.zeros((5, 1, m, n), dtype=bool)
        del x

        List = np.zeros((5, 1, m, n), dtype=np.float32)

        tmp1 = (mask_front[ii] == 0) * (mask_ztop[ii] == 0)  # ##top+front
        tmp2 = (mask_front[ii] == 1) * (mask_zbottom[ii] == 0)  # ##bottom+front
        tmp3 = (mask_back[ii] == 0) * (mask_ztop[ii] == 1)  # ##top+back
        tmp4 = (mask_back[ii] == 1) * (mask_zbottom[ii] == 1)  # ##bottom+back

        vnameList = ["1", "2", "3", "4"]

        flag_nsct = 0
        for vname in vnameList:
            maskList[int(vname)] += locals()["tmp" + vname] * (
                ~locals()["tmp" + listPair1[vname]]
            )
            if vnameList.index(vname) < vnameList.index(listPair1[vname]):
                v = locals()["tmp" + vname] * locals()["tmp" + listPair1[vname]]
                if sum(sum(v)):
                    v_labeled, num = measure.label(v, connectivity=2, return_num=True)
                    if flag_nsct == 0:
                        F1, _, _ = decModel.nsctDec(
                            xtorch[int(vname)][None], 1, _forFeatures=True
                        )
                        F2, _, _ = decModel.nsctDec(
                            xtorch[int(listPair1[vname])][None], 1, _forFeatures=True
                        )
                    for vv in range(1, num + 1):
                        v_s = v_labeled == vv
                        if ((F1 - F2) * v_s).sum() >= 0:
                            maskList[int(vname)] += v_s
                        else:
                            maskList[int(listPair1[vname])] += v_s
                    flag_nsct = 1
        maskList[0] = 1 - maskList[1:].sum(0)
        if maskList[0].sum() > 0:
            if flag_nsct == 0:
                F1, _, _ = decModel.nsctDec(xtorch[1][None], 1, _forFeatures=True)
                F2, _, _ = decModel.nsctDec(xtorch[4][None], 1, _forFeatures=True)
            v_labeled, num = measure.label(maskList[0], connectivity=2, return_num=True)
            for vv in range(1, num + 1):
                v_s = v_labeled == vv
                if ((F1 - F2) * v_s).sum() >= 0:
                    maskList[1] += v_s
                else:
                    maskList[4] += v_s
        maskList = np.concatenate(
            (maskList[1:2] + maskList[2:3], maskList[3:4] + maskList[4:5]), 0
        )
        boundary_mask[ii] = maskList[0, 0, :, :]

    # np.save("boundary_mask1.npy", boundary_mask)
    _mask_small_tmp = boundary_mask[:, :-2:3, :-2:3]
    _mask_small = np.zeros(
        (s, _mask_small_tmp.shape[1] * 2, _mask_small_tmp.shape[2] * 2), dtype=bool
    )
    _mask_small[:, ::2, ::2] = _mask_small_tmp

    with tqdm.tqdm(
        total=((_mask_small.shape[1] - 1) // 10 + 1)
        * ((_mask_small.shape[2] - 1) // 10 + 1),
        desc="refine along z: ",
        leave=False,
    ) as pbar:
        for i in range((_mask_small.shape[1] - 1) // 10 + 1):
            for j in range((_mask_small.shape[2] - 1) // 10 + 1):
                _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = (
                    morphology.remove_small_objects(
                        _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10],
                        5,
                    )
                )
                pbar.update(1)
    r = copy.deepcopy(_mask_small[:, ::2, ::2])
    _mask_small[:] = 1
    _mask_small[:, ::2, ::2] = r

    with tqdm.tqdm(
        total=((_mask_small.shape[1] - 1) // 10 + 1)
        * ((_mask_small.shape[2] - 1) // 10 + 1),
        desc="refine along z: ",
    ) as pbar:
        for i in range((_mask_small.shape[1] - 1) // 10 + 1):
            for j in range((_mask_small.shape[2] - 1) // 10 + 1):
                _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = (
                    morphology.remove_small_holes(
                        _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10],
                        5,
                    )
                )
                pbar.update(1)
    boundary_mask[:, : _mask_small_tmp.shape[1] * 3, : _mask_small_tmp.shape[1] * 3] = (
        np.repeat(np.repeat(_mask_small[:, ::2, ::2], 3, 1), 3, 2)
    )
    boundary_mask[invalid_region] = 1

    # boundary_mask = np.load("boundary_mask2.npy")
    s_f = illu_front.shape[0]
    s_b = illu_back.shape[0]
    if s_f < s_b:
        illu_front = np.concatenate((illu_front, illu_back[-(s_b - s_f) :, :, :]), 0)

    l_temp = np.concatenate(
        (
            np.arange(GFr[0] // 2, 0, -1),
            np.arange(s),
            np.arange(s - GFr[0] // 2, s - GFr[0] + 1, -1),
        ),
        0,
    )

    for ii in tqdm.tqdm(range(2, len(l_temp) - 2), desc="fusion: "):  # topVol.shape[0]
        l_s = l_temp[ii - GFr[0] // 2 : ii + GFr[0] // 2 + 1]

        bottomMask = 1 - boundary_mask[None, l_s, :, :]
        topMask = boundary_mask[None, l_s, :, :]

        ind = ii - GFr[0] // 2
        if save_separate_results:
            data = np.stack(
                (
                    illu_front[l_s, :, :].astype(np.float32),
                    illu_front[l_s, :, :].astype(np.float32),
                    illu_back[l_s, :, :].astype(np.float32),
                    illu_back[l_s, :, :].astype(np.float32),
                ),
                0,
            )
            mask = np.concatenate(
                (
                    topMask * (1 - mask_front[None, l_s, :, :]),
                    topMask * mask_front[None, l_s, :, :],
                    bottomMask * (1 - mask_back[None, l_s, :, :]),
                    bottomMask * mask_back[None, l_s, :, :],
                ),
                0,
            )
        else:
            data = np.stack(
                (
                    illu_front[l_s, :, :].astype(np.float32),
                    illu_back[l_s, :, :].astype(np.float32),
                ),
                0,
            )
            mask = np.concatenate(
                (
                    topMask,
                    bottomMask,
                ),
                0,
            )
        a, c = fusion_perslice(
            data,
            mask,
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
        reconVol[ind] = a

    del mask_front, mask_ztop, mask_back, mask_zbottom
    del illu_front, illu_back
    return reconVol


def load_dat_slices_blockwise(
    h5_dataset,
    z_indices,
):
    z_sorted = sorted(z_indices)
    z_min, z_max = z_sorted[0], z_sorted[-1] + 1
    block = h5_dataset[z_min:z_max]  # 连续读取
    return np.array(block)


@torch.no_grad()
def volumeTranslate_compose(
    inputs,
    T1,
    T2,
    padding_z,
    save_path,
    flip_axes,
    device,
    xy_spacing,
    z_spacing,
    large_vol=False,
):

    s, m, n = inputs.shape

    if T1 is not None:
        T1 = torch.from_numpy(T1).to(device)
    else:
        T1 = torch.eye(4).to(device)
    if T2 is not None:
        T2 = torch.from_numpy(T2.astype(np.float32)).to(device)
    else:
        T2 = torch.eye(4).to(device)

    T_compose = torch.matmul(T1, T2)

    inputs_dtype = inputs.dtype

    z_list = np.arange(s)
    if np.isin(0, flip_axes):
        z_list = z_list[::-1]
        flip_z = True
    else:
        flip_z = False

    if np.ceil(padding_z) > len(z_list):
        z_list = np.concatenate(
            (
                z_list,
                np.ones(int(np.ceil(padding_z) - len(z_list)), dtype=np.int32) * -1,
            )
        )

    commonData = np.zeros(
        (len(z_list), m, n),
        dtype=inputs_dtype,
    )

    yy, xx = torch.meshgrid(
        torch.arange(n).to(torch.float).to(device) * xy_spacing,
        torch.arange(m).to(torch.float).to(device) * xy_spacing,
        indexing="ij",
    )
    xx, yy = xx.T[None], yy.T[None]

    ss = torch.split(torch.arange(commonData.shape[0]), 10)
    mid_p_x = m // 2
    mid_p_y = n // 2

    norm = torch.from_numpy(
        np.array([z_spacing, xy_spacing, xy_spacing, 1]).astype(np.float32)
    ).to(device)[:, None, None, None]

    for s in tqdm.tqdm(ss, desc="projecting: "):
        start, end = s[0], s[-1] + 1
        start, end = start.item(), end.item()
        coor = torch.ones(4, end - start, m, n).to(device)
        coor[0, ...] = (
            z_spacing
            * torch.ones_like(xx)
            * torch.arange(start, end, dtype=torch.float, device=device)[:, None, None]
        )
        coor[1, ...] = xx
        coor[2, ...] = yy

        coor_translated = torch.einsum("ab,bcde->acde", T_compose, coor)

        coor_outlier = (
            torch.einsum(
                "ab,bn->an",
                T_compose,
                torch.tensor(
                    [
                        [start, 0, (n - 1), 1],
                        [start, (m - 1), (n - 1), 1],
                        [start, (m - 1), 0, 1],
                        [start, 0, 0, 1],
                        [(end - 1), 0, (n - 1), 1],
                        [(end - 1), (m - 1), (n - 1), 1],
                        [(end - 1), (m - 1), 0, 1],
                        [(end - 1), 0, 0, 1],
                    ],
                    device=device,
                ).T
                * torch.tensor(
                    [z_spacing, xy_spacing, xy_spacing, 1],
                    device=device,
                )[:, None],
            )
            / norm[:, :, 0, 0]
        )

        coor_translated /= norm

        coor_translated = coor_translated[:-1, ...]
        del coor
        if coor_outlier[0, ...].max() < 0:
            continue
        if coor_outlier[0, ...].min() >= len(z_list):
            continue
        minn = int(torch.clip(torch.floor(coor_outlier[0, ...].min()), 0, None))
        maxx = int(
            torch.clip(torch.ceil(coor_outlier[0, ...].max()), None, len(z_list))
        )
        coor_translated[0, ...] = coor_translated[0, ...] - minn

        z_list_small = z_list[minn : maxx + 1]

        ind = np.where(z_list_small != -1)[0]

        if ind.size == 0:
            continue

        z_use = np.flip(z_list_small[ind])[::-1] if flip_z else z_list_small[ind]
        tmp = load_dat_slices_blockwise(
            inputs,
            z_use.tolist(),
        )

        if np.isin(1, flip_axes):
            coor_translated[1, ...] = (m - 1) - coor_translated[1, ...]
        if np.isin(2, flip_axes):
            coor_translated[2, ...] = (n - 1) - coor_translated[2, ...]

        coor_translated[0, ...] -= min(ind)

        if z_use.tolist() == sorted(z_use.tolist()):
            pass
        else:
            coor_translated[0, ...] = (tmp.shape[0] - 1) - coor_translated[0, ...]

        translatedDatasmall = np.zeros(coor_translated.shape[1:], dtype=np.float32)

        for c_x_s, c_x_e, c_y_s, c_y_e in zip(
            [0, 0, mid_p_x, mid_p_x],
            [mid_p_x, mid_p_x, None, None],
            [0, mid_p_y, 0, mid_p_y],
            [mid_p_y, None, mid_p_y, None],
        ):
            coor_translated_small = coor_translated[:, :, c_x_s:c_x_e, c_y_s:c_y_e]
            bounding_x = coor_translated_small[1, [0, -1], :, :]  # .to(torch.int)
            bounding_y = coor_translated_small[2, [0, -1], :, :]  # .to(torch.int)

            if (bounding_x.max() < 0) or (bounding_y.max() < 0):
                pass
            else:
                tmp_small = torch.from_numpy(
                    tmp[
                        :,
                        int(max(bounding_x.min() - 10, 0)) : int(bounding_x.max() + 10),
                        int(max(bounding_y.min() - 10, 0)) : int(bounding_y.max() + 10),
                    ].astype(np.float32)
                ).to(device)

                coor_translated_small[:] = (
                    coor_translated_small
                    - torch.tensor(
                        [
                            0,
                            int(max(bounding_x.min() - 10, 0)),
                            int(max(bounding_y.min() - 10, 0)),
                        ],
                        device=device,
                    )[:, None, None, None]
                )
                coor_translated_small[:] = (
                    coor_translated_small
                    / torch.tensor(
                        [tmp_small.shape[0], tmp_small.shape[-2], tmp_small.shape[-1]],
                        device=device,
                    )[:, None, None, None]
                    - 0.5
                ) * 2
                translatedDatasmall[:, c_x_s:c_x_e, c_y_s:c_y_e] = coordinate_mapping(
                    tmp_small,
                    coor_translated_small[[2, 1, 0], ...],
                )
                del tmp_small

        if translatedDatasmall.ndim == 2:
            translatedDatasmall = translatedDatasmall[None]
        commonData[start:end, ...] = translatedDatasmall
        del coor_translated, translatedDatasmall

    gc.collect()

    if save_path is not None:
        print("save...")
        if ".tif" == os.path.splitext(save_path)[1]:
            if large_vol is False:
                tifffile.imwrite(save_path, commonData)
            else:
                save_path_dat = os.path.splitext(save_path)[0] + ".dat"

                OmeTiffWriter.save(commonData, save_path, dim_order="ZYX")

                Z, Y, X = commonData.shape
                dtype = commonData.dtype

                if os.path.exists(save_path_dat):
                    os.remove(save_path_dat)

                mm = np.memmap(save_path_dat, dtype=dtype, mode="w+", shape=(Z, Y, X))

                mm[:] = commonData
                mm.flush()
                return mm
        elif ".npy" == os.path.splitext(save_path)[1]:
            np.save(save_path, commonData)
            return commonData
        else:
            pass
    else:
        return commonData


def coordinate_mapping(
    smallData,
    coor_translated,
    padding_mode="zeros",
):
    translatedDataCupy = F.grid_sample(
        smallData[None, None, :, :],
        coor_translated.permute(1, 2, 3, 0)[None],
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    translatedDatasmall = translatedDataCupy.squeeze().cpu().data.numpy()
    del translatedDataCupy, smallData
    return translatedDatasmall


def boundaryInclude(
    ft,
    t,
    m,
    n,
    spacing,
):
    z = copy.deepcopy(t)
    while 1:
        transformed_point = np.matmul(
            ft["reg_matrix_inv"],
            np.array([[z, 0, 0, 1], [z, 0, n, 1], [z, m, 0, 1], [z, m, n, 1]]).T,
        )[0, ...]
        zz = min(transformed_point)
        if zz > t + 1:
            break
        z += spacing
    return z


def numpy_affine_to_ants_transform(
    T,
    center=None,
    dimension=3,
    output_path="tx.mat",
):
    if T.shape == (3, 3):
        inferred_dim = 2
    elif T.shape == (4, 4):
        inferred_dim = 3
    else:
        raise ValueError("Affine matrix must be 3x3 (2D) or 4x4 (3D).")

    dim = dimension if dimension is not None else inferred_dim

    A = T[:dim, :dim].flatten()
    t = T[:dim, -1]

    tx = ants.create_ants_transform(
        transform_type="AffineTransform",
        dimension=dim,
    )
    tx.set_parameters(np.concatenate([A, t]))

    if center is None:
        tx.set_fixed_parameters(np.zeros(dim))
    else:
        center = np.asarray(center)
        if center.shape[0] != dim:
            raise ValueError(f"Center must have length {dim}. Got {len(center)}")
        tx.set_fixed_parameters(center)

    ants.write_transform(tx, output_path)
    return tx


def antsreg_to_matrix(
    AffineTransform_float_3_3_inverse,
    fixed_inverse,
):
    AffineTransform = AffineTransform_float_3_3_inverse[:, 0]
    afixed = fixed_inverse[:, 0]
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(AffineTransform[:9].astype(np.float64))
    affine.SetTranslation(AffineTransform[9:].astype(np.float64))
    affine.SetCenter(afixed.astype(np.float64))
    A = np.array(affine.GetMatrix()).reshape(3, 3)
    c = np.array(affine.GetCenter())
    t = np.array(affine.GetTranslation())
    T = np.eye(4, dtype=np.float32)
    T[0:3, 0:3] = A
    T[0:3, 3] = -np.dot(A, c) + t + c
    return T


def fineReg(
    respective_view_uint16_pad,
    moving_view_uint16_pad,
    xcs,
    xce,
    ycs,
    yce,
    AffineMapZXY,
    z_spacing,
    xy_spacing,
    registration_params,
):
    mass_translation_mat = np.eye(4)
    mass_translation_mat[:3, -1] = [
        AffineMapZXY[0],
        AffineMapZXY[1],
        AffineMapZXY[2],
    ]
    numpy_affine_to_ants_transform(mass_translation_mat, dimension=3)

    respective_view_cropped = respective_view_uint16_pad[:, xcs:xce, ycs:yce]
    moving_view_cropped = moving_view_uint16_pad[:, xcs:xce, ycs:yce]
    del respective_view_uint16_pad, moving_view_uint16_pad
    respective_view_uint8 = respective_view_cropped.astype(np.float32)
    moving_view_uint8 = moving_view_cropped.astype(np.float32)

    respective_view_mask = np.ones(respective_view_uint8.shape, dtype=bool)
    moving_view_mask = np.ones(respective_view_uint8.shape, dtype=bool)
    if AffineMapZXY[0] > 0:
        moving_view_mask[: int(AffineMapZXY[0] / z_spacing)] = 0
        respective_view_mask[-int(AffineMapZXY[0] / z_spacing) :] = 0
    if AffineMapZXY[0] < 0:
        moving_view_mask[int(AffineMapZXY[0] / z_spacing) :] = 0
        respective_view_mask[: -int(AffineMapZXY[0] / z_spacing)] = 0
    moving_view_mask *= moving_view_uint8 > 0
    respective_view_mask *= respective_view_uint8 > 0

    size = (
        sys.getsizeof(respective_view_uint8)
        / registration_params["axial_downsample"]
        / (registration_params["lateral_downsample"]) ** 2
    )
    if size < 209715344 * 4:
        s = 0
        e = None
    else:
        r = (
            209715344
            * 4
            // sys.getsizeof(
                np.ones(
                    int(
                        moving_view_uint8[0].size
                        / (registration_params["lateral_downsample"]) ** 2
                    ),
                    dtype=np.float32,
                )
            )
        )

        s = (moving_view_uint8.shape[0] - r) // 2
        e = -s
        print("only [{}, {}] slices will be used for registration...".format(s, e))

    del moving_view_cropped, respective_view_cropped

    print("to ANTS...")
    staticANTS = ants.from_numpy(
        respective_view_uint8[
            s : e : registration_params["axial_downsample"],
            :: registration_params["lateral_downsample"],
            :: registration_params["lateral_downsample"],
        ]
    )
    movingANTS = ants.from_numpy(
        moving_view_uint8[
            s : e : registration_params["axial_downsample"],
            :: registration_params["lateral_downsample"],
            :: registration_params["lateral_downsample"],
        ]
    )

    staticANTS = ants.from_numpy(
        respective_view_uint8[
            s : e : registration_params["axial_downsample"],
            :: registration_params["lateral_downsample"],
            :: registration_params["lateral_downsample"],
        ]
    )
    respective_view_mask_ants = ants.from_numpy(
        respective_view_mask[
            s : e : registration_params["axial_downsample"],
            :: registration_params["lateral_downsample"],
            :: registration_params["lateral_downsample"],
        ]
    )
    moving_view_mask_ants = ants.from_numpy(
        moving_view_mask[
            s : e : registration_params["axial_downsample"],
            :: registration_params["lateral_downsample"],
            :: registration_params["lateral_downsample"],
        ]
    )

    movingANTS.set_spacing(
        (
            z_spacing * registration_params["axial_downsample"],
            xy_spacing * registration_params["lateral_downsample"],
            xy_spacing * registration_params["lateral_downsample"],
        )
    )
    staticANTS.set_spacing(
        (
            z_spacing * registration_params["axial_downsample"],
            xy_spacing * registration_params["lateral_downsample"],
            xy_spacing * registration_params["lateral_downsample"],
        )
    )

    respective_view_mask_ants.set_spacing(
        (
            z_spacing * registration_params["axial_downsample"],
            xy_spacing * registration_params["lateral_downsample"],
            xy_spacing * registration_params["lateral_downsample"],
        )
    )
    moving_view_mask_ants.set_spacing(
        (
            z_spacing * registration_params["axial_downsample"],
            xy_spacing * registration_params["lateral_downsample"],
            xy_spacing * registration_params["lateral_downsample"],
        )
    )

    del moving_view_uint8, respective_view_uint8

    print("registration...")
    regModel = ants.registration(
        staticANTS,
        movingANTS,
        mask=respective_view_mask_ants,
        moving_mask=moving_view_mask_ants,
        type_of_transform="Affine",
        mask_all_stages=True,
        random_seed=2022,
        initial_transform=["tx.mat"],
        verbose=True,
        aff_shrink_factors=(6, 4, 2),
        aff_iterations=(2100, 1200, 1200),
        aff_smoothing_sigmas=(3, 2, 1),
    )

    moving = regModel["warpedmovout"].numpy()
    static = staticANTS.numpy()

    ss = torch.split(torch.arange(moving.shape[0]), 20)

    mask = np.zeros(static.shape, dtype=bool)

    for sss in tqdm.tqdm(ss, desc="masking: ", leave=False):
        start, end = sss[0], sss[-1] + 1
        start, end = start.item(), end.item()
        input_batch_1 = static[start:end]
        input_batch_2 = moving[start:end]
        th = filters.threshold_otsu(
            np.maximum(input_batch_1[:, ::5, ::5], input_batch_2[:, ::5, ::5])
        )
        mask[start:end] = (input_batch_1 > th) * (input_batch_2 > th)

    maskANTS = ants.from_numpy(mask)

    maskANTS.set_spacing(
        (
            z_spacing * registration_params["axial_downsample"],
            xy_spacing * registration_params["lateral_downsample"],
            xy_spacing * registration_params["lateral_downsample"],
        )
    )

    regModel2 = ants.registration(
        staticANTS,
        regModel["warpedmovout"],
        mask=maskANTS,
        moving_mask=maskANTS,
        type_of_transform="Affine",
        mask_all_stages=True,
        random_seed=2022,
        initial_transform="Identity",
        verbose=True,
        aff_shrink_factors=(1,),
        aff_iterations=(100,),
        aff_smoothing_sigmas=(0,),
        aff_sampling=256,
    )

    rfile_1 = scipyio.loadmat(regModel["fwdtransforms"][0])
    rfile_inverse_1 = scipyio.loadmat(regModel["invtransforms"][0])
    rfile_2 = scipyio.loadmat(regModel2["fwdtransforms"][0])
    rfile_inverse_2 = scipyio.loadmat(regModel2["invtransforms"][0])

    reg_matrix_1_inv = antsreg_to_matrix(
        rfile_inverse_1["AffineTransform_float_3_3"],
        rfile_inverse_1["fixed"],
    )
    reg_matrix_2_inv = antsreg_to_matrix(
        rfile_inverse_2["AffineTransform_float_3_3"],
        rfile_inverse_2["fixed"],
    )

    reg_matrix_1 = antsreg_to_matrix(
        rfile_1["AffineTransform_float_3_3"],
        rfile_1["fixed"],
    )
    reg_matrix_2 = antsreg_to_matrix(
        rfile_2["AffineTransform_float_3_3"],
        rfile_2["fixed"],
    )

    reg_matrix = np.matmul(reg_matrix_2, reg_matrix_1)
    reg_matrix_inv = np.matmul(reg_matrix_1_inv, reg_matrix_2_inv)

    del regModel, movingANTS, staticANTS
    os.remove("tx.mat")
    return {
        "AffineMapZXY": AffineMapZXY,
        "reg_matrix": reg_matrix,
        "reg_matrix_inv": reg_matrix_inv,
        "region_for_reg": np.array([xcs, xce, ycs, yce]),
    }


def coarseRegistrationXY(
    front,
    back,
    z_spacing,
    xy_spacing,
):
    front = ants.from_numpy(front.astype(np.float32))
    back = ants.from_numpy(back.astype(np.float32))
    front.set_spacing((xy_spacing, xy_spacing))
    back.set_spacing((xy_spacing, xy_spacing))
    regModel = ants.registration(
        front,
        back,
        type_of_transform="Translation",
        random_seed=2022,
    )
    f = scipyio.loadmat(regModel["fwdtransforms"][0])["AffineTransform_float_2_2"]
    AffineMapZXY = np.zeros(3)
    AffineMapZXY[1:] = f[-2:][:, 0]

    return AffineMapZXY, front.numpy(), regModel["warpedmovout"].numpy()


def coarseRegistrationZX(
    front,
    back,
    z_spacing,
    xy_spacing,
    AffineMapZXY,
):
    mass_translation_mat = np.eye(3)
    mass_translation_mat[:2, -1] = [
        AffineMapZXY[0],
        AffineMapZXY[1],
    ]
    numpy_affine_to_ants_transform(mass_translation_mat, dimension=2)

    front = ants.from_numpy(front.astype(np.float32))
    back = ants.from_numpy(back.astype(np.float32))
    front.set_spacing((z_spacing, xy_spacing))
    back.set_spacing((z_spacing, xy_spacing))

    back = ants.apply_transforms(
        fixed=front,
        moving=back,
        transformlist=["tx.mat"],
        interpolator="linear",
    )

    regModel = ants.registration(
        front,
        back,
        mask=front > 0,
        moving_mask=back > 0,
        type_of_transform="Translation",
        random_seed=2022,
        restrict_transformation=(1, 0),
    )
    f = scipyio.loadmat(regModel["fwdtransforms"][0])["AffineTransform_float_2_2"]
    AffineMapZXY[:2] += f[-2:][:, 0]
    os.remove("tx.mat")

    return AffineMapZXY


def strip_ext(path, with_ext=0):
    if isinstance(path, str):
        if not with_ext:
            return os.path.splitext(os.path.basename(path))[0]
        else:
            return os.path.basename(path)
    else:
        return path


def read_with_bioio(
    path,
    T_flag=0,
):
    data_handle = BioImage(path)
    data = data_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
    return data


def extract_leaf_file_paths_from_file(yaml_file_path):
    def recurse(d, result):
        if isinstance(d, dict):
            if "path" in d and "contents" not in d:
                file_name = os.path.basename(d["path"])
                if file_name in result:
                    if isinstance(result[file_name], list):
                        result[file_name].append(d["path"])
                    else:
                        result[file_name] = [result[file_name], d["path"]]
                else:
                    result[file_name] = d["path"]
            else:
                for v in d.values():
                    recurse(v, result)
        elif isinstance(d, list):
            for item in d:
                recurse(item, result)

    with open(yaml_file_path, "r", encoding="utf-8") as f:
        yaml_dict = yaml.safe_load(f)

    result = OrderedDict()
    recurse(yaml_dict, result)
    for key, value in result.items():
        if key.startswith("illuFusionResult") and ("_reg" not in key):
            if isinstance(value, list):
                pass
            else:
                result[key] = [value]
        if key.startswith("fusionBoundary_z") and ("_reg" not in key):
            if isinstance(value, list):
                pass
            else:
                result[key] = [value]
        if key.startswith("fusionBoundary_xy") and ("_reg" not in key):
            if isinstance(value, list):
                pass
            else:
                result[key] = [value]
    return result


def normalize_path(path):
    return str(PurePosixPath(path.replace("\\", "/")))


def get_template_path(filename):
    return Path(__file__).parent / "templates" / filename


def format_yaml_blocks(yaml_text: str) -> str:
    lines = yaml_text.strip().splitlines()
    formatted = []

    def is_top_level_key(line: str) -> bool:
        return bool(re.match(r'^[\'"]\S+["\']\s*:\s*$', line.strip()))

    for i, line in enumerate(lines):
        if is_top_level_key(line) and formatted and formatted[-1].strip() != "":
            formatted.append("")  # 插入一行空行（仅当上行非空）
        formatted.append(line)

    return "\n".join(formatted)


def render_yaml_template(
    template_path: str,
    context: dict,
    output_path: str = None,
) -> dict:
    with open(template_path, "r", encoding="utf-8") as f:
        template_str = f.read()

    template = Template(template_str, trim_blocks=True, lstrip_blocks=True)
    rendered = template.render(**context)

    rendered = format_yaml_blocks(rendered)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write(rendered)

    return yaml.safe_load(rendered)


def parse_yaml_illu(
    data_path,
    top_illu_data,
    bottom_illu_data,
    left_illu_data,
    right_illu_data,
    save_path,
    save_folder,
    save_separate_results,
    sparse_sample,
    cam_pos,
    camera_position,
    display,
    train_params,
    file_name,
):
    context = {
        "result_folder": save_folder,
        "save_path": normalize_path(os.path.join(save_path, save_folder)),
        "include_fuse_illu_mask": save_separate_results,
    }
    context.update(train_params)

    context["illu_fuse_result"] = "illuFusionResult{}.tif".format(
        ("" if train_params["require_segmentation"] else "_without_segmentation")
    )
    context["illu_boundary_result"] = "fusionBoundary_xy{}.tif".format(
        "" if train_params["require_segmentation"] else "_without_segmentation"
    )

    if (left_illu_data is not None) and (right_illu_data is not None):
        context["top_illu_orient"] = "left"
        context["bottom_illu_orient"] = "right"
        context["illu_horizontal"] = True
        if isinstance(left_illu_data, str):
            context["top_folder"] = splitext(left_illu_data)[0]
            context["left_illu_data"] = normalize_path(
                os.path.join(
                    data_path,
                    left_illu_data,
                )
            )
        else:
            context["top_folder"] = "left_illu{}".format(
                "+" + camera_position if len(camera_position) != 0 else ""
            )
            context["left_illu_data"] = "ArrayLike"
        if isinstance(right_illu_data, str):
            context["bottom_folder"] = splitext(right_illu_data)[0]
            context["right_illu_data"] = normalize_path(
                os.path.join(
                    data_path,
                    right_illu_data,
                )
            )
        else:
            context["bottom_folder"] = "right_illu{}".format(
                "+" + camera_position if len(camera_position) != 0 else ""
            )
            context["right_illu_data"] = "ArrayLike"

    elif (top_illu_data is not None) and (bottom_illu_data is not None):
        context["illu_horizontal"] = False
        context["top_illu_orient"] = "top"
        context["bottom_illu_orient"] = "bottom"
        if isinstance(top_illu_data, str):
            context["top_folder"] = splitext(top_illu_data)[0]
            context["top_illu_data"] = normalize_path(
                os.path.join(
                    data_path,
                    top_illu_data,
                )
            )
        else:
            context["top_folder"] = "top_illu{}".format(
                "+" + camera_position if len(camera_position) != 0 else ""
            )
            context["top_illu_data"] = "ArrayLike"

        if isinstance(bottom_illu_data, str):
            context["bottom_folder"] = splitext(bottom_illu_data)[0]
            context["bottom_illu_data"] = normalize_path(
                os.path.join(
                    data_path,
                    bottom_illu_data,
                )
            )
        else:
            context["bottom_folder"] = "bottom_illu{}".format(
                "+" + camera_position if len(camera_position) != 0 else ""
            )
            context["bottom_illu_data"] = "ArrayLike"
    else:
        print("input(s) missing, please check.")
        return

    context["sparse_sample"] = sparse_sample

    yaml_path = os.path.join(
        save_path,
        save_folder,
        file_name,
    )
    _ = render_yaml_template(
        get_template_path("fuse_illu.yaml"),
        context,
        output_path=yaml_path,
    )
    return yaml_path


def parse_yaml_det(
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
    z_spacing,  # axial
    xy_spacing,  # lateral
    left_right,
    xy_downsample_ratio,
    z_downsample_ratio,
    display,
    train_params,
    registration_params,
    file_name,
):
    context = {
        "result_folder": save_folder,
        "save_path": normalize_path(os.path.join(save_path, save_folder)),
        "include_fuse_det_mask": save_separate_results,
        "require_refine_reg": registration_params["require_reg_finetune"],
    }
    train_params.update({"registration_params": registration_params})
    context.update(train_params)

    context["det_fuse_result"] = "quadrupleFusionResult{}.tif".format(
        ("" if train_params["require_segmentation"] else "_without_segmentation")
    )
    context["det_boundary_result"] = "fusionBoundary_z{}.tif".format(
        "" if train_params["require_segmentation"] else "_without_segmentation"
    )
    context["illu_fuse_result"] = "illuFusionResult{}.tif".format(
        ("" if train_params["require_segmentation"] else "_without_segmentation")
    )
    context["illu_boundary_result"] = "fusionBoundary_xy{}.tif".format(
        "" if train_params["require_segmentation"] else "_without_segmentation"
    )

    if (
        (top_illu_ventral_det_data is not None)
        and (bottom_illu_ventral_det_data is not None)
        and (top_illu_dorsal_det_data is not None)
        and (bottom_illu_dorsal_det_data is not None)
    ):
        context["coarse_registered_vol"] = (
            os.path.splitext(context["illu_fuse_result"])[0] + "_coarse_reg.tif"
        )
        context["fine_registered_vol"] = (
            os.path.splitext(context["illu_fuse_result"])[0] + "_reg.tif"
        )
        context["illu_boundary_result_reg"] = (
            os.path.splitext(context["illu_boundary_result"])[0] + "_reg.npy"
        )
        context["illu_horizontal"] = False
        template_name = "fuse_4_det.yaml"
        if isinstance(top_illu_ventral_det_data, str):
            context["top_ventral_folder"] = splitext(top_illu_ventral_det_data)[0]
            context["top_illu_orient"] = "top"
            context["top_illu_ventral_det_data"] = normalize_path(
                os.path.join(
                    data_path,
                    top_illu_ventral_det_data,
                )
            )
        else:
            context["top_ventral_folder"] = "top_illu+ventral_det"
            context["top_illu_orient"] = "top"
            context["top_illu_ventral_det_data"] = "ArrayLike"
        if isinstance(bottom_illu_ventral_det_data, str):
            context["bottom_ventral_folder"] = splitext(bottom_illu_ventral_det_data)[0]
            context["bottom_ventral_illu_orient"] = "bottom"
            context["bottom_illu_ventral_det_data"] = normalize_path(
                os.path.join(
                    data_path,
                    bottom_illu_ventral_det_data,
                )
            )
        else:
            context["bottom_ventral_folder"] = "bottom_illu+ventral_det"
            context["bottom_illu_orient"] = "bottom"
            context["bottom_illu_ventral_det_data"] = "ArrayLike"
        if isinstance(top_illu_dorsal_det_data, str):
            context["top_dorsal_folder"] = splitext(top_illu_dorsal_det_data)[0]
            context["top_illu_orient"] = "top"
            context["top_illu_dorsal_det_data"] = normalize_path(
                os.path.join(
                    data_path,
                    top_illu_dorsal_det_data,
                )
            )
        else:
            context["top_dorsal_folder"] = "top_illu+dorsal_det"
            context["top_illu_orient"] = "top"
            context["top_illu_dorsal_det_data"] = "ArrayLike"
        if isinstance(bottom_illu_dorsal_det_data, str):
            context["bottom_dorsal_folder"] = splitext(bottom_illu_dorsal_det_data)[0]
            context["bottom_illu_orient"] = "bottom"
            context["bottom_illu_dorsal_det_data"] = normalize_path(
                os.path.join(
                    data_path,
                    bottom_illu_dorsal_det_data,
                )
            )
        else:
            context["bottom_dorsal_folder"] = "bottom_illu+dorsal_det"
            context["bottom_illu_orient"] = "bottom"
            context["bottom_illu_dorsal_det_data"] = "ArrayLike"
        context["top_dorsal_reg"] = context["top_dorsal_folder"] + "_reg.tif"
        context["bottom_dorsal_reg"] = context["bottom_dorsal_folder"] + "_reg.tif"
    elif (
        (left_illu_ventral_det_data is not None)
        and (right_illu_ventral_det_data is not None)
        and (left_illu_dorsal_det_data is not None)
        and (right_illu_dorsal_det_data is not None)
    ):
        context["coarse_registered_vol"] = (
            os.path.splitext(context["illu_fuse_result"])[0] + "_coarse_reg.tif"
        )
        context["fine_registered_vol"] = (
            os.path.splitext(context["illu_fuse_result"])[0] + "_reg.tif"
        )
        context["illu_boundary_result_reg"] = (
            os.path.splitext(context["illu_boundary_result"])[0] + "_reg.tif"
        )
        context["illu_horizontal"] = True
        template_name = "fuse_4_det.yaml"
        if isinstance(left_illu_ventral_det_data, str):
            context["top_ventral_folder"] = splitext(left_illu_ventral_det_data)[0]
            context["top_illu_orient"] = "left"
            context["left_illu_ventral_det_data"] = normalize_path(
                os.path.join(
                    data_path,
                    left_illu_ventral_det_data,
                )
            )
        else:
            context["top_ventral_folder"] = "left_illu+ventral_det"
            context["top_illu_orient"] = "left"
            context["left_illu_ventral_det_data"] = "ArrayLike"
        if isinstance(right_illu_ventral_det_data, str):
            context["bottom_ventral_folder"] = splitext(right_illu_ventral_det_data)[0]
            context["bottom_illu_orient"] = "right"
            context["right_illu_ventral_det_data"] = normalize_path(
                os.path.join(
                    data_path,
                    right_illu_ventral_det_data,
                )
            )
        else:
            context["bottom_ventral_folder"] = "right_illu+ventral_det"
            context["bottom_illu_orient"] = "right"
            context["right_illu_ventral_det_data"] = "ArrayLike"
        if isinstance(left_illu_dorsal_det_data, str):
            context["top_dorsal_folder"] = splitext(left_illu_dorsal_det_data)[0]
            context["top_illu_orient"] = "left"
            context["left_illu_dorsal_det_data"] = normalize_path(
                os.path.join(
                    data_path,
                    left_illu_dorsal_det_data,
                )
            )
        else:
            context["top_dorsal_folder"] = "left_illu+dorsal_det"
            context["top_illu_orient"] = "left"
            context["left_illu_dorsal_det_data"] = "ArrayLike"
        if isinstance(right_illu_dorsal_det_data, str):
            context["bottom_dorsal_folder"] = splitext(right_illu_dorsal_det_data)[0]
            context["bottom_illu_orient"] = "right"
            context["right_illu_dorsal_det_data"] = normalize_path(
                os.path.join(
                    data_path,
                    right_illu_dorsal_det_data,
                )
            )
        else:
            context["bottom_dorsal_folder"] = "right_illu+dorsal_det"
            context["bottom_illu_orient"] = "right"
            context["right_illu_dorsal_det_data"] = "ArrayLike"
        context["top_dorsal_reg"] = context["top_dorsal_folder"] + "_reg.tif"
        context["bottom_dorsal_reg"] = context["bottom_dorsal_folder"] + "_reg.tif"
    elif (ventral_det_data is not None) and ((dorsal_det_data is not None)):
        if (xy_downsample_ratio is None) and (z_downsample_ratio is None):
            template_name = "fuse_2_det.yaml"
        else:
            template_name = "fuse_2_high_res_det.yaml"
            context["require_refine_reg_hr"] = registration_params[
                "skip_refine_registration"
            ]
        if left_right is None:
            print("left-right marker is missing.")
            return
        if left_right is True:
            context["ventral_illu_orient"] = "left-right"
        else:
            context["ventral_illu_orient"] = "top-bottom"
        if isinstance(ventral_det_data, str):
            context["ventral_folder"] = splitext(ventral_det_data)[0]
            context["ventral_det_data"] = normalize_path(
                os.path.join(
                    data_path,
                    ventral_det_data,
                )
            )
        else:
            context["ventral_folder"] = "ventral_det"
            context["ventral_det_data"] = "ArrayLike"
        if isinstance(dorsal_det_data, str):
            context["dorsal_folder"] = splitext(dorsal_det_data)[0]
            context["dorsal_det_data"] = normalize_path(
                os.path.join(
                    data_path,
                    dorsal_det_data,
                )
            )
        else:
            context["dorsal_folder"] = "dorsal_det"
            context["dorsal_det_data"] = "ArrayLike"
        context["coarse_registered_vol"] = context["dorsal_folder"] + "_coarse_reg.tif"
        context["fine_registered_vol"] = context["dorsal_folder"] + "_reg.tif"
    else:
        pass

    context["require_flipping_along_illu_for_dorsaldet"] = (
        require_flipping_along_illu_for_dorsaldet
    )
    context["require_flipping_along_det_for_dorsaldet"] = (
        require_flipping_along_det_for_dorsaldet
    )
    context["require_registration"] = require_registration
    context["sparse_sample"] = sparse_sample
    context["z_spacing"] = z_spacing if require_registration else "n.a."
    context["xy_spacing"] = xy_spacing if require_registration else "n.a."
    if xy_downsample_ratio is None:
        context["xy_downsample_ratio"] = "n.a."
    else:
        context["xy_downsample_ratio"] = xy_downsample_ratio
    if z_downsample_ratio is None:
        context["z_downsample_ratio"] = "n.a."
    else:
        context["z_downsample_ratio"] = z_downsample_ratio

    yaml_path = os.path.join(
        save_path,
        save_folder,
        file_name,
    )
    _ = render_yaml_template(
        get_template_path(template_name),
        context,
        output_path=yaml_path,
    )
    return yaml_path


def fusion_perslice(
    x,
    mask,
    GFr,
    device,
):
    n, c, m, n = x.shape
    GF = GuidedFilter(r=GFr, eps=1)
    x = torch.from_numpy(x).to(device)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).to(device).to(torch.float)

    result, num = GF(x, mask)
    num = num == (2 * GFr[1] + 1) * (2 * GFr[1] + 1) * GFr[0]
    result[num] = 1
    result = result / result.sum(0, keepdim=True)
    minn, maxx = x.min(), x.max()
    y_seg = x[:, c // 2 : c // 2 + 1, :, :] * result
    y = torch.clip(y_seg.sum(0), minn, maxx)

    return (
        y.squeeze().cpu().data.numpy().astype(np.uint16),
        result.squeeze().cpu().data.numpy(),
    )


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def diff_x(self, input, r):
        left = input[:, :, r : 2 * r + 1]
        middle = input[:, :, 2 * r + 1 :] - input[:, :, : -2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], dim=2)
        return output

    def diff_y(self, input, r):
        left = input[:, :, :, r : 2 * r + 1]
        middle = input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], dim=3)
        return output

    def forward(self, x):
        return self.diff_y(
            self.diff_x(x.sum(1, keepdims=True).cumsum(dim=2), self.r[1]).cumsum(dim=3),
            self.r[1],
        )


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()
        if isinstance(r, list):
            self.r = r
        else:
            self.r = [r, r]
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        mean_y_tmp = self.boxfilter(y)
        x, y = 0.001 * x, 0.001 * y
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()
        N = self.boxfilter(torch.ones_like(x))
        mean_x = self.boxfilter(x) / N
        mean_y = self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N
        return (
            mean_A * x[:, c_x // 2 : c_x // 2 + 1, :, :] + mean_b
        ) / 0.001, mean_y_tmp


def extendBoundary2(
    boundary,
    window_size,
):
    for i in range(boundary.shape[0]):
        tmp = copy.deepcopy(boundary[i, :])
        p = np.where(tmp != 0)[0]
        if len(p) != 0:
            p0, p1 = p[0], p[-1]
            valid_slice = tmp[p0 : p1 + 1]
            g_left = [0] * window_size * 2 + [
                valid_slice[1] - valid_slice[0]
            ] * window_size * 2
            left_ext = np.cumsum(
                np.append(
                    -1
                    * signal.savgol_filter(g_left, window_size, 1)[
                        window_size
                        + window_size // 2
                        + 1 : 2 * window_size
                        + window_size // 2
                    ],
                    valid_slice[0],
                )[::-1]
            )[::-1]
            if p0 + 1 - len(left_ext) > 0:
                left_ext = np.pad(left_ext, [(p0 + 1 - len(left_ext), 0)], mode="edge")
            else:
                left_ext = left_ext[-(p0 + 1) :]
            boundary[i, : p0 + 1] = left_ext
            g_right = [valid_slice[-1] - valid_slice[-2]] * window_size * 2 + [
                0
            ] * window_size * 2
            right_ext = np.cumsum(
                np.concatenate(
                    (
                        np.array([valid_slice[-1]]),
                        signal.savgol_filter(g_right, window_size, 1)[
                            window_size
                            + window_size // 2
                            + 1 : 2 * window_size
                            + window_size // 2
                        ],
                    )
                )
            )
            if len(tmp[p1:]) - len(right_ext) > 0:
                right_ext = np.pad(
                    right_ext,
                    [(0, len(tmp[p1:]) - len(right_ext))],
                    mode="edge",
                )
            else:
                right_ext = right_ext[: len(tmp[p1:])]
            boundary[i, p1:] = right_ext

    return boundary


def extendBoundary(
    boundary,
    window_size_list,
    poly_order_list,
    spacing,
    _xy,
):
    # boundaryEM = copy.deepcopy(boundary)
    if _xy is True:
        mask = morphology.binary_dilation(
            boundary != 0, np.ones((1, window_size_list[1]))
        )
        for dim in [1]:
            # window_size = window_size_list[dim]
            # poly_order = poly_order_list[dim]
            for i in range(boundary.shape[0]):
                p = np.where(boundary[i, :] != 0)[0]
                if len(p) != 0:
                    p0, p1 = p[0], p[-1]
                    boundary[i, :p0] = boundary[i, p0]
                    boundary[i, p1:] = boundary[i, p1]
                else:
                    boundary[i, :] = boundary[i, :]
    else:
        mask = boundary != 0

    boundary[~mask] = 0

    dist, ind = scipy.ndimage.distance_transform_edt(
        boundary == 0, return_distances=True, return_indices=True, sampling=spacing
    )
    boundary[boundary == 0] = boundary[ind[0], ind[1]][boundary == 0]

    return boundary


def EM2DPlus(
    segMask,
    f0,
    f1,
    window_size,
    poly_order,
    kernel2d,
    maxEpoch,
    device,
    _xy,
):

    def preComputePrior(seg, f0, f1):
        A = torch.cumsum(seg * f0, 0) + torch.flip(
            torch.cumsum(torch.flip(seg * f1, [0]), 0), [0]
        )
        return A

    def maskfor2d(segMask, window_size, m1, s1, n1):

        min_boundary = (
            torch.from_numpy(
                first_nonzero(
                    segMask,
                    None,
                    0,
                    segMask.shape[0] * 2,
                )
            )
            .to(torch.float)
            .to(device)
        )
        max_boundary = (
            torch.from_numpy(
                last_nonzero(
                    segMask,
                    None,
                    0,
                    -segMask.shape[0] * 2,
                )
            )
            .to(torch.float)
            .to(device)
        )
        if _xy is True:
            tmp = copy.deepcopy(
                min_boundary.cpu().data.numpy()
            )  # min_boundary.cpu().data.numpy()
            tmp[tmp != m1 * 2] = 0
            _, ind = scipy.ndimage.distance_transform_edt(
                tmp, return_distances=True, return_indices=True, sampling=[1, 1e3]
            )
            min_boundary[min_boundary == m1 * 2] = min_boundary[ind[0], ind[1]][
                min_boundary == m1 * 2
            ]
            tmp = copy.deepcopy(
                max_boundary.cpu().data.numpy()
            )  # max_boundary.cpu().data.numpy()
            tmp[tmp != -m1 * 2] = 0
            _, ind = scipy.ndimage.distance_transform_edt(
                tmp, return_distances=True, return_indices=True, sampling=[1, 1e3]
            )
            max_boundary[max_boundary == -m1 * 2] = max_boundary[ind[0], ind[1]][
                max_boundary == -m1 * 2
            ]

            mm = ~scipy.ndimage.binary_dilation(
                (segMask.sum(0) != 0).cpu().data.numpy(),
                np.ones((window_size[0] * 2 + 1, 1)),
            )
        else:
            mm = ~(segMask.sum(0) != 0).cpu().data.numpy()

        min_boundary[mm] = 0
        max_boundary[mm] = m1

        validMask = (segMask.sum(0) != 0).cpu().data.numpy().astype(np.float32)
        tmp = np.repeat(np.arange(s1)[:, None], n1, 1)
        maskrow = (tmp >= first_nonzero(validMask, None, axis=0, invalid_val=-1)) * (
            tmp <= last_nonzero(validMask, None, axis=0, invalid_val=-1)
        )
        tmp = np.repeat(np.arange(n1)[None, :], s1, 0)
        maskcol = (
            tmp >= first_nonzero(validMask, None, axis=1, invalid_val=-1)[:, None]
        ) * (tmp <= last_nonzero(validMask, None, axis=1, invalid_val=-1)[:, None])
        validMask[:] = copy.deepcopy(maskcol * maskrow).astype(np.float32)

        missingMask = (
            (segMask.sum(0) == 0).cpu().data.numpy()
            * (
                np.arange(n1)[None, :]
                >= first_nonzero(validMask, None, axis=1, invalid_val=-1)[:, None]
            )
            * (
                np.arange(n1)[None, :]
                <= last_nonzero(validMask, None, axis=1, invalid_val=-1)[:, None]
            )
        )
        validMask[missingMask] = 1

        valid_edge2 = skimage.util.view_as_windows(
            np.pad(
                validMask,
                (
                    (window_size[0] // 2, window_size[0] // 2),
                    (window_size[1] // 2, window_size[1] // 2),
                ),
                "constant",
                constant_values=0,
            ),
            window_size,
        ).sum((-2, -1))
        valid_edge2 = (
            (valid_edge2 < window_size[0] * window_size[1])
            * (valid_edge2 > 0)
            * validMask.astype(bool)
        )

        validMask_stripe = copy.deepcopy(validMask)
        validFor2D = skimage.util.view_as_windows(
            np.pad(
                validMask_stripe,
                (
                    (window_size[0] // 2, window_size[0] // 2),
                    (window_size[1] // 2, window_size[1] // 2),
                ),
                "constant",
                constant_values=0,
            ),
            window_size,
        ).sum((-2, -1))
        validFor2D = validFor2D == window_size[0] * window_size[1]
        validFor2D += valid_edge2

        return (
            torch.from_numpy(validFor2D).to(device),
            torch.arange(m1, device=device)[:, None, None],
            min_boundary,
            max_boundary,
        )

    def missingBoundary(boundaryTMP, s1, n1):
        boundaryTMP[boundaryTMP == 0] = np.nan
        a1 = np.isnan(boundaryTMP)
        boundaryTMP[np.isnan(boundaryTMP).sum(1) >= (n - 1), :] = 0
        for i in range(s1):
            boundaryTMP[i] = (
                pd.DataFrame(boundaryTMP[i])
                .interpolate("polynomial", order=1)
                .values[:, 0]
            )

        a2 = np.isnan(boundaryTMP)
        boundaryTMP[np.isnan(boundaryTMP)] = 0
        return boundaryTMP, (a2 == 0) * (a1)

    def selected_filter(x, validFor2D, min_boundary, max_boundary, kernel_high):
        w1, w2 = window_size
        dim0 = torch.zeros((1, n), dtype=torch.int).to(device)
        dim1 = torch.arange(0, n).to(device)
        y = torch.zeros_like(x)
        x_pad = F.pad(
            x[None, None], (w2 // 2, w2 // 2, w1 // 2, w1 // 2), mode="reflect"
        )[0, 0]
        min_boundary_pad = F.pad(
            min_boundary[None, None],
            (w2 // 2, w2 // 2, w1 // 2, w1 // 2),
            mode="reflect",
        )[0, 0]
        max_boundary_pad = F.pad(
            max_boundary[None, None],
            (w2 // 2, w2 // 2, w1 // 2, w1 // 2),
            mode="reflect",
        )[0, 0]
        for ind, i in enumerate(range(w1 // 2, s + w1 // 2)):
            xs = x_pad[i - w1 // 2 : i + w1 // 2 + 1, :]
            min_boundary_s = min_boundary_pad[i - w1 // 2 : i + w1 // 2 + 1, :]
            max_boundary_s = max_boundary_pad[i - w1 // 2 : i + w1 // 2 + 1, :]
            xs_unfold = xs.unfold(0, w1, 1).unfold(1, w2, 1)
            mask1 = (xs_unfold >= min_boundary[ind : ind + 1, :, None, None]) * (
                xs_unfold <= max_boundary[ind : ind + 1, :, None, None]
            )
            min_boundary_s_unfold = min_boundary_s.unfold(0, w1, 1).unfold(1, w2, 1)
            max_boundary_s_unfold = max_boundary_s.unfold(0, w1, 1).unfold(1, w2, 1)
            mask2 = (x[ind : ind + 1, :, None, None] >= min_boundary_s_unfold) * (
                x[ind : ind + 1, :, None, None] <= max_boundary_s_unfold
            )
            mask = mask1 * mask2
            mask[validFor2D[ind : ind + 1, :], :, :] = 1
            mask[:, :, w1 // 2, w2 // 2] = 1
            K = mask * kernel_high
            y[i - w1 // 2] = (K * xs_unfold).sum((-2, -1)) / K.sum((-2, -1))
            s_m = K.sum((-2, -1)) < 0.5
            if s_m.sum() > 0:
                xs_unfold_sort = torch.sort(
                    xs_unfold.reshape(1, n, -1), dim=-1, descending=True
                )[
                    0
                ]  # [:, :, :, 59//2-26:59//2+26+1]
                med_ind = (
                    xs_unfold_sort.shape[-1]
                    // 2
                    * torch.ones((1, n), dtype=torch.int).to(device)
                )
                median_result = xs_unfold_sort[dim0, dim1, med_ind]
                y[i - w1 // 2 : i - w1 // 2 + 1, :][s_m] = median_result[s_m]
        return y

    def init(x, validFor2D, bg_mask, min_boundary, max_boundary):
        dim0 = torch.zeros((1, n), dtype=torch.int).to(device)
        dim1 = torch.arange(0, n).to(device)
        w1, w2 = window_size
        y = torch.zeros_like(x)
        validFor2D0 = copy.deepcopy(validFor2D)

        x_pad_cpu = np.pad(x.cpu().data.numpy(), ((w2 // 2, w2 // 2)), mode="reflect")
        x_cpu = x.cpu().data.numpy()

        validFor2D = scipy.ndimage.binary_dilation(
            validFor2D.cpu().data.numpy(), np.ones((1, w2))
        )
        for i in range(s):
            t = torch.where(validFor2D0[i] > 0)[0]
            if len(t) > 0:
                a = t[0]
                b = t[-1]
                x_pad_cpu[i, a : b + w2] = np.pad(
                    x_cpu[i][a : b + 1], (w2 // 2, w2 // 2), mode="reflect"
                )
        x = torch.from_numpy(x_pad_cpu[:, w2 // 2 : -w2 // 2 + 1]).to(device)
        x_pad = F.pad(
            x[None, None], (w2 // 2, w2 // 2, w1 // 2, w1 // 2), mode="reflect"
        )[0, 0]

        validFor2D = torch.from_numpy(validFor2D).to(device)
        validFor2D = (
            F.pad(
                validFor2D[None, None] + 0.0,
                (w2 // 2, w2 // 2, w1 // 2, w1 // 2),
                mode="reflect",
            )[0, 0]
            > 0
        )
        for ind, i in enumerate(range(w1 // 2, s + w1 // 2)):
            xs = x_pad[i - w1 // 2 : i + w1 // 2 + 1, :]
            validFor2D_s = validFor2D[i - w1 // 2 : i + w1 // 2 + 1, :]
            xs_unfold = xs.unfold(0, w1, 1).unfold(1, w2, 1)  # .reshape(1, n, -1)
            validFor2D_unfold = validFor2D_s.unfold(0, w1, 1).unfold(1, w2, 1)
            mask = validFor2D_unfold
            mask[:, :, w1 // 2, w2 // 2] = 1
            med_ind = mask.sum((-2, -1)) // 2
            xs_unfold_sort = torch.sort(
                (mask * xs_unfold).reshape(1, n, -1), dim=-1, descending=True
            )[0]
            y[i - w1 // 2] = xs_unfold_sort[dim0, dim1, med_ind]
            y[i - w1 // 2][y[i - w1 // 2] == 0] = x[ind, :][y[i - w1 // 2] == 0]
        return y * validFor2D0

    m, s, n = segMask.shape
    segMask = segMask != 0
    bg_mask = segMask.sum(0) == 0
    feature = torch.zeros(m, s, n).to(device)

    for ss in range(s):
        feature[:, ss, :] = preComputePrior(
            segMask[:, ss, :],
            f0[:, ss, :],
            f1[:, ss, :],
        )

    cn = 0

    (
        validFor2D,
        coorMask,
        min_boundary,
        max_boundary,
    ) = maskfor2d(segMask, window_size, m, s, n)
    boundary0 = torch.argmax(feature, 0).cpu().data.numpy().astype(np.float32)

    # tmp = np.arange(m)[:, None, None] > boundary0[None, :, :]

    boundary, _ = missingBoundary(copy.deepcopy(boundary0), s, n)
    if _xy:
        boundaryLS = (
            init(
                torch.from_numpy(boundary).to(device),
                validFor2D,
                bg_mask,
                min_boundary,
                max_boundary,
            )
            .cpu()
            .data.numpy()
        )
    else:
        boundaryLS = copy.deepcopy(boundary)

    boundaryLS = extendBoundary(
        boundaryLS,
        window_size,
        poly_order,
        [window_size[1] / window_size[0], 1.0],
        _xy=_xy,
    )
    # tmp0 = np.arange(m)[:, None, None] > boundaryLS[None, :, :]
    boundary = torch.from_numpy(boundary).to(device)
    boundaryLS = torch.from_numpy(boundaryLS).to(device)
    boundaryOld = copy.deepcopy(boundary)

    boundaryLS = torch.maximum(boundaryLS, min_boundary)
    boundaryLS = torch.minimum(boundaryLS, max_boundary)

    w1, w2 = window_size
    for e in range(maxEpoch):
        Lambda = feature.max() / ((boundaryLS - boundary) ** 2 + 1).max()
        boundary[:] = torch.argmax(feature - Lambda * (boundaryLS - coorMask) ** 2, 0)
        changes = (
            100
            if e == 0
            else torch.quantile(torch.abs((boundaryOld - boundary) * (~bg_mask)), 0.99)
        )
        boundaryOld[:] = copy.deepcopy(boundary)
        boundaryLS[:] = selected_filter(
            boundary, bg_mask, min_boundary, max_boundary, kernel2d
        )

        cn = cn + 1 if changes < (5 if _xy else 2) else 0
        print(
            "\rNo.{:0>3d} iteration EM: maximum changes = {}".format(
                e, changes if e > 0 else "--"
            ),
            end="",
        )
    del feature, f0, f1

    boundaryLS = boundaryLS.cpu().data.numpy()

    return boundaryLS


def waterShed(
    xo,
    thresh,
    maxv,
    minv,
    m,
    n,
):
    x = np.zeros((m, n), dtype=np.float32)
    fg, bg = np.zeros((m, n), dtype=np.uint8), np.zeros((m, n), dtype=np.uint8)
    marker32, mm = np.zeros((m, n), dtype=np.int32), np.zeros((m, n), dtype=np.uint8)
    tmpMask = np.zeros((m, n), dtype=bool)
    if xo.max() > 0:
        x[:] = 255 * np.clip((xo - minv) / (maxv - minv), 0, 1)
    else:
        pass
    fg[:] = 255 * thresh.astype(bool)
    _, bg[:] = cv2.threshold(
        cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=10),
        1,
        128,
        1,
    )
    marker32[:] = cv2.watershed(
        cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_GRAY2BGR),
        np.int32(cv2.add(fg, bg)),
    )
    mm[:] = cv2.convertScaleAbs(marker32)
    tmpMask[:] = (mm != 128) * (mm != 1)

    tmpMask[::2, :] = skimage.morphology.binary_dilation(
        tmpMask[::2, :], np.ones((1, 3))
    )

    del thresh, x, fg, bg, marker32, mm
    return tmpMask


def refineShape(
    segMaskTop,
    segMaskBottom,
    topF,
    bottomF,
    s,
    m,
    n,
    r,
    _xy,
    max_seg,
):

    def missingBoundary(x, mask):
        data = copy.deepcopy(x) + 0.0
        data[mask] = np.nan
        data = pd.DataFrame(data).interpolate("polynomial", order=1).values[:, 0]
        return np.isinf(data) * (mask == 1)

    def outlierFilling(segMask):
        first = first_nonzero(segMask, None, 0, n - 1)
        last = last_nonzero(segMask, None, 0, 0)
        B = np.cumsum(segMask, 1)
        C = np.cumsum(segMask[:, ::-1], 1)[:, ::-1]
        D = (B > 0) * (C > 0) * (segMask == 0)
        D = D.astype(np.float32)
        E = copy.deepcopy(D)
        F = copy.deepcopy(D)
        b1 = (D[:, 1:] - D[:, :-1]) == 1
        b1 = np.concatenate((np.zeros((b1.shape[0], 1)), b1), 1).astype(np.uint8)

        b2 = (D[:, :-1] - D[:, 1:]) == 1
        b2 = np.concatenate((b2, np.zeros((b2.shape[0], 1))), 1).astype(np.uint8)

        b1 = (
            scipy.signal.convolve(b1, np.ones((11, 1)).astype(np.uint8), mode="same")
            == 11
        ) + 0.0  # 左
        b2 = (
            scipy.signal.convolve(b2, np.ones((11, 1)).astype(np.uint8), mode="same")
            == 11
        ) + 0.0  # 右

        for ii in range(b1.shape[0]):
            aa = np.where(b1[ii, :] == 1)[0]
            bb = np.where(b2[ii, :] == 1)[0]
            for a in aa:
                for b in bb:
                    if b >= a:
                        if (D[ii, a : b + 1] == 1).sum() == (b - a + 1):
                            E[ii, a : b + 1] = 0
                            break

        Mask = (F - E) == 1  # ((-b1+b2) == -1)+((b1+b2) == 2)

        testa2 = measure.label(Mask, connectivity=2)
        props = measure.regionprops(testa2)
        A = np.zeros(Mask.shape, bool)

        for ind, p in enumerate(props):
            bbox = p.bbox
            if (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]) < 2:
                first[bbox[1] : bbox[3]] = np.minimum(
                    np.linspace(first[bbox[1]], first[bbox[3]], bbox[3] - bbox[1]),
                    first[bbox[1] : bbox[3]],
                )
                last[bbox[1] : bbox[3]] = np.maximum(
                    np.linspace(last[bbox[1]], last[bbox[3]], bbox[3] - bbox[1]),
                    last[bbox[1] : bbox[3]],
                )
                mm = np.isin(testa2, ind + 1)
                mm[:, mm.sum(0) < r] = 0
                mm[:] = scipy.ndimage.binary_erosion(mm, np.ones((3, 1)))
                A[mm] = 1
        tmp = np.arange(segMask.shape[0])[:, None]
        result = (tmp >= first) * ((tmp <= last))
        result[A] = 1
        return segMask ^ result

    _mask, _maskl, _maskm = (
        np.zeros((s, m, n), dtype=bool),
        np.zeros((s, m, n), dtype=bool),
        np.zeros((s, m, n), dtype=bool),
    )

    for i in tqdm.tqdm(
        range(0, s), desc="refine pair-wise segmentation result: ", leave=False
    ):
        temp = np.linspace(0, m - 1, m, dtype=np.int32)[:, None]
        temp_top = segMaskTop[i, :, :]
        temp_bottom = segMaskBottom[i, :, :]

        boundaryCoordsBottom = last_nonzero(
            None, temp_bottom, axis=0, invalid_val=np.inf
        )
        boundaryCoordsTop = first_nonzero(None, temp_top, axis=0, invalid_val=np.inf)
        boundaryCoordsBottom2 = np.full(boundaryCoordsBottom.shape, np.inf)
        tmp = morphology.remove_small_objects(~np.isinf(boundaryCoordsBottom), r)
        boundaryCoordsBottom2[tmp] = boundaryCoordsBottom[tmp]
        boundaryCoordsTop2 = np.full(boundaryCoordsBottom.shape, np.inf)
        tmp = morphology.remove_small_objects(~np.isinf(boundaryCoordsTop), r)
        boundaryCoordsTop2[tmp] = boundaryCoordsTop[tmp]
        mask_bottom = np.isinf(boundaryCoordsBottom2) * (~np.isinf(boundaryCoordsTop2))
        mask_top = (~np.isinf(boundaryCoordsBottom2)) * np.isinf(boundaryCoordsTop2)
        mask_bottom_s = missingBoundary(boundaryCoordsBottom2, mask_bottom)
        mask_top_s = missingBoundary(boundaryCoordsTop2, mask_top)
        mask_bottom_l = mask_bottom_s ^ mask_bottom
        mask_top_l = mask_top_s ^ mask_top
        np.nan_to_num(boundaryCoordsBottom, copy=False, nan=0, posinf=0)
        np.nan_to_num(boundaryCoordsTop, copy=False, nan=m - 1, posinf=m - 1)

        segMask = (temp >= boundaryCoordsTop) * ((temp <= boundaryCoordsBottom))
        segMaskk = segMask.sum(0) > 0
        segMask += outlierFilling(segMask)
        segMask[:, mask_bottom_l] += temp_top[:, mask_bottom_l]
        segMask[:, mask_top_l] += temp_bottom[:, mask_top_l]
        _segMask = fillHole(segMask[None])[0]

        _maskl[i] = _segMask

        if i < max(max_seg):
            f = first_nonzero(None, _segMask, axis=0, invalid_val=0)
            l_temp = last_nonzero(None, _segMask, axis=0, invalid_val=m - 1)

            bottom_labeled = measure.label(temp_bottom, connectivity=2)
            top_labeled = measure.label(temp_top, connectivity=2)

            boundary_top_ind = top_labeled[f, np.arange(n)]
            boundary_bottom_ind = bottom_labeled[l_temp, np.arange(n)]

            boundary_patch_bottom = bottom_labeled == boundary_bottom_ind[None, :]
            boundary_patch_top = top_labeled == boundary_top_ind[None, :]

            num_bottom = boundary_patch_bottom.sum(0)
            num_top = boundary_patch_top.sum(0)
            error_bottom = ((topF[i] > bottomF[i]) * boundary_patch_bottom).sum(0)
            error_top = ((topF[i] < bottomF[i]) * boundary_patch_top).sum(0)
            boundary_bottom_ind[
                ~(
                    (error_bottom < 11)
                    * (
                        num_bottom
                        / np.clip(((temp_bottom + temp_top) * _segMask).sum(0), 1, None)
                        < 0.2
                    )
                )
            ] = (boundary_bottom_ind.max() + 1)
            boundary_top_ind[
                ~(
                    (error_top < 11)
                    * (
                        num_top
                        / np.clip(((temp_bottom + temp_top) * _segMask).sum(0), 1, None)
                        < 0.2
                    )
                )
            ] = (boundary_top_ind.max() + 1)
            temp_bottom = (bottom_labeled == boundary_bottom_ind[None, :]) ^ temp_bottom
            temp_top = (top_labeled == boundary_top_ind[None, :]) ^ temp_top
            temp_bottom = morphology.remove_small_objects(temp_bottom, 121)
            temp_top = morphology.remove_small_objects(temp_top, 121)

            boundaryCoordsBottom = last_nonzero(
                None, temp_bottom, axis=0, invalid_val=0
            )
            boundaryCoordsTop = first_nonzero(None, temp_top, axis=0, invalid_val=m - 1)

            segMask = (temp >= boundaryCoordsTop) * ((temp <= boundaryCoordsBottom))
            mask = (segMask.sum(0) > 0) ^ segMaskk
            A = (boundaryCoordsBottom == 0) + (boundaryCoordsTop == (m - 1))
            boundaryCoordsBottom[A] = 0
            boundaryCoordsTop[A] = m - 1
            boundaryCoordsTop = boundaryCoordsTop[
                scipy.ndimage.distance_transform_edt(
                    boundaryCoordsTop == (m - 1),
                    return_distances=False,
                    return_indices=True,
                )
            ][
                0
            ]  # [(segMask.sum(0)>0)^segMaskk]
            boundaryCoordsBottom = boundaryCoordsBottom[
                scipy.ndimage.distance_transform_edt(
                    boundaryCoordsBottom == 0,
                    return_distances=False,
                    return_indices=True,
                )
            ][
                0
            ]  # [(segMask.sum(0)>0)^segMaskk]
            segMask[:, mask] = (
                (temp >= boundaryCoordsTop) * ((temp <= boundaryCoordsBottom))
            )[:, mask]
            segMask += outlierFilling(segMask)
            segMask[:, mask_bottom_l] += temp_top[:, mask_bottom_l]
            segMask[:, mask_top_l] += temp_bottom[:, mask_top_l]
            segMask = fillHole(segMask[None])[0] * _segMask

            _maskm[i] = segMask
        else:
            pass
    t = (np.arange(s)[:, None] >= np.array(max_seg)[None, :])[:, None, :].repeat(m, 1)
    _mask[t] = _maskl[t]
    _mask[~t] = _maskm[~t]
    if max(max_seg) > 0:
        for i in tqdm.tqdm(range(0, s), desc="refine pair-wise segmentation result: "):
            _mask[i] += outlierFilling(_mask[i])
    if _xy is False:
        return fillHole(_mask.transpose(1, 2, 0))
    else:
        _mask_small_tmp = _mask[:, :-1:2, :-1:2]

        _mask_small = np.zeros(
            (s, _mask_small_tmp.shape[1] * 2, _mask_small_tmp.shape[2] * 2), dtype=bool
        )
        _mask_small[:, ::2, ::2] = _mask_small_tmp

        with tqdm.tqdm(
            total=((_mask_small.shape[1] - 1) // 10 + 1)
            * ((_mask_small.shape[2] - 1) // 10 + 1),
            desc="refine along z: ",
            leave=False,
        ) as pbar:
            for i in range((_mask_small.shape[1] - 1) // 10 + 1):
                for j in range((_mask_small.shape[2] - 1) // 10 + 1):
                    _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = (
                        morphology.remove_small_objects(
                            _mask_small[
                                :, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10
                            ],
                            5,
                        )
                    )
                    pbar.update(1)
        r = copy.deepcopy(_mask_small[:, ::2, ::2])
        _mask_small[:] = 1
        _mask_small[:, ::2, ::2] = r

        with tqdm.tqdm(
            total=((_mask_small.shape[1] - 1) // 10 + 1)
            * ((_mask_small.shape[2] - 1) // 10 + 1),
            desc="refine along z: ",
        ) as pbar:
            for i in range((_mask_small.shape[1] - 1) // 10 + 1):
                for j in range((_mask_small.shape[2] - 1) // 10 + 1):
                    _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = (
                        morphology.remove_small_holes(
                            _mask_small[
                                :, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10
                            ],
                            5,
                        )
                    )
                    pbar.update(1)
        _mask[:, : _mask_small_tmp.shape[1] * 2, : _mask_small_tmp.shape[2] * 2] = (
            np.repeat(np.repeat(_mask_small[:, ::2, ::2], 2, 1), 2, 2)
        )

        _mask[:] = fillHole(_mask)
        return _mask


def fillHole(segMask):
    z, h, w = segMask.shape
    h += 2
    w += 2
    result = np.zeros(segMask.shape, dtype=bool)
    for i in range(z):
        _mask = np.pad(segMask[i], ((1, 1), (1, 1)))
        im_floodfill = 255 * (_mask.astype(np.uint8)).copy()
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(im_floodfill, mask, seedPoint=(0, 0), newVal=255)
        result[i, :, :] = (segMask[i] + (~im_floodfill)[1:-1, 1:-1]).astype(bool)
    return result


def sgolay2dkernel(
    window_size,
    order,
):
    # n_terms = (order + 1) * (order + 2) / 2.0
    half_size = window_size // 2
    exps = []
    for row in range(order[0] + 1):
        for column in range(order[1] + 1):
            if (row + column) > max(*order):
                continue
            exps.append((row, column))
    indx = np.arange(-half_size[0], half_size[0] + 1, dtype=np.float64)
    indy = np.arange(-half_size[1], half_size[1] + 1, dtype=np.float64)
    dx = np.repeat(indx, window_size[1])
    dy = np.tile(indy, [window_size[0], 1]).reshape(
        window_size[0] * window_size[1],
    )
    A = np.empty((window_size[0] * window_size[1], len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])
    return np.linalg.pinv(A)[0].reshape((window_size[0], -1))


def first_nonzero(
    arr,
    mask,
    axis,
    invalid_val=np.nan,
):
    if mask is None:
        mask = arr != 0
    if type(mask) is not np.ndarray:
        mask = mask.cpu().detach().numpy()
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(
    arr,
    mask,
    axis,
    invalid_val=np.nan,
):
    if mask is None:
        mask = arr != 0
    if type(mask) is not np.ndarray:
        mask = mask.cpu().detach().numpy()
    val = mask.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)
