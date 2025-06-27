#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This sample script will get deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""

import argparse
import logging
import sys
import traceback

from leonardo_toolset.fusion import FUSE_illu, get_module_version

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

###############################################################################


def list_of_floats(arg):
    return list(map(int, arg.split(",")))


def bool_args(arg):
    if ("false" == arg) or ("False" == arg):
        return False
    elif ("true" == arg) or ("True" == arg):
        return True


class Args(argparse.Namespace):

    DEFAULT_FIRST = 10
    DEFAULT_SECOND = 20

    def __init__(self):
        # Arguments that could be passed in through the command line
        self.first = self.DEFAULT_FIRST
        self.second = self.DEFAULT_SECOND
        self.debug = False
        #
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(
            prog="run_dualillufuse",
            description="run dualillufuse for LSFM images",
        )

        p.add_argument(
            "-v",
            "--version",
            action="version",
            version="%(prog)s " + get_module_version(),
        )

        p.add_argument(
            "--require_precropping",
            type=bool_args,
            default="True",
        )

        p.add_argument(
            "--precropping_params",
            type=list_of_floats,
            action="store",
            default=[],
        )

        p.add_argument(
            "--resample_ratio",
            action="store",
            dest="resample_ratio",
            default=2,
            type=int,
        )

        p.add_argument(
            "--window_size",
            type=list_of_floats,
            action="store",
            default=[5, 59],
        )

        p.add_argument(
            "--poly_order",
            type=list_of_floats,
            action="store",
            default=[2, 2],
        )

        p.add_argument(
            "--n_epochs",
            action="store",
            dest="n_epochs",
            default=50,
            type=int,
        )

        p.add_argument(
            "--require_segmentation",
            type=bool_args,
            default="True",
        )

        p.add_argument(
            "--device",
            action="store",
            dest="device",
            default="cuda",
            type=str,
        )

        p.add_argument(
            "--data_path",
            action="store",
            dest="data_path",
            type=str,
        )

        p.add_argument(
            "--sample_name",
            action="store",
            dest="sample_name",
            type=str,
        )

        p.add_argument(
            "--top_illu_data",
            action="store",
            dest="top_illu_data",
            default=None,
            type=str,
        )

        p.add_argument(
            "--bottom_illu_data",
            action="store",
            dest="bottom_illu_data",
            default=None,
            type=str,
        )

        p.add_argument(
            "--left_illu_data",
            action="store",
            dest="left_illu_data",
            default=None,
            type=str,
        )

        p.add_argument(
            "--right_illu_data",
            action="store",
            dest="right_illu_data",
            default=None,
            type=str,
        )

        p.add_argument(
            "--save_path",
            action="store",
            dest="save_path",
            type=str,
        )

        p.add_argument(
            "--save_folder",
            action="store",
            dest="save_folder",
            type=str,
        )

        p.add_argument(
            "--camera_position",
            action="store",
            dest="camera_position",
            default="",
            type=str,
        )

        p.add_argument(
            "--cam_pos",
            action="store",
            dest="cam_pos",
            default="front",
            type=str,
        )

        p.add_argument(
            "--sparse_sample",
            type=bool_args,
            default="False",
        )

        p.add_argument(
            "--save_separate_results",
            type=bool_args,
            default="False",
        )

        p.add_argument(
            "--debug",
            action="store_true",
            dest="debug",
            help=argparse.SUPPRESS,
        )
        p.parse_args(namespace=self)


###############################################################################


def main():
    try:
        args = Args()
        dbg = args.debug

        exe = FUSE_illu(
            args.require_precropping,
            args.precropping_params,
            args.resample_ratio,
            args.window_size,
            args.poly_order,
            args.n_epochs,
            args.require_segmentation,
            args.device,
        )
        _ = exe.train(
            args.data_path,
            args.sample_name,
            args.top_illu_data,
            args.bottom_illu_data,
            args.left_illu_data,
            args.right_illu_data,
            args.save_path,
            args.save_folder,
            args.save_separate_results,
            args.sparse_sample,
            args.cam_pos,
            args.camera_position,
        )

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
