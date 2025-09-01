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
import json
import re
from leonardo_toolset.destripe import DeStripe, get_module_version

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

###############################################################################


def list_of_floats(arg):
    return list(map(float, arg.split(",")))


def bool_args(arg):
    if ("false" == arg) or ("False" == arg):
        return False
    elif ("true" == arg) or ("True" == arg):
        return True


def _smart_cast(s):
    sl = str(s).lower()
    if sl in {"true", "t", "yes", "y", "1"}:
        return True
    if sl in {"false", "f", "no", "n", "0"}:
        return False
    if re.fullmatch(r"[+-]?\d+", str(s)):
        return int(s)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+)", str(s)):
        return float(s)
    try:
        return json.loads(s)
    except Exception:
        return s


def parse_unknown_as_kwargs(unknown_tokens):
    out = {}
    i = 0
    while i < len(unknown_tokens):
        tok = unknown_tokens[i]
        if tok.startswith("--"):
            if "=" in tok:
                k, v = tok[2:].split("=", 1)
                values = [v]
                i += 1
            else:
                k = tok[2:]
                values = []
                j = i + 1
                while j < len(unknown_tokens) and not unknown_tokens[j].startswith(
                    "--"
                ):
                    values.append(unknown_tokens[j])
                    j += 1
                if not values:
                    out[k] = True
                    i += 1
                    continue
                i = j

            if k.startswith("angle_offset"):

                if len(values) == 1 and "," in values[0]:
                    out[k] = list_of_floats(values[0])
                else:
                    out[k] = list_of_floats(",".join(values))
            elif k.startswith("illu_orient"):
                if len(values) != 1:
                    raise argparse.ArgumentTypeError(f"{k} expects a single value.")
                out[k] = values[0]
            elif k.startswith("x_"):
                if len(values) != 1:
                    raise argparse.ArgumentTypeError(f"{k} expects a single path.")
                out[k] = values[0]
            else:
                out[k] = (
                    _smart_cast(values[0])
                    if len(values) == 1
                    else [_smart_cast(v) for v in values]
                )
        else:
            i += 1
    return out


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
            prog="run_destripe",
            description="run destripe for LSFM images",
        )

        p.add_argument(
            "-v",
            "--version",
            action="version",
            version="%(prog)s " + get_module_version(),
        )

        p.add_argument(
            "--resample_ratio",
            action="store",
            dest="resample_ratio",
            default=3,
            type=int,
        )

        p.add_argument(
            "--guided_upsample_kernel",
            action="store",
            dest="guided_upsample_kernel",
            default=49,
            type=int,
        )

        p.add_argument(
            "--hessian_kernel_sigma",
            action="store",
            dest="hessian_kernel_sigma",
            default=1,
            type=float,
        )

        p.add_argument(
            "--lambda_masking_mse",
            action="store",
            dest="lambda_masking_mse",
            default=1,
            type=float,
        )

        p.add_argument(
            "--lambda_tv",
            action="store",
            dest="lambda_tv",
            default=1,
            type=float,
        )

        p.add_argument(
            "--lambda_hessian",
            action="store",
            dest="lambda_hessian",
            default=1,
            type=float,
        )

        p.add_argument(
            "--inc",
            action="store",
            dest="inc",
            default=16,
            type=int,
        )

        p.add_argument(
            "--n_epochs",
            action="store",
            dest="n_epochs",
            default=300,
            type=int,
        )

        p.add_argument(
            "--wedge_degree",
            action="store",
            dest="wedge_degree",
            default=29,
            type=float,
        )

        p.add_argument(
            "--n_neighbors",
            action="store",
            dest="n_neighbors",
            default=16,
            type=int,
        )

        p.add_argument(
            "--backend",
            action="store",
            dest="backend",
            default="jax",
            type=str,
        )

        p.add_argument(
            "--device",
            action="store",
            dest="device",
            default=None,
            type=str,
        )

        # .train()

        p.add_argument(
            "--is_vertical",
            action="store",
            dest="is_vertical",
            type=bool_args,
            default=None,
        )

        p.add_argument(
            "--x",
            action="store",
            dest="x",
            default=None,
            type=str,
        )

        p.add_argument(
            "--mask",
            action="store",
            dest="mask",
            default=None,
            type=str,
        )

        p.add_argument(
            "--fusion_mask",
            action="store",
            dest="fusion_mask",
            default=None,
            type=str,
        )

        p.add_argument(
            "--illu_orient",
            action="store",
            dest="illu_orient",
            default=None,
            type=str,
        )

        p.add_argument(
            "--angle_offset",
            type=list_of_floats,
            action="store",
            default=None,
        )

        p.add_argument(
            "--display",
            type=bool_args,
            default=False,
        )

        p.add_argument(
            "--display_angle_orientation",
            type=bool_args,
            default=False,
        )

        p.add_argument(
            "--non_positive",
            type=bool_args,
            default=False,
        )

        p.add_argument(
            "--debug",
            action="store_true",
            dest="debug",
            help=argparse.SUPPRESS,
        )

        p.add_argument(
            "--save_path",
            action="store",
            dest="save_path",
            type=str,
            required=True,
        )

        ns, unknown = p.parse_known_args(namespace=self)
        self.kwargs = parse_unknown_as_kwargs(unknown)


###############################################################################


def main():
    try:
        args = Args()
        dbg = args.debug

        exe = DeStripe(
            args.resample_ratio,
            args.guided_upsample_kernel,
            args.hessian_kernel_sigma,
            args.lambda_masking_mse,
            args.lambda_tv,
            args.lambda_hessian,
            args.inc,
            args.n_epochs,
            args.wedge_degree,
            args.n_neighbors,
            args.backend,
            args.device,
        )
        _ = exe.train(
            args.save_path,
            args.is_vertical,
            args.x,
            args.mask,
            args.fusion_mask,
            args.illu_orient,
            args.angle_offset,
            args.display,
            args.display_angle_orientation,
            args.non_positive,
            **getattr(args, "kwargs", {}),
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
