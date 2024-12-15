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

from leonardo import workflow_wrapper, get_module_version

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

###############################################################################


class Args(argparse.Namespace):

    def __init__(self):
        # Arguments that could be passed in through the command line
        self.debug = False
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(
            prog="run_leonardo_workflows",
            description="batch running for leonardo workflows",
        )

        p.add_argument(
            "-v",
            "--version",
            action="version",
            version="%(prog)s " + get_module_version(),
        )
        p.add_argument(
            "--input",
            dest="input_dir",
            required=True,
            help="path to the input folder",
        )
        p.add_argument(
            "--output",
            dest="output_dir",
            required=True,
            help="path to save the results",
        )
        p.add_argument(
            "--workflow",
            dest="workflow_type",
            default="destripe_fuse",
            help=(
                "select the type of workflow: destripe_fuse (default), "
                "destripe_only, fuse_only"
            ),
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

        exe = workflow_wrapper(args.workflow_type, args.input_dir, args.output_dir)
        exe.process()

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
