#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging

# Third party

# Relative

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class workflow_wrapper(object):
    """
    The wrapper for different workflows

    Parameters
    ----------
    workflow_type: str
        the type of workflow to execute
    input_dir: str or Path
        input path
    output_dir: str or Path
        output path
    """

    def __init__(self, workflow_type, input_dir, output_dir):
        # TODO: currently, it is a placeholder
        print(workflow_type)
        print(input_dir)
        print(output_dir)

    def process(self):
        # TODO: currently, it is a placeholder
        if self.workflow_type == "destripe_fuse":
            print(self.input_dir)
            print(self.output_dir)
