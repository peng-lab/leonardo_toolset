Tutorials
=========
This section contains various tutorials showcasing light-sheet data processing with :mod:`Leonardo`.

Basic Tutorials
----------------
This section contains notebook tutorials showcasing how to do stripe removal using :mod:`Leonardo-DeStripe`, 
how to fuse datasets with dual-sided illumination and/or dual-sided detection using :mod:`Leonardo-FUSE`, 
and how to build an optimal workflow to resolve sample-induced aberrations using :mod:`Leonardo-DeStripe-FUSE`.

.. toctree::
    :maxdepth: 1

    tutorials/notebook/Leonardo_DeStripe
    tutorials/notebook/Leonardo_FUSE_along_illumination
    tutorials/notebook/Leonardo_FUSE_along_detection
    tutorials/notebook/Leonardo_FUSE_DeStripe

Advanced Tutorials
--------------------
This section contains video tutorials showcasing how to use our Napari interface to easily realize aforementioned functions.

.. video:: tutorials/video/destripe.mp4
    :width: 100%
    :caption: Leonardo-DeStripe in Napari
    :align: center

.. video:: tutorials/video/fuse-ill.mp4
    :width: 100%
    :caption: Leonardo-Fuse (along illumination) in Napari
    :align: center

.. video:: tutorials/video/fuse-det.mp4
    :width: 100%
    :caption: Leonardo-Fuse (along detection) in Napari
    :align: center