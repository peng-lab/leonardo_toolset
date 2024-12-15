# Leonardo: a toolset to remove sample-induced aberrations in light sheet microscopy images 

[![Build Status](https://github.com/peng-lab/leonardo/workflows/Build%20Main/badge.svg)](https://github.com/peng-lab/leonardo/actions)
---

*Leonardo* is a toolbox able to resolve all sample-induced aberrations in selective plane illumination microscopy (SPIM, also called light-sheet fluorescence microscopy, LSFM) by using two major modules: (1) **DeStripe** removes the stripe artifacts caused by light absorption; (2) **FUSE** reconstructs one single high-quality image from dual-sided illumination and/or dual-sided detection while eliminating optical distortions (ghosts) caused by light refraction. 

## Documentation and Tutorials:

https://leonardo-lsfm.readthedocs.io/en/latest/installation.html

## Installation

**Stable Release:** `pip install leonardo`<br>
**Development Head:** `pip install git+https://github.com/peng-lab/leonardo.git`

**Full software including napari plugins:** `pip install leonardo[napari]`<br>

This **Leonardo** package contains everything you need for using the toolset, including batch processing support for different workflows, such as running destriping then fusion. If you need only one specific component, you can find more info below:

<details>
    <summary>More details about the Leonardo package</summary>

    ## packages for core componets:

    * Leonardo-DeStripe: https://github.com/peng-lab/lsfm_destripe
    * Leonardo-FUSE: https://github.com/peng-lab/lsfm_fuse

    ## packages for napari plugins:

    * plugin for Leonardo-DeStripe: https://github.com/peng-lab/lsfm_destripe_napari 
    * plugin for Leonardo-FUSE: https://github.com/peng-lab/lsfm_fusion_napari

</details>


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT license**

