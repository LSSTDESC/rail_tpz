# pz-rail-tpz

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/LSSTDESC/pz-rail-tpz/branch/main/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/pz-rail-tpz)
[![PyPI](https://img.shields.io/pypi/v/rail_tpz?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/rail_tpz/)

"Lite" version of Matias Carrasco-Kind's TPZ (Trees for Photo-z) regression-tree-based photo-z code.  This initial version **only** implements the regression-tree mode, it does not implement the classification tree or SOM-based photo-z estimators.

For the regression tree mode, the current implementation includes generation of "random" data via Gaussian scatter on each of the attributes that contain an uncertainty, but it does **not** implement the out-of-bag error or varImportance sampling that are included in the full MLZ/TPZ package.

## RAIL: Redshift Assessment Infrastructure Layers

This package is part of the larger ecosystem of Photometric Redshifts
in [RAIL](https://github.com/LSSTDESC/RAIL).

### Citing RAIL

This code, while public on GitHub, has not yet been released by DESC and is
still under active development. Our release of v1.0 will be accompanied by a
journal paper describing the development and validation of RAIL.

If you make use of the ideas or software in RAIL, please cite the repository 
<https://github.com/LSSTDESC/RAIL>. You are welcome to re-use the code, which
is open source and available under terms consistent with the MIT license.

External contributors and DESC members wishing to use RAIL for non-DESC projects
should consult with the Photometric Redshifts (PZ) Working Group conveners,
ideally before the work has started, but definitely before any publication or 
posting of the work to the arXiv.

### Citing this package

If you use this package, you should also cite the appropriate papers for each
code used.  A list of such codes is included in the 
[Citing RAIL](https://lsstdescrail.readthedocs.io/en/stable/source/citing.html)
section of the main RAIL Read The Docs page.
