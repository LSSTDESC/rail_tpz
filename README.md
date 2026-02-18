# pz-rail-tpz

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/LSSTDESC/pz-rail-tpz/branch/main/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/pz-rail-tpz)
[![PyPI](https://img.shields.io/pypi/v/rail_tpz?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/rail_tpz/)

"Lite" version of Matias Carrasco-Kind's TPZ (Trees for Photo-z) regression-tree-based photo-z code.  This initial version **only** implements the regression-tree mode, it does not implement the classification tree or SOM-based photo-z estimators.  All credit on algorithm development and initial coding goes to Matias Carrasco-Kind.

If you use TPZ for any publication, in addition to RAIL, you should cite Matias' TPZ paper:
Carrasco Kind, M., & Brunner, R. J., 2013 “TPZ : Photometric redshift PDFs and ancillary information by using prediction trees and random forests”, MNRAS, 432, 1483 [Link](https://ui.adsabs.harvard.edu/abs/2013MNRAS.432.1483C/abstract)

For more details on the algorith, see Matias's MLZ website:
http://matias-ck.com/mlz/

For the regression tree mode, the current implementation includes generation of "random" data via Gaussian scatter on each of the attributes that contain an uncertainty, but it does **not** implement the out-of-bag error or varImportance sampling that are included in the full MLZ/TPZ package.

## RAIL: Redshift Assessment Infrastructure Layers

This package is part of the larger ecosystem of Photometric Redshifts
in [RAIL](https://github.com/LSSTDESC/RAIL).

### Citing RAIL

RAIL is open source and may be used according to the terms of its [LICENSE](https://github.com/LSSTDESC/RAIL/blob/main/LICENSE) [(BSD 3-Clause)](https://opensource.org/licenses/BSD-3-Clause).
If you used RAIL in your study, please cite this repository <https://github.com/LSSTDESC/RAIL>, and RAIL Team et al. (2025) <https://arxiv.org/abs/2505.02928>
```
@ARTICLE{2025arXiv250502928T,
       author = {{The RAIL Team} and {van den Busch}, Jan Luca and {Charles}, Eric and {Cohen-Tanugi}, Johann and {Crafford}, Alice and {Crenshaw}, John Franklin and {Dagoret}, Sylvie and {De-Santiago}, Josue and {De Vicente}, Juan and {Hang}, Qianjun and {Joachimi}, Benjamin and {Joudaki}, Shahab and {Bryce Kalmbach}, J. and {Kannawadi}, Arun and {Liang}, Shuang and {Lynn}, Olivia and {Malz}, Alex I. and {Mandelbaum}, Rachel and {Merz}, Grant and {Moskowitz}, Irene and {Oldag}, Drew and {Ruiz-Zapatero}, Jaime and {Rahman}, Mubdi and {Rau}, Markus M. and {Schmidt}, Samuel J. and {Scora}, Jennifer and {Shirley}, Raphael and {St{\"o}lzner}, Benjamin and {Toribio San Cipriano}, Laura and {Tortorelli}, Luca and {Yan}, Ziang and {Zhang}, Tianqing and {the Dark Energy Science Collaboration}},
        title = "{Redshift Assessment Infrastructure Layers (RAIL): Rubin-era photometric redshift stress-testing and at-scale production}",
      journal = {arXiv e-prints},
     keywords = {Instrumentation and Methods for Astrophysics, Cosmology and Nongalactic Astrophysics, Astrophysics of Galaxies},
         year = 2025,
        month = may,
          eid = {arXiv:2505.02928},
        pages = {arXiv:2505.02928},
          doi = {10.48550/arXiv.2505.02928},
archivePrefix = {arXiv},
       eprint = {2505.02928},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250502928T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
Please consider also inviting the developers as co-authors on publications resulting from your use of RAIL by [making an issue](https://github.com/LSSTDESC/rail/issues/new/choose).
A convenient list of what to cite may be found under [Citing RAIL](https://rail-hub.readthedocs.io/en/latest/source/citing.html) on ReadTheDocs.
Additionally, several of the codes accessible through the RAIL ecosystem must be cited if used in a publication.

### Citing this package

If you use this package, you should also cite the appropriate papers for each
code used.  A list of such codes is included in the 
[Citing RAIL](https://lsstdescrail.readthedocs.io/en/stable/source/citing.html)
section of the main RAIL Read The Docs page.
