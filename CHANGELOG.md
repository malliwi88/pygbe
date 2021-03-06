# PyGBe Change Log
----
## Current development
---

### Added

### Changed

### Fixed

### Removed

## 0.2.1
---
### Added

* All documentation is available on github pages
* Support for Cuda 7.5
* One liner for setting up Py2.7 environment (sans PyCUDA)

### Changed
* `config` and `param` files are now globbed for so they can have a name different
  than the folder which contains them.  
* Updated license with new contributors

### Fixed

* Uncaught exception when pygbe doesnt run correctly
* Wrong error type in regression test master script
* Performance runs work on non-X backends

### Removed

## 0.2
---
### Added
* `setup.py` installer
* `argparse` ArgumentParser to handle command line arguments (all optional)
  * `-c` to specify config file
  * `-p` to specify param file
  * `-o` to specify output folder
  * `-g` to specify geometry folder
* Docstrings to all functions
* Checks for NVCC version and to warn if user doesn't have NVCC on PATH
* Sphinx documentation
* In addition to text output, numerical results are stored to a pickled dictionary for easy access
  

### Changed
* Repo structure altered to match Python packaging guidelines.
* Modularized code and removed all relative imports
* All `import *` (excepting files in `scripts/`) have been removed and changed to explicit imports
* Problems are now grouped-by-folder.  A given problem will have the format:
```
lys 
  ˫ lys.param
  ˫ lys.config
  ˫ built_parse.pqr
  ˫ geometry/Lys1.face
  ˫ geometry/Lys1.vert
  ˫ output/

* Support running in current directory by passing '.' as path
```
* Refactored regression tests, added simple caching to avoid test repeats
* Move many, many functions around so that individual `.py` filenames are more descriptive and accurate


### Removed
* Makefiles (functionality replaced by `setup.py`)
* `pygbe_matrix` and `scripts` folder -- to be relocated to a more appropriate repo somewhere
