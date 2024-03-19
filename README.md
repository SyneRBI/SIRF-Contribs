# SIRF-Contribs
Users contributions to SIRF.

This space is for contributions that are not quite ready yet to be incorporated into SIRF's main repository. We welcome the sharing of any contributions that might help other researchers. We hope that widely used features can then be finalised, and then added to the main SIRF repository.

Please add your contributions as a pull request. 

- Jupyter notebooks can be added to [./src/notebooks](./src/notebooks). 
- Python code should be added to [./src/Python/sirf/contrib](./src/Python/sirf/contrib) with an `__init__.py` file. In this fashion, once a user has checked out this repository, they can use it in their code with, for example:

```
from sirf.contrib import kcl
```

# Current contents

- [KCL priors](./src/Python/sirf/contrib/kcl) - de Pierro, Bowsher, etc. priors
- [brainweb](./src/Python/sirf/contrib/brainweb-utilities) - preliminary script to create brainweb data with some extra features.
- [MCIR](./src/Python/sirf/contrib/MCIR) - scripts for the MR and PET MCIR reconstructions.
- [Grappa_and_CIL](./src/notebooks/Grappa_and_CIL.ipynb) - notebook demonstrating CIL integration with SIRF, with the use case of a GRAPPA reconstruction of MR data.
- [BSREM](./src/Python/sirf/contrib/BSREM) example functions and [BSREM_illustration notebook](./src/notebooks/BSREM_illustration.ipynb) for MAP optimisation with PET (or SPECT).
