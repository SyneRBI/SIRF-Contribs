# MCIR

The code in this folder was used for the publication on using SIRF for PET/MR MCIR (DOI TBC):

*Motion estimation and correction for simultaneous PET/MR using SIRF*<br>
Richard Brown, Christoph Kolbitsch, Claire Delplancke, Evangelos Papoutsellis, Johannes Mayer, Evgueni Ovtchinnikov, Edoardo Pasca, Radhouene Neji, Casper da Costa-Luis, Ashley G. Gillman, Matthias J. Ehrhardt, Jamie McClelland, Bjoern Eiben, and Kris Thielemans<br>
Special issue on Synergistic Image Reconstruction (Part 2), Philosophical Transactions of the Royal Society A

It contains two main scripts, detailed below, for the MR and PET MCIR reconstructions, respectively.

## MR MCIR
The script is located here: [MR\_recon\_file.py](./path_to_MR_recon.py)

- The results of this script are given in Section 3a of the publication.

## PET MCIR
The script is located here: [PET\_recon\_file.py](./PET_recon_file.py)

- PET MCIR using PDHG and SPDHG algorithms. For SPDHG, the subsetting takes place over the motion gates, although an untested implementation of subsetting over the views is also included. The results of this script is given in Section 3b of the publication.

## Code dependencies

At the time of writing, these scripts used some features that were not quite in a finalised state such that they could be incorporated into the main branches of the SIRF and CIL repositories. We hope that over time, these will be merged in, simplifying the following steps somewhat.

The best bet is to checkout a commit of the SIRF-SuperBuild that was used at the time of writing, as this will install as many dependencies as possible with matching versions that were used. This can be done with:

```
git checkout https://github.com/SyneRBI/SIRF-SuperBuild.git
cd SIRF-SuperBuild
git checkout 0365a51041f885e652d560ad7d4848555ad19fe0
mkdir ~/devel/build
cd ~/devel/build
cmake ../SIRF-SuperBuild
make
```

As per the rest of SIRF, this code should work all versions of python3.x, which has CI/CD for various python versions.

**N.B.**: Unfortunately, there is no guarantee that this will work, as we cannot guarantee that old versions of SIRF's dependencies will successfully build on newer versions of operating systems.


### PET dependencies

  - SIRF branch [pet\_am\_subset](https://github.com/SyneRBI/SIRF/tree/pet_am_subset) – adds functionality for projecting a subset of views. This functionality is not currently used in the MCIR process, so perhaps the current `master` branch would work.
  - CIL tag [20.11.1](https://github.com/vais-ral/CCPi-Framework/releases/tag/v20.11.1) 
  - SIRF-SuperBuild branch [update\_cil\_2011](https://github.com/SyneRBI/SIRF-SuperBuild/tree/update_cil_2011): Gadgetron is not built.

### MR dependencies

With the SuperBuild successfully installed, the MR code requires the current branches to be checked out:

- SIRF branch [rpe-encoding](https://github.com/SyneRBI/SIRF/tree/rpe-encoding): enables handling of non-Cartesian k-space data.
- CIL tag [20.04](https://github.com/vais-ral/CCPi-Framework/releases/tag/v20.04)
