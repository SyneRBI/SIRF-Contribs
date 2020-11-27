# MCIR

The code in this folder was used for the publication on using SIRF for PET/MR MCIR (DOI).

It contains two main scripts, detailed below, for the MR and PET MCIR reconstructions, respectively.

## MR MCIR
The script is located here: [MR\_recon\_file.py](./path_to_MR_recon.py)

- The results of this script are given in Section 3a of the publication.

## PET MCIR
The script is located here: [PET\_recon\_file.py](./path_to_MR_recon.py)

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

**N.B.**: Unfortunately, there is no guarantee that this will work, as we cannot guarantee that old versions of SIRF's dependencies will successfully build on newer versions of operating systems.

### MR dependencies

With the SuperBuild successfully installed, the MR code requires the current branches to be checked out:

### MR dependencies

With the SuperBuild successfully installed, the PET code requires the current branches to be checked out: