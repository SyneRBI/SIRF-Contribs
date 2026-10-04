# Batch Dynamic PET Simulator

This contribution is based on work described in the following MSc reports:

- Miao Su, “Using Deep Learning to Correct Parametric Images for Motion in
  Dynamic PET,” Master's report, University College London, 2026.
- Yiming Li, “Deep Learning-Based Six-Degree-of-Freedom Rigid Motion
  Estimation for Simulated Dynamic PET,” MSc report, 2026.

We thank William Wei for his earlier work, which informed the design of this
simulation pipeline. The design was also informed by earlier work by Haoran Lu,
but this contribution does not copy source code from either work.

See [SIMULATION.md](SIMULATION.md) for a detailed description of the simulation methodology.

## Requirements

SIRF with STIR support must already be installed and configured before the
Python dependencies below are installed. This contribution does not install
SIRF/STIR automatically.

From the repository root directory, install the Python dependencies with:

```bash
python -m pip install -r src/Python/sirf/contrib/dynamicPETsimulation/requirements.txt
```

## Included input files

The example configuration refers to the following files, which are included in
this directory:

```text
input/AIF_FDG.mat
input/XCAT_Mask_400x400x900_act_NoLes.mat
input/XCAT_mask_look_up_table.txt
```

These paths are interpreted relative to the current working directory.
Absolute paths can instead be specified in the JSON configuration file.

## Running the simulator

The simulator source, example configuration, and input data have been preserved
unchanged from the source repository. The command-line interface and its
available options are defined in `pet_sim/cli.py`.

Exact invocation examples are not included here because the original entry
point has not been modified or revalidated as part of this documentation-only
migration. Use the invocation already configured for your SIRF environment.
When invoking the simulator from a different working directory, use absolute
input paths in the configuration file.
