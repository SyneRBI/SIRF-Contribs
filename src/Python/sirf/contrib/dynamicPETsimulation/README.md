# Batch Dynamic PET Simulator

This contribution was migrated from the `data_simulation` directory on the
`yiming` branch of
[`KrisThielemans/Using-Deep-Learning-to-correct-parametric-images-for-motion-in-dynamic-PET`](https://github.com/KrisThielemans/Using-Deep-Learning-to-correct-parametric-images-for-motion-in-dynamic-PET/tree/yiming/data_simulation).

The simulation design was informed by earlier work by William Wei and Haoran
Lu, but this contribution does not copy their source code.

See [SIMULATION.md](SIMULATION.md) for a detailed description of the simulation methodology.

## Requirements

Run the simulator in a Python environment with SIRF/STIR already installed and configured. SIRF/STIR is not installed automatically by this project.

Install SIRF-Contribs from the repository root directory:

```bash
python -m pip install .
```

For an editable development installation, use `python -m pip install -e .`.

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
