# Dynamic PET Simulation Method

This simulator generates dynamic FDG PET data with known inter-frame rigid
motion. It combines an XCAT anatomical label volume [1], tissue-specific tracer
kinetics, PET acquisition modelling, count noise, iterative reconstruction,
and Patlak analysis in a reproducible batch pipeline.

This document is adapted from the project's original *Data Simulation*
chapter. Implementation-specific details were checked against the contributed
code and `config.example.json`.

The processing stages are:

1. crop and randomly reposition an XCAT label volume;
2. place a synthetic lesion inside the configured target organ;
3. generate tissue and lesion time–activity curves (TACs) with a
   two-tissue-compartment model (2TCM);
4. construct frame-averaged, noise-free activity volumes and a static
   attenuation map;
5. apply persistent rigid-motion events to the activity and attenuation
   volumes;
6. forward project each activity frame with its motion-matched attenuation
   map, add a uniform background, and generate Poisson-noisy projection data;
7. reconstruct each frame with OSEM while using the static reference
   attenuation map; and
8. fit reconstructed Patlak maps and save both reconstruction-derived and
   formula-derived parametric images.

All quantities described as *example values* below come from
`config.example.json`. They are configuration choices, not fixed properties of
the implementation.

## Inputs and spatial conventions

The simulation reads three inputs:

- an arterial input function (AIF) from a MATLAB file containing an `aif`
  structure with time samples `tt`, interpreted in minutes, and concentration
  samples `dat`;
- a three-dimensional XCAT tissue-label array named `xcat_dat`, stored in
  `(x, y, z)` order; and
- a lookup table that maps XCAT label values to tissue names and the kinetic
  parameters $K_1$, $k_2$, $k_3$, $k_4$, and $V_b$.

Arrays are converted to `(z, y, x)` order before they are stored or passed to
SIRF. Dynamic arrays use `(t, z, y, x)`, and two-channel parametric arrays use
`(channel, z, y, x)`.

The target-organ label is used to centre the axial crop and to define the
region in which a synthetic lesion can be placed. It does not restrict the
simulation to that organ: all labels inside the cropped field of view remain
available for activity and attenuation assignment. In the example
configuration, the liver (label 13) centres a 127-slice crop of a
`400 x 400 x 900` label volume. The example source voxel dimensions are
`(x, y, z) = (0.203642, 0.203642, 0.2025)` cm, giving a cropped label array of
`(x, y, z) = (400, 400, 127)` before conversion to SIRF order.

## Anatomical variation and lesion generation

Each sample receives one global affine repositioning of the cropped label
volume. The transform consists of an isotropic scale factor, rotations about
the three spatial axes, and translations along those axes. Rotation and
scaling are performed about the volume centre; translation in centimetres is
converted to voxel displacement using the configured voxel sizes. Nearest-
neighbour interpolation preserves the integer labels, and points mapped
outside the source volume are assigned label zero.

This transform is fixed for the complete dynamic sequence. It represents
between-sample anatomical and positioning variation and is distinct from the
frame-wise rigid motion introduced later. The example configuration samples:

- an isotropic scale factor from 0.95 to 1.05;
- translations independently within $\pm(5, 5, 4)$ cm in `(x, y, z)`; and
- rotations independently within $\pm(10, 10, 10)$ degrees about `(x, y, z)`.

After repositioning, a random ellipsoidal lesion is generated inside the
target-organ mask. Its three radii are expressed in voxels. The example uses a
base radius of `(6, 6, 4)` voxels, with an independently sampled multiplier
from 0.2 to 1.2 for each radius. The centre is selected from locations far
enough inside the organ to contain the complete ellipsoid when possible. If
the organ is too small to satisfy that constraint, the ellipsoid is clipped to
the organ mask.

Lesion kinetic parameters are sampled by multiplying configurable base values
by parameter-specific random factors. The example base parameters are
$K_1=1.056$, $k_2=1.029$, $k_3=0.32$, $k_4=0$, and $V_b=0.205$. The lesion
overrides the activity of its underlying tissue and is assigned a configurable
attenuation class (`Body` in the example).

## Tissue kinetics and frame averaging

All tissues use the same interpolated plasma input function $C_p(t)$, while
their kinetic parameters come from the label lookup table. For compartment
concentrations $C_1(t)$ and $C_2(t)$, the implemented 2TCM is

$$
\frac{dC_1(t)}{dt}
= K_1 C_p(t) - (k_2 + k_3)C_1(t) + k_4 C_2(t),
$$

$$
\frac{dC_2(t)}{dt}
= k_3 C_1(t) - k_4 C_2(t).
$$

The tissue TAC includes the vascular fraction:

$$
C_T(t) = (1-V_b)\left[C_1(t)+C_2(t)\right] + V_b C_p(t).
$$

The differential equations are solved from time zero through the end of the
last requested frame. A separate TAC is generated for each distinct set of
lookup-table parameters and for the sampled lesion parameters. If a non-zero
label has no kinetic entry, the default behaviour is to use the configured
replacement tissue (`body_activity` in the example). When replacement is
disabled, those labels receive zero activity.

For frame $n$, spanning $[t_n^s,t_n^e]$, the simulator uses the mean activity
over the full acquisition interval rather than a value sampled at one time:

$$
\bar{C}_{T,j}^{(n)}
= \frac{1}{t_n^e-t_n^s}
  \int_{t_n^s}^{t_n^e} C_{T,j}(t)\,dt.
$$

The activity assigned to voxel $x$ is therefore

$$
I_n(x)=\bar{C}_{T,L(x)}^{(n)},
$$

where $L(x)$ is the transformed XCAT label at that voxel. Voxels in the lesion
mask instead receive the lesion frame average.

The number, duration, and temporal placement of frames are configurable. The
frame start times are evenly spaced between `start_scan_seconds` and the latest
start time that lets the final frame end at `end_scan_seconds`; consequently,
the frames need not be contiguous. The example configuration creates eight
one-minute frames distributed from 20 to 60 minutes after time zero. These
settings produce eight noise-free, motion-free activity volumes with common
anatomy but time-varying tracer distributions.

## Attenuation map

The transformed XCAT labels are grouped into configurable attenuation classes.
The example classes and linear attenuation coefficients are:

| Class | Coefficient (cm⁻¹) |
| --- | ---: |
| Air | 0.0000 |
| Lung | 0.0267 |
| Body | 0.0927 |
| Bone | 0.1305 |

Label zero is treated as air. A non-zero label without an explicit class uses
the configured fallback class (`Body` in the example). The resulting map is
the static, reference-pose attenuation map.

## Frame-wise rigid motion

Inter-frame motion is modelled as six-degree-of-freedom rigid motion: three
translations and three rotations, without scaling or non-rigid deformation.
Frame zero is always the reference frame and cannot contain a motion event.

The number of events is sampled uniformly from the configured inclusive range,
and distinct event frames are sampled from frames 1 through $T-1$. At each
event, translations and rotations are sampled independently from symmetric
ranges about zero. An event changes the pose at its selected frame, and that
change persists in all later frames. If multiple events are enabled, their
transforms are composed, so each later pose contains all preceding changes.
The example configuration uses exactly one event with maximum translations of
`(1, 1, 1)` cm and maximum rotations of `(10, 10, 5)` degrees.

For each frame, the composed pose is converted to the output-to-input backward
mapping expected by SciPy. Rotation is about the volume centre, and translation
is converted from centimetres to axis-specific voxel displacement. If $B_n$
denotes that backward map, the moved activity is

$$
I_n^{\mathrm{motion}}(x)=I_n^{\mathrm{clean}}(B_n(x)).
$$

The composed transform is applied once to each motion-free activity frame,
using linear interpolation and zero outside the source field of view. The same
frame transform is also applied to the static attenuation map. Both the
individual event parameters and the cumulative frame poses are saved as motion
ground truth.

## PET acquisition model and count noise

The SIRF framework [3] and its Python interface to STIR [2] provide the
acquisition geometry, attenuation model, forward projection, and
reconstruction. The example configuration uses a Siemens mMR acquisition
template with span 11 and maximum ring difference 60. The source activity and
attenuation volumes are resampled to the scanner image grid while preserving
voxel values.

Forward projection uses the moved activity volume and the attenuation map
moved by the same frame transform. Thus, the activity and attenuation
distributions are spatially aligned during acquisition simulation. The
implementation uses `AcquisitionModelUsingParallelproj` with attenuation
sensitivity.

When additive background is enabled, one uniform value is calculated for each
frame:

$$
b_n = f_b \max_i\left[(A_{\mu_n}I_n^{\mathrm{motion}})_i\right],
$$

where $f_b$ is `projection.background_fraction`, $A_{\mu_n}$ is the forward
model using the moved attenuation map, and $i$ indexes sinogram bins. The same
$b_n$ is added to every bin:

$$
\bar{y}_n = A_{\mu_n}I_n^{\mathrm{motion}} + b_n\mathbf{1}.
$$

This uniform additive term approximates scatter and random coincidence counts;
those processes are not simulated explicitly. The example background fraction
is 0.05. SIRF's `PoissonNoiseGenerator` then generates noisy projection data
using the configured scaling factor and `preserve_mean=True`. The example
scaling factor is 20. A reproducible seed is
set separately for each frame. Noisy projections can be retained as Interfile
data or deleted after reconstruction.

## OSEM reconstruction and attenuation mismatch

Each noisy dynamic frame is reconstructed independently. The reconstruction
uses a Poisson log-likelihood objective and SIRF's `OSMAPOSLReconstructor`,
initialised with a uniform image. The number of subsets and subiterations are
configurable; the example uses 7 subsets and 63 subiterations.

Unlike the forward model, reconstruction uses the static reference-pose
attenuation map for every frame. The known uniform background for each frame is
also supplied to the reconstruction model when background simulation is
enabled. For frames affected by motion, the attenuation factors used during
reconstruction therefore differ from those used to generate the projection
data. This deliberately introduces the attenuation mismatch that occurs when a
single reference attenuation map is used after the emission anatomy changes
pose.

The reconstructed dynamic sequence consequently contains tracer-kinetic
variation, inter-frame rigid motion, attenuation mismatch, Poisson count noise,
scanner sampling, and finite-iteration reconstruction effects.

## Patlak analysis and learning pairs

Patlak analysis uses only frames completely contained in the configured fit
window. At least two such frames are required. The example fit window is 20 to
60 minutes. For every selected frame, the implementation computes the
frame-average plasma concentration and the frame average of the cumulative AIF.
It then forms

$$
x_n =
\frac{\overline{\int_0^t C_p(\tau)\,d\tau}^{\,(n)}}
     {\bar{C}_p^{(n)}},
\qquad
y_n(x) = \frac{\bar{C}_T^{(n)}(x)}{\bar{C}_p^{(n)}}.
$$

An ordinary least-squares fit of $y_n(x)=K_i(x)x_n+V_d(x)$ is applied
voxel-wise to the reconstructed frames. The slope is saved as the reconstructed
$K_i$ map and the intercept as the reconstructed $V_d$ map; negative fitted
values are not clipped.

Reference parametric maps are calculated directly from each label's kinetic
parameters, rather than fitted from the simulated frames:

$$
K_i = (1-V_b)\frac{K_1k_3}{k_2+k_3},
$$

$$
V_d = (1-V_b)\frac{K_1k_2}{(k_2+k_3)^2}+V_b.
$$

The lesion uses its sampled kinetic parameters. These formulas are the
configured irreversible-Patlak ground truth; the implementation still applies
them and emits a warning if a lookup-table tissue has non-zero $k_4$. The
source-grid ground-truth maps are also resampled to the scanner grid while
preserving voxel values.

For downstream learning, the simulator writes two unnormalised arrays with
channel order `[Ki, Vd]`:

- `unet/input_czyx`: Patlak maps fitted from the motion-affected OSEM sequence;
- `unet/target_czyx`: formula-derived ground-truth maps on the scanner grid.

## Example configuration at a glance

The main settings used by `config.example.json` are summarised here. Every row
can be changed in the JSON configuration or with a command-line override.

| Stage | Example setting |
| --- | --- |
| Anatomy | liver label 13; 127 axial slices; scale 0.95–1.05 |
| Initial pose | up to `(5, 5, 4)` cm translation and `(10, 10, 10)` degrees rotation |
| Frames | 8 frames; 60 seconds each; distributed over 20–60 minutes |
| Lesion | base radii `(6, 6, 4)` voxels; radius multipliers 0.2–1.2 |
| Inter-frame motion | exactly 1 event; up to `(1, 1, 1)` cm and `(10, 10, 5)` degrees |
| Scanner | Siemens mMR; span 11; maximum ring difference 60 |
| Projection | 5% uniform background; noise scaling factor 20 |
| Reconstruction | 7 subsets; 63 subiterations |
| Patlak | complete frames within 20–60 minutes |

## HDF5 output

Each completed sample contains one HDF5 project file (named
`project_pet_dynamic.h5` by the example configuration). The principal datasets
are listed below; additional arrays and group attributes record the resolved
geometry, random seeds, configuration choices, and processing conventions.

| Group | Principal contents |
| --- | --- |
| `gt/` | motion-free activity `emi_clean_tzyx`, static attenuation `atn_static_zyx`, and transformed XCAT labels `label_bed_zyx` |
| `masks/` | target-organ and lesion masks |
| `aif/` | AIF time and concentration samples |
| `frames/` | frame start, end, midpoint, and duration |
| `tac/` | frame-averaged tissue, target-organ, and lesion TACs; continuous organ and lesion TACs; label-to-kinetics mapping; kinetic parameters |
| `motion/` | moved activity and attenuation sequences; selected event frames; event and cumulative translations, rotations, and affine matrices; initial anatomy transform |
| `sirf/` | source and scanner-grid shapes and voxel sizes |
| `projection/` | per-frame emission-only, expected, and noisy projection sums; background values; noise settings and seed base |
| `recon/` | OSEM dynamic sequence, reconstruction sums, and the static attenuation map on the scanner grid |
| `patlak/` | reconstructed and ground-truth $K_i$/$V_d$ maps, selected frames, and regression variables |
| `unet/` | two-channel `[Ki, Vd]` input and target arrays |
| `metadata/` | sample identity, crop, lesion geometry, voxel size, kinetic conventions, and provenance fields |

The HDF5 file does not contain complete sinograms. When
`projection.save_noisy_projections` is true, noisy projection data are kept as
Interfile files in the sample's `sirf_noisy_projection/` directory. When
`reconstruction.save_interfile` is true, reconstructed frames are additionally
written to `sirf_osem_reconstruction/`.

Each sample directory also contains its resolved `sample_config.json`, a
machine-readable `status.json`, and `run.log`. At batch level, the simulator
writes `resolved_config.json`, `batch_manifest.jsonl`, and `batch.log`. The
sample seed is derived from the master seed and sample index, so generating a
subset of a batch does not change the random sequence for a given sample index.

## References

1. W. P. Segars, G. Sturgeon, S. Mendonca, J. Grimes, and B. M. W. Tsui,
   “4D XCAT phantom for multimodality imaging research,” *Medical Physics*,
   37(9), 4902–4915 (2010). [doi:10.1118/1.3480985](https://doi.org/10.1118/1.3480985)
2. K. Thielemans, C. Tsoumpas, S. Mustafovic, T. Beisel, P. Aguiar, N. Dikaios,
   and M. W. Jacobson, “STIR: software for tomographic image reconstruction
   release 2,” *Physics in Medicine & Biology*, 57(4), 867–883 (2012).
   [doi:10.1088/0031-9155/57/4/867](https://doi.org/10.1088/0031-9155/57/4/867)
3. E. Ovtchinnikov et al., “SIRF: Synergistic Image Reconstruction Framework,”
   *Computer Physics Communications*, 249, 107087 (2020).
   [doi:10.1016/j.cpc.2019.107087](https://doi.org/10.1016/j.cpc.2019.107087)
