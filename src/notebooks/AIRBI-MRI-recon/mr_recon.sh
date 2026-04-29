#!/bin/bash

. /opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh
gadgetron &
# python /workdir/reco_scripts/mr_direct_recon.py /input /output
python stgeorges_pipeline.py