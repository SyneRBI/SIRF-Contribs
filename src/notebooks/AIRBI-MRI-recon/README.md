## St Georges Pipeline

### On a running docker container

```
docker exec sirf-jupyter bash -lc 'source /opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh && cd work/SIRF-Contribs/src/notebooks/AIRBI-MRI-recon && python stgeorges_pipeline.py '
```

Copy the DICOM files
```
scp -r ubuntu@172.16.111.141:workdir/data/recon ~/Data/Emily/
```

### As executable

```
docker build . --no-cache
docker compose build
docker compose up
```

Make sure that `/input` and `/output` are bound to the container as this is where the script expects to find files

