FROM ghcr.io/synerbi/sirf:edge-gpu

USER jovyan
RUN pip install pydicom

RUN git clone --branch downsampled-MRI-recon-stgeorge --depth 1 https://github.com/synerbi/SIRF-Contribs.git

WORKDIR /home/jovyan/work/SIRF-Contribs/src/notebooks/AIRBI-MRI-recon
CMD ["python" , "stgeorges_pipeline.py"]
