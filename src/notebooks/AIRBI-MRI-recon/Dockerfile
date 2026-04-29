FROM ghcr.io/synerbi/sirf:edge-gpu

USER jovyan
RUN pip install pydicom
RUN pip install ismrmrd

RUN git clone --branch downsampled-MRI-recon-stgeorge --depth 1 https://github.com/paskino/SIRF-Contribs.git

WORKDIR /home/jovyan/SIRF-Contribs/src/notebooks/AIRBI-MRI-recon
CMD ["bash" , "mr_recon.sh"]
# CMD ["id", "-a"]
