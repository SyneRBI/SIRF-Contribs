FROM ghcr.io/synerbi/sirf:edge

USER jovyan
RUN pip install pydicom
RUN pip install ismrmrd
RUN pip install nii2dcm

# RUN git clone --branch downsampled-MRI-recon-stgeorge --depth 1 https://github.com/paskino/SIRF-Contribs.git

RUN mkdir -p /home/jovyan/recon_scripts
COPY mr_recon.sh /home/jovyan/recon_scripts/
COPY stgeorges_pipeline.py /home/jovyan/recon_scripts/
COPY stgeorges_utils.py /home/jovyan/recon_scripts/

WORKDIR /home/jovyan/recon_scripts

CMD ["bash" , "mr_recon.sh"]
# CMD ["id", "-a"]
