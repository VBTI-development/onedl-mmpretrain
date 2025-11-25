ARG CU_VERSION="129"
ARG TORCH_VERSION="280"
ARG ONEDL_MMCV_VERSION="2.3.2"

FROM vbti/onedl-mmcv-cu${CU_VERSION}-torch${TORCH_VERSION}:${ONEDL_MMCV_VERSION}

# Install OneDL-MMPretrain
COPY . /onedl/onedl-mmpretrain

WORKDIR /onedl/onedl-mmpretrain
RUN mim install --no-cache-dir -e .[multimodal,optional] \
    && python -m compileall -q /onedl/onedl-mmpretrain

# Verify installation
RUN python -c "import mmpretrain; print('mmpretrain version:', mmpretrain.__version__)"
