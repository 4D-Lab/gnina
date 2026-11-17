# Build of gnina against ROCm, for AMD GPUs.
#
# The image builds libmolgrid from source too, because the upstream libmolgrid
# is CUDA-only and has to be hipified before gnina can link it -- see
# docs/AMD_GPU.md.
#
# Run with:
#   docker run --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined ...

FROM rocm/dev-ubuntu-24.04:6.2-complete

WORKDIR /root
ARG DEBIAN_FRONTEND="noninteractive"

# Which AMD GPUs to generate code for. gfx90a = MI210/MI250, gfx942 = MI300,
# gfx1100 = RX 7900. Narrow this to your own card to cut build time.
ARG GPU_ARCHS="gfx90a;gfx942;gfx1030;gfx1100"

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    build-essential cmake git wget curl swig \
    libboost-all-dev libeigen3-dev libgoogle-glog-dev \
    libprotobuf-dev protobuf-compiler libhdf5-dev libatlas-base-dev \
    libjsoncpp-dev librdkit-dev \
    python3-dev python3-numpy python3-pip python3-pytest

# The ROCm build of torch, not the CUDA one.
RUN pip3 install --break-system-packages --index-url https://download.pytorch.org/whl/rocm6.2 \
    torch torchvision torchaudio
RUN pip3 install --break-system-packages scikit-image pyquaternion google-api-python-client six

RUN git clone https://github.com/openbabel/openbabel.git && \
    cd openbabel && mkdir build && cd build && \
    cmake -DWITH_MAEPARSER=OFF -DWITH_COORDGEN=OFF -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON .. && \
    make -j$(nproc) && make install

# libmolgrid: upstream is CUDA-only, and porting it needs more than hipify --
# the CMakeLists declares the CUDA language, the python/docs/test subdirectories
# are added unconditionally, and HIP's float3 is not an aggregate. scripts/
# hipify_libmolgrid.sh does all of it and audits what is left.
COPY scripts/hipify_libmolgrid.sh /root/hipify_libmolgrid.sh
RUN git clone https://github.com/gnina/libmolgrid.git && \
    /root/hipify_libmolgrid.sh /root/libmolgrid && \
    cmake -S libmolgrid -B libmolgrid/build \
      -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_HIP_ARCHITECTURES="${GPU_ARCHS}" \
      -DAMDGPU_TARGETS="${GPU_ARCHS}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_STATIC=ON -DBUILD_SHARED=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build libmolgrid/build -j$(nproc) --target install

COPY . /root/gnina

RUN cd /root/gnina && mkdir -p build && cd build && \
    cmake .. \
      -DGNINA_GPU_BACKEND=HIP \
      -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_HIP_ARCHITECTURES="${GPU_ARCHS}" \
      -DLIBMOLGRID_LIBRARY=/usr/local/lib/libmolgrid.a \
      -DLIBMOLGRID_INCLUDE_DIR=/usr/local/include && \
    make -j$(nproc) && make install

ENV PATH="/usr/local/bin:${PATH}"
