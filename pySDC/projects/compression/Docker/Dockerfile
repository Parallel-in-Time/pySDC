FROM spack/ubuntu-jammy

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update -y && apt install -y cmake ccache pkg-config python3 pip mpich

# Install libpressio with Python and SZ3 support
RUN spack install libpressio+python+sz3

# Prepare an entrypoint that installs pySDC
COPY docker_entrypoint.sh /bin/
ENTRYPOINT ["docker_entrypoint.sh"]
