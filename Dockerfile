# Use your MinkowskiEngine image as the base
FROM yarinpour/minkowski_engine

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    libopenblas-dev \
    wget \
    && apt-get clean

# Install Miniconda (skip if already installed)
ENV CONDA_DIR /opt/conda
RUN if [ ! -d "$CONDA_DIR" ]; then \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p $CONDA_DIR && \
    rm /miniconda.sh; \
    else \
    echo "Miniconda already installed at $CONDA_DIR"; \
    fi

# Update PATH environment variable
ENV PATH $CONDA_DIR/bin:$PATH

# Copy the environment file and create the conda environment
COPY env.yml /tmp/env.yml
RUN conda env create -f /tmp/env.yml && conda clean -a

# Set the entrypoint to use the growsp environment
ENTRYPOINT ["conda", "run", "-n", "growsp", "/bin/bash"]
