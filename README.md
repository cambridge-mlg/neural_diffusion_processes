# Geometric neural diffusion processes

This repository contains the code for the paper 'Geometric neural diffusion processes'.
This paper theoretically and practically extends denoising diffusion models to function spaces.

## How to install

Clone repo
```
git clone -b neural-diffusion-processes URL
```

Create virtual environment, either with `virtualenv`
```
virtualenv -p python3.9 venv
source venv/bin/activate
```

or with `conda` (convenient to install jax)
```
conda create -n venv python=3.9
source activate venv
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

Install the runtime dependencies
```
pip install -r requirements.txt
```

Install the package
```
pip install -e .
```

Additionally install the experiment specific dependencies
```
pip install -r experiments/*/requirements.txt
```

## Code structure


## Reproducing experiments

### 1D regression over stationary scalar fields
```
python experiments/
```

### Regression over Gaussian process vector field
With non-equivariant score network
```
python experiments/steerable_gp/main.py net=e3nn
```
With E(2)-equivariant score network
```
python experiments/steerable_gp/main.py net=mattn
```

### Global tropical cyclone trajectory predictio

