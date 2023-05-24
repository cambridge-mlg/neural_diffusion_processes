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

The main folder is `/neural_diffusion_processes` with
- `sde*.py` files with the noising and densoing process, along with associated sampling and likelihood evaluation functions.
- `/utils`: lots of different kind of helpers.
- `/data`: for dataloaders and synthetic dataset generation
- `/models`: different neural network architectures based on `haiku` for parameterising the score network.

Then, `/experiments` has three folders, one for each subsection of the experimental section of the paper
- `/regression1d`: for the 1D datasets
- `/steerable_gp`: for the synthetic 2D vector fields from steerable GPs
- `/storm`: for the hurricane trajectories

## Experiments

### 1D regression over stationary scalar fields
With white noise as limiting process
```
python experiments/regression1d/main.py
```
With squared-exponential kernel
```
python experiments/regression1d/main.py kernel=se kernel.params.lengthscale=0.1
```

### Regression over Gaussian process vector field
With non-equivariant score network
```
python experiments/steerable_gp/main.py net=mattn
```
With E(2)-equivariant score network
```
python experiments/steerable_gp/main.py net=e3nn
```

### Global tropical cyclone trajectory prediction
Additionally requires the `basemap` package.
```
python experiments/storm/main.py
```
