# How to install

Clone repo
```
git clone -b neural-diffusion-processes https://github.com/oxcsml/score-sde.git
```

Create virtual environment, either with `virtualenv`
```
virtualenv -p python3.9 venv
source venv/bin/activate
pip install jax[cuda]==0.4.6 --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

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