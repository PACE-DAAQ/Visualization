# The visualization tools for MMM PACE project
This repo contains quick functions and Jupyter notebooks to analyze the results from MPAS-GOCART2G-JEDI

More things are adding to this repo

## Create your Python virtual environment with daaq package
*Conda @ Derecho*

```
module load conda
conda create -n <venv> --clone npl-2025b
conda activate <venv>
cd <repo>
pip install -e .
```

Then you should be able to import it as a package.
