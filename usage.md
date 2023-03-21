
### Prerequisite.
1. Linux/MacOS only.
1. Install gurobi and prepare a license

2. Prepare Python environment
We recommend to use anaconda.

``` shell
conda env create -n SPREAD -f environment.yml
```

3. Run

Use the appropriate flag to run an experiment or generate an evaluation plot: 

``` shell
python main.py [--FLAG]
```


### Directory Structure
``` 
├── results
│   ├── data        # experiment data
│   └── figures     # evaluation figures
├── src             # source code
├── environment.yml # conda environment file
└── main.py         # CLI entrance

```
