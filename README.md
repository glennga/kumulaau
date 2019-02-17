# Coalescent Based Microsatellite Simulator + Analysis for Different Demographic Models
This repository holds research toward the modeling and analysis of microsatellite mutations.

## Usage
1. Clone this repository!
```bash
git clone https://github.com/glennga/micro-coa.git
```

2. Create an environment variable to this project called `MICRO_SAT_PATH`:
```bash
export MICRO_SAT_PATH=/.../micro-coa
```

3. This project uses Python 3.7, Matplotlib, Numpy, Numba, Scipy, and GSL. An `environment.yml` file is provided to build a clone of the Python environment used to develop this:
```bash
conda env create -f environment.yml
conda activate micro
conda list
```

4. Install GSL. For instructions on how to do so, follow here: http://www2.lawrence.edu/fast/GREGGJ/CMSC210/gsl/gsl.html

5. Build our C extensions by running our setup script. This should create a 'build' folder in this project's top level directory.
```bash
cd $MICRO_SAT_PATH
python setup.py build
python setup.py install
```

6. Populate the observed database `observed.db` by running the ALFRED script:
```bash
cd $MICRO_SAT_PATH
./data/alfred/alfred.sh
``` 

7. For now, simulation and analysis only exists for a single population. Modify and run ABC-MCMC script:
```bash
# If working without the HPC:
cd $MICRO_SAT_PATH
./script/methoda.sh data/methoda.db

# If you are on the UH HPC w/ SLURM:
sbatch ./script/methoda.slurm --array=0-19
```