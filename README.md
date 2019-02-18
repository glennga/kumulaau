# Coalescent Based Microsatellite Simulator + Analysis for Different Demographic Models
This repository holds research toward the modeling and analysis of microsatellite mutations, and a package for microsatellite demographic model inference. The name _kumulaau_ comes from the Hawaiian word for tree, which are the backbone for our speedy simulation.

## Getting Started
1. Clone this repository!
```bash
git clone https://github.com/glennga/kumulaau.git
```

2. Create an environment variable to this project called `MICRO_SAT_PATH`:
```bash
export MICRO_SAT_PATH=/.../kumulaau
```

3. This project uses Python 3.7, Matplotlib, Numpy, Numba, Scipy, and GSL. An `environment.yml` file is provided to build a clone of the Python environment used to develop this:
```bash
conda env create -f environment.yml
conda activate micro
conda list
```

4. Install GSL. For instructions on how to do so, follow here: http://www2.lawrence.edu/fast/GREGGJ/CMSC210/gsl/gsl.html

5. Install our module. This will build our C extensions.
```bash
conda activate micro
cd $MICRO_SAT_PATH
pip install .
```

6. Populate the observed database `observed.db` by running the ALFRED script:
```bash
cd $MICRO_SAT_PATH
./data/alfred/alfred.sh
```

7. For now, simulation and analysis only exists for a single population. Modify and run the ABC-MCMC script:
```bash
# If working without the HPC:
cd $MICRO_SAT_PATH
./script/wasteful/wasteful.sh data/wasteful.db

# If you are on the UH HPC w/ SLURM:
sbatch ./script/wasteful.slurm --array=0-19
```

## Usage

### `kumulaau/population.py`
#### Standalone Program
Simulate the evolution of single population. 
```bash
usage: population.py [-h] [-image IMAGE] [-accel_c {0,1}] [-i_0 I_0 [I_0 ...]]
                     [-n N] [-f F] [-c C] [-d D] [-kappa KAPPA] [-omega OMEGA]
```
| Parameter          | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| *image (optional)* | *Image file to save resulting repeat length distribution (histogram) to.* |
| accel_c            | Use C extension to run the simulation. Toggle w/ 1/0.        |
| i_0                | Repeat lengths of starting ancestors.                        |
| n                  | Starting population size.                                    |
| f                  | Scaling factor for total mutation rate.                      |
| c                  | Constant bias for the upward mutation rate.                  |
| d                  | Linear bias for the downward mutation rate.                  |
| kappa              | Lower bound of repeat lengths.                               |
| omega              | Upper bound of repeat lengths.                               |

#### Module

1. Defines the `BaseParameters` class, which holds all parameters associated with evolving a single population.
2. Defines the `Population` class, which (a) traces the evolutionary tree for n diploid individuals and (b) evolves said tree given a set of seeds lengths.

Example usage of the two classes are given below:
```python
from numpy import array

args = BaseParameters(n=100, f=100, c=0.001, d=0.0001, kappa=3, omega=30)

# Trace our tree through instantiation. We enable usage of the C extension.
p = Population(theta=args, accel_c=True)

# Evolve the tree, with a common ancestor length of $\ell = 15$.
evolved_100 = p.evolve(array([15]))
```

### `kumulaau/distance.py`
#### Standalone Program
Sample a simulated population and compare this to an observed data set.
```bash
usage: distance.py [-h] [-odb ODB] [-rdb RDB] [-function {COSINE,EUCLIDEAN}]
                   [-uid_observed UID_OBSERVED]
                   [-locus_observed LOCUS_OBSERVED]
```
|Parameter|Description|
|---|---|
|*odb (optional)*|*Location of the observed database file (default = `data/observed.db`).*|
|*rdb (optional)*|*Location of the database to record data to (default = `data/delta.db`).*|
|function|Distance function to use.|
|uid_observed|ID of the observed sample to compare to.|
|locus_observed|Locus of the observed sample to compare to.|

#### Module
1. Defines the `Distance` class, an ABC used for quantifying the distance between an observed and generated distribution.
2. A `Distance` object is used to determine the likelihood of some parameter set $\theta$, given a set of observations (columns in the $\Delta$ matrix) and the number of simulation runs (rows in $\Delta$ matrix).

Example usage of the `Distance` class with the `Cosine` implementing it is given below:
```python
from sqlite3 import connect

# Connect to our observed database.
connection_o = connect('/.../{MICRO_SAT_PATH}/data/observed.db')
cursor_o = connection_o.cursor()

# Collect our observed dataset.
main_observed_frequency = connection_o.execute(""" -- Pull the frequency from the observed database. --
        SELECT ELL, ELL_FREQ
        FROM OBSERVED_ELL
        WHERE SAMPLE_UID LIKE ?
        AND LOCUS LIKE ?
    """, ("SA000289T", "D20S481")).fetchall()
    
# Create our accumulator.
main_accumulator = Cosine(sql_observed=[main_observed_frequency], kappa=3, omega=30, simulation_n=500)
args = BaseParameters(n=100, f=100, c=0.001, d=0.0001, kappa=3, omega=30)

# Compute the approximate likelihood of these parameters, using $\epislon = 0.1$ as our definition of a match.
expected_delta = main_accumulator.fill_matrices(args, epsilon=0.1)
```

### `script/wasteful/wasteful.py`
ABC MCMC for microsatellite mutation model parameter estimation.
```bash
usage: wasteful.py [-h] [-odb ODB] [-mdb MDB]
                   [-uid_observed UID_OBSERVED [UID_OBSERVED ...]]
                   [-locus_observed LOCUS_OBSERVED [LOCUS_OBSERVED ...]]
                   [-simulation_n SIMULATION_N] [-iterations_n ITERATIONS_N]
                   [-epsilon EPSILON] [-flush_n FLUSH_N] [-seed SEED] [-n N]
                   [-f F] [-c C] [-d D] [-kappa KAPPA] [-omega OMEGA]
                   [-n_sigma N_SIGMA] [-f_sigma F_SIGMA] [-c_sigma C_SIGMA]
                   [-d_sigma D_SIGMA] [-kappa_sigma KAPPA_SIGMA]
                   [-omega_sigma OMEGA_SIGMA]
```

| Parameter        | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| *odb (optional)* | *Location of the observed database file (default = `data/observed.db`).* |
| *mdb (optional)* | *Location of the database to record to (default = `data/method-a.db`).* |
| uid_observed     | ID of the observed samples to compare to.                    |
| locus_observed   | Loci of observed samples (must match with uids).             |
| simulation_n     | Number of simulations to use to obtain a distance.           |
| iterations_n     | Number of iterations to run MCMC for.                        |
| epsilon          | Maximum acceptance value for distance between [0, 1].        |
| flush_n          | Number of iterations to run MCMC before flushing to disk.    |
| seed             | 1 -> last recorded "mdb" position is used (TIME_R, PROPOSED_TIME), otherwise 0. |
| n                | Starting sample size (population size).                      |
| f                | Scaling factor for total mutation rate.                      |
| c                | Constant bias for the upward mutation rate.                  |
| d                | Linear bias for the downward mutation rate.                  |
| kappa            | Lower bound of repeat lengths.                               |
| omega            | Upper bound of repeat lengths.                               |
| n_sigma          | Step size of n when changing parameters.                     |
| f_sigma          | Step size of f when changing parameters.                     |
| c_sigma          | Step size of c when changing parameters.                     |
| d_sigma          | Step size of d when changing parameters.                     |
| kappa_sigma      | Step size of kappa when changing parameters.                 |
| omega_sigma      | Step size of omega when changing parameters.                 |

### `script/plot/plot.py`

Display the results of MCMC scripts.

```bash
usage: plot.py [-h] [-db DB] [-burn_in BURN_IN] [-function {1,2,3,4}]
               [-image_file IMAGE_FILE] [-params PARAMS [PARAMS ...]]
```

| Parameter               | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| db                      | Location of the database required to operate on.             |
| burn_in                 | Burn in period, in terms of iterations.                      |
| function                | Visualization function to use: <br><br>*[1 <- Waiting times histogram of mutation model MCMC.]<br>[2 <- Probability of our mutation model parameters given our data (histogram & MCMC).]<br>[3 <- Trace plot of our parameters for the mutation model MCMC.]<br>[4 <- Log-likelihood curves of our parameters for the mutation model MCMC.]* |
| *image_file (optional)* | *Image file to save resulting figure to.*                    |
| params                  | Parameters associated with function of use:<br><br>*[1 <- Step sizes of histogram in following order: N, F, C, D, KAPPA, OMEGA.]<br>[2 <- Step sizes of histogram in following order: N, F, C, D, KAPPA, OMEGA.]<br>[3 <- None.]<br>[4 <- None.]* |
