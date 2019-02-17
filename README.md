# Coalescent Based Microsatellite Simulator + Analysis for Different Demographic Models
This repository holds research toward the modeling and analysis of microsatellite mutations.

## Getting Started
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

7. For now, simulation and analysis only exists for a single population. Modify and run the ABC-MCMC script:
```bash
# If working without the HPC:
cd $MICRO_SAT_PATH
./script/methoda.sh data/methoda.db

# If you are on the UH HPC w/ SLURM:
sbatch ./script/methoda.slurm --array=0-19
```

## Usage

### population.py
#### Standalone Program
Simulate the evolution of single population. 
```bash
usage: population.py [-h] [-image IMAGE] [-accel_c {0,1}] [-i_0 I_0 [I_0 ...]]
                     [-n N] [-f F] [-c C] [-d D] [-kappa KAPPA] [-omega OMEGA]
                     
  -h, --help          show this help message and exit
  -image IMAGE        Image file to save resulting repeat length distribution (histogram) to.
  -accel_c {0,1}      Use C extension to run the simulation. Toggle w/ 1/0.
  -i_0 I_0 [I_0 ...]  Repeat lengths of starting ancestors.
  -n N                Starting population size.
  -f F                Scaling factor for total mutation rate.
  -c C                Constant bias for the upward mutation rate.
  -d D                Linear bias for the downward mutation rate.
  -kappa KAPPA        Lower bound of repeat lengths.
  -omega OMEGA        Upper bound of repeat lengths.
```
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

### distance.py
#### Standalone Program
Sample a simulated population and compare this to an observed data set.
```bash
usage: distance.py [-h] [-odb ODB] [-rdb RDB] [-function {COSINE,EUCLIDEAN}]
                   [-uid_observed UID_OBSERVED]
                   [-locus_observed LOCUS_OBSERVED]

  -h, --help                     show this help message and exit
  -odb ODB                       Location of the observed database file.
  -rdb RDB                       Location of the database to record data to.
  -function {COSINE,EUCLIDEAN}   Distance function to use.
  -uid_observed UID_OBSERVED     ID of the observed sample to compare to.
  -locus_observed LOCUS_OBSERVED Locus of the observed sample to compare to. 
```

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

### methoda.py
#### Standalone Program
ABC MCMC for microsatellite mutation model parameter estimation.
```bash
usage: methoda.py [-h] [-odb ODB] [-mdb MDB]
                  [-uid_observed UID_OBSERVED [UID_OBSERVED ...]]
                  [-locus_observed LOCUS_OBSERVED [LOCUS_OBSERVED ...]]
                  [-simulation_n SIMULATION_N] [-iterations_n ITERATIONS_N]
                  [-epsilon EPSILON] [-flush_n FLUSH_N] [-seed SEED] [-n N]
                  [-f F] [-c C] [-d D] [-kappa KAPPA] [-omega OMEGA]
                  [-n_sigma N_SIGMA] [-f_sigma F_SIGMA] [-c_sigma C_SIGMA]
                  [-d_sigma D_SIGMA] [-kappa_sigma KAPPA_SIGMA]
                  [-omega_sigma OMEGA_SIGMA]
             
  -h, --help                                          show this help message and exit
  -odb ODB                                            Location of the observed database file.
  -mdb MDB                                            Location of the database to record to.
  -uid_observed UID_OBSERVED [UID_OBSERVED ...]       IDs of observed samples to compare to.
  -locus_observed LOCUS_OBSERVED [LOCUS_OBSERVED ...] Loci of observed samples (must match with uid).
  -simulation_n SIMULATION_N                          Number of simulations to use to obtain a distance.
  -iterations_n ITERATIONS_N                          Number of iterations to run MCMC for.
  -epsilon EPSILON                                    Maximum acceptance value for distance between [0, 1].
  -flush_n FLUSH_N                                    Number of iterations to run MCMC before flushing to disk.
  -seed SEED                                          1 -> last recorded "mdb" position is used (TIME_R, PROPOSED_TIME).
  -n N                                                Starting sample size (population size).
  -f F                                                Scaling factor for total mutation rate.
  -c C                                                Constant bias for the upward mutation rate.
  -d D                                                Linear bias for the downward mutation rate.
  -kappa KAPPA                                        Lower bound of repeat lengths.
  -omega OMEGA                                        Upper bound of repeat lengths.
  -n_sigma N_SIGMA                                    Step size of n when changing parameters.
  -f_sigma F_SIGMA                                    Step size of f when changing parameters.
  -c_sigma C_SIGMA                                    Step size of c when changing parameters.
  -d_sigma D_SIGMA                                    Step size of d when changing parameters.
  -kappa_sigma KAPPA_SIGMA                            Step size of kappa when changing parameters.
  -omega_sigma OMEGA_SIGMA                            Step size of omega when changing parameters.     
```

### plot.py
Display the results of MCMC scripts.

```bash
usage: plot.py [-h] [-db DB] [-burn_in BURN_IN] [-function {1,2,3,4}]
               [-image_file IMAGE_FILE] [-params PARAMS [PARAMS ...]]
               
  -h, --help                  show this help message and exit
  -db DB                      Location of the database required to operate on.
  -burn_in BURN_IN            Burn in period, in terms of iterations.
  -function {1,2,3,4}         Visualization function to use: [1 <- Waiting times
                              histogram of mutation model MCMC.] [2 <- Probability
                              of our mutation model parameters given our data
                              (histogram & MCMC).] [3 <- Trace plot of our
                              parameters for the mutation model MCMC.] [4 <- Log-
                              likelihood curves of our parameters for the mutation
                              model MCMC.]
  -image_file IMAGE_FILE      Image file to save resulting figure to.
  -params PARAMS [PARAMS ...] Parameters associated with function of use: [1 <- Step
                              sizes of histogram in following order: N, F, C, D,
                              KAPPA, OMEGA.] [2 <- Step sizes of histogram in
                              following order: N, F, C, D, KAPPA, OMEGA.] [3 <-
                               None.] [4 <- None.]
```