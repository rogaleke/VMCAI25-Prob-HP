# Efficient Discovery of Actual Causality with in Stochastic Systems

## Introduction
In this repository you can explore our case studies and their implementations.  Here we provide a guide for the implementation of our algorithm on three case studies: Mountain Car for OpenAI Gym, Lunar Lander from OpenAI Gym, and an F16 autopilot simulator from [here](https://github.com/stanleybak/AeroBenchVVPython.git).

## Experiments
Before running this code, ensure you have installed the required packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Combined Benchmarks
We have provided a convenient script `demo\run_benchmarks.py` which will by default run each experiment on their respective DTMCs a single time each, but if you wish to run on different DTMCs or average over larger sample sizes you can simply change the parameters to any of the experiment function calls located at the end of the file.  For example, the driver code for the Mountain Car experiment is `mountain_car_benchmark(dir_path = "environments/mountain_car/dtmcs", repeats = 1)`, where `dir_path` is the directory containing the DTMCs in .csv form, and `repeats` is the number of executions to average the results over for each DTMC.

### Mountain Car Environment
#### Run the Experiment
To run this experiment on a sample DTMC simply run `demo\mountain_car_experiment.py` using python from this repo's root. If you wish to specify a DTMC to run this experiment on, simply change the `dtmc_path` variable to the path of the desired DTMC.

### Lunar Lander
#### Run the Experiment
To run this experiment on a sample DTMC simply run `demo\lunar_lander_experiment.py` using python from this repo's root. If you wish to specify a DTMC to run this experiment on, simply change the `dtmc_path` variable to the path of the desired DTMC

### F16 Autopilot Simulator
#### Run the Experiments
To run this experiment on a sample DTMC simply run `demo\f16_experiment_scenario1.py` for scenario 1 and `demo\f16_experiment_scenario2.py` for scenario 2 using python from this repo's root. If you wish to specify a DTMC to run this experiment on, simply change the `dtmc_path` variable to the path of the desired DTMC
