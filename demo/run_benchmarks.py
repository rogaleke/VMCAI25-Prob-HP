import sys
sys.path.append(".")
import mountain_car_experiment
import f16_experiment_scenario1
import f16_experiment_scenario2
import lunar_lander_experiment
from pathlib import Path
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def mountain_car_benchmark(dir_path = "data/mountain_car/M_CAR", output_path = "demo/results/mcar.csv", timeout = 60, repeats = 1, skip_concrete=True):
    # Lists to store the times for each performance metric.
    dtmc_sizes = []
    dtmc_times = []
    ar_times = []
    refinement_counts = []
    files = []

    benchmark_path = Path(dir_path)

    time_pattern = re.compile(r'.*_time(\d+)\.csv$')

    for file in benchmark_path.glob("*.csv"):
        sample_dtmc_time = []
        sample_dtmc_size = []
        sample_abs_ref_time = []
        sample_ar_steps = []
        match = time_pattern.match(file.name)
        n = pd.read_csv(file).shape[0]
        print(f"{file} -- Number of States: {n}")
        if match:
            max_time_step = int(match.group(1))
        else:
            print(f"No time step found in {file.name}")
            continue
        for i in range(repeats):
            dtmc_time, abs_time, rf_steps = mountain_car_experiment.run_profile_mcar(file, max_time_step, max(n//20, 10), max(2, n//20), split_method='rand_split_single', split_ratio=0.61, skip_concrete=skip_concrete, timeout=timeout)
            if dtmc_time != None:
                sample_dtmc_size.append(n)
                sample_dtmc_time.append(dtmc_time)
                sample_abs_ref_time.append(abs_time)
                sample_ar_steps.append(rf_steps)
            else:
                print(f"No Cause found for {file}.")
                continue

        if sample_dtmc_time:
            files.append(file.name)
            dtmc_times.append(sum(sample_dtmc_time)/len(sample_dtmc_time))
            dtmc_sizes.append(sum(sample_dtmc_size)/len(sample_dtmc_size))
            ar_times.append(sum(sample_abs_ref_time)/len(sample_abs_ref_time))
            refinement_counts.append(sum(sample_ar_steps)/len(sample_ar_steps))

            results = pd.DataFrame({
                'File': files,
                'DTMC Size': dtmc_sizes,
                'DTMC Time': dtmc_times,
                'AR Time': ar_times,
                'Refinement Steps': refinement_counts
            })

            #save the results to a CSV file
            results.to_csv(output_path, index=False)

        print(f"Done with {file}")

        
    results = pd.DataFrame({
        'File': files,
        'DTMC Size': dtmc_sizes,
        'DTMC Time': dtmc_times,
        'AR Time': ar_times,
        'Refinement Steps': refinement_counts
    })

    #save the results to a CSV file
    results.to_csv(output_path, index=False)

    print("Results:")
    for i in range(len(dtmc_times)):
        print(f"DTMC Size: {dtmc_sizes[i]}, DTMC Time: {dtmc_times[i]}, AR Time: {ar_times[i]}, Steps: {refinement_counts[i]}")

def f16_scen1_benchmark(dir_path = "data/f16/first_scenario_f16", output_path = "demo/results/f16_1.csv", repeats = 3, skip_concrete=True):

    dtmc_sizes = []
    dtmc_times = []
    ar_times = []
    refinement_counts = []
    files = []

    benchmark_path = Path(dir_path)

    time_pattern = re.compile(r'.*_time(\d+)\.csv$')

    for file in benchmark_path.glob("*.csv"):
        sample_dtmc_time = []
        sample_dtmc_size = []
        sample_abs_ref_time = []
        sample_ar_steps = []
        match = time_pattern.match(file.name)
        n = pd.read_csv(file).shape[0]
        print(f"{file} -- Number of States: {n}")
        if match:
            max_time_step = int(match.group(1))
        else:
            print(f"No time step found in {file.name}")
            continue
        for i in range(repeats):
            dtmc_time, abs_time, rf_steps = f16_experiment_scenario1.run_profile_f16_scen1(file, split_method='rand_split_single', split_ratio=0.61, skip_concrete=skip_concrete)
            if dtmc_time != None:
                sample_dtmc_size.append(n)
                sample_dtmc_time.append(dtmc_time)
                sample_abs_ref_time.append(abs_time)
                sample_ar_steps.append(rf_steps)
            else:
                print(f"No Cause found for {file}.")
                continue

        if sample_dtmc_time:
            files.append(file.name)
            dtmc_times.append(sum(sample_dtmc_time)/len(sample_dtmc_time))
            dtmc_sizes.append(sum(sample_dtmc_size)/len(sample_dtmc_size))
            ar_times.append(sum(sample_abs_ref_time)/len(sample_abs_ref_time))
            refinement_counts.append(sum(sample_ar_steps)/len(sample_ar_steps))

        results = pd.DataFrame({
        'File': files,
        'DTMC Size': dtmc_sizes,
        'DTMC Time': dtmc_times,
        'AR Time': ar_times,
        'Refinement Steps': refinement_counts
        })
        
        results.to_csv(output_path, index=False)

        print(f"Done with {file}")

        
    results = pd.DataFrame({
        'File': files,
        'DTMC Size': dtmc_sizes,
        'DTMC Time': dtmc_times,
        'AR Time': ar_times,
        'Refinement Steps': refinement_counts
    })

    #save the results to a CSV file
    results.to_csv(output_path, index=False)

    print("Results:")
    for i in range(len(dtmc_times)):
        print(f"DTMC Size: {dtmc_sizes[i]}, DTMC Time: {dtmc_times[i]}, AR Time: {ar_times[i]}, Steps: {refinement_counts[i]}")

def f16_scen2_benchmark(dir_path = "data/f16/second_scenario_f16", output_path = "demo/results/f16_2.csv", repeats = 3, skip_concrete=True):
    # Lists to store the times for each performance metric.
    dtmc_sizes = []
    dtmc_times = []
    ar_times = []
    refinement_counts = []
    files = []

    benchmark_path = Path(dir_path)

    time_pattern = re.compile(r'.*_time(\d+)\.csv$')

    for file in benchmark_path.glob("*.csv"):
        sample_dtmc_time = []
        sample_dtmc_size = []
        sample_abs_ref_time = []
        sample_ar_steps = []
        match = time_pattern.match(file.name)
        n = pd.read_csv(file).shape[0]
        print(f"{file} -- Number of States: {n}")
        if match:
            max_time_step = int(match.group(1))
        else:
            print(f"No time step found in {file.name}")
            continue
        for i in range(repeats):
            dtmc_time, abs_time, rf_steps = f16_experiment_scenario2.run_profile_f16_scen2(file, split_method='rand_split_single', split_ratio=0.60, skip_concrete=skip_concrete)
            if dtmc_time != None:
                sample_dtmc_size.append(n)
                sample_dtmc_time.append(dtmc_time)
                sample_abs_ref_time.append(abs_time)
                sample_ar_steps.append(rf_steps)
            else:
                print(f"No Cause found for {file}.")
                continue

        if sample_dtmc_time:
            files.append(file.name)
            dtmc_times.append(sum(sample_dtmc_time)/len(sample_dtmc_time))
            dtmc_sizes.append(sum(sample_dtmc_size)/len(sample_dtmc_size))
            ar_times.append(sum(sample_abs_ref_time)/len(sample_abs_ref_time))
            refinement_counts.append(sum(sample_ar_steps)/len(sample_ar_steps))

        results = pd.DataFrame({
        'File': files,
        'DTMC Size': dtmc_sizes,
        'DTMC Time': dtmc_times,
        'AR Time': ar_times,
        'Refinement Steps': refinement_counts
        })

        #save the results to a CSV file
        results.to_csv(output_path, index=False)

    print(f"Done with {file}")

        
    results = pd.DataFrame({
        'File': files,
        'DTMC Size': dtmc_sizes,
        'DTMC Time': dtmc_times,
        'AR Time': ar_times,
        'Refinement Steps': refinement_counts
    })

    #save the results to a CSV file
    results.to_csv(output_path, index=False)

    print("Results:")
    for i in range(len(dtmc_times)):
        print(f"DTMC Size: {dtmc_sizes[i]}, DTMC Time: {dtmc_times[i]}, AR Time: {ar_times[i]}, Steps: {refinement_counts[i]}")

def lunar_lander_benchmark(dir_path = "data/lunar_lander/LunarLander", output_path = "demo/results/lunar.csv", repeats = 3, skip_concrete=True):
    # Lists to store the times for each performance metric.
    dtmc_sizes = []
    dtmc_times = []
    ar_times = []
    refinement_counts = []
    files = []

    benchmark_path = Path(dir_path)

    time_pattern = re.compile(r'time(\d+)_.*\.csv$')

    for file in benchmark_path.glob("*.csv"):
        sample_dtmc_time = []
        sample_dtmc_size = []
        sample_abs_ref_time = []
        sample_ar_steps = []
        match = time_pattern.match(file.name)
        n = pd.read_csv(file).shape[0]
        pos_amt = max(n//10, 10)
        vel_amt = max(2, n//40)
        print(f"{file} -- Number of States: {n}")
        if match:
            pass
        else:
            print(f"No time step found in {file.name}")
            continue
        for i in range(repeats):
            dtmc_time, abs_time, rf_steps = lunar_lander_experiment.run_profile_lander(file, 2, 5, 5, 2, split_method='rand_split_single', split_ratio=0.7, skip_concrete=skip_concrete)
            if dtmc_time != None:
                sample_dtmc_size.append(n)
                sample_dtmc_time.append(dtmc_time)
                sample_abs_ref_time.append(abs_time)
                sample_ar_steps.append(rf_steps)
            else:
                print(f"No Cause found for {file}.")
                continue

        if sample_dtmc_time:
            files.append(file.name)
            dtmc_times.append(sum(sample_dtmc_time)/len(sample_dtmc_time))
            dtmc_sizes.append(sum(sample_dtmc_size)/len(sample_dtmc_size))
            ar_times.append(sum(sample_abs_ref_time)/len(sample_abs_ref_time))
            refinement_counts.append(sum(sample_ar_steps)/len(sample_ar_steps))

            results = pd.DataFrame({
            'File': files,
            'DTMC Size': dtmc_sizes,
            'DTMC Time': dtmc_times,
            'AR Time': ar_times,
            'Refinement Steps': refinement_counts
            })

            #save the results to a CSV file
            results.to_csv(output_path, index=False)

        print(f"Done with {file}")


    results = pd.DataFrame({
        'File': files,
        'DTMC Size': dtmc_sizes,
        'DTMC Time': dtmc_times,
        'AR Time': ar_times,
        'Refinement Steps': refinement_counts
    })

    #save the results to a CSV file
    results.to_csv(output_path, index=False)
    print("Results:")
    for i in range(len(dtmc_times)):
        print(f"DTMC Size: {dtmc_sizes[i]}, DTMC Time: {dtmc_times[i]}, AR Time: {ar_times[i]}, Steps: {refinement_counts[i]}")


if __name__ == '__main__':
    mountain_car_benchmark(dir_path = "environments/mountain_car/dtmcs", repeats = 1)
    f16_scen1_benchmark(dir_path = "environments/f16/dtmcs_scen1", repeats = 1)
    f16_scen2_benchmark(dir_path = "environments/f16/dtmcs_scen2", repeats = 1)
    lunar_lander_benchmark(dir_path = "environments/lunar_lander/LunarLander_2/dtmcs", repeats = 1)