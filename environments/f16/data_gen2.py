import os
import subprocess
import warnings

warnings.filterwarnings('ignore')

pwd = os.getcwd()
direct = os.path.join(pwd, 'second_scenario_f16_test')
os.makedirs(direct, exist_ok=True)

setpoints_extesion = [500]
times = [18]
init_vels = [700]
init_alts = [1000]
for i in range(1, 4):
    for alt in init_alts:
        for vel in init_vels:
            for setpoint in setpoints_extesion:
                for time in times:
                    filename = f'alt{alt}_vel{vel}_ex{setpoint}_time{time}_{i}.csv'
                    print(f"Generating file: {filename}")
                    file_path = os.path.join(direct, filename)
                    command = [
                        "python3", "gen_dtmc_sc2.py",
                        "--init_vel", str(vel),
                        "--init_alt", str(alt),
                        "--init_setpoint", str(setpoint),
                        "--init_time", str(time),
                        "--file_name", filename,
                        "--dir", direct,
                        "--max_noise", str(0.2),
                        "--max_n_dist", str(2)
                    ]
                    try:
                        subprocess.run(command, timeout=800, check=True)
                    except subprocess.TimeoutExpired:
                        print(f"timed out: {' '.join(command)}")
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                        continue
                    except subprocess.CalledProcessError as e:
                        print(f"{e}")
                        continue
