import os
import subprocess
import warnings

warnings.filterwarnings('ignore')

pwd = os.getcwd()
direct = os.path.join(pwd, 'fisrt_scenario_f16')
os.makedirs(direct, exist_ok=True)

setpoints_extesion = [100,150,200, 300, 500]
times = [10, 15, 20, 40, 60]
init_vels = [600, 800, 1000, 1200]

for vel in init_vels:
    for setpoint in setpoints_extesion:
        for time in times:
            filename = f'vel{vel}_ex{setpoint}_time{time}.csv'
            file_path = os.path.join(direct, filename)
            command = [
                "python3", "gen_dtmc_sc1.py",
                "--init_vel", str(vel),
                "--init_setpoint", str(setpoint),
                "--init_time", str(time),
                "--file_name", filename,
                "--dir", direct,
                "--max_noise", str(0.1),
                "--max_n_dist", str(2)
            ]
            try:
                subprocess.run(command, timeout=500, check=True)
            except subprocess.TimeoutExpired:
                print(f"timed out: {' '.join(command)}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                continue
            except subprocess.CalledProcessError as e:
                print(f"{e}")
                continue
