import os
import subprocess
import warnings

warnings.filterwarnings('ignore')

pwd = os.getcwd()
direct = os.path.join(pwd, 'M_CAR')
os.makedirs(direct, exist_ok=True)

init_pos = [0.4, 0.3, 0.2, 0.1]
times = [10, 15, 20, 25, 30]
init_vels = [0.07, 0.06, 0.04, 0.03, 0.02]

for vel in init_vels:
    for pos in init_pos:
        for time in times:
            filename = f'vel{vel}pos{pos}_time{time}.csv'
            file_path = os.path.join(direct, filename)
            command = [
                "python3", "gen_dtmc.py",
                "--init_vel", str(vel),
                "--init_pos", str(pos),
                "--init_time", str(time),
                "--file_name", filename,
                "--dir", direct,
                "--max_noise", str(0.15),
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
