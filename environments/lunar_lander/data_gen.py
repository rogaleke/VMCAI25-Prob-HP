import os
import subprocess
import warnings

warnings.filterwarnings('ignore')

pwd = os.getcwd()
direct = os.path.join(pwd, 'LunarLander_causal')
os.makedirs(direct, exist_ok=True)

times = [10]

for i in range(5):
    for time in times:
        filename = f'time{time}_{i}.csv'
        print(f"Generating file: {filename}")
        file_path = os.path.join(direct, filename)
        command = [
            "python3", "gen_dtmc.py",
            "--init_time", str(time),
            "--file_name", filename,
            "--dir", direct,
            "--max_noise", str(0.15),
            "--max_n_dist", str(2)
        ]
        try:
            subprocess.run(command, timeout=3000, check=True)
        except subprocess.TimeoutExpired:
            print(f"timed out: {' '.join(command)}")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            continue
        except subprocess.CalledProcessError as e:
            print(f"{e}")
            continue
