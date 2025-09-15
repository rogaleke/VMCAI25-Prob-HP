from MCar import CustomMountainCarEnv
import numpy as np
import random
from stable_baselines3 import DQN
import time
import json
import pandas as pd
import warnings
import argparse
warnings.filterwarnings('ignore')

import os

current_directory = os.getcwd()


class ID_Assign:
    def __init__(self):
      self.id_mapping = {}  
      self.next_id = 0      

    def get_list_id(self, lst):
      key = tuple(lst)
      if key in self.id_mapping:
          return self.id_mapping[key]
      else:
          self.id_mapping[key] = self.next_id
          self.next_id += 1
          return self.id_mapping[key]
        
    def exists(self, lst):
      return tuple(lst) in self.id_mapping
        

def element_exists(lst, target):
    return any(target == item for sublist in lst for subsublist in sublist for item in subsublist)


def distribution(n, precision =2):
    weights = [random.random() for _ in range(n)]
    total_weight = sum(weights)
    return [round(w / total_weight, precision) for w in weights]

def add_noise(a, noise_min=0.0, noise_max=0.15, seed =None):
    rng = np.random.default_rng(seed)
    a = np.array(a)
    percentages = rng.uniform(noise_min, noise_max, size=a.shape)
    signs = rng.choice([-1, 1], size=a.shape)
    noise = a * percentages * signs
    noisy_a = a + noise
    return noisy_a.tolist()

def float_format(numbers, precisions):

    if len(numbers) != len(precisions):
        raise ValueError("The lists 'numbers' and 'precisions' must have the same length.")
    return [round(num, prec) for num, prec in zip(numbers, precisions)]



def generate(env, model, state, step, assigner, done=False, max_step=20, dataframe=None , dtmc=None, seed=None, max_noise= 0.15, output_file = 'M_car_test.csv'):
    df = dataframe
    
    #print(f"step: {step}")
    if dtmc is None:
        dtmc = {}

    flag = assigner.exists(state+ [step])
    state_id = assigner.get_list_id(state)

    if step > max_step or state[0]>0.5:
        log = [state[0], state[1], step, int(state_id), None, None, None, None, None, None, None, None]
        df.loc[len(dataframe)] = log
        df.rename(columns={'vel':'V1', 'pos':'V2', 'Step':'V3'}, inplace=True)
        df['V3'] = df['V3'].astype('Int32')
        df['State'] = df['State'].astype('Int32')
        df['Next_State_1'] = df['Next_State_1'].astype('Int32')
        df['Next_State_2'] = df['Next_State_2'].astype('Int32')
        df['Next_State_3'] = df['Next_State_3'].astype('Int32')
        df['Next_State_4'] = df['Next_State_4'].astype('Int32')
        df.to_csv(output_file, index=False)
        return 0

    # print("state")
    # print(state)

    action, _states = model.predict(state, deterministic=True)
    env.reset(state)
    next_state, _, _, _, _ = env.step(action)

    n = random.randint(1, 3)
    probabilities = distribution(n)
    dist = []
    for i in range(n):
        #noisy_obs = add_noise(next_state, seed=(i + 1) * 20, noise_max=0.15, noise_min=0.00)
        noisy_obs = add_noise(next_state, noise_max=max_noise, noise_min=0.00)
        formatted_state = float_format(noisy_obs, [2, 3])
        state_done = formatted_state[0] >= 0.5
        dist.append([probabilities[i], assigner.get_list_id(formatted_state ), formatted_state, step, state_done])

    dtmc[state_id] = dist

    new_step = step + 1

    new_log = [state[0], state[1], int(step), int(state_id)]
    new_log.extend([int(dist[i][1]) if i < len(dist) else None for i in range(4)])
    new_log.extend([dist[i][0] if i < len(dist) else None for i in range(4)])
    df.loc[len(dataframe)] = new_log
    for item in dist:
        next_done = item[2][0] >= 0.5
        # generate(env, model, item[2], new_step, assigner, next_done, max_step, dataframe=df)
        generate(env = env, model = model, state=item[2], step= new_step, assigner = assigner, max_step= max_step, dataframe=df, done=next_done, max_noise=max_noise, output_file=output_file) 

#Call this from any file in the main folder
def generate_random_dtmc(file, max_noise = 0.19, max_steps = 10, starting_pos = 0.35, starting_vel = 0.05):
    
    env = CustomMountainCarEnv()
    init = [starting_pos, starting_vel]
    assigner = ID_Assign()

    model = DQN.load("models/DQN-0/120000_steps")

    columns = ['vel', 'pos', 'Step', 'State', 'Next_State_1', 'Next_State_2', 'Next_State_3', 'Next_State_4',
              'Prob_1', 'Prob_2', 'Prob_3', 'Prob_4']

    df = pd.DataFrame(columns=columns)
    current_state = env.reset(init)

    generate(env = env, model = model, state=current_state.tolist(), step= 0, assigner= assigner, max_step=max_steps, dataframe=df, max_noise=max_noise, output_file=file) 
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name",help="file name of csv file",default="f16")
    parser.add_argument("--init_vel", help="init vel", default=-1000)
    parser.add_argument("--init_pos", help="onit pos", default=-100)
    parser.add_argument("--dir", help="directory of csv file", default='f16_sc_1')
    parser.add_argument("--init_time", help="time of running", default=10)
    parser.add_argument("--max_noise", help="Maximum noise added", default=0.15)
    parser.add_argument("--max_n_dist", help="Maximum size of dist", default=3)



    args = parser.parse_args()

    init_vel = float(args.init_vel)
    init_pos = float(args.init_pos)
    init_time = int(args.init_time)
    init_max_noise = float(args.max_noise)
    init_max_n_dist = int(args.max_n_dist)
    filename_factual = os.path.join(args.dir, args.file_name)

    generate_random_dtmc(filename_factual, max_noise = init_max_noise, max_steps = init_time, starting_pos = init_pos, starting_vel = init_vel)



    
    
    
  

    


  



    


        
        
    

    