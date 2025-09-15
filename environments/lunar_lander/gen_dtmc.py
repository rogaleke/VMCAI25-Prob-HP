import numpy as np
import random
from stable_baselines3 import PPO
import time
import json
import pandas as pd
import warnings
import gymnasium as gym
import argparse
import os
import warnings
from LunarLander import CustomLunarLander
warnings.filterwarnings('ignore')


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


def distribution(n, precision=3):
    # 1) generate random weights and normalize to [0,1]
    weights = [random.random() for _ in range(n)]
    total = sum(weights)
    probs = [w/total for w in weights]

    # 2) round all but the last
    rounded = [round(p, precision) for p in probs[:-1]]
    # 3) make the last one whatever's needed to reach exactly 1
    last = round(1 - sum(rounded), precision)
    return rounded + [last]

def add_noise(a, noise_min=0.0, noise_max=0.1):
    a = np.array(a)
    percentages = np.random.uniform(noise_min, noise_max, size=a.shape)
    signs = 1
    noise = a * percentages * signs
    noisy_a = a + noise
    return noisy_a.tolist()

def float_format(numbers, precisions):

    if len(numbers) != len(precisions):
        raise ValueError("The lists 'numbers' and 'precisions' must have the same length.")
    return [round(num, prec) for num, prec in zip(numbers, precisions)]



def generate(env, model, state, step, assigner, max_step=20, dataframe=None, dtmc=None, violate = False, observation= None, flag= False):
    # Create a unique state identifier by appending the step to the state tuple.
    state_with_step = state + [step]



    state_id = assigner.get_list_id(state_with_step)

    state1 = env.normalize_state(env.get_state())

    # print("state:",state)
    # print("State:", env.get_state())
    # print("State1:", state1)
    

    if (abs(state1[0])> 0.5 or abs(state1[1])> 0.5 ) and step >= max_step:
        violate = True
        log = [state[0], state[1],state[2], state[3], 0 ,violate, step, state_id] + [None] * 8
        dataframe.loc[len(dataframe)] = log
        return
    if (abs(state1[0])< 0.5 and abs(state1[1])< 0.5  ):
        violate = False
        log = [state[0], state[1],state[2], state[3], 0 ,violate, step, state_id] + [None] * 8
        dataframe.loc[len(dataframe)] = log
        return
    if flag:
        return
    

    for j in range(30):
        action, _states = model.predict(np.array(env.normalize_state(env.get_state())), deterministic=True)
        next_state, _, done, _, _ = env.step(action)

    n = random.randint(1, 3)
    probabilities = distribution(n)
    dist = []
    new_step = step + 1

    next_state = env.get_state()

    # print("Next State:", next_state[:4])

    for i in range(n):
        noisy_obs = add_noise(next_state[:4], noise_max=0.5, noise_min=0.00)
        # print("Noisy Obs:",i,noisy_obs)
        formatted_state = float_format(noisy_obs, [2, 2,3,3])
        new_state_with_step = formatted_state + [new_step]
        flag1 = assigner.exists(new_state_with_step)
        new_state_id = assigner.get_list_id(new_state_with_step)
        dist.append([probabilities[i], new_state_id, formatted_state, step, flag1 ])

    # Record the current state's transitions.

    new_log = [state[0], state[1],state[2], state[3], action ,violate, step, state_id]
    new_log.extend([dist[i][1] if i < len(dist) else None for i in range(4)])
    new_log.extend([dist[i][0] if i < len(dist) else None for i in range(4)])
    dataframe.loc[len(dataframe)] = new_log

    # Mark the current state as processed in dtmc.
    if dtmc is None:
        dtmc = {}
    dtmc[state_id] = dist

    # Recurse on each child state.
    for item in dist:
        obs = item[2] + env.get_state()[4:]

        env.set_state(obs)
        
        generate(env=env, model=model, state=item[2], step=new_step, assigner=assigner, 
                 max_step=max_step, dataframe=dataframe, dtmc=dtmc, violate = False, observation=env.normalize_state(obs), flag = item[4])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name",help="file name of csv file",default="Lunar.csv")
    parser.add_argument("--dir", help="directory of csv file", default='')
    parser.add_argument("--init_time", help="time of running", default=10)
    parser.add_argument("--max_noise", help="Maximum noise added", default=0.15)
    parser.add_argument("--max_n_dist", help="Maximum size of dist", default=2)



    args = parser.parse_args()

    init_time = int(args.init_time)
    init_max_noise = float(args.max_noise)
    init_max_n_dist = int(args.max_n_dist)
    filename_factual = os.path.join(args.dir, args.file_name)


    env = CustomLunarLander()

    env.reset()

    state = env.get_state()[:4]

    assigner = ID_Assign()

    model = PPO.load("models/PPO_1/4000000_steps")

    columns = ['x', 'y', 'vel_x', 'vel_y','action', 'violate' ,'Step', 'State', 'Next_State_1', 'Next_State_2', 'Next_State_3', 'Next_State_4',
              'Prob_1', 'Prob_2', 'Prob_3', 'Prob_4']


    df = pd.DataFrame(columns=columns)

    st = float_format(list(state), [3, 3,3,3])
    generate(env = env, model = model, state=st, step= 0, assigner= assigner, max_step=init_time, dataframe=df, observation = env.normalize_state(env.get_state())) 
    
    # Once the simulation is done, adjust data types if needed and write to CSV once.
    df.rename(columns={'x': 'V1', 'y': 'V2','vel_x':'V3','vel_y':'V4', 'action': 'V5' ,'violate': 'V6' ,'Step': 'V7'}, inplace=True)
    df['V7'] = df['V7'].astype('Int32')
    df['State'] = df['State'].astype('Int32')
    df['Next_State_1'] = df['Next_State_1'].astype('Int32')
    df['Next_State_2'] = df['Next_State_2'].astype('Int32')
    df['Next_State_3'] = df['Next_State_3'].astype('Int32')
    df['Next_State_4'] = df['Next_State_4'].astype('Int32')
    df.to_csv(filename_factual, index=False)

    
    
    
  

    


  



    


        
        
    

    