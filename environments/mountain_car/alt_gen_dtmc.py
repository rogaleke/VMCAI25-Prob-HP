from MCar import CustomMountainCarEnv
import numpy as np
import random
from stable_baselines3 import DQN
import time
import json
import pandas as pd
import warnings
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


def distribution(n, precision =2):
    weights = [random.random() for _ in range(n)]
    total_weight = sum(weights)
    return [round(w / total_weight, precision) for w in weights]

def add_noise(a, noise_min=0.0, noise_max=0.15):
    a = np.array(a)
    percentages = np.random.uniform(noise_min, noise_max, size=a.shape)
    signs = np.random.choice([-1, 1], size=a.shape)
    noise = a * percentages * signs
    noisy_a = a + noise
    return noisy_a.tolist()

def float_format(numbers, precisions):

    if len(numbers) != len(precisions):
        raise ValueError("The lists 'numbers' and 'precisions' must have the same length.")
    return [round(num, prec) for num, prec in zip(numbers, precisions)]



def generate(env, model, state, step, assigner, done=False, max_step=20, dataframe=None, dtmc=None):
    # Create a unique state identifier by appending the step to the state tuple.
    state_with_step = state + [step]
    state_id = assigner.get_list_id(state_with_step)
    
    # If the state (by its id) is already in the DataFrame, do not continue simulating this branch.
    if state_id in dataframe['State'].values:
        return

    # Terminate if we've exceeded max steps or the state is terminal.
    if step > max_step or done:
        log = [state[0], state[1], step, state_id] + [None] * 8
        dataframe.loc[len(dataframe)] = log
        return

    # Predict next action and simulate the next state.
    action, _states = model.predict(state, deterministic=True)
    env.reset(state)
    next_state, _, _, _, _ = env.step(action)

    # Decide how many transitions to simulate (2 or 3).
    n = random.randint(2, 3)
    probabilities = distribution(n)
    dist = []
    new_step = step + 1

    for i in range(n):
        noisy_obs = add_noise(next_state, noise_max=0.19, noise_min=0.04)
        formatted_state = float_format(noisy_obs, [3, 3])
        state_done = formatted_state[0] >= 0.5
        formatted_state[0] = min(0.6, max(-1.2, formatted_state[0]))
        formatted_state[1] = min(0.07, max(-0.07, formatted_state[1]))
        # Create a unique key for the new state including the new step.
        new_state_with_step = formatted_state + [new_step]
        new_state_id = assigner.get_list_id(new_state_with_step)
        dist.append([probabilities[i], new_state_id, formatted_state, step, state_done])

    # Record the current state's transitions.
    new_log = [state[0], state[1], step, state_id]
    new_log.extend([dist[i][1] if i < len(dist) else None for i in range(4)])
    new_log.extend([dist[i][0] if i < len(dist) else None for i in range(4)])
    dataframe.loc[len(dataframe)] = new_log

    # Mark the current state as processed in dtmc.
    if dtmc is None:
        dtmc = {}
    dtmc[state_id] = dist

    # Recurse on each child state.
    for item in dist:
        next_done = item[4]  # terminal flag from the noisy state
        generate(env=env, model=model, state=item[2], step=new_step, assigner=assigner, 
                 max_step=max_step, dataframe=dataframe, dtmc=dtmc)
if __name__ == "__main__":

    env = CustomMountainCarEnv()

    init = [0.3 , 0.06]

    prc = [3,3]

    assigner = ID_Assign()

    model = DQN.load("data/mountain_car/models/DQN-0/120000_steps")

    columns = ['vel', 'pos', 'Step', 'State', 'Next_State_1', 'Next_State_2', 'Next_State_3', 'Next_State_4',
              'Prob_1', 'Prob_2', 'Prob_3', 'Prob_4']


    df = pd.DataFrame(columns=columns)

    current_state = env.reset(init)
    generate(env = env, model = model, state=current_state.tolist(), step= 0, assigner= assigner, max_step=20, dataframe=df) 
    
    # Once the simulation is done, adjust data types if needed and write to CSV once.
    df.rename(columns={'vel': 'V1', 'pos': 'V2', 'Step': 'V3'}, inplace=True)
    df['V3'] = df['V3'].astype('Int32')
    df['State'] = df['State'].astype('Int32')
    df['Next_State_1'] = df['Next_State_1'].astype('Int32')
    df['Next_State_2'] = df['Next_State_2'].astype('Int32')
    df['Next_State_3'] = df['Next_State_3'].astype('Int32')
    df['Next_State_4'] = df['Next_State_4'].astype('Int32')
    df.to_csv('data/mountain_car/Mcar.csv', index=False)
    df.to_csv('mcar_preprocessed.csv', index=False)



    
    
    
  

    


  



    


        
        
    

    