'''
Stanley Bak

Engine controller specification checking
'''
import numpy as np
from numpy import deg2rad
from RunF16Sim import RunF16Sim
from PassFailAutomaton import AirspeedPFA, FlightLimits
from CtrlLimits import CtrlLimits
from LowLevelController import LowLevelController
from Autopilot import FixedSpeedAutopilot
from controlledF16 import controlledF16
from plot import plot2d
import random
import pandas as pd
import argparse
import warnings
import os 
import sys
warnings.filterwarnings('ignore')
sys.setrecursionlimit(100000)





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
    if seed is not None:
        np.random.seed(seed)

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


def check(states, nzs, pss):
    for state in states:
        alt, alpha, beta, pitch1, pitch2 = state[:5]
        if not (300 <= alt <= 2500):
            return True
        if not (-10 <= alpha <= 45):
            return True
        if not (-30 <= beta <= 30):
            return True
        if not (-30 <= pitch2 <= 30):  
            return True

    if any(nz < -5 or nz > 9 for nz in nzs):
        return True
    if any(p < -500 or p > 500 for p in pss):
        return True

    return False

def generate(state, ctrlLimits, flightLimits, llc , initialState, assigner ,setpoint , p_gain , f16_plant , time, acc, step , dataframe, violate=False, max_noise = 0.15, n_dist=3, filename  ='f16_csv' ):


    state_id = assigner.get_list_id(state)

    if step > time and ( state[0] < (setpoint - setpoint * (acc / 100.0)) or state[0] > (setpoint + setpoint * (acc / 100.0))):
        violate = True
        log = [state[0], state[1], state[2], state[3],  int(step), violate, int(state_id), None, None, None, None, None, None, None, None]

        df.loc[len(dataframe)] = log
        df.rename(columns={'vel':'V1', 'alt':'V2', 'power':'V3', 'trot': 'V4', 'step': 'V5', 'violate':'V6'}, inplace=True)
        df['V5'] = df['V5'].astype('Int32')
        df['state'] = df['state'].astype('Int32')
        df['Next_State_1'] = df['Next_State_1'].astype('Int32')
        df['Next_State_2'] = df['Next_State_2'].astype('Int32')
        df['Next_State_3'] = df['Next_State_3'].astype('Int32')
        df['Next_State_4'] = df['Next_State_4'].astype('Int32')
        df.to_csv(filename, index=False)

        return 0
    if step <= time and ( state[0] > (setpoint - setpoint * (acc / 100.0)) and state[0] < (setpoint + setpoint * (acc / 100.0))):
        violate = False
        log = [state[0], state[1], state[2], state[3],  int(step), violate, int(state_id), None, None, None, None, None, None, None, None]

        df.loc[len(dataframe)] = log
        df.rename(columns={'vel':'V1', 'alt':'V2', 'power':'V3', 'trot': 'V4', 'step': 'V5', 'violate':'V6'}, inplace=True)
        df['V5'] = df['V5'].astype('Int32')
        df['state'] = df['state'].astype('Int32')
        df['Next_State_1'] = df['Next_State_1'].astype('Int32')
        df['Next_State_2'] = df['Next_State_2'].astype('Int32')
        df['Next_State_3'] = df['Next_State_3'].astype('Int32')
        df['Next_State_4'] = df['Next_State_4'].astype('Int32')
        df.to_csv(filename, index=False)
        return 0
    

    elif step > time :
        log = [state[0], state[1], state[2], state[3],  int(step), violate, int(state_id), None, None, None, None, None, None, None, None]

        df.loc[len(dataframe)] = log
        df.rename(columns={'vel':'V1', 'alt':'V2', 'power':'V3', 'trot': 'V4', 'step': 'V5', 'violate':'V6'}, inplace=True)
        df['V5'] = df['V5'].astype('Int32')
        df['state'] = df['state'].astype('Int32')
        df['Next_State_1'] = df['Next_State_1'].astype('Int32')
        df['Next_State_2'] = df['Next_State_2'].astype('Int32')
        df['Next_State_3'] = df['Next_State_3'].astype('Int32')
        df['Next_State_4'] = df['Next_State_4'].astype('Int32')
        df.to_csv(filename, index=False)

        return 0


    def der_func(t, y):
        'derivative function'

        der = controlledF16(t, y, f16_plant, ap, llc)[0]

        rv = np.zeros((y.shape[0],))

        rv[0] = der[0] # speed
        rv[12] = der[12] # power lag term

        return rv


    ap = FixedSpeedAutopilot(setpoint, p_gain, llc.xequil, llc.uequil, flightLimits, ctrlLimits)
    pass_fail = AirspeedPFA(time - step  , setpoint, acc)

    

    # todo check time below 

    passed, times, states, modes, ps_list, Nz_list, u_list = RunF16Sim(initialState, time + 1, der_func, f16_plant, ap, llc, pass_fail, sim_step=1)

    next_state = [
        states[1][0], # Vt
        states[1][11], # alt
        states[1][12], # power lag term
        u_list[1][0] # throttle
    ]
    n = random.randint(1, n_dist)
    probabilities = distribution(n)
    dist = []


    for i in range(n):
        noisy_obs = add_noise(next_state, seed=(i + 1) * 20, noise_max=max_noise, noise_min=0.00)
        formatted_state = float_format(noisy_obs, [0, 0, 1, 2])
        formatted_state[2] = min(1, max(0, formatted_state[2]))
        formatted_state[3] = min(1, max(0, formatted_state[3]))
        state_done = formatted_state[0] >= setpoint
        dist.append([probabilities[i], assigner.get_list_id(formatted_state), formatted_state, step, state_done])

    new_step = step + 1

    new_log = [state[0], state[1], state[2], state[3],  int(step), violate, int(state_id)]
    new_log.extend([int(dist[i][1]) if i < len(dist) else None for i in range(4)])
    new_log.extend([dist[i][0] if i < len(dist) else None for i in range(4)])
    df.loc[len(dataframe)] = new_log
    for item in dist:


        initialState = states[1][:13].copy()
        initialState[0] = item[2][0]
        initialState[11] = item[2][1]
        initialState[12] = item[2][2]

        state = [item[2][0], item[2][1], item[2][2], item[2][3]]
        
        # generate(env, model, item[2], new_step, assigner, next_done, max_step, dataframe=df)
        generate( state, ctrlLimits, flightLimits, llc , initialState, assigner, setpoint, p_gain, f16_plant, time, acc, step = new_step, dataframe = dataframe, max_noise = max_noise, n_dist= n_dist, filename=filename)

    


    



if __name__ == '__main__':

    


    # initial parameters

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name",help="file name of csv file",default="f16")
    parser.add_argument("--init_vel", help="init vel", default=-1000)
    parser.add_argument("--init_setpoint", help="init setpoint extended to init vel", default=-100)
    parser.add_argument("--dir", help="directory of csv file", default='f16_sc_1')
    parser.add_argument("--init_time", help="time of running", default=10)
    parser.add_argument("--max_noise", help="Maximum noise added", default=0.2)
    parser.add_argument("--max_n_dist", help="Maximum size of dist", default=3)



    args = parser.parse_args()

    init_vel = int(args.init_vel)
    init_setpoint = int(args.init_setpoint)
    init_time = int(args.init_time)
    init_max_noise = float(args.max_noise)
    init_max_n_dist = int(args.max_n_dist)
    filename_factual = os.path.join(args.dir, args.file_name)



    power = 0 # Power

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    alt =5000  # Initial Attitude
    Vt = init_vel # Initial Speed
    phi = 0 #(pi/2)*0.5           # Roll angle from wings level (rad)
    theta = 0 #(-pi/2)*0.8        # Pitch angle from nose level (rad)
    psi = 0 #-pi/4                # Yaw angle from North (rad)

    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initialState = [Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]


    ctrlLimits = CtrlLimits()
    flightLimits = FlightLimits()
    llc = LowLevelController(ctrlLimits)

    assigner = ID_Assign()

    state = [
        Vt, # Vt
        alt, # alt
        power, # power lag term
        0 # throttle
    ]

    columns = ['vel', 'alt', 'power', 'trot', 'step', 'violate','state', 'Next_State_1', 'Next_State_2', 'Next_State_3', 'Next_State_4',
              'Prob_1', 'Prob_2', 'Prob_3', 'Prob_4']
    
    df = pd.DataFrame(columns=columns)



    generate(state, ctrlLimits, flightLimits, llc , initialState, assigner,setpoint = init_vel + init_setpoint, p_gain = 0.01, f16_plant = 'morelli', time= init_time, acc = 5, dataframe =df ,step = 0, max_noise=init_max_noise, n_dist=init_max_n_dist, filename = filename_factual)

    
