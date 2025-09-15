'''
Stanley Bak

Engine controller specification checking
'''

import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt

from RunF16Sim import RunF16Sim
from PassFailAutomaton import FlightLimitsPFA, FlightLimits
from CtrlLimits import CtrlLimits
from LowLevelController import LowLevelController
from Autopilot import FixedAltitudeAutopilot
from controlledF16 import controlledF16
import argparse
from plot import plot2d
import random
import pandas as pd
import os

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


def check(states):

    alpha = np.rad2deg(states[1])
    alt = states[4]
    nz = states[6]
    vel = states[0]

    if vel< 300 or vel>2500:
        return True
    if not (1 <= alt <= 45000):
        return True
    if not (-10 <= alpha <= 45):
        return True
    if (nz < -5 or nz > 9 ):
        return True

    return False


def generate(state, ctrlLimits, flightLimits, llc , initialState, assigner ,setpoint ,f16_plant , time, step , dataframe, violation = False, filename='f16_2.csv'):


    state_id = assigner.get_list_id(state)

    violate = check(state)


        # columns = ['vel', 'alpha', 'theta', 'pitch','alt','power','G-force','step',' violation', 'state', 'Next_State_1', 'Next_State_2', 'Next_State_3', 'Next_State_4',
        #       'Prob_1', 'Prob_2', 'Prob_3', 'Prob_4']

    


    if (step >= time and (state[4] <  setpoint - setpoint * (5 / 100.0) or state[4] >  setpoint + setpoint * (5 / 100.0))) or violate:
        log = state + [step, True, state_id] + [None, None, None, None, None, None, None, None]
        df.loc[len(dataframe)] = log
        return 0
    
    if step <= time and ((state[4] >  setpoint - setpoint * (5 / 100.0) and state[4] <  setpoint + setpoint * (5 / 100.0))):
        log = state + [step, False, state_id] + [None, None, None, None, None, None, None, None]
        df.loc[len(dataframe)] = log

        
        return 0
    
    if (step > time):
        log = state + [step, False, state_id] + [None, None, None, None, None, None, None, None]
        df.loc[len(dataframe)] = log


        return 0
    
    

    def der_func(t, y):
        'derivative function for RK45'

        der = controlledF16(t, y, f16_plant, ap, llc)[0]

        rv = np.zeros((y.shape[0],))

        rv[0] = der[0] # air speed
        rv[1] = der[1] # alpha
        rv[4] = der[4] # pitch angle
        rv[7] = der[7] # pitch rate
        rv[11] = der[11] # altitude
        rv[12] = der[12] # power lag term
        rv[13] = der[13] # Nz integrator

        return rv


    ap = FixedAltitudeAutopilot(setpoint, llc.xequil, llc.uequil, flightLimits, ctrlLimits)
    pass_fail = FlightLimitsPFA(flightLimits)
    pass_fail.break_on_error = False

    

    

    passed, times, states, modes, ps_list, Nz_list, u_list =  RunF16Sim(initialState, time-step , der_func, f16_plant, ap, llc, pass_fail, sim_step=1)


    if len(states)>1:
        next_state = [
        states[1][0],#Vt
        states[1][1], #alpha
        states[1][4], # pitch angle
        states[1][7], # pitch rate
        states[1][11], # alt
        states[1][12], # power lag term
        Nz_list[1] # Nz integrator
    ]
    else:

        next_state = [
        states[0][0],#Vt
        states[0][1], #alpha
        states[0][4], # pitch angle
        states[0][7], # pitch rate
        states[0][11], # alt
        states[0][12], # power lag term
        Nz_list[0] # Nz integrator
    ]



    
    # print(violate)
    
    n = random.randint(1,4)
    probabilities = distribution(n)
    dist = []


    for i in range(n):
        noisy_obs = add_noise(next_state, seed=(i + 1) * 20, noise_max=0.1, noise_min=0.00)
        formatted_state = float_format(noisy_obs, [2, 2, 2, 2,1,1,2])
        dist.append([probabilities[i], assigner.get_list_id(formatted_state), formatted_state, step])

    new_step = step + 1

    new_log = state + [step, violation, state_id]
    
    new_log.extend([int(dist[i][1]) if i < len(dist) else None for i in range(4)])
    new_log.extend([dist[i][0] if i < len(dist) else None for i in range(4)])
    df.loc[len(dataframe)] = new_log
    for item in dist:

        initialState = states[0][:13].copy()
        initialState[0] = item[2][0]
        initialState[1] = item[2][1]
        initialState[4] = item[2][2]
        initialState[7] = item[2][3]
        initialState[11] = item[2][4]
        initialState[12] = item[2][5]


        state = [item[2][0], item[2][1], item[2][2], item[2][3], item[2][4], item[2][5], item[2][6]]


        generate( state, ctrlLimits, flightLimits, llc , initialState, assigner, setpoint, f16_plant, time, step = new_step, dataframe = dataframe, filename  = filename)

    


    



if __name__ == '__main__':



    # initial parameters

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name",help="file name that the controller fails",default="f16")
    parser.add_argument("--init_alt", help="init alt ", default=-1000)
    parser.add_argument("--init_setpoint", help="init setpoint extended to init alt", default=-500)
    parser.add_argument("--dir", help="directory file that csv file will be save", default='f16_sc_2')
    parser.add_argument("--init_time", help="time of running", default=10)
    parser.add_argument("--max_noise", help="Maximum noise added", default=0.2)
    parser.add_argument("--max_n_dist", help="Maximum size of dist", default=3)
    parser.add_argument("--init_vel", help="init vel", default=-1000)



    args = parser.parse_args()

    init_vel = int(args.init_vel)
    init_alt = int(args.init_alt)
    init_setpoint = int(args.init_setpoint)
    init_time = int(args.init_time)
    init_max_noise = float(args.max_noise)
    init_max_n_dist = int(args.max_n_dist)
    filename_factual = os.path.join(args.dir, args.file_name)

    


    power = 0 # Power

    # Default alpha & beta
    alpha = deg2rad(0) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    alt = init_alt # Initial Attitude
    Vt = init_vel # Initial Speed
    phi = 0 #(pi/2)*0.5           # Roll angle from wings level (rad)
    theta = 0 #(-pi/2)*0.8        # Pitch angle from nose level (rad)
    psi = 0 #-pi/4                # Yaw angle from North (rad)

    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initialState = [Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    assigner = ID_Assign()

    state = [
        Vt,#Vt
        alpha, #alpha
        theta, # pitch angle
        0, # pitch rate
        alt, # alt
        power, # power lag term
        0 # Nz integrator
    ]

    columns = ['vel', 'alpha', 'theta', 'pitch','alt','power','G-force','step','violation', 'state', 'Next_State_1', 'Next_State_2', 'Next_State_3', 'Next_State_4',
              'Prob_1', 'Prob_2', 'Prob_3', 'Prob_4']
    
    df = pd.DataFrame(columns=columns)

    ctrlLimits = CtrlLimits()
    flightLimits = FlightLimits()
    llc = LowLevelController(ctrlLimits)



    generate(state, ctrlLimits, flightLimits, llc , initialState, assigner,setpoint = init_alt +init_setpoint , f16_plant = 'stevens', time= init_time, dataframe =df ,step = 0, filename=filename_factual )

    # 368.0,-0.22,-0.09,-0.73,2281.4,1,-2.71,5,False

    df.rename(columns={'vel':'V1', 'alpha':'V2', 'theta':'V3', 'pitch': 'V4', 'alt': 'V5', 'power': 'V6', 'G-force': 'V7','step': 'V8','violation': 'V9'}, inplace=True)
    df['V8'] = df['V8'].astype('Int32')
    df['state'] = df['state'].astype('Int32')
    df['Next_State_1'] = df['Next_State_1'].astype('Int32')
    df['Next_State_2'] = df['Next_State_2'].astype('Int32')
    df['Next_State_3'] = df['Next_State_3'].astype('Int32')
    df['Next_State_4'] = df['Next_State_4'].astype('Int32')
    df.to_csv(filename_factual, index=False)

    
