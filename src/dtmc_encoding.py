from z3 import *
from utils import get_float

EPSILON = 0.000001

def find_cause(dtmc_dict, initial, target,  EPSILON, debug= False):

    all_states = set(dtmc_dict.keys())
    for transitions in dtmc_dict.values():
        for (dest, prob) in transitions:
            all_states.add(dest)

    P = { state: Real("p_{}".format(state)) for state in all_states }

    solver = Solver()

    for state in all_states:
        if state in target:
            # Reachable (inside state)
            solver.add(P[state] == 1)
        elif state not in dtmc_dict:
            # No outgoing Transition
            solver.add(P[state] == 0)
        else:
            transition_sum = 0
            for (dest, prob) in dtmc_dict[state]:
                transition_sum += prob * P[dest]
                # print(dest, prob, transition_sum)
            solver.add(P[state] == transition_sum)

    solver.add(P[initial]>EPSILON)

    candiates = list()

    if solver.check() == sat:
        

        model = solver.model()

        if debug == True:
            print(model)

        for state in all_states:
            
            if get_float(model.eval(P[state]))> get_float(model.eval(P[initial])) and state not in target:
                candiates.append([state, get_float(model.eval(P[state]))])

    else:
        print("Cause Not Found!")
        return None

    cause = list()

    for can_state in candiates:

        P = { state: Real("p_{}".format(state)) for state in all_states }
        solver = Solver()

        for state in all_states:
            if state in target:
                # Reachable (inside state)
                solver.add(P[state] == 1)
            elif state not in dtmc_dict:
                # No outgoing Transition
                solver.add(P[state] == 0)
            else:
                transition_sum = 0
                for (dest, prob) in dtmc_dict[state]:
                    if state == can_state[0]:
                        transition_sum += 0 * P[dest]
                    else:
                        transition_sum += prob * P[dest]
                    # print(dest, prob, transition_sum)
                solver.add(P[state] == transition_sum)
                
        solver.add(P[initial]>EPSILON)

        if solver.check() == sat:
            model = solver.model()
            flag = 0
            for state in all_states: 
                if get_float(model.eval(P[state])) > can_state[1] and state not in target:
                    flag = 1
            if flag == 0:
                cause.append(can_state)

    if len(cause)>0:
        return cause
    else:
        print("Cause Not Found!")
        return None