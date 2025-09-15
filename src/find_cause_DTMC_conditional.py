from z3 import *
import sys
sys.path.append(".")
from src.utils import get_float

def concrete_conditional_cause(dtmc_dict, initial, target, EPSILON, first_cause = True, debug=False):

    all_states = set(dtmc_dict.keys())

    for transitions in dtmc_dict.values():
        for (dest, prob) in transitions:
            all_states.add(dest)

    p = { state: Real("p_{}".format(state)) for state in all_states }
    solver = Solver()
    
    for state in all_states:
        if state in target:
            solver.add(p[state] == 1)
        # Dead-end (non-target) states: no outgoing transitions
        elif state not in dtmc_dict:
            solver.add(p[state] == 0)
        else:
            solver.add(p[state] == Sum([RealVal(prob) * p[dest] for (dest, prob) in dtmc_dict[state]]))
    # Ensure the initial state has a nontrivial reachability.
    solver.add(p[initial] > EPSILON)
    
    cause_can = []   # Will hold candidates as [state, p_value]

    if solver.check() == sat:
        model = solver.model()
        for state in all_states:
            # Candidate if state is not in the target and its reachability exceeds that from the initial.
            if state not in target and get_float(model.eval(p[state])) > get_float(model.eval(p[initial])):
                cause_can.append([state, get_float(model.eval(p[state]))])
    else:
        if debug:
            print("Cause Not Found!")
        return []
    
    # --- Phase 2: For each candidate, disable its outgoing transitions and check the effect ---
    cause = []
    for candidate in cause_can:
        p_mod = { state: Real("p_mod_{}".format(state)) for state in all_states }
        p_can = candidate[1]
        solver = Solver()
        for state in all_states:
            if state in target:
                solver.add(p_mod[state] == 1)
            elif state not in dtmc_dict:
                solver.add(p_mod[state] == 0)
            else:
                if state == candidate[0]:
                    # Disable candidate: force its probability to 0.
                    solver.add(p_mod[state] == 0)
                else:
                    solver.add(p_mod[state] == Sum([RealVal(prob) * p_mod[dest] for dest, prob in dtmc_dict[state]]))
                    solver.add(p_mod[state] <= p_can)

        solver.add(p_mod[initial] > EPSILON)
    
        if solver.check() == sat:
            cause.append(candidate[0])
            if first_cause:
                break
    
    if len(cause) > 0:
        return cause
    else:
        if debug:
            print("Cause Not Found!!")
        return None