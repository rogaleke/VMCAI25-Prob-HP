from z3 import *
import sys
sys.path.append(".")
from src.utils import z3_max, z3_min, get_float


def cause_smt_mdp(mdp_dict, initial, target,  EPSILON, max_splits=1, first_cause=True, debug= False):

    #mdp_dict = {state: actions for state, actions in mdp_dict.items()
    #            if any(transitions for transitions in actions.values())}

    all_states = set(mdp_dict.keys())

    #select state with most actions
    split_candidates = [k for k in mdp_dict if k not in target and len(mdp_dict[k]) > 1]
    if split_candidates:
        split_state = [max(split_candidates, key=lambda k: len(mdp_dict[k]))]
    else:
        split_state = None

    for actions in mdp_dict.values():
        for transitions in actions.values():
            for (dest, prob) in transitions:
                all_states.add(dest)


    P_max = { state: Real("p_max_{}".format(state)) for state in all_states }
    P_min = { state: Real("p_min_{}".format(state)) for state in all_states }
    P_range = {state: Real("p_range_{}".format(state)) for state in all_states }
    Candidate_flags = {state: Bool("c_flag_{}".format(state)) for state in all_states}

    solver = Solver()


    for state in all_states:
        if state in target:
            solver.add(P_max[state] == 1)
            solver.add(P_min[state] == 1)
        elif state not in mdp_dict:
            solver.add(P_max[state] == 0)
            solver.add(P_min[state] == 0)
        else:
            actions = mdp_dict[state]
            solver.add(P_max[state] == z3_max([
                Sum([prob * P_max[dest] for dest, prob in transitions])
                for transitions in mdp_dict[state].values()]))
            solver.add(P_min[state] == z3_min([
                Sum([prob * P_min[dest] for dest, prob in transitions])
                for transitions in mdp_dict[state].values()]))
            solver.add(P_range[state] == P_max[state] - P_min[state])
            solver.add(Candidate_flags[state] == (P_min[state] > P_min[initial]))


    solver.add(P_max[initial] > EPSILON)



    cause_can = list()
    cause = list()
    state_range = list()

    if solver.check() == sat:


        model = solver.model()

        if debug:
            print(model)
        range_list = [[state, get_float(model.eval(P_range[state]))]
                        for state in all_states 
                        if state in mdp_dict.keys() and len(mdp_dict[state]) > 1 and state not in target]

        cause_can = [[state, get_float(model.eval(P_min[state])), get_float(model.eval(P_max[state]))]
                      for state, v in Candidate_flags.items() if model.eval(v) == True]
    else:
        if debug:
            print("Cannot Reach Effect")
        return None, split_state
    


    for candidate in cause_can:

        P_max = { state: Real("p_max_{}".format(state)) for state in all_states }
        can_min = RealVal(candidate[1])

        solver = Solver()

        for state in all_states:
            if state in target:               
                solver.add(P_max[state] == 1)
            elif state not in mdp_dict:
                solver.add(P_max[state] == 0)
            else:
                
                actions = mdp_dict[state] 
                action_vals_max = [Sum([prob * P_max[dest] for (dest, prob) in transitions])  if state != candidate[0] else 0 for transitions in actions.values()]
            
                solver.add(P_max[state] == z3_max(action_vals_max))
                solver.add(P_max[state] <= can_min)

        solver.add(P_max[initial] > EPSILON)

        if solver.check() == sat:
            cause.append(candidate[0])
            if first_cause:
                break

    #state_range_sorted = [-x for x in heapq.nsmallest(max_splits, state_range, key=lambda z: z[1])]
    #split_state = [item[0] for item in state_range_sorted]
    
    if len(cause) > 0:
        return cause, None
    else:
        if debug:
            print("Cause Not Found!!")
        return None, split_state



if __name__ == "__main__":

    EPSILON = 0.000001


    mdp_dict = {
        0: {
            "s1": [(1, 0.3), (2, 0.7)],
        },
        1: {
            "s2": [(3, 1.0)],
            "s3": [(4, 0.5), (5, 0.2)],
        },
        2: {
            "s3": [(6, 0.6), (5, 0.4)],
            "s2": [(3, 0.9), (5, 0.1)],
        }

    }

    # mdp_dict = {
    #     0: { "a": [(1, 0.25), (2, 0.25), (3, 0.25), (4, 0.25)] },
    #     1: { "a": [(5, 0.5), (6, 0.7)] },
    #     2: { "a": [(5, 0.5), (6, 0.5)] },
    #     3: { "a": [(5, 0.5), (6, 0.1)] },
    #     4: { "a": [(5, 0.5), (6, 0.5)] }
    # }

    # mdp_dict = {
    #     0: { "a": [(1, 0.25), (2, 0.75)] },
    #     1: { "a": [(5, 0.5), (6, 0.7)] },
    #     2: { "a": [(5, 0.5), (6, 0.5)],
    #          "b": [(5, 0.5), (6, 0.1)],
    #          "c": [(5, 0.5), (6, 0.5)]}
    #}

    # mdp_dict = {
    #     0: {
    #         "a": [(1, 1)],
    #         "b": [(2, 0.001), (3,0.999)]
    #     },
    #     2: {
    #         "a": [(10, 1)],
    #     },

    #     1: {
    #         "s3": [(4,1)],
    #     },
    #     4: {
    #         "a": [(6, 0.99), (7, 0.01)],
    #         "b": [(8, 0.01), (7, 0.99)],

    #     }

    # }

    target = {5}
    initial = 0

    cause, x  = cause_smt_mdp(mdp_dict, initial, target, EPSILON)

    print(cause)
    print(x)