from z3 import *

def compute_mapping_by_cardinality(dtmc_dict, initial, target, max_cause_cardinality, min_cause_cardinality=1, debug=False):

    # Normalize target to a set.
    if isinstance(target, set):
        target_set = target
    else:
        target_set = {target}
    
    states = set(dtmc_dict.keys())
    for transitions in dtmc_dict.values():
        for (succ, _) in transitions:
            states.add(succ)
    states = list(states)
    
    trans_map = {}
    for state, transitions in dtmc_dict.items():
        for (succ, prob) in transitions:
            trans_map[(state, succ)] = prob
    
    # Initially, candidate forbidden states: all states except initial and target(s).
    candidate_forbidden = set(states) - {initial} - target_set
    
    s = Solver()
    
    P = { state: Real(f"P_{state}") for state in states }
    for state in states:
        s.add(P[state] >= 0, P[state] <= 1)
    
    # Create a Boolean variable for each candidate forbidden state.
    # If forb[state] is True, then that state’s reachability is forced to 0.
    forb = { state: Bool(f"forb_{state}") for state in candidate_forbidden }
    

    for t in target_set:
        s.add(P[t] == 1)

    s.add(P[initial] > 0) 
    
    # Add reachability equations for every non-target state.
    for state in states:
        if state in target_set:
            continue
        if state not in dtmc_dict:
            # Dead-end state: no outgoing transitions.
            s.add(P[state] == 0)
            continue
        # Compute weighted sum over all successors (if transition missing, contributes 0).
        trans_sum = Sum([RealVal(trans_map.get((state, succ), 0)) * P[succ] for succ in states])
        if state in candidate_forbidden:
            s.add(P[state] == If(forb[state], 0, trans_sum))
        else:
            s.add(P[state] == trans_sum)
    
    # ---- Compute baseline P(effect) with no forbidden states ----
    s.push()
    # Force no candidate is forbidden.
    for state in candidate_forbidden:
        s.add(forb[state] == False)
    if s.check() == sat:
        m0 = s.model()
        p_effect = m0.evaluate(P[initial])
    else:
        raise Exception("Baseline model unsat!")
    s.pop()
    

    forb = { state: Bool(f"forb_{state}") for state in candidate_forbidden }

    n = len(candidate_forbidden)
    results = []
    found_true = False  # Flag to stop if we find an assignment meeting the condition.
    
    for k in range(max(min_cause_cardinality, 1), min(max_cause_cardinality, n)+1):
        s.push()  # Save current context.
        # Add constraint: exactly k candidate booleans are True.
        cardinality_constraint = Sum([If(forb[state], 1, 0) for state in candidate_forbidden]) == k
        s.add(cardinality_constraint)
        
        cardinality_results = []
        while s.check() == sat:
            m = s.model()
            assignment = { state: is_true(m[forb[state]]) for state in candidate_forbidden }
            p_val = m.evaluate(P[initial])
            check = simplify(p_val < (p_effect / RealVal(2)))
            mapping_value = is_true(check)
            if debug:
                print("Baseline p_effect:", p_effect, "P_cf(initial):", p_val)
            cardinality_results.append((assignment, mapping_value, p_val))
            if mapping_value:
                found_true = True
            # Block current model.
            block = []
            for state in candidate_forbidden:
                b = forb[state]
                if is_true(m[b]):
                    block.append(b)
                else:
                    block.append(Not(b))
            s.add(Not(And(block)))
        
        s.pop()
        if cardinality_results:
            results.extend(cardinality_results)
        if found_true:
            break
    
    # Prepare final output: list of lists of candidate state IDs (the ones set to True) for assignments meeting the condition.
    final_list = []
    for assignment, mapping_value, p_val in results:
        if mapping_value:
            true_states = [state for state, val in assignment.items() if val]
            final_list.append(true_states)
    
    return final_list, p_effect
