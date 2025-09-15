import sys
sys.path.append(".")
from src import dtmc_abstraction
from src import mdp_encoding
from src import utils
import pprint as pp
import time


def AbsRef(file, initial, target, initial_predicates=None, partition_override = [], split_method = 'single', debug = False, split_ratio = 0.5, timeout=180, state_col='State'):

    start_time = time.time()
    df, var_columns, state_id_column, next_state_columns, prob_columns = utils.parse_dtmc_csv(file, state_col=state_col)
    df = df.drop_duplicates()
    if not partition_override:
        parts = dtmc_abstraction.partition_states(df, initial_predicates, state_col=state_id_column)
    else:
        parts = partition_override

    abstract_mdp, initial_abs, target_abs, name_map = dtmc_abstraction.build_abstract_mdp_output(df, parts, next_state_columns, prob_columns, initial, target, state_id_col=state_col)

    EPSILON = 0.0001

    t0 = time.time()
    causes, state_to_split = mdp_encoding.cause_smt_mdp(abstract_mdp, initial_abs, {target_abs},  EPSILON, debug= False)
    t1 = time.time()
    print(f"Initial Step - Could not find cause (Number of states: {len(name_map.keys())}) \t Step: {t1-t0}s | Total: {t1 - start_time}s")

    refinement_count = 0

    if causes:
        if debug:
            print(f"Cause: {causes}")
        return {c: name_map[c] for c in causes}, refinement_count
    else:
        while not causes:
            t0 = time.time()
            refinement_count += 1 
            if not state_to_split:
                print("No states left to refine")
                if debug:
                    print("FINAL MDP:")
                    pp.pprint(refined_abstract_mdp)
                return {}, refinement_count
            # give abstrace_mdp, initial_abs, target_abs, name_map to refinement, get state to refine
            state_to_refine = state_to_split[0]
            
            if debug:
                print(f"State to Refine: {state_to_refine}")
            refined_parts = dtmc_abstraction.split_state(name_map, state_to_refine, method=split_method, split_ratio=split_ratio)
            refined_abstract_mdp, refined_initial_abs, refined_target_abs, refined_name_map = dtmc_abstraction.build_abstract_mdp_output(df, refined_parts, next_state_columns, prob_columns, initial, target, state_id_col=state_id_column)
            if refined_target_abs == None:
                #raise Exception("Something went wront with abstraction, no target state")
                return None, None
            if debug:
                print(f"Refined Abstract MDP:")
                pp.pprint(refined_abstract_mdp)
            causes, state_to_split = mdp_encoding.cause_smt_mdp(refined_abstract_mdp, refined_initial_abs, {refined_target_abs},  EPSILON, debug= False)
            name_map = refined_name_map
            t1 = time.time()
            if not causes:
                print(f"Step {refinement_count} - Could not find cause (Number of states: {len(name_map.keys())}) \t Step: {t1-t0}s | Total: {t1 - start_time}s")
                if t1 - start_time > timeout:
                    print(f"Timeout after {timeout} seconds")
                    return None, None
        return {c: refined_name_map[c] for c in causes}, refinement_count