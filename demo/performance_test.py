import sys
import warnings
warnings.filterwarnings('ignore')
import time
sys.path.append(".")
from src import find_cause_DTMC_conditional
from src import abstraction_refinement
from src import utils

def performance_test(file, partitions, initial, abs_target, refinement_method = 'rand_half', initial_partitions = 'min_num_states', split_ratio = 0.5, debug = False, skip_concrete = False, timeout=1200, state_col='State'):
    dtmc_time = 0.0
    abstraction_refinement_time = 0.0
    steps = 0
    if not skip_concrete:
        t0 = time.time()
        dtmc_df, vars, state_ids, next_states, prob_cols = utils.parse_dtmc_csv(file, state_col=state_col)
        dtmc_df = dtmc_df.drop_duplicates()
        dtmc = utils.df_to_dtmc_dict(dtmc_df, next_states, prob_cols, state_col=state_col)
        concrete_cause = find_cause_DTMC_conditional.concrete_conditional_cause(dtmc, initial, set(partitions[abs_target]), EPSILON=0.000001, debug=False, first_cause=False)
        t1 = time.time()
        dtmc_time = t1 - t0      
    if not skip_concrete and not concrete_cause:
        print(f"Could not find cause for {file}")
        return None, None, None
    elif skip_concrete or concrete_cause:
        if not skip_concrete:
            print(f"Concrete found cause for {file} : {concrete_cause}")
        t0 = time.time()
        causes, steps = abstraction_refinement.AbsRef(file, initial, set(partitions[abs_target]), partition_override=partitions, split_method=refinement_method, split_ratio=split_ratio, timeout=timeout, state_col=state_col)
        t1 = time.time()
        abstraction_refinement_time = t1 - t0
        if causes != None and causes:
            print(f"Abstract found cause for {file} in {steps} refinement steps.")
            print(f"Cause: {causes}")
        else:
            return None, None, None
    return dtmc_time, abstraction_refinement_time, steps
