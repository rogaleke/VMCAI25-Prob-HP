import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append(".")
import demo.performance_test as performance_test

#returns a list of initial predicates for the df over non initial and non effect states.
def get_initial_predicates(df, columns, group_counts, state_col='state', method='cut', exclude = None):
    
    df_copy = df.copy()

    if exclude is not None:
        df_copy = df_copy.query(exclude)
    # Create group columns for each of the passed columns.
    for col, n_groups in zip(columns, group_counts):
        if method == 'cut':
            # Equal-width binning. Labels will be integers 0 to n_groups-1.
            df_copy[f'{col}_group'] = pd.cut(df_copy[col], bins=n_groups, labels=False)
        elif method == 'qcut':
            # Quantile-based binning. This gives roughly equal number of observations per bin.
            df_copy[f'{col}_group'] = pd.qcut(df_copy[col], q=n_groups, labels=False, duplicates='drop')
        else:
            raise ValueError("method must be either 'cut' or 'qcut'")
    
    # List of the new group columns.
    group_columns = [f'{col}_group' for col in columns]
    
    # Identify the column (or index) that contains state names.
    if state_col is None:
        state_series = df_copy.index
    else:
        state_series = df_copy[state_col]
    
    # Group by the new group columns and collect the state names for each group.
    grouped_states = df_copy.groupby(group_columns).apply(lambda x: list(state_series.loc[x.index]))
    
    return grouped_states.tolist()

def run_profile_f16_scen2(file, split_method='rand_split_single', split_ratio = 0.61, skip_concrete=False):

    file = file

    df = pd.read_csv(file)
    #df = df.drop_duplicates()

    initial_predicate = f'state == 0'
    initial_state_predicates = df.query(initial_predicate)['state'].tolist()
    effect_pred = f'V9 == True'
    effects = df.query(effect_pred)['state'].tolist()

    exclusion = f'not ({effect_pred}) and not ({initial_predicate})'
    cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']
    groups = [2, 5, 2, 2, 1, 1, 4]
    result = get_initial_predicates(df, cols, groups, state_col='state', method='cut', exclude=exclusion)

    result.insert(0, initial_state_predicates)
    result.append(effects)
    initial_predicates = {x: result[x] for x in range(len(result))}

    ef_count = len(effects)
    if not ef_count:
        print(f"NO EFFECTS FOUND {file}")
        return None, None, None
    print(f"LEN EFFECT STATES: {ef_count}")

    #sanitize empty predicates:
    initial_predicates = {k:v for k,v in initial_predicates.items() if v}

    dtmc_time, abs_time, rf_steps = performance_test.performance_test(file, initial_predicates, 0, len(result)-1, refinement_method=split_method, split_ratio=split_ratio, debug=False, skip_concrete=skip_concrete, state_col='state')
    if dtmc_time and abs_time and rf_steps:
        print(f"Concrete took {dtmc_time}s \t\t Abs took {abs_time}s in {rf_steps} steps of refinement")
    return dtmc_time, abs_time, rf_steps

if __name__ == "__main__":
    dtmc_path = r'environments\f16\dtmcs_scen2\alt1000_vel1000_ex250_time10.csv'
    run_profile_f16_scen2(dtmc_path, split_method='rand_split_single', split_ratio=0.4, skip_concrete=False)