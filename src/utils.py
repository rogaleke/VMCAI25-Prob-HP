import csv
from z3 import * 
import pandas as pd
import re

#Helper Functions
def z3_max(exprs):
    if not exprs:
        return None
    res = exprs[0]
    for e in exprs[1:]:
         res = If(res >= e, res, e)
    return res

def z3_min(exprs):
    if not exprs:
        return None
    res = exprs[0]
    for e in exprs[1:]:
         res = If(res <= e, res, e)
    return res

def get_float(z3_rel):
    num = z3_rel.numerator_as_long()
    denom = z3_rel.denominator_as_long()
    float_val = num / denom
    return float_val


def parse_dtmc_csv(file, state_col='State'):
    df = pd.read_csv(file)

    var_columns = sorted(
        [col for col in df.columns if re.match(r'^V\d+$', col)],
        key = lambda x: int(x[1:])
    )

    state_id_column = state_col

    next_state_columns = sorted(
        [col for col in df.columns if re.match(r'^Next_State_\d+$', col)],
        key = lambda x: int(x.split('_')[-1])
    )

    prob_columns = sorted(
        [col for col in df.columns if re.match(r'^Prob_\d+$', col)],
        key = lambda x: int(x.split('_')[-1])
    )

    df = df.fillna(value=-1)
    
    df[var_columns] = df[var_columns].apply(pd.to_numeric, errors='coerce')
    df[state_id_column] = pd.to_numeric(df[state_id_column], errors='coerce').astype(int)
    df[next_state_columns] = df[next_state_columns].apply(lambda x: pd.to_numeric(x, errors='coerce').astype(int))
    df[prob_columns] = df[prob_columns].apply(pd.to_numeric, errors='coerce')

    return df, var_columns, state_id_column, next_state_columns, prob_columns

def df_to_dtmc_dict(df, next_state_cols, prob_cols, state_col='State'):
    dtmc = {}
    for _, row in df.iterrows():
        src = int(row[state_col])
        transitions = []
        # For each transition pair, add it if both destination and probability are present.
        for ns_col, prob_col in zip(next_state_cols, prob_cols):
            ns_val = row[ns_col]
            prob_val = row[prob_col]
            if pd.isna(ns_val) or pd.isna(prob_val):
                continue
            transitions.append((int(ns_val), float(prob_val)))
        
        # If no outgoing transitions ignore
        if not transitions:
            continue
        
        dtmc[src] = transitions
    return dtmc

def CSV2DTMC(file, depth = 5):
    dtmc_dict = {}

    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            
            state = int(row['State'].strip())
            transitions = []
            
            for i in range(1, depth):
                next_state_col = f"Next_State_{i}"
                prob_col = f"Prob_{i}"
                
                next_state_val = row[next_state_col].strip() if row[next_state_col] else ""
                prob_val = row[prob_col].strip() if row[prob_col] else ""
                if next_state_val:
                    
                    next_state = int(next_state_val)
                    prob = float(prob_val) if prob_val else 0.0
                    transitions.append((next_state, prob))
            
            if transitions:
                dtmc_dict[state] = transitions

    return dtmc_dict