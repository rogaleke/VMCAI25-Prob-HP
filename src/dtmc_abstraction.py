import pandas as pd
import numpy as np
import re
import pprint
import random

def partition_states(df, predicates, state_col='State'):
    remaining = df.copy()
    partitions = {}

    for pred in predicates:
        try:
            matched = remaining.query(pred)
        except Exception as e:
            print(f'Error on predicate: {e}')
            continue

        if not matched.empty:
            partitions[pred] = matched[state_col].tolist()
            remaining = remaining.drop(matched.index)
    
    for i, row in remaining.iterrows():
        state = row[state_col]
        partitions[f"not_abs_{state}"] = [state.astype(int).item()]

    return partitions

def build_abstract_mdp_output(df, partitions, next_state_columns, prob_columns, initial_state, target_set, state_id_col='State'):

    target_abs = None
    partition_representative = {part_label: f'abs_{states[0]}' for part_label, states in partitions.items() if states}
    partitions = {k:v for k,v in partitions.items() if v}
    for part_label, part_set in partitions.items():
        if initial_state in part_set:
            initial_abs = partition_representative[part_label]
        #only works if all effects are in the same partition
        elif target_set.intersection(set(part_set)) == target_set:
            target_abs = partition_representative[part_label]

    # Build mapping from each original state id to its partition label
    state_to_abs = {}
    for part_label, states in partitions.items():
        for s in states:
            state_to_abs[s] = part_label

    state_external = {}
    for idx, row in df.iterrows():
        state_id = row[state_id_col]
        current_part = state_to_abs.get(state_id)
        ext_trans = {}
        
        # Check each transition pair.
        for ns_col, prob_col in zip(next_state_columns, prob_columns):
            ns_val = row[ns_col]
            prob_val = row[prob_col]
            if pd.isna(ns_val) or pd.isna(prob_val):
                continue
            ns_val = int(ns_val)
            target_part = state_to_abs.get(ns_val)
            if target_part is None:
                continue
            if target_part != current_part:
                # Use id for the target abstract state.
                target_rep = partition_representative[target_part]
                ext_trans[target_rep] = ext_trans.get(target_rep, 0.0) + prob_val
        state_external[state_id] = ext_trans

    #group states by their external transition signature.
    mdp = {}
    for part_label, states in partitions.items():
        rep_id = partition_representative[part_label]
        # Group states by signature: a sorted tuple of (target, rounded probability) pairs.
        signature_groups = {}
        for s in states:
            trans = state_external.get(s, {})
            # Create signature: empty tuple if no external transitions.
            sig = tuple(sorted(
                (target, round(prob, 6).item() if isinstance(prob, np.floating) else round(prob, 6))
                for target, prob in trans.items()
            ))
            signature_groups.setdefault(sig, []).append(s)
        
        # For each signature group, record one action.
        actions = {}
        action_num = 0
        for sig, state_group in signature_groups.items():
            # The action's transitions is just the list of (target, probability) pairs.
            # (If sig is empty, the action will have an empty list. (terminal states))
            transitions = list(sig)
            actions[f'act_{action_num}'] = transitions
            action_num += 1
        
        mdp[rep_id] = actions
        partition_mapping = {partition_representative[part_label]: states for part_label, states in partitions.items() if states}
    
    return mdp, initial_abs, target_abs, partition_mapping


def split_state(parts, state_to_refine, method='single', split_ratio = 0.5):
    original_state = parts[state_to_refine]
    new_parts = parts.copy()
    del new_parts[state_to_refine]
    if method == 'single':
        for state in original_state:
            new_parts[f'abs_{state}'] = [state]
    elif method == 'rand_half':
        new1 = sorted(random.sample(original_state, max(1,len(original_state)//2)))
        new2 = sorted(list(set(original_state) - set(new1)))
        new_parts[f'abs_{new1[0]}'] = new1
        new_parts[f'abs_{new2[0]}'] = new2
    elif method == 'rand_split':
        new1 = sorted(random.sample(original_state, max(1,int(len(original_state)*split_ratio))))
        new2 = sorted(list(set(original_state) - set(new1)))
        new_parts[f'abs_{new1[0]}'] = new1
        new_parts[f'abs_{new2[0]}'] = new2
    elif method == 'rand_split_single':
        new1 = sorted(random.sample(original_state, max(1,int(len(original_state)*split_ratio))))
        new_temp = sorted(list(set(original_state) - set(new1)))
        new_parts[f'abs_{new1[0]}'] = new1
        for state in new_temp:
            new_parts[f'abs_{state}'] = [state]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'single' or 'rand_half'.")
    return new_parts
