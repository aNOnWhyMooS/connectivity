import os, re
from typing import Dict, List, Tuple

def get_interpolation_wise_data(lines):    
    interpol_wise = []
    for line in lines:
        match_obj = re.match(r"Euclidean distance between (.+) and (.+): (\d+\.\d+).", line)
        if match_obj is not None:
            interpol_wise.append([])
        if len(interpol_wise)!=0:
            interpol_wise[-1].append(line)
    return interpol_wise

def get_data_from_strs(interpol_strs, idx=1):
    match_obj = re.match(r"Euclidean distance between (.+) and (.+): (\d+\.\d+).", interpol_strs[0])
    interpol_from = match_obj[2]
    interpol_to = match_obj[1]
    euc_dist = float(match_obj[3])
    loss_vals = []

    #print(interpol_strs)
    for i, st in enumerate(interpol_strs):
        if re.fullmatch(r"(-{3,100}\s+)+-{3,100}", st) is not None:
            loss_vals.append(float(interpol_strs[i+1].split()[idx]))
    if len(loss_vals)!=10:
        print(interpol_strs)
        raise ValueError(f"Number of loss values read, must be 10! Instead read: {loss_vals}\
            between {interpol_from}, {interpol_to}")
    return interpol_from, interpol_to, euc_dist, loss_vals

def load_file(filename, idx=1):
    assert idx in [1,2]
    
    vals = dict()
    
    with open(filename) as f:
        lines = [elem.strip() for elem in f.readlines()]
    
    interpol_wise = get_interpolation_wise_data(lines)
    for interpolation_data in interpol_wise:
        i, j, dist, losses = get_data_from_strs(interpolation_data, idx)
        vals[(i,j)] = (dist, losses)
        vals[(j,i)] = (dist, losses[::-1])
        if (i,i) not in vals:
            vals[(i,i)] = (0, [losses[0]]*10)
        else:
            assert vals[(i,i)]==(0, [losses[0]]*10)
        if (j,j) not in vals:
            vals[(j,j)] = (0, [losses[-1]]*10)
        else:
            assert vals[(j,j)]==(0, [losses[-1]]*10)
    return vals

def load_logs(directory, metric="loss") -> Dict[Tuple[str,str], Tuple[float, List[float]]]:
    """
    Returns:
        A dictionary mapping model pairs (i,j) to a tuple of (euclidean distance between models, 
        and a list of metric values on the interpolation between the two models).
    Note:
        1. Entry for (i,j) has metric values encountered when going from i to j. 
        2. Separate entries for (i,j) and (j,i) are there.
    """
    vals = dict()
    idx = 1 if metric=="loss" else 2
    for filename in os.listdir(directory):
        filename = os.path.join(directory, filename)
        if (not os.path.isfile(filename) or 
                filename.endswith(".err") or 
                filename.endswith(".pkl")):
            continue
        vals.update(load_file(filename, idx))
    return vals