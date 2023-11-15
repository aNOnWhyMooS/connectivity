"""Plots variation of loss/accuracy on non-linear path connecting 2 models.
"""
import re
import ast
import argparse
import warnings
from typing import List, Dict, Tuple, Optional, Literal
from functools import partial

import matplotlib.pyplot as plt

from constellations.utils.load_interpols import load_logs

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_files',
        nargs = '+',
        type = str,
        help = ('File having logs measuring performance on the non-linear path.'
                 'All files must correspond to same path.')
     )
     
    parser.add_argument(
        '--metric',
        default = 'loss',
        type = str,
        help = 'Metric to plot.'
    )
     
    parser.add_argument(
        '--overlay_linear',
        action='store_true',
        help='Linear interpolation will be overlayed.'
    )

    return parser

def get_dataset(lines: List[str]) -> str:
  for line in lines:
     match_obj = re.fullmatch(r'Dataset: ([A-Za-z0-9]+)', line)
     if match_obj is not None:
          return match_obj.group(1)
  raise AssertionError('Couldn\'t find a line starting with "Dataset :"')

def get_dists(lines: List[str]) -> List[float]:
    for line in lines:
        match_obj = re.fullmatch(r'Distances between points on bent line: (.*)',
                                         line)
        if match_obj is not None:
            return ast.literal_eval(match_obj.group(1))
    raise AssertionError('Couldn\'t find a line starting'
                                ' mentioning the distances between lines.')

def get_indices(lines: List[str]) -> Tuple[int, int]:
    for line in lines:
        match_obj = re.fullmatch(r'Base Model Prefix: (.*)', line)
        if match_obj is None:
            continue
        prefix = match_obj.group(1)
        match_obj = re.search(r'/indices_(\d+)_(\d+)/', prefix)
        if match_obj is None:
            raise AssertionError(f'The base model prefix: {prefix}'
                                        f'doesn\'t have indices.')
        return (int(match_obj.group(1)), int(match_obj.group(2)))

def get_steps(lines: List[str]) -> Tuple[int, int]:
    for line in lines:
        match_obj = re.fullmatch(r'Base Model Prefix: (.*)', line)
        if match_obj is None:
            continue
        prefix = match_obj.group(1)
        match_obj = re.search(r'/num_steps_(\d+)_(\d+)/', prefix)
        if match_obj is None:
            raise AssertionError(f'The base model prefix: {prefix}'
                                        f'doesn\'t have num_steps.')
        return (int(match_obj.group(1)), int(match_obj.group(2)))


def get_metrics(lines: List[str]) -> List[str]:
    for i, line in enumerate(lines):
        match_obj = re.fullmatch(r'([-]+\s\s)+[-]+', line)
        if match_obj is not None:
            if i==0:
                continue
            return lines[i-1].split()
    raise AssertionError('Couldn\'t find any metric names.')
        

def split_segment_wise(lines: List[str]) -> List[List[str]]:
    segment_wise_lines = []
    for line in lines:
        match_obj = re.fullmatch(r'Interpolating in segment from (\d+) to (\d+)', line)
        if match_obj is not None:
            segment_wise_lines.append([])
        elif len(segment_wise_lines)==0:
            continue
        else:
            segment_wise_lines[-1].append(line)
    return segment_wise_lines

def segment_to_dict(lines: List[str], fields: List[str],
                          metric: Optional[str]=None) -> List[Dict[str, float]]:
    metrics = fields if metric is None else [metric]
    segment = []
    for line in lines[3::4]:
        segment.append({
            field: float(e)
            for field, e in zip(fields, line.split())
            if field in metrics
        })
    return segment


def parse_data(lines: List[str], metric: str) -> Tuple[str, List[List[float]]]:
    dataset = get_dataset(lines)
    fields = get_metrics(lines)
    segment_wise_lines = split_segment_wise(lines)
    segment_dicts = map(partial(segment_to_dict, fields=fields, 
                                         metric=metric),
                              segment_wise_lines)
    segment_wise_metrics= [[e[metric] for e in segment]
                                  for segment in segment_dicts]
    return dataset, segment_wise_metrics

def check_files(args) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    prev_steps, prev_indices = None, None
    for f in args.log_files:
        with open(f, 'r') as g:
            lines = [line.strip() for line in g.readlines()]
        
        steps = get_steps(lines)
        indices = get_indices(lines)
        
        if not ((prev_steps is None or prev_steps==steps) and
                  (prev_indices is None or prev_indices==indices)):
            warnings.warn('Got files with different non-linear paths.')
            return None
        
        prev_steps = steps
        prev_indices = indices
    return steps, indices

def plot_bent_lines(args, xaxis: Literal['point_num', 'dist'] = 'point_num') -> int:

    to_plot = []
    end_points =  {}
    for f in args.log_files:
        with open(f, 'r') as g:
            lines = [line.strip() for line in g.readlines()]
        dataset, vals = parse_data(lines, args.metric)
        dists = get_dists(lines)
        steps = get_steps(lines)
        indices = get_indices(lines)
        to_plot.append((dataset, (dists, vals, (*steps, *indices))))
    
    for dataset, (dists, vals, sni) in to_plot:
        joined_segments = [e for lis in vals for e in lis]
        t = len(joined_segments)
        
        if xaxis == 'point_num':
            xs = range(t)
            
            x = 0
            for i, lis in enumerate(vals[:-1]):
                x += len(lis)
                plt.axvline(x=x, linestyle='--',)
            end_points[sni] = x + len(vals[-1])
        
        else:
            curve_length = sum(dists)
            xs = [curve_length*(i/(t-1)) for i in range(t)]

            x = 0
            for i, dist in enumerate(dists[:-1]):
                x += dist
                plt.axvline(x=x, linestyle='--',)
            end_points[sni] = x + dists[-1]

        label = f'NL. path {sni[2]}@{sni[0]} -> {sni[3]}@{sni[1]} on {dataset}.'
        print(f'plotting for : {label}')
        plt.plot(xs, joined_segments, label=label)
        
    plt.xlabel('Point number' if xaxis=='point_num' else 'Distance along curve')
    return end_points

def get_linear(metric, steps, indices):
    assert steps[0]==steps[1]
    interpol_logs = load_logs(('../../conss/logs/NLI/long_qqp_berts/'
                               f'qqp_interpol@{steps[0]}steps/'),
                              metric=metric)
    return interpol_logs[tuple(map(str, indices))]

def uniform_spaced(tot_dist, num_points, add_last_point=False):
    dists = [(i/num_points)*tot_dist for i in range(num_points)]
    if add_last_point:
        dists.append(tot_dist)
    return dists

def plot_linear(args, steps: Tuple[int, int],
                indices: Tuple[int, int], tot_points=None):
    interpol_vals = get_linear(args.metric, steps, indices)
    if tot_points is None:
        tot_points = interpol_vals[0]
    xs = uniform_spaced(tot_points, 9, add_last_point=True)
    ys = interpol_vals[1]
    plt.plot(xs, ys,
             label=f'L. path {indices[0]}@{steps[0]} -> {indices[1]}@{steps[1]}.',
             alpha=0.5)

def main(args):
    out = check_files(args)
    if out is not None:
        steps, indices = out
        tot_points = plot_bent_lines(args, xaxis='point_num')[(*steps, *indices)]
        if args.overlay_linear:
            plot_linear(args, steps, indices, tot_points=tot_points)
        plt.title(f'Bent path between model {indices[0]}@{steps[0]} steps'
                    f' and model {indices[1]}@{steps[1]} steps')    
        plt.savefig(f'{indices[0]}_{steps[0]}_{indices[1]}_{steps[1]}.pdf')
    else:
        end_points = plot_bent_lines(args, xaxis='dist')
        if args.overlay_linear:
            for f in args.log_files:
                with open(f, 'r') as g:
                    lines = [line.strip() for line in g.readlines()]
                steps, indices = get_steps(lines), get_indices(lines)
                print(end_points.keys())
                plot_linear(args, steps, indices, end_points[(*steps, *indices)])
        plt.title('Bent path between models')
        plt.savefig('bent_paths.pdf')
    plt.legend()
    plt.ylabel(f'{args.metric}')
    plt.show()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)