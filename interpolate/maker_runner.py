cmd = "\"python3 interpolate_1d.py --base_model {9} --from_model_type {8}\
    --suffix_pairs {0} --base_models_prefix {1} --save_file {2} --dataset {3} --split {4}\
    --num_steps {5} --experiment_id interpolate_{7}_{3}_{4}_{5}_{6} #--permute_wts\""

import argparse  
import subprocess
import shlex

def get_parser():
    parser=argparse.ArgumentParser()
    
    parser.add_argument(
        "--all_suffixes",
        type=str,
        nargs="+"
    )
    
    parser.add_argument(
        "--base_models_prefix",
        type=str,
        required=True,
        help="Common prefix of models to be loaded",
    )
    
    parser.add_argument(
        "--save_file",
        type=str,
        help="Name of file where to save the interpolation values,\
        using pickle."
    )

    
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnli",
        choices=["mnli", "hans", "cola", "qqp"],
        help="dataset [mnli, hans, cola, qqp]",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "in_domain_dev", "out_domain_dev", "validation"],
        help="dataset [train, test](mnli/hans) or \
            [train, in_domain_dev, out_domain_dev, validation](cola).",
    )
    
    parser.add_argument(
        "--num_steps",
        type=int,
        help="Will pick models from commit having these number of steps\
            mentioned in its commit message. path_or_name can't be local\
            directory if specifying this argument."
    )

    parser.add_argument(
        "--free_overlays_begin",
        type=int,
        help="Free overlay numbers begin idx-1",
        required=True
    )
    
    parser.add_argument(
        "--user_name",
        type=str,
        default="rb5139",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory within the logs directory,\
            where experiment results are to be logged."
    )

    parser.add_argument(
        "--total_jobs",
        type=int,
        default=20,
    )
    
    parser.add_argument(
       "--from_model_type",
       type=str,
       choices=["pt", "flax", "tf"],
       help="Type of model on HF hub",
       default="flax",
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        help="The base model from which the collection of models was finetuned.\
                Its tokenizer will be used.",
        required=True,
    )
    return parser

if __name__=="__main__":
    parser=get_parser()
    args = parser.parse_args()
    assert args.save_file.endswith(".pkl")
    
    to_interpolate = args.all_suffixes
    suffix_pairs = []
    for i, elem1 in enumerate(to_interpolate):
        for j, elem2 in enumerate(to_interpolate):
            if i<j:
                suffix_pairs.append(elem1+","+elem2)
    
    cmds = []
    overlay_nums = [str(args.free_overlays_begin)]
    
    total_interpols = len(suffix_pairs)
    step_size = total_interpols//(args.total_jobs-1)
    
    for i in range(args.total_jobs):
        cmds.append(cmd.format(" ".join([str(elem) for elem in suffix_pairs[0:step_size]]),
                                args.base_models_prefix, args.save_file[:-4]+str(i)+".pkl", args.dataset, args.split,
                                str(args.num_steps), str(i), args.base_models_prefix.replace("/", "_"), args.from_model_type,
                                args.base_model))
        
        suffix_pairs = suffix_pairs[step_size:]
        
        overlay_nums.append(str(int(overlay_nums[-1])+1))
    

overlay_nums = overlay_nums[1:]

models_from = args.base_models_prefix.strip("/").split("/")[-1]
command = ("python3 sbatch_maker.py --cmds "+" ".join(cmds)+" --overlay_file_nums "+" ".join(overlay_nums)+f" --sbatch_file_prefix {args.dataset}_{args.split}_{models_from}_ft-"
            +f" --user_name {args.user_name}"+f" --log_dir {args.log_dir}")
            
print(command)
subprocess.run(shlex.split(command))
