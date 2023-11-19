cmd = "\"python3 finetune_from_pt_ckpt.py --hub_token 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 \
    --ft_seeds \\\\\\\"{0}\\\\\\\" --pt_model_name_or_path {1} --output_dir {2} --task_name {3} \
    --save_eval_steps {4}\""

import argparse  
import subprocess
import shlex

def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--all_ft_seeds",
        type=int,
        nargs="+"
    )
    parser.add_argument(
        "--pt_model_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--models_save_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="mnli",
    )
    parser.add_argument(
        "--save_eval_steps",
        type=int,
        default=10000,
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


    return parser

if __name__=="__main__":
    parser=get_parser()
    args = parser.parse_args()
    to_finetune = args.all_ft_seeds
    cmds = []
    overlay_nums = [str(args.free_overlays_begin)]

    while len(to_finetune)!=0:
        cmds.append(cmd.format(" ".join([str(elem) for elem in to_finetune[:4]]),
                                args.pt_model_name_or_path, args.models_save_dir, 
                                args.task_name, args.save_eval_steps))
        
        to_finetune = to_finetune[4:]
        
        overlay_nums.append(str(int(overlay_nums[-1])+1))

overlay_nums = overlay_nums[1:]

command = ("python3 sbatch_maker.py --cmds "+" ".join(cmds)+" --overlay_file_nums "+" ".join(overlay_nums)+f" --sbatch_file_prefix {args.task_name}_ft-"
            +f" --user_name {args.user_name}")
            
print(command)
subprocess.run(shlex.split(command))