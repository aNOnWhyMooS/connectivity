import argparse
import shlex
import subprocess

def get_parser():
    parser = argparse.ArgumentParser(description="Fine-tune a single pt-checkpoint's multiple \
                                                  times with different seeds on MNLI")
    parser.add_argument("--hub_token", type=str, required=True,
                        help="The token of HF-hub to use for pushing models.")
    parser.add_argument("--ft_seeds", type=str, 
                        help="Space separated values ft_seeds(all ints).",
                        required=True)
    parser.add_argument("--pt_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="mnli")
    parser.add_argument("--save_eval_steps", type=str, default="10000")
    return parser

def is_git_lfs_available():
    ret_code = subprocess.run(shlex.split("git lfs")).returncode
    if ret_code==0:
        return True
    elif ret_code==1:
        return False
    else:
        raise EnvironmentError(f"git lfs is not properly installed. Got \
            unexpected return code {ret_code} when running \"git lfs\"")
    

def install_git_lfs():
    out = subprocess.run(shlex.split("wget https://github.com/git-lfs/git-lfs/releases/download/v3.0.2/git-lfs-linux-amd64-v3.0.2.tar.gz"), capture_output=True)
    if out.returncode!=0:
        raise EnvironmentError(f"""Failed to download git-lfs via wget. Got return code {out.returncode}.\
            Output from run: {out.stdout.decode("UTF-8")} \n {out.stderr.decode("UTF-8")}""")
    out = subprocess.run(shlex.split("tar -xzvf git-lfs-linux-amd64-v3.0.2.tar.gz"), capture_output=True)
    if out.returncode!=0:
        raise EnvironmentError(f"""Failed to untar git-lfs. Got return code {out.returncode}.\
            Output from run: {out.stdout.decode("UTF-8")} \n {out.stderr.decode("UTF-8")}""")
    out = subprocess.run(shlex.split("bash install.sh"), capture_output=True)
    if out.returncode!=0:
        raise EnvironmentError(f"""Failed to install git-lfs. Got return code {out.returncode}.\
            Output from run: {out.stdout.decode("UTF-8")} \n {out.stderr.decode("UTF-8")}""")
    
    subprocess.run("rm -rf man/")
    subprocess.run("rm git-lfs-linux-amd64-v3.0.2.tar.gz")
    
    return True

if __name__=="__main__":
    parser = get_parser()
    ft_command = "python run_flax_glue.py \
        --model_name_or_path {2} \
        --task_name {5} \
        --max_seq_length 128 \
        --learning_rate 2e-5 \
        --num_train_epochs 12 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 512 \
        --eval_steps {6} --save_steps {6}\
        --output_dir {3}long_cpts_corr_init_shuff_clipped_warmed_seq_len_128_{4}_{5}_ft_{1}/ \
        --seed {1} --push_to_hub --hub_token {0} --warmup_steps {7} --weight_decay 0.01"

    if not is_git_lfs_available():
        install_git_lfs()

    args = parser.parse_args()
    output_dir = args.output_dir.rstrip("/")+"/"
    model_name = args.pt_model_name_or_path.split("/")[-1]
    
    warmup_steps = {"mnli": 3681, "qqp": 3411, "cola": 80,}.get(args.task_name, 0)
    
    print(f"Finetuning with num warmup steps:", warmup_steps, flush=True)

    for seed in args.ft_seeds.split():
        final_cmd = ft_command.format(args.hub_token, seed, args.pt_model_name_or_path, output_dir, model_name, args.task_name, args.save_eval_steps, warmup_steps)
        print("Running Command: ", final_cmd)
        subprocess.call(shlex.split(final_cmd))
        print(f"Completed fine-tuning of with seed {seed} of {args.pt_model_name_or_path}", flush=True)
