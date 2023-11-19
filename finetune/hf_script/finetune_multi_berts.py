import argparse
import shlex
import subprocess

def get_parser():
    parser = argparse.ArgumentParser(description="Fine-tune multi-berts on MNLI")
    parser.add_argument("--seeds", type=int, nargs="+", 
                        help="The multibert models with these seeds will be pretrained.")
    parser.add_argument("--models_per_multibert", type=int, default=5,
                        help="Number of fine-tuned models per multi-bert pretrained model.\
                            Finetuning runs will use seeds from 0, 1...models_per_multibert-1.")
    parser.add_argument("--hub_token", type=str, required=True,
                        help="The token of HF-hub to use for pushing models.")
    parser.add_argument("--seed_pairs", type=str, help="Space separated entries of form pt_seed,ft_seed\
                        .To specify particular ft seeds to use for particular pretrained models.")
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
        --model_name_or_path google/multiberts-seed_{0} \
        --task_name mnli \
        --max_seq_length 512 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 32 \
        --eval_steps 10000 --save_steps 10000\
        --output_dir ./multibert_mnli_ft/multiberts_seed_{0}_ft_{2}/ \
        --seed {2} --push_to_hub --hub_token {3}"

    if not is_git_lfs_available():
        install_git_lfs()

    args = parser.parse_args()
    
    if args.seeds is not None:
        for multibert_seed in args.seeds:
            for ft_seed in range(args.models_per_multibert):
                dir_name = str(multibert_seed)+"_"+str(ft_seed)
                final_cmd = ft_command.format(multibert_seed, dir_name, ft_seed, args.hub_token)
                subprocess.call(shlex.split(final_cmd))
                print(f"Completed fine-tuning of multibert seed {multibert_seed} with ft seed {ft_seed}", flush=True)
    
    if args.seed_pairs is not None:
        seed_pairs = [elem.split(",") for elem in args.seed_pairs.split()]
        for pair in seed_pairs:
            multibert_seed = int(pair[0])
            ft_seed = int(pair[1])
            dir_name = str(multibert_seed)+"_"+str(ft_seed)
            final_cmd = ft_command.format(multibert_seed, dir_name, ft_seed, args.hub_token)
            subprocess.call(shlex.split(final_cmd))
            print(f"Completed fine-tuning of multibert seed {multibert_seed} with ft seed {ft_seed}", flush=True)