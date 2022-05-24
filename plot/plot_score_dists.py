import argparse
from constellations.plot_utils import plot_score_dists

from constellations.utils.load_eval_logs import get_metrics

def get_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--grp1_name",
        required=True,
    )
    
    parser.add_argument(
        "--grp2_name",
        required=True,
    )
    
    parser.add_argument(
        "--eval_dir_mnli1",
        required=True,
        help="The directory having mnli .json model evaluation files \
            for model grp 1."
    )

    parser.add_argument(
        "--eval_dir_mnli2",
        required=True,
        help="The directory having mnli .json model evaluation files \
            for model grp 2."
    )

    parser.add_argument(
        "--eval_dir_hans1",
        required=True,
        help="The directory having hans .json model evaluation files \
            for model grp 1."
    )

    parser.add_argument(
        "--eval_dir_hans2",
        required=True,
        help="The directory having hans .json model evaluation files \
            for model grp 2."
    )
    
    parser.add_argument(
        "--save_file_prefix",
        required=True,
    )
    
    parser.add_argument(
        "--same_x_scale",
        action="store_true",
        help="Plot with same X-axis scale."
    )
    
    parser.add_argument(
        "--eval_metric",
        type=str,
        help="Metric to use to rank models.",
        default="accuracy",
    )

    return parser


def get_loss_and_accs(args):
    losses1 = {}
    accs1 = {}
    losses2 = {}
    accs2 = {}
    
    for heuristic in ["subsequence", "lexical_overlap", "constituent"]:
        for entailing in [True, False]:        
            key = heuristic+("_onlyEntailing" if entailing else "_onlyNonEntailing")
            
            metrics1 = get_metrics(args.eval_dir_hans1, heuristic, entailing)
            accs1[key] = [v["accuracy"] for k, v in metrics1.items()]
            losses1[key] = [v["loss"] for k, v in metrics1.items()]
            
            metrics2 = get_metrics(args.eval_dir_hans2, heuristic, entailing)
            accs2[key] = [v["accuracy"] for k, v in metrics2.items()]
            losses2[key] = [v["loss"] for k, v in metrics2.items()]
            print("done")
    
    key = "mnli"
    metrics1 = get_metrics(args.eval_dir_mnli1)
    accs1[key] = [v["accuracy"] for k, v in metrics1.items()]
    losses1[key] = [v["loss"] for k, v in metrics1.items()]
            
    metrics2 = get_metrics(args.eval_dir_mnli2)
    accs2[key] = [v["accuracy"] for k, v in metrics2.items()]
    losses2[key] =  [v["loss"] for k, v in metrics2.items()]
    
    return losses1, accs1, losses2, accs2

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    losses1, accs1, losses2, accs2 = get_loss_and_accs(args)
    
    plot_score_dists(losses1, losses2, args.grp1_name, 
                     args.grp2_name, args.save_file_prefix+"_losses.pdf",
                     sharex=args.same_x_scale)
    
    plot_score_dists(accs1, accs2, args.grp1_name, args.grp2_name,
                     args.save_file_prefix+"_accs.pdf",
                     sharex=args.same_x_scale)