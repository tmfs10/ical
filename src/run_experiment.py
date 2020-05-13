import argparse
import numpy as np
import random
import sys
import torch
from gpu_utils.utils import gpu_init, nvidia_smi

from acquisition_method import AcquisitionMethod
from context_stopwatch import ContextStopwatch
from dataset_enum import DatasetEnum, get_targets, get_experiment_data, train_model
from random_fixed_length_sampler import RandomFixedLengthSampler
from torch_utils import get_base_indices
import torch.utils.data as data
import numpy as np
import random
from acquisition_functions import AcquisitionFunction, compute_pair_mi, generate_sample, compute_mi_sample
from reduced_consistent_mc_sampler import reduced_eval_consistent_bayesian_model
from blackhc import laaos

import blackhc.notebook

import functools
import itertools

import os


def create_experiment_config_argparser(parser):
    parser.add_argument("--batch_size", type=int, default=4, help="input batch size for training")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--scoring_batch_size", type=int, default=256, help="input batch size for scoring")
    parser.add_argument("--test_batch_size", type=int, default=256, help="input batch size for testing")
    parser.add_argument(
        "--validation_set_size",
        type=int,
        default=128,
        help="validation set size (0 for len(test_dataset) or whatever we got from the dataset)",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=1, help="# patience epochs for early stopping per iteration"
    )
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train")
    parser.add_argument("--epoch_samples", type=int, default=5056, help="number of epochs to train")
    parser.add_argument("--num_inference_samples", type=int, default=5, help="number of samples for inference")
    parser.add_argument(
        "--available_sample_k",
        type=int,
        default=10,
        help="number of active samples to add per active learning iteration",
    )
    parser.add_argument("--target_num_acquired_samples", type=int, default=800, help="max number of samples to acquire")
    parser.add_argument("--target_accuracy", type=float, default=0.98, help="max accuracy to train to")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--quickquick", action="store_true", default=False, help="uses a very reduced dataset")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--fix_numpy_python_seed", action="store_true", default=False, help="fixes seed for numpy and python as well")
    parser.add_argument("--cudnn_deterministic", action="store_true", default=False, help="fixes seed for numpy and python as well")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="how many batches to wait before logging training status"
    )
    parser.add_argument(
        "--initial_samples_per_class",
        type=int,
        default=2,
        help="how many samples per class should be selected for the initial training set",
    )
    parser.add_argument(
        "--initial_sample",
        dest="initial_samples",
        type=int,
        action="append",
        help="sample that needs to be part of the initial samples (instead of sampling initial_samples_per_class)",
        default=None,
    )
    parser.add_argument(
        "--file_with_initial_samples",
        dest="file_with_initial_samples",
        type=str,
        default="",
    )
    parser.add_argument(
        "--type",
        type=AcquisitionFunction,
        default=AcquisitionFunction.bald,
        help=f"acquisition function to use (options: {[f.name for f in AcquisitionFunction]})",
    )
    parser.add_argument(
        "--acquisition_method",
        type=AcquisitionMethod,
        default=AcquisitionMethod.multibald,
        help=f"acquisition method to use (options: {[f.name for f in AcquisitionMethod]})",
    )
    parser.add_argument(
        "--dataset",
        type=DatasetEnum,
        default=DatasetEnum.mnist,
        help=f"dataset to use (options: {[f.name for f in DatasetEnum]})",
    )
    parser.add_argument(
        "--min_remaining_percentage",
        type=int,
        default=100,
        help="how much of the available dataset should remain after culling in BatchBALD",
    )
    parser.add_argument(
        "--min_candidates_per_acquired_item",
        type=int,
        default=20,
        help="at least min_candidates_per_acquired_item*acqusition_size should remain after culling in BatchBALD",
    )
    parser.add_argument(
        "--initial_percentage",
        type=int,
        default=100,
        help="how much of the available dataset should be kept before scoring (cull randomly for big datasets)",
    )
    parser.add_argument(
        "--reduce_percentage",
        type=int,
        default=0,
        help="how much of the available dataset should be culled after each iteration",
    )
    parser.add_argument(
        "--balanced_validation_set",
        action="store_true",
        default=False,
        help="uses a balanced validation set (instead of randomly picked)"
        "(and if no validation set is provided by the dataset)",
    )
    parser.add_argument(
        "--balanced_test_set",
        action="store_true",
        default=False,
        help="force balances the test set---use with CAUTION!",
    )

    # HSIC arguments
    parser.add_argument(
        "--hsic_compute_batch_size",
        type=int,
        default=1000,
    )

    # HSIC arguments
    parser.add_argument(
        "--max_batch_compute_size",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--ical_max_greedy_iterations",
        type=int,
        default=0,
    )

    def str2bool(v):
        if v.lower() in ('true', '1', 'y', 'yes'):
            return True
        elif v.lower() in ('false', '0', 'n', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected. Got' + v)

    parser.add_argument(
        "--hsic_resample",
        type=str2bool,
        default="True",
    )

    parser.add_argument(
        "--hsic_kernel_name",
        type=str,
        default='mixrq',
    )
    
    # FASS arguments
    parser.add_argument(
        "--fass_entropy_bag_size_factor",
        type=float,
        default=30,
    )

    parser.add_argument(
        "--max_num_batch_init_samples_to_read",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--num_to_condense",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--num_inference_for_marginal_stat",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--random_ical_minibatch",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--use_orig_condense",
        action="store_true",
        default=False,
    )

    return parser



def main():
    parser = argparse.ArgumentParser(
        description="BatchBALD", formatter_class=functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=120)
    )
    parser.add_argument("--experiment_task_id", type=str, default=None, help="experiment id")
    parser.add_argument(
        "--experiments_laaos", type=str, default=None, help="Laaos file that contains all experiment task configs"
    )
    parser.add_argument(
        "--experiment_description", type=str, default="Trying stuff..", help="Description of the experiment"
    )
    parser = create_experiment_config_argparser(parser)
    args = parser.parse_args()

    if args.gpu == -1:
        gpu_id = gpu_init(best_gpu_metric="mem")
    else:
        gpu_id = gpu_init(gpu_id=args.gpu)
    print("Running on GPU " + str(gpu_id))

    if args.experiments_laaos is not None:
        config = laaos.safe_load(
            args.experiments_laaos, expose_symbols=(AcquisitionFunction, AcquisitionMethod, DatasetEnum)
        )
        # Merge the experiment config with args.
        # Args take priority.
        args = parser.parse_args(namespace=argparse.Namespace(**config[args.experiment_task_id]))

    # DONT TRUNCATE LOG FILES EVER AGAIN!!! (OFC THIS HAD TO HAPPEN AND BE PAINFUL)
    reduced_dataset = args.quickquick
    if args.experiment_task_id:
        store_name = args.experiment_task_id
        if reduced_dataset:
            store_name = "quickquick_" + store_name
    else:
        store_name = "results"

    # Make sure we have a directory to store the results in, and we don't crash!
    os.makedirs("./laaos", exist_ok=True)
    store = laaos.create_file_store(
        store_name,
        suffix="",
        truncate=False,
        type_handlers=(blackhc.laaos.StrEnumHandler(), blackhc.laaos.ToReprHandler()),
    )
    store["args"] = args.__dict__
    store["cmdline"] = sys.argv[:]

    print("|".join(sys.argv))
    print(args.__dict__)

    acquisition_method: AcquisitionMethod = args.acquisition_method

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    if args.fix_numpy_python_seed:
        np.random.seed(args.seed)
        random.seed(args.seed)
    if args.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Using {device} for computations")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    dataset: DatasetEnum = args.dataset
    samples_per_class = args.initial_samples_per_class
    validation_set_size = args.validation_set_size
    balanced_test_set = args.balanced_test_set
    balanced_validation_set = args.balanced_validation_set

    if args.file_with_initial_samples != "":
        if args.initial_samples is None:
            args.initial_samples = []
        num_read = 0
        with open(args.file_with_initial_samples) as f:
            for line in f:
                cur_samples = []
                if line.startswith("store['initial_samples']"):
                    cur_samples = [int(k) for k in line.strip().split('=')[1][1:-1].split(',')]
                    num_read += 1
                elif "chosen_targets" in line:
                    line = line.strip().split("'chosen_targets': [")[1]
                    line = line.split("]")[0]
                    cur_samples = [int(k) for k in line.split(',')]
                    num_read += 1
                args.initial_samples += cur_samples
                if num_read >= args.max_num_batch_init_samples_to_read:
                    break

    experiment_data = get_experiment_data(
        data_source=dataset.get_data_source(),
        num_classes=dataset.num_classes,
        initial_samples=args.initial_samples,
        reduced_dataset=reduced_dataset,
        samples_per_class=samples_per_class,
        validation_set_size=validation_set_size,
        balanced_test_set=balanced_test_set,
        balanced_validation_set=balanced_validation_set,
    )

    test_loader = torch.utils.data.DataLoader(
        experiment_data.test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    train_loader = torch.utils.data.DataLoader(
        experiment_data.train_dataset,
        sampler=RandomFixedLengthSampler(experiment_data.train_dataset, args.epoch_samples),
        batch_size=args.batch_size,
        **kwargs,
    )

    available_loader = torch.utils.data.DataLoader(
        experiment_data.available_dataset, batch_size=args.scoring_batch_size, shuffle=False, **kwargs
    )

    validation_loader = torch.utils.data.DataLoader(
        experiment_data.validation_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    store["iterations"] = []
    # store wraps the empty list in a storable list, so we need to fetch it separately.
    iterations = store["iterations"]

    store["initial_samples"] = experiment_data.initial_samples

    acquisition_function: AcquisitionFunction = args.type
    max_epochs = args.epochs

    for iteration in itertools.count(1):

        def desc(name):
            return lambda engine: "%s: %s (%s samples)" % (name, iteration, len(experiment_data.train_dataset))

        with ContextStopwatch() as train_model_stopwatch:
            early_stopping_patience = args.early_stopping_patience
            num_inference_samples = args.num_inference_samples
            log_interval = args.log_interval

            model, num_epochs, test_metrics = dataset.train_model(
                train_loader,
                test_loader,
                validation_loader,
                num_inference_samples,
                max_epochs,
                early_stopping_patience,
                desc,
                log_interval,
                device,
            )
        target_size = max(args.min_candidates_per_acquired_item * args.available_sample_k, len(available_loader.dataset) * args.min_remaining_percentage // 100)
        result = reduced_eval_consistent_bayesian_model(
            bayesian_model=model,
            acquisition_function=AcquisitionFunction.predictive_entropy,
            num_classes=dataset.num_classes,
            k=args.num_inference_samples,
            initial_percentage=args.initial_percentage,
            reduce_percentage=args.reduce_percentage,
            target_size=target_size,
            available_loader=available_loader,
            device=device,
        )
        print("entropy score shape:",result.scores_B.numpy().shape)
        entropy_score = result.scores_B.numpy().mean()
        to_store = {}
        with ContextStopwatch() as batch_acquisition_stopwatch:
            ret = acquisition_method.acquire_batch(
                bayesian_model=model,
                acquisition_function=acquisition_function,
                available_loader=available_loader,
                num_classes=dataset.num_classes,
                k=args.num_inference_samples,
                b=args.available_sample_k,
                min_candidates_per_acquired_item=args.min_candidates_per_acquired_item,
                min_remaining_percentage=args.min_remaining_percentage,
                initial_percentage=args.initial_percentage,
                reduce_percentage=args.reduce_percentage,
                max_batch_compute_size=args.max_batch_compute_size,
                hsic_compute_batch_size=args.hsic_compute_batch_size,
                hsic_kernel_name=args.hsic_kernel_name,
                fass_entropy_bag_size_factor=args.fass_entropy_bag_size_factor,
                hsic_resample=args.hsic_resample,
                ical_max_greedy_iterations=args.ical_max_greedy_iterations,
                device=device,
                store=to_store,
                random_ical_minibatch=args.random_ical_minibatch,
                num_to_condense=args.num_to_condense,
                num_inference_for_marginal_stat=args.num_inference_for_marginal_stat,
                use_orig_condense=args.use_orig_condense,
            )
            if type(ret) is tuple:
                batch, time_taken = ret
            else:
                batch = ret

        probs_B_K_C = result.logits_B_K_C.exp()
        B, K, C = list(probs_B_K_C.shape) # (pool size, num samples, num classes)
        probs_B_C = probs_B_K_C.mean(dim=1)
        prior_entropy = -(probs_B_C * probs_B_C.log()).sum(dim=-1) # (batch_size,)
        mi = 0.
        post_entropy = 0.
        sample_B_K_C = generate_sample(probs_B_K_C)
        for i, idx in enumerate(batch.indices[:100]):
            cur_post_entropy = compute_pair_mi(idx, i, probs_B_K_C, probs_B_C, sample_B_K_C)
            mi += (prior_entropy[i]-cur_post_entropy)/len(batch.indices)
            post_entropy += cur_post_entropy/len(batch.indices)
        mi = mi.item()
        post_entropy = post_entropy.item()
        print('post_entropy', post_entropy, 'mi', mi)

        prior_entropy, mi = compute_mi_sample(probs_B_K_C, sample_B_K_C, num_samples=50)
        print('prior_entropy', prior_entropy, 'unpooled interdependency', mi)


        original_batch_indices = get_base_indices(experiment_data.available_dataset, batch.indices)
        print(f"Acquiring indices {original_batch_indices}")
        targets = get_targets(experiment_data.available_dataset)
        acquired_targets = [int(targets[index]) for index in batch.indices]
        print(f"Acquiring targets {acquired_targets}")

        iterations.append(
            dict(
                num_epochs=num_epochs,
                test_metrics=test_metrics,
                active_entropy=entropy_score,
                chosen_targets=acquired_targets,
                chosen_samples=original_batch_indices,
                chosen_samples_score=batch.scores,
                chosen_samples_orignal_score=batch.orignal_scores,
                train_model_elapsed_time=train_model_stopwatch.elapsed_time,
                batch_acquisition_elapsed_time=batch_acquisition_stopwatch.elapsed_time,
                prior_pool_entropy=prior_entropy,
                batch_pool_mi=mi,
                **to_store,
            )
        )

        experiment_data.active_learning_data.acquire(batch.indices)

        num_acquired_samples = len(experiment_data.active_learning_data.active_dataset) - len(
            experiment_data.initial_samples
        )
        if num_acquired_samples >= args.target_num_acquired_samples:
            print(f"{num_acquired_samples} acquired samples >= {args.target_num_acquired_samples}")
            break
        if test_metrics["accuracy"] >= args.target_accuracy:
            print(f'accuracy {test_metrics["accuracy"]} >= {args.target_accuracy}')
            break

    with ContextStopwatch() as train_model_stopwatch:
        early_stopping_patience = args.early_stopping_patience
        num_inference_samples = args.num_inference_samples
        log_interval = args.log_interval

        model, num_epochs, test_metrics = dataset.train_model(
            train_loader,
            test_loader,
            validation_loader,
            num_inference_samples,
            max_epochs,
            early_stopping_patience,
            desc,
            log_interval,
            device,
        )
    target_size = max(args.min_candidates_per_acquired_item * args.available_sample_k, len(available_loader.dataset) * args.min_remaining_percentage // 100)
    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=model,
        acquisition_function=AcquisitionFunction.predictive_entropy,
        num_classes=dataset.num_classes,
        k=args.num_inference_samples,
        initial_percentage=args.initial_percentage,
        reduce_percentage=args.reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    probs_B_K_C = result.logits_B_K_C.exp()
    B, K, C = list(probs_B_K_C.shape) # (pool size, num samples, num classes)
    probs_B_C = probs_B_K_C.mean(dim=1)
    prior_entropy = -(probs_B_C * probs_B_C.log()).sum(dim=-1) # (batch_size,)

    print('post_entropy', prior_entropy.mean().item(), 'mi', mi)

    print("DONE")


if __name__ == "__main__":
    main()
