import enum

from torch import nn as nn

import independent_batch_acquisition
import multi_bald
from acquisition_batch import AcquisitionBatch
from acquisition_functions import AcquisitionFunction


class AcquisitionMethod(enum.Enum):
    independent = "independent"
    multibald = "multibald"
    ical = "ical"
    icalpointwise = "icalpointwise"
    fass = "fass"
    acsfw = "acsfw"

    def acquire_batch(
        self,
        bayesian_model: nn.Module,
        acquisition_function: AcquisitionFunction,
        available_loader,
        num_classes,
        k,
        b,
        min_candidates_per_acquired_item,
        min_remaining_percentage,
        initial_percentage,
        reduce_percentage,
        max_batch_compute_size=0,
        hsic_compute_batch_size=None,
        hsic_kernel_name=None,
        hsic_resample=True,
        fass_entropy_bag_size_factor=2.0,
        ical_max_greedy_iterations=0,
        device=None,
        store=None,
        random_ical_minibatch=False,
        num_to_condense=200,
        num_inference_for_marginal_stat=100,
        use_orig_condense=False,
    ) -> AcquisitionBatch:
        target_size = max(
            min_candidates_per_acquired_item * b, len(available_loader.dataset) * min_remaining_percentage // 100
        )

        if self == self.independent:
            return independent_batch_acquisition.compute_acquisition_bag(
                bayesian_model=bayesian_model,
                acquisition_function=acquisition_function,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                available_loader=available_loader,
                device=device,
            )
        elif self == self.multibald:
            return multi_bald.compute_multi_bald_batch(
                bayesian_model=bayesian_model,
                available_loader=available_loader,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                target_size=target_size,
                device=device,
            )
        elif self == self.ical:
            return multi_bald.compute_ical(
                bayesian_model=bayesian_model,
                available_loader=available_loader,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                target_size=target_size,
                max_batch_compute_size=max_batch_compute_size,
                hsic_compute_batch_size=hsic_compute_batch_size,
                hsic_kernel_name=hsic_kernel_name,
                hsic_resample=hsic_resample,
                max_greedy_iterations=ical_max_greedy_iterations,
                device=device,
                store=store,
                num_to_condense=num_to_condense,
            )
        elif self == self.icalpointwise:
            return multi_bald.compute_ical_pointwise(
                bayesian_model=bayesian_model,
                available_loader=available_loader,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                target_size=target_size,
                max_batch_compute_size=max_batch_compute_size,
                hsic_compute_batch_size=hsic_compute_batch_size,
                hsic_kernel_name=hsic_kernel_name,
                hsic_resample=hsic_resample,
                max_greedy_iterations=ical_max_greedy_iterations,
                device=device,
                store=store,
                num_to_condense=num_to_condense,
                num_inference_for_marginal_stat=num_inference_for_marginal_stat,
                use_orig_condense=use_orig_condense,
            )
        elif self == self.fass:
            return multi_bald.compute_fass_batch(
                bayesian_model=bayesian_model,
                available_loader=available_loader,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                target_size=target_size,
                max_entropy_bag_size=int(b*fass_entropy_bag_size_factor),
                fass_compute_batch_size=hsic_compute_batch_size,
                device=device,
            )
        elif self == self.acsfw:
            return multi_bald.compute_acs_fw_batch(
                bayesian_model=bayesian_model,
                available_loader=available_loader,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                target_size=target_size,
                max_entropy_bag_size=int(b*fass_entropy_bag_size_factor),
                device=device,
            )
        else:
            raise NotImplementedError(f"Unknown acquisition method {self}!")
