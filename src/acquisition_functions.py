import torch
from blackhc.progress_bar import with_progress_bar

import torch_utils
import torch.distributions as tdist

import enum

def compute_mi_probs(logits_B_K_C, num_samples=500):
    assert False, 'something is wrong here'
    probs_B_K_C = logits_B_K_C.exp()
    B, K, C = list(logits_B_K_C.shape) # (pool size, num samples, num classes)

    probs_B_C = probs_B_K_C.mean(dim=1)
    prior_entropy = -(probs_B_C * probs_B_C.log()).sum(dim=-1) # (batch_size,)

    mi = 0.
    sample_prior_entropy = 0.
    for i_sample in range(num_samples):
        i1 = torch.randint(low=0, high=B, size=(1,)).item()
        i2 = torch.randint(low=0, high=B, size=(1,)).item()
        if i2 == i1:
            i2 += 1

        post_entropy = (probs_B_K_C[i1, :]*(-probs_B_K_C[i2] * logits_B_K_C[i2]).sum(dim=-1, keepdim=True)).mean()
        cur_mi = prior_entropy[i2]-post_entropy
        sample_prior_entropy += prior_entropy[i2].item()/num_samples
        mi += cur_mi.item()/num_samples
    return sample_prior_entropy, mi


def generate_sample(probs_B_K_C, one_hot=True):
    B, K, C = list(probs_B_K_C.shape) # (pool size, num samples, num classes)
    dist_B_K_C = tdist.categorical.Categorical(probs_B_K_C.view(-1, C)) # shape B*K x C
    sample_B_K_C = dist_B_K_C.sample([1]) # shape 1 x B*K
    assert list(sample_B_K_C.shape) == [1, B*K]
    sample_B_K_C = sample_B_K_C[0]

    if one_hot:
        oh_sample = torch.eye(C)[sample_B_K_C] # B*K x C
        oh_sample = oh_sample.view(B, K, C)
        sample_B_K_C = oh_sample

    return sample_B_K_C


def compute_pair_mi(i1, i2, probs_B_K_C, probs_B_C, sample_B_K_C):
    B, K, C = list(probs_B_K_C.shape) # (pool size, num samples, num classes)
    post_entropy = 0.
    for c in range(C):
        idxes_mask = ((sample_B_K_C[i1, :, c] == 1).long() > 0)
        assert list(idxes_mask.shape) == [K], idxes_mask.shape
        counts = sample_B_K_C[i2][idxes_mask].sum(0).float()
        assert list(counts.shape) == [C], counts.shape
        sample_probs = counts/idxes_mask.long().sum()

        temp = 0
        for c2 in range(C):
            if sample_probs[c2] > 0:
                temp += -sample_probs[c2]*sample_probs[c2].log()
        post_entropy += probs_B_C[i1, c] * temp
    return post_entropy


def compute_mi_sample(probs_B_K_C, sample_B_K_C, num_samples=500):
    B, K, C = list(probs_B_K_C.shape) # (pool size, num samples, num classes)

    probs_B_C = probs_B_K_C.mean(dim=1)
    prior_entropy = -(probs_B_C * probs_B_C.log()).sum(dim=-1) # (batch_size,)

    mi = 0.
    sample_prior_entropy = 0.
    for i_sample in range(num_samples):
        i1 = torch.randint(low=0, high=B, size=(1,)).item()
        i2 = torch.randint(low=0, high=B, size=(1,)).item()
        if i2 == i1:
            i2 += 1

        post_entropy = compute_pair_mi(i1, i2, probs_B_K_C, probs_B_C, sample_B_K_C)

        cur_mi = prior_entropy[i2]-post_entropy
        sample_prior_entropy += prior_entropy[i2].item()/num_samples
        mi += cur_mi.item()/num_samples
    return sample_prior_entropy, mi



def random_acquisition_function(logits_b_K_C):
    # If we use this together with a heuristic, make it small, so the heuristic takes over after the
    # first random pick.
    return torch.rand(logits_b_K_C.shape[0], device=logits_b_K_C.device) * 0.00001


def variation_ratios(logits_b_K_C):
    # torch.max yields a tuple with (max, argmax).
    return torch.ones(logits_b_K_C.shape[0], dtype=logits_b_K_C.dtype, device=logits_b_K_C.device) - torch.exp(
        torch.max(torch_utils.logit_mean(logits_b_K_C, dim=1, keepdim=False), dim=1, keepdim=False)[0]
    )


def mean_stddev_acquisition_function(logits_b_K_C):
    return torch_utils.mean_stddev(logits_b_K_C)


def max_entropy_acquisition_function(logits_b_K_C):
    return torch_utils.entropy(torch_utils.logit_mean(logits_b_K_C, dim=1, keepdim=False), dim=-1)


def bald_acquisition_function(logits_b_K_C):
    return torch_utils.mutual_information(logits_b_K_C)


class AcquisitionFunction(enum.Enum):
    random = "random"
    predictive_entropy = "predictive_entropy"
    bald = "bald"
    variation_ratios = "variation_ratios"
    mean_stddev = "mean_stddev"

    @property
    def scorer(self):
        if self == AcquisitionFunction.random:
            return random_acquisition_function
        elif self == AcquisitionFunction.predictive_entropy:
            return max_entropy_acquisition_function
        elif self == AcquisitionFunction.bald:
            return bald_acquisition_function
        elif self == AcquisitionFunction.variation_ratios:
            return variation_ratios
        elif self == AcquisitionFunction.mean_stddev:
            return mean_stddev_acquisition_function
        else:
            return NotImplementedError(f"{self} not supported yet!")

    def compute_scores(self, logits_B_K_C, available_loader, device):
        scorer = self.scorer

        if self == AcquisitionFunction.random:
            return scorer(logits_B_K_C, None).double()

        B, K, C = logits_B_K_C.shape

        # We need to sample the predictions from the bayesian_model n times and store them.
        with torch.no_grad():
            scores_B = torch.empty((B,), dtype=torch.float64)

            if device.type == "cuda":
                torch_utils.gc_cuda()
                KC_memory = K * C * 8
                batch_size = min(torch_utils.get_cuda_available_memory() // KC_memory, 8192)
            else:
                batch_size = 4096

            for scores_b, logits_b_K_C in with_progress_bar(
                torch_utils.split_tensors(scores_B, logits_B_K_C, batch_size), unit_scale=batch_size
            ):
                scores_b.copy_(scorer(logits_b_K_C.to(device)), non_blocking=True)

        return scores_B
