import torch
import numpy as np
import torch.nn as nn

from blackhc.progress_bar import with_progress_bar

import torch.distributions as tdist
import joint_entropy.exact as joint_entropy_exact
import joint_entropy.sampling as joint_entropy_sampling
import torch_utils
import math
import time
import hsic

from acquisition_batch import AcquisitionBatch

from acquisition_functions import AcquisitionFunction, compute_mi_sample
from reduced_consistent_mc_sampler import reduced_eval_consistent_bayesian_model


compute_multi_bald_bag_multi_bald_batch_size = None

class ProjectedFrankWolfe(object):
    def __init__(self, py, logits, J, **kwargs):
        """
        Constructs a batch of points using ACS-FW with random projections. Note the slightly different interface.
        :param data: (ActiveLearningDataset) Dataset.
        :param J: (int) Number of projections.
        :param kwargs: (dict) Additional arguments.
        """
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.device = logits.device

        self.ELn, self.entropy = self.get_projections(py, logits, J, **kwargs)
        squared_norm = torch.sum(self.ELn * self.ELn, dim=-1)
        self.sigmas = torch.sqrt(squared_norm + 1e-6)
        self.sigma = self.sigmas.sum()
        self.EL = torch.sum(self.ELn, dim=0)

    def get_projections(self, py, logits, J, projection='two', gamma=0, **kwargs):
        """
        Get projections for ACS approximate procedure
        :param J: (int) Number of projections to use
        :param projection: (str) Type of projection to use (currently only 'two' supported)
        :return: (torch.tensor) Projections
        """
        assert logits.shape[1] == J
        C = logits.shape[-1]
        ent = lambda py: torch.distributions.Categorical(probs=py).entropy()
        projections = []
        feat_x = []
        with torch.no_grad():
            ent_x = ent(py)
            if projection == 'two':
                for j in range(J):
                    cur_logits = logits[:, j, :]
                    ys = torch.ones_like(cur_logits).type(torch.LongTensor) * torch.arange(C, device=logits.device)[None, :]
                    ys = ys.t()
                    loglik = torch.stack([-self.cross_entropy(cur_logits, y) for y in ys]).t()
                    loglik = torch.sum(py * loglik, dim=-1, keepdim=True)
                    projections.append(loglik + gamma * ent_x[:, None])
            else:
                raise NotImplementedError

        return torch.sqrt(1 / torch.DoubleTensor([J], device=logits.device)) * torch.cat(projections, dim=1), ent_x

    def _init_build(self, M, **kwargs):
        pass  # unused

    def build(self, M=1, **kwargs):
        """
        Constructs a batch of points to sample from the unlabeled set.
        :param M: (int) Batch size.
        :param kwargs: (dict) Additional parameters.
        :return: (list of ints) Selected data point indices.
        """
        self._init_build(M, **kwargs)
        w = torch.zeros([len(self.ELn), 1], device=self.device).double()
        norm = lambda weights: (self.EL - (self.ELn.t() @ weights).squeeze()).norm()
        for m in range(M):
            w = self._step(m, w)

        # print(w[w.nonzero()[:, 0]].cpu().numpy())
        print('|| L-L(w)  ||: {:.4f}'.format(norm(w)))
        print('|| L-L(w1) ||: {:.4f}'.format(norm((w > 0).double())))
        print('Avg pred entropy (pool): {:.4f}'.format(self.entropy.mean().item()))
        print('Avg pred entropy (batch): {:.4f}'.format(self.entropy[w.flatten() > 0].mean().item()))

        return w.nonzero()[:, 0].cpu().numpy()

    def _step(self, m, w, **kwargs):
        """
        Applies one step of the Frank-Wolfe algorithm to update weight vector w.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return: (numpy array) Weight vector after adding m-th data point to the batch.
        """
        #print(self.ELn.type(), w.type())
        self.ELw = (self.ELn.t() @ w).squeeze()
        scores = (self.ELn / self.sigmas[:, None]) @ (self.EL - self.ELw)
        f = torch.argmax(scores)
        gamma, f1 = self.compute_gamma(f, w)
        # print('f: {}, gamma: {:.4f}, score: {:.4f}'.format(f, gamma.item(), scores[f].item()))
        if np.isnan(gamma.cpu()):
            raise ValueError

        w = (1 - gamma) * w + gamma * (self.sigma / self.sigmas[f]) * f1
        return w

    def compute_gamma(self, f, w):
        """
        Computes line-search parameter gamma.
        :param f: (int) Index of selected data point.
        :param w: (numpy array) Current weight vector.
        :return: (float, numpy array) Line-search parameter gamma and f-th unit vector [0, 0, ..., 1, ..., 0]
        """
        f1 = torch.zeros_like(w)
        f1[f] = 1
        Lf = (self.sigma / self.sigmas[f] * f1.t() @ self.ELn).squeeze()
        Lfw = Lf - self.ELw
        numerator = Lfw @ (self.EL - self.ELw)
        denominator = Lfw @ Lfw
        return numerator / denominator, f1

def compute_acs_fw_batch(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    max_entropy_bag_size,
    device=None,
) -> AcquisitionBatch:
    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    start_time = time.process_time()

    B, K, C = list(result.logits_B_K_C.shape) # (pool size, mc dropout samples, classes)
    py = result.logits_B_K_C.exp_().mean(dim=1)

    num_projections=10
    gamma=0.7
    assert K >= num_projections
    cs = ProjectedFrankWolfe(py, result.logits_B_K_C[:, :num_projections, :], num_projections, gamma=gamma)

    end_time = time.process_time()
    global_acquisition_bag = cs.build(M=b).tolist()
    s = set(global_acquisition_bag)
    perm = torch.randperm(B)
    bi = 0
    while len(global_acquisition_bag) < b:
        k = perm[bi].item()
        if k not in s:
            global_acquisition_bag += [k]
            s.add(k)
        bi += 1
    time_taken = end_time-start_time
    print('ack time taken', time_taken)
    acquisition_bag_scores = []

    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None), time_taken

def compute_fass_batch(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    max_entropy_bag_size,
    fass_compute_batch_size,
    device=None,
) -> AcquisitionBatch:
    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    start_time = time.process_time()

    B, K, C = list(result.logits_B_K_C.shape)
    probs_B_C = result.logits_B_K_C.exp_().mean(dim=1)
    preds_B = probs_B_C.max(dim=-1)[1]
    entropy = -(probs_B_C * probs_B_C.log()).sum(dim=-1)

    ack_bag = []
    global_acquisition_bag = []
    acquisition_bag_scores = []

    score_sort = torch.sort(entropy, descending=True)
    score_sort_idx = score_sort[1]
    score_sort = score_sort[0]

    cand_pts_idx = set(score_sort_idx[:max_entropy_bag_size].cpu().numpy().tolist())

    cand_X = []
    cand_X_preds = []
    cand_X_idx = []
    for i, (batch, labels) in enumerate(
        with_progress_bar(available_loader, unit_scale=available_loader.batch_size)
    ):
        lower = i * available_loader.batch_size
        upper = min(lower + available_loader.batch_size, B)
        idx_to_extract = np.array(list(set(range(lower, upper)).intersection(cand_pts_idx)), dtype=np.int32)
        cand_X_preds += [preds_B[idx_to_extract]]
        cand_X_idx += [torch.from_numpy(idx_to_extract).long()]
        idx_to_extract -= lower

        batch = batch.view(batch.shape[0], -1) # batch_size x num_features
        cand_X += [batch[idx_to_extract]]

    cand_X = torch.cat(cand_X, dim=0).unsqueeze(1).to(device)
    cand_X_preds = torch.cat(cand_X_preds, dim=0).to(device)
    cand_X_idx = torch.cat(cand_X_idx, dim=0).to(device)

    num_cands = cand_X.shape[0]
    if num_cands > fass_compute_batch_size and fass_compute_batch_size > 0:
        sqdist = []
        for bs in range(0, num_cands, fass_compute_batch_size):
            be = min(num_cands, bs+fass_compute_batch_size)
            sqdist += [hsic.sqdist(cand_X[bs:be], cand_X).mean(-1).cpu()]
    else:
        sqdist = hsic.sqdist(cand_X, cand_X).mean(-1) # cand_X size x cand_X size
    sqdist = torch.cat(sqdist, dim=0).to(device)
    max_dist = sqdist.max()

    cand_min_dist = torch.ones((cand_X.shape[0],), device=device) * max_dist
    ack_bag = []
    global_acquisition_bag = []
    for ackb_i in range(b):
        cand_distance = torch.ones((cand_X.shape[0],), device=device) * max_dist
        for c in range(C):
            cand_c_idx = cand_X_preds == c
            if cand_c_idx.long().sum() == 0:
                continue
            temp2 = []
            for bs in range(0, sqdist.shape[1], 5000):
                be = min(sqdist.shape[1], bs+5000)
                bl = be-bs
                temp = torch.cat(
                        [
                            cand_min_dist[cand_c_idx].unsqueeze(-1).repeat([1, bl]).unsqueeze(-1), 
                            sqdist[cand_c_idx, bs:be].unsqueeze(-1)
                        ], dim=-1)
                temp2 += [torch.min(temp, dim=-1)[0].detach()]
                del temp
                torch.cuda.empty_cache()
            temp2 = torch.cat(temp2, dim=1).mean(1).detach()
            cand_distance[cand_c_idx] = temp2
        cand_distance[ack_bag] = max_dist
        winner_index = cand_distance.argmin().item()
        ack_bag += [winner_index]
        #print('cand_distance.shape', cand_distance.shape, winner_index, cand_X_idx.shape)
        winner_index = cand_X_idx[winner_index].item()
        global_acquisition_bag.append(result.subset_split.get_dataset_indices([winner_index]).item())

    assert len(ack_bag) == b
    np.set_printoptions(precision=3, suppress=True)
    #print('Acquired predictions')
    #for i in range(len(ack_bag)):
    #    print('ack_i', i, probs_B_K_C[ack_bag[i]].cpu().numpy())

    end_time = time.process_time()
    time_taken = end_time-start_time
    print('ack time taken', time_taken)
    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None), time_taken

def compute_ical(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    max_batch_compute_size,
    hsic_compute_batch_size,
    hsic_kernel_name,
    max_greedy_iterations=0,
    hsic_resample=True,
    device=None,
    store=None,
    num_to_condense=200,
) -> AcquisitionBatch:
    assert hsic_compute_batch_size is not None
    assert hsic_kernel_name is not None

    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    start_time = time.process_time()

    probs_B_K_C = result.logits_B_K_C.exp_()
    B, K, C = list(result.logits_B_K_C.shape)

    dist_B_K_C = tdist.categorical.Categorical(result.logits_B_K_C.view(-1, C)) # shape B*K x C
    sample_B_K_C = dist_B_K_C.sample([1]) # shape 1 x B*K
    assert list(sample_B_K_C.shape) == [1, B*K]
    sample_B_K_C = sample_B_K_C[0]
    oh_sample = torch.eye(C)[sample_B_K_C] # B*K x C
    oh_sample = oh_sample.view(B, K, C)
    sample_B_K_C = oh_sample#.to(device)

    kernel_fn = getattr(hsic, hsic_kernel_name+'_kernels')

    dist_matrices = []
    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrix = hsic.sqdist(sample_B_K_C[bs:be].permute([1, 0, 2])) # n=K, d=B, k=C
        dist_matrices += [dist_matrix]
        bs = be
    dist_matrices = torch.cat(dist_matrices, dim=-1)#.to(device)

    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrices[:, :, bs:be] = kernel_fn(dist_matrices[:, :, bs:be])
        bs = be
    kernel_matrices = dist_matrices.permute([2, 0, 1]).to(device) # B, K, K
    assert list(kernel_matrices.shape) == [B, K, K], "%s == %s" % (kernel_matrices.shape, [B, K, K])

    ack_bag = []
    global_acquisition_bag = []
    acquisition_bag_scores = []
    batch_kernel = None
    print('Computing HSIC for', B, 'points')

    score_sort = torch.sort(result.scores_B, descending=True)
    score_sort_idx = score_sort[1]
    score_sort = score_sort[0]
    indices_to_condense = np.random.randint(low=0, high=score_sort_idx.shape[0], size=num_to_condense)

    if max_greedy_iterations == 0:
        max_greedy_iterations = b
    assert b % max_greedy_iterations == 0, "acquisition batch size must be a multiple of (ical_)max_greedy_iterations!"
    greedy_ack_batch_size = b//max_greedy_iterations
    print('max_greedy_iterations', max_greedy_iterations, 'greedy_ack_batch_size', greedy_ack_batch_size)

    for ackb_i in range(max_greedy_iterations):
        bs = 0
        hsic_scores = []
        condense_kernels = kernel_matrices[indices_to_condense].permute([1, 2, 0]).mean(dim=-1, keepdim=True).unsqueeze(0) # 1, K, K, 1
        while bs < B:
            be = min(B, bs+hsic_compute_batch_size)
            m = be-bs

            if batch_kernel is None:
                hsic_scores += [hsic.total_hsic_parallel(
                    torch.cat([
                        condense_kernels.repeat([m, 1, 1, 1]),
                        kernel_matrices[bs:be].unsqueeze(-1),
                    ],
                    dim=-1
                    ).to(device)
                )]
            else:
                hsic_scores += [hsic.total_hsic_parallel(
                    torch.cat([
                        condense_kernels.repeat([m, 1, 1, 1]), # M, K, K, 1
                        torch.cat([
                            batch_kernel.unsqueeze(0).repeat([m, 1, 1, 1]), # M, K, K, max_batch_compute_size
                            kernel_matrices[bs:be].unsqueeze(-1), # M, K, K, 1
                        ], dim=-1).mean(dim=-1, keepdim=True),
                    ],
                    dim=-1
                    ).to(device)
                )]
            bs = be

        hsic_scores = torch.cat(hsic_scores)
        hsic_scores[ack_bag] = -math.inf

        _, sorted_idxes = torch.sort(hsic_scores, descending=True)
        winner_idxes = []
        g_ack_i = 0
        while len(winner_idxes) < greedy_ack_batch_size:
            assert g_ack_i < sorted_idxes.shape[0]
            idx = sorted_idxes[g_ack_i].item()
            g_ack_i += 1
            if idx in ack_bag:
                continue
            winner_idxes += [idx]

        ack_bag += winner_idxes
        global_acquisition_bag += [i.item() for i in result.subset_split.get_dataset_indices(winner_idxes)]
        acquisition_bag_scores += [s.item() for s in hsic_scores[winner_idxes]]
        print('winner score', result.scores_B[winner_idxes].mean().item(), ', hsic_score', hsic_scores[winner_idxes].mean().item(), ', ackb_i', ackb_i)
        if batch_kernel is None:
            batch_kernel = kernel_matrices[winner_idxes].permute([1, 2, 0]) # K, K, L
        else:
            batch_kernel = torch.cat([batch_kernel, kernel_matrices[winner_idxes].permute([1, 2, 0])], dim=-1) # K, K, ack_size
            assert len(batch_kernel.shape) == 3
        if batch_kernel.shape[-1] >= max_batch_compute_size and max_batch_compute_size != 0:
            idxes = np.random.choice(batch_kernel.shape[-1], size=max_batch_compute_size, replace=False)
            batch_kernel = batch_kernel[:, :, idxes]

        result.scores_B[winner_idxes] = -math.inf
        score_sort = torch.sort(result.scores_B, descending=True)
        score_sort_idx = score_sort[1]
        score_sort = score_sort[0]
        if hsic_resample:
            indices_to_condense = np.random.randint(low=0, high=score_sort_idx.shape[0], size=num_to_condense)

    assert len(ack_bag) == b
    np.set_printoptions(precision=3, suppress=True)

    end_time = time.process_time()
    time_taken = end_time-start_time
    print('ack time taken', time_taken)
    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None), time_taken

def compute_ical_pointwise(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    max_batch_compute_size,
    hsic_compute_batch_size,
    hsic_kernel_name,
    max_greedy_iterations=0,
    hsic_resample=True,
    device=None,
    store=None,
    num_to_condense=200,
    num_inference_for_marginal_stat=0,
    use_orig_condense=False,
) -> AcquisitionBatch:
    assert hsic_compute_batch_size is not None
    assert hsic_kernel_name is not None

    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )

    start_time = time.process_time()

    probs_B_K_C = result.logits_B_K_C.exp_()
    B, K, C = list(result.logits_B_K_C.shape)

    dist_B_K_C = tdist.categorical.Categorical(result.logits_B_K_C.view(-1, C)) # shape B*K x C
    sample_B_K_C = dist_B_K_C.sample([1]) # shape 1 x B*K
    assert list(sample_B_K_C.shape) == [1, B*K]
    sample_B_K_C = sample_B_K_C[0]
    oh_sample = torch.eye(C)[sample_B_K_C] # B*K x C
    oh_sample = oh_sample.view(B, K, C)
    sample_B_K_C = oh_sample#.to(device)

    kernel_fn = getattr(hsic, hsic_kernel_name+'_kernels')

    dist_matrices = []
    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrix = hsic.sqdist(sample_B_K_C[bs:be].permute([1, 0, 2])) # n=K, d=B, k=C
        dist_matrices += [dist_matrix]
        bs = be
    dist_matrices = torch.cat(dist_matrices, dim=-1)#.to(device)

    bs = 0
    while bs < B:
        be = min(B, bs+hsic_compute_batch_size)
        dist_matrices[:, :, bs:be] = kernel_fn(dist_matrices[:, :, bs:be])
        bs = be
    kernel_matrices = dist_matrices.permute([2, 0, 1]).to(device) # B, K, K
    assert list(kernel_matrices.shape) == [B, K, K], "%s == %s" % (kernel_matrices.shape, [B, K, K])

    ack_bag = []
    global_acquisition_bag = []
    acquisition_bag_scores = []
    batch_kernel = None
    print('Computing HSIC for', B, 'points')

    score_sort = torch.sort(result.scores_B, descending=True)
    score_sort_idx = score_sort[1]
    score_sort = score_sort[0]
    indices_to_condense = np.random.randint(low=0, high=score_sort_idx.shape[0], size=num_to_condense)

    if max_greedy_iterations == 0:
        max_greedy_iterations = b
    assert b % max_greedy_iterations == 0, "acquisition batch size must be a multiple of (ical_)max_greedy_iterations!"
    greedy_ack_batch_size = b//max_greedy_iterations
    print('max_greedy_iterations', max_greedy_iterations, 'greedy_ack_batch_size', greedy_ack_batch_size)

    div_condense_num = 10

    for ackb_i in range(max_greedy_iterations):
        bs = 0
        hsic_scores = []
        condense_kernels = kernel_matrices[indices_to_condense].permute([1, 2, 0]).mean(dim=-1, keepdim=True).unsqueeze(0) # 1, K, K, 1
        div_indices_to_condense = np.random.randint(low=0, high=score_sort_idx.shape[0], size=max_batch_compute_size)
        if use_orig_condense:
            div_indices_to_condense = indices_to_condense
        div_size = div_indices_to_condense.shape[0]
        div_condense_kernels = kernel_matrices[div_indices_to_condense].unsqueeze(1) # div_size, 1, K, K
        while bs < B:
            be = min(B, bs+hsic_compute_batch_size)
            m = be-bs

            if batch_kernel is None:
                hsic_scores += [hsic.total_hsic_parallel(
                    torch.cat([
                        condense_kernels.repeat([m, 1, 1, 1]),
                        kernel_matrices[bs:be].unsqueeze(-1),
                    ],
                    dim=-1
                    ).to(device)
                )]
            else:
                num_ack = len(ack_bag)
                if num_inference_for_marginal_stat > 0:
                    marginal_stat_K_idx = torch.randperm(K)[:num_inference_for_marginal_stat]
                else:
                    marginal_stat_K_idx = torch.arange(K)
                K2 = marginal_stat_K_idx.shape[0]

                if K2 < K:
                    cur_og_batch_kernel = batch_kernel[marginal_stat_K_idx][:, marginal_stat_K_idx][None, :, :, None].repeat([m, 1, 1, 1]) # M, K2, K2, 1
                    cur_batch_kernel = (cur_og_batch_kernel*num_ack + kernel_matrices[bs:be][:, marginal_stat_K_idx][:, :, marginal_stat_K_idx].unsqueeze(-1))/(num_ack+1) # M, K2, K2, 1
                    cur_div_condense_kernels = div_condense_kernels[:, :, marginal_stat_K_idx][:, :, :, marginal_stat_K_idx].repeat([1, m, 1, 1]).view(-1, K2, K2, 1) # div_size*M, K, K, 1
                else:
                    cur_og_batch_kernel = batch_kernel[None, :, :, None].repeat([m, 1, 1, 1]) # M, K2, K2, 1
                    cur_batch_kernel = (cur_og_batch_kernel*num_ack + kernel_matrices[bs:be].unsqueeze(-1))/(num_ack+1) # M, K2, K2, 1
                    cur_div_condense_kernels = div_condense_kernels.repeat([1, m, 1, 1]).view(-1, K2, K2, 1) # div_size*M, K, K, 1
                assert list(cur_batch_kernel.shape) == [m, K2, K2, 1], cur_batch_kernel.shape
                assert list(cur_div_condense_kernels.shape) == [m*div_size, K2, K2, 1], cur_div_condense_kernels.shape
                hsic_scores1 = hsic.total_hsic_parallel(
                    torch.cat([
                        cur_div_condense_kernels,
                        cur_batch_kernel.unsqueeze(0).repeat([div_size, 1, 1, 1, 1]).view(-1, K2, K2, 1), # div_size*M, K2, K2, 1
                    ], # div_size*M, K2, K2, 2
                    dim=-1
                    ).to(device)
                )
                hsic_scores2 = hsic.total_hsic_parallel(
                    torch.cat([
                        cur_div_condense_kernels,
                        cur_og_batch_kernel.unsqueeze(0).repeat([div_size, 1, 1, 1, 1]).view(-1, K2, K2, 1),
                    ],
                    dim=-1
                    ).to(device)
                )

                if not use_orig_condense:
                    to_add = max(hsic_scores1.min().item(), hsic_scores2.min().item())
                    hsic_scores1 += to_add + 1e-8
                    hsic_scores2 += to_add + 1e-8
                    scores = (hsic_scores1/hsic_scores2).view(div_size, m)
                    scores = torch.max(scores, torch.tensor(1., device=scores.device))
                    marginal_improvement_ratio = scores.mean(0) # marginal fractional improvement in dependency

                if K2 == K:
                    hsic_scores1 = hsic.total_hsic_parallel(
                        torch.cat([
                            condense_kernels.repeat([m, 1, 1, 1]), # M, K, K, 1
                            cur_batch_kernel,
                        ],
                        dim=-1
                        ).to(device)
                    )
                else:
                    cur_og_batch_kernel = batch_kernel[None, :, :, None].repeat([m, 1, 1, 1]) # M, K, K, 1
                    cur_batch_kernel = (cur_og_batch_kernel*num_ack + kernel_matrices[bs:be].unsqueeze(-1))/(num_ack+1) # M, K, K, 1
                    hsic_scores1 = hsic.total_hsic_parallel(
                        torch.cat([
                            condense_kernels.repeat([m, 1, 1, 1]), # M, K, K, 1
                            cur_batch_kernel, # M, K, K, 1
                        ],
                        dim=-1
                        ).to(device)
                    )
                if use_orig_condense:
                    scores = hsic_scores1-hsic_scores2.view(div_size, m)
                    scores = torch.max(scores, torch.tensor(0., device=scores.device))
                    hsic_scores += [scores.mean(0)]
                else:
                    hsic_scores1 *= (marginal_improvement_ratio-1)
                    hsic_scores += [hsic_scores1]
                torch.cuda.empty_cache()
            bs = be

        hsic_scores = torch.cat(hsic_scores)
        hsic_scores[ack_bag] = -math.inf

        _, sorted_idxes = torch.sort(hsic_scores, descending=True)
        winner_idxes = []
        for g_ack_i in range(greedy_ack_batch_size):
            winner_idxes += [sorted_idxes[g_ack_i].item()]

        old_num_acks = len(ack_bag)
        ack_bag += winner_idxes
        new_num_acks = len(ack_bag)
        global_acquisition_bag += [i.item() for i in result.subset_split.get_dataset_indices(winner_idxes)]
        acquisition_bag_scores += [s.item() for s in hsic_scores[winner_idxes]]
        print('winner score', result.scores_B[winner_idxes].mean().item(), ', hsic_score', hsic_scores[winner_idxes].mean().item(), ', ackb_i', ackb_i)
        if batch_kernel is None:
            batch_kernel = kernel_matrices[winner_idxes].mean(0) # K, K
        else:
            batch_kernel = (batch_kernel*old_num_acks + kernel_matrices[winner_idxes].sum(0))/new_num_acks
        assert len(batch_kernel.shape) == 2

        result.scores_B[winner_idxes] = -math.inf
        score_sort = torch.sort(result.scores_B, descending=True)
        score_sort_idx = score_sort[1]
        score_sort = score_sort[0]
        if hsic_resample:
            indices_to_condense = np.random.randint(low=0, high=score_sort_idx.shape[0], size=num_to_condense)

    assert len(ack_bag) == b
    np.set_printoptions(precision=3, suppress=True)

    end_time = time.process_time()
    time_taken = end_time-start_time
    print('ack time taken', time_taken)
    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None), time_taken

def compute_multi_bald_batch(
    bayesian_model: nn.Module,
    available_loader,
    num_classes,
    k,
    b,
    target_size,
    initial_percentage,
    reduce_percentage,
    device=None,
) -> AcquisitionBatch:
    result = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        initial_percentage=initial_percentage,
        reduce_percentage=reduce_percentage,
        target_size=target_size,
        available_loader=available_loader,
        device=device,
    )
    start_time = time.process_time()

    subset_split = result.subset_split

    partial_multi_bald_B = result.scores_B
    # Now we can compute the conditional entropy
    conditional_entropies_B = joint_entropy_exact.batch_conditional_entropy_B(result.logits_B_K_C)

    # We turn the logits into probabilities.
    probs_B_K_C = result.logits_B_K_C.exp_()

    # Don't need the result anymore.
    result = None

    torch_utils.gc_cuda()
    # torch_utils.cuda_meminfo()

    with torch.no_grad():
        num_samples_per_ws = 40000 // k
        num_samples = num_samples_per_ws * k

        if device.type == "cuda":
            # KC_memory = k*num_classes*8
            sample_MK_memory = num_samples * k * 8
            MC_memory = num_samples * num_classes * 8
            copy_buffer_memory = 256 * num_samples * num_classes * 8
            slack_memory = 2 * 2 ** 30
            multi_bald_batch_size = (
                torch_utils.get_cuda_available_memory() - (sample_MK_memory + copy_buffer_memory + slack_memory)
            ) // MC_memory

            global compute_multi_bald_bag_multi_bald_batch_size
            if compute_multi_bald_bag_multi_bald_batch_size != multi_bald_batch_size:
                compute_multi_bald_bag_multi_bald_batch_size = multi_bald_batch_size
                print(f"New compute_multi_bald_bag_multi_bald_batch_size = {multi_bald_batch_size}")
        else:
            multi_bald_batch_size = 16

        subset_acquisition_bag = []
        global_acquisition_bag = []
        acquisition_bag_scores = []

        # We use this for early-out in the b==0 case.
        MIN_SPREAD = 0.1

        if b == 0:
            b = 100
            early_out = True
        else:
            early_out = False

        prev_joint_probs_M_K = None
        prev_samples_M_K = None
        for i in range(b):
            torch_utils.gc_cuda()

            if i > 0:
                # Compute the joint entropy
                joint_entropies_B = torch.empty((len(probs_B_K_C),), dtype=torch.float64)

                exact_samples = num_classes ** i
                if exact_samples <= num_samples:
                    prev_joint_probs_M_K = joint_entropy_exact.joint_probs_M_K(
                        probs_B_K_C[subset_acquisition_bag[-1]][None].to(device),
                        prev_joint_probs_M_K=prev_joint_probs_M_K,
                    )

                    # torch_utils.cuda_meminfo()
                    batch_exact_joint_entropy(
                        probs_B_K_C, prev_joint_probs_M_K, multi_bald_batch_size, device, joint_entropies_B
                    )
                else:
                    if prev_joint_probs_M_K is not None:
                        prev_joint_probs_M_K = None
                        torch_utils.gc_cuda()

                    # Gather new traces for the new subset_acquisition_bag.
                    prev_samples_M_K = joint_entropy_sampling.sample_M_K(
                        probs_B_K_C[subset_acquisition_bag].to(device), S=num_samples_per_ws
                    )

                    # torch_utils.cuda_meminfo()
                    for joint_entropies_b, probs_b_K_C in with_progress_bar(
                        torch_utils.split_tensors(joint_entropies_B, probs_B_K_C, multi_bald_batch_size),
                        unit_scale=multi_bald_batch_size,
                    ):
                        joint_entropies_b.copy_(
                            joint_entropy_sampling.batch(probs_b_K_C.to(device), prev_samples_M_K), non_blocking=True)

                        # torch_utils.cuda_meminfo()

                    prev_samples_M_K = None
                    torch_utils.gc_cuda()

                partial_multi_bald_B = joint_entropies_B - conditional_entropies_B
                joint_entropies_B = None

            # Don't allow reselection
            partial_multi_bald_B[subset_acquisition_bag] = -math.inf

            winner_index = partial_multi_bald_B.argmax().item()

            # Actual MultiBALD is:
            actual_multi_bald_B = partial_multi_bald_B[winner_index] - torch.sum(
                conditional_entropies_B[subset_acquisition_bag]
            )
            actual_multi_bald_B = actual_multi_bald_B.item()

            print(f"Actual MultiBALD: {actual_multi_bald_B}")

            # If we early out, we don't take the point that triggers the early out.
            # Only allow early-out after acquiring at least 1 sample.
            if early_out and i > 1:
                current_spread = actual_multi_bald_B[winner_index] - actual_multi_bald_B.median()
                if current_spread < MIN_SPREAD:
                    print("Early out")
                    break

            acquisition_bag_scores.append(actual_multi_bald_B)

            subset_acquisition_bag.append(winner_index)
            # We need to map the index back to the actual dataset.
            global_acquisition_bag.append(subset_split.get_dataset_indices([winner_index]).item())

            print(f"Acquisition bag: {sorted(global_acquisition_bag)}, num_ack: {i}")

    end_time = time.process_time()
    time_taken = end_time-start_time
    print('ack time taken', time_taken)

    return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None), time_taken


def batch_exact_joint_entropy(probs_B_K_C, prev_joint_probs_M_K, chunk_size, device, out_joint_entropies_B):
    """This one switches between devices, too."""
    for joint_entropies_b, probs_b_K_C in with_progress_bar(
        torch_utils.split_tensors(out_joint_entropies_B, probs_B_K_C, chunk_size), unit_scale=chunk_size
    ):
        joint_entropies_b.copy_(
            joint_entropy_exact.batch(probs_b_K_C.to(device), prev_joint_probs_M_K), non_blocking=True
        )

    return joint_entropies_b


def batch_exact_joint_entropy_logits(logits_B_K_C, prev_joint_probs_M_K, chunk_size, device, out_joint_entropies_B):
    """This one switches between devices, too."""
    for joint_entropies_b, logits_b_K_C in with_progress_bar(
        torch_utils.split_tensors(out_joint_entropies_B, logits_B_K_C, chunk_size), unit_scale=chunk_size
    ):
        joint_entropies_b.copy_(
            joint_entropy_exact.batch(logits_b_K_C.to(device).exp(), prev_joint_probs_M_K), non_blocking=True
        )

    return joint_entropies_b
