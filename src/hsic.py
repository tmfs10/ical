import torch
import numpy as np
import sys
from typing import Sequence, Callable, Optional, Union
import ops


def sqdist_parallel(X1):
    """
    X is of shape (m, n, d, k)
    return is of shape (m, n, n, d)
    """
    return ((X1.unsqueeze(1) - X1.unsqueeze(2)) ** 2).mean(-1)


def sqdist(X1, X2=None, do_mean=False, collect=True):
    if X2 is None:
        """
        X is of shape (n, d, k) or (n, d)
        return is of shape (n, n, d)
        """
        if X1.ndimension() == 2:
            return (X1.unsqueeze(0) - X1.unsqueeze(1)) ** 2
        else:
            assert X1.ndimension() == 3, X1.shape
            if not collect:
                return ((X1.unsqueeze(0) - X1.unsqueeze(1)) ** 2) # (n, n, d, k)

            if do_mean:
                return ((X1.unsqueeze(0) - X1.unsqueeze(1)) ** 2).mean(-1)
            else:
                sq = ((X1.unsqueeze(0) - X1.unsqueeze(1)) ** 2)
                #assert not ops.is_inf(sq)
                #assert not ops.is_nan(sq)
                #assert (sq.view(-1) < 0).sum() == 0, str((sq.view(-1) < 0).sum())
                return sq.sum(-1)
    else:
        """
        X1 is of shape (n, d, k) or (n, d)
        X2 is of shape (m, d, k) or (m, d)
        return is of shape (n, m, d)
        """
        assert X1.ndimension() == X2.ndimension()
        if X1.ndimension() == 2:
            # (n, d)
            return (X2.unsqueeze(0) - X1.unsqueeze(1)) ** 2
        else:
            # (n, d, k)
            assert X1.ndimension() == 3
            if not collect:
                return ((X2.unsqueeze(0) - X1.unsqueeze(1)) ** 2) # (n, n, d, k)
            if do_mean:
                return ((X2.unsqueeze(0) - X1.unsqueeze(1)) ** 2).mean(-1)
            else:
                return ((X2.unsqueeze(0) - X1.unsqueeze(1)) ** 2).sum(-1)


def mixrbf_kernels(dist_matrix, bws=[.01, .1, .2, 1, 5, 10, 100], weights=None):
    # input is (n, n, d), output is also (n, n, d)
    assert dist_matrix.ndimension() == 3, str(dist_matrix.shape)
    #assert (dist_matrix.view(-1) <= 0).sum() == 0
    dist_matrix = dist_matrix.unsqueeze(0)
    bws = dist_matrix.new_tensor(bws).view(-1, 1, 1, 1)
    weights = weights or 1 / len(bws)
    weights = weights * dist_matrix.new_ones(len(bws))

    parts = torch.exp(dist_matrix / (-2 * bws ** 2))
    #assert (parts.view(-1) <= 0).sum() == 0
    return torch.einsum("w,wijd->ijd", (weights, parts))


def mixrq_kernels(dist_matrix, alphas=[.2, .5, 1, 2, 5], weights=None):
    # input is (n, n, d), output is also (n, n, d)
    if dist_matrix.ndimension() == 4:
        return mixrq_kernels2(dist_matrix, alphas, weights)

    assert dist_matrix.ndimension() == 3, str(dist_matrix.shape)
    #assert (dist_matrix.contiguous().view(-1) < 0).sum() == 0
    dist_matrix = dist_matrix.unsqueeze(0)
    alphas = dist_matrix.new_tensor(alphas).view(-1, 1, 1, 1)
    weights = weights or 1.0 / len(alphas)
    weights = weights * dist_matrix.new_ones(len(alphas))

    logs = torch.log1p(dist_matrix / (2 * alphas))
    #     assert torch.isfinite(logs).all()
    #assert not ops.is_inf(logs)
    #assert not ops.is_nan(logs)
    #assert (logs.contiguous().view(-1) < 0).sum() == 0
    return torch.einsum("w,wijd->ijd", (weights, torch.exp(-alphas * logs)))


def mixrq_kernels_parallel(dist_matrix, alphas=[.2, .5, 1, 2, 5], weights=None):
    # input is (m, n, n, d), output is also (m, n, n, d)
    assert dist_matrix.ndimension() == 4, str(dist_matrix.shape)
    #assert (dist_matrix.contiguous().view(-1) < 0).sum() == 0
    dist_matrix = dist_matrix.unsqueeze(1)
    weights = weights or 1.0 / len(alphas)
    weights = weights * dist_matrix.new_ones(len(alphas))
    alphas = dist_matrix.new_tensor(alphas).view(1, -1, 1, 1, 1)

    logs = torch.log1p(dist_matrix / (2 * alphas))
    #     assert torch.isfinite(logs).all()
    #assert not ops.is_inf(logs)
    #assert not ops.is_nan(logs)
    #assert (logs.contiguous().view(-1) < 0).sum() == 0
    return torch.einsum("w,mwijd->mijd", (weights, torch.exp(-alphas * logs)))


def two_vec_mixrbf_kernels(X1, X2=None, bws=[.01, .1, .2, 1, 5, 10, 100], weights=None, do_mean=False):
    # input is of shape (n,) or (n, k)
    # output is of shape (n, n, 1)
    if X2 is None:
        if X1.ndimension() == 1:
            X1 = X1.unsqueeze(-1).unsqueeze(-1)
        elif X1.ndimension() == 2:
            X1 = X1.unsqueeze(1)

        assert X1.ndimension() == 3

        dist_matrix = sqdist(X1, do_mean=do_mean)
        assert dist_matrix.shape[0] == X1.shape[0], str(dist_matrix.shape) + ", " + str(X1.shape)
        assert dist_matrix.shape[1] == X1.shape[0], str(dist_matrix.shape) + ", " + str(X1.shape)
        assert dist_matrix.shape[2] == 1, str(dist_matrix.shape)

        # (n, n, 1)
        return mixrbf_kernels(dist_matrix, bws, weights)
    else:
        if X1.ndimension() == 1:
            X1 = X1.unsqueeze(-1).unsqueeze(-1)
        elif X1.ndimension() == 2:
            X1 = X1.unsqueeze(1)

        if X2.ndimension() == 1:
            X2 = X2.unsqueeze(-1).unsqueeze(-1)
        elif X2.ndimension() == 2:
            X2 = X2.unsqueeze(1)

        assert X1.ndimension() == 3
        assert X2.ndimension() == 3

        dist_matrix = sqdist(X1, X2, do_mean)
        return mixrbf_kernels(dist_matrix, bws, weights)


def two_vec_mixrq_kernels(X1, X2=None, bws=[.01, .1, .2, 1, 5, 10, 100], weights=None, do_mean=False):
    # input is of shape (n,) or (n, k)
    # output is of shape (n, n, 1)
    if X2 is None:
        if X1.ndimension() == 1:
            X1 = X1.unsqueeze(-1).unsqueeze(-1)
        elif X1.ndimension() == 2:
            X1 = X1.unsqueeze(1)

        assert X1.ndimension() == 3

        dist_matrix = sqdist(X1, do_mean=do_mean)
        assert dist_matrix.shape[0] == X1.shape[0], str(dist_matrix.shape) + ", " + str(X1.shape)
        assert dist_matrix.shape[1] == X1.shape[0], str(dist_matrix.shape) + ", " + str(X1.shape)
        assert dist_matrix.shape[2] == 1, str(dist_matrix.shape)

        # (n, n, 1)
        return mixrq_kernels(dist_matrix, bws, weights)
    else:
        if X1.ndimension() == 1:
            X1 = X1.unsqueeze(-1).unsqueeze(-1)
        elif X1.ndimension() == 2:
            X1 = X1.unsqueeze(1)

        if X2.ndimension() == 1:
            X2 = X2.unsqueeze(-1).unsqueeze(-1)
        elif X2.ndimension() == 2:
            X2 = X2.unsqueeze(1)

        assert X1.ndimension() == 3
        assert X2.ndimension() == 3

        dist_matrix = sqdist(X1, X2=X2, do_mean=do_mean)

        # (n, n, 1)
        return mixrq_kernels(dist_matrix, bws, weights)


def dimwise_mixrbf_kernels(X, bws=[.01, .1, .2, 1, 5, 10, 100], weights=None):
    """Mixture of RBF kernels between each dimension of X.

    If X is shape (n, d), returns shape (n, n, d).

    Kernel is sum_i wt_i exp(- (x - y)^2 / (2 bw^2)).
    If wts is not passed, uses 1 for each alpha.
    """

    bws = X.new_tensor(bws).view(-1, 1, 1, 1)
    weights = weights or 1 / len(bws)
    weights = weights * X.new_ones(len(bws))

    # sqdists = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).unsqueeze(0)
    sqdists = sqdist(X).unsqueeze(0)

    parts = torch.exp(sqdists / (-2 * bws ** 2))
    return torch.einsum("w,wijd->ijd", (weights, parts))


def dimwise_mixrq_kernels(
    X: torch.Tensor, alphas: Sequence[float] = (.2, .5, 1, 2, 5), weights=None
) -> torch.Tensor:
    """
    Mixture of RQ kernels between each dimension of X.

    If X is shape (n, d), returns shape (n, n, d).
    n is the number of samples from each RV and d is the number of RVs.
    k_ijk = RQK(X[i, k], X[j, k]) (the kernel of the i'th and j'th samples of RV k).

    Kernel is sum_i wt_i (1 + (x - y)^2 / (2 alpha_i))^{-alpha_i}.
    If weights is not passed, each alpha is weighted equally.

    A vanilla RQ kernel is k(x, x') = sigma^2 (1 + (x - x')^2 / (2 alpha l^2)) ^ -alpha
    Here we (by default) use a weighted combination of multiple alphas and set l = sigma = 1.
    """

    n_dims = X.ndimension()
    assert n_dims in (2, 3), X.ndimension()

    alphas = X.new_tensor(alphas).view(-1, 1, 1, 1)
    weights = weights or 1.0 / len(alphas)
    weights = weights * X.new_ones(len(alphas))

    # dims are (alpha, x, y, dim)
    sqdists = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).unsqueeze(0)

    if n_dims == 3:
        sqdists = sqdists.sum(dim=-1)

    # 30% faster without asserts in some quick tests (n = 250, d = 1K)
    #     assert (sqdists >= 0).all()

    logs = torch.log1p(sqdists / (2 * alphas))
    #     assert torch.isfinite(logs).all()
    return torch.einsum("w,wijd->ijd", (weights, torch.exp(-alphas * logs)))


def linear_kernel(x: torch.Tensor, c: float = 1):
    """
    :param x: n x d for n samples from d RVs
    """
    return x.unsqueeze(0) * x.unsqueeze(1) + c


def dimwise_distance_kernels(x: torch.Tensor):
    """
    :param x: n x d for n samples from d RVs
    """
    return (x.unsqueeze(0) - x.unsqueeze(1)).abs()


################################################################################
# HSIC estimators


def total_hsic(kernels, logspace=True):
    """(Biased) estimator of total independence.

    kernels should be shape (n, n, d) to test for total
    independence among d variables with n paired samples.
    """
    # formula from section 4.4 of
    # https://papers.nips.cc/paper/4893-a-kernel-test-for-three-variable-interactions.pdf
    shp = kernels.shape
    assert len(shp) == 3
    assert shp[0] == shp[1], "%s == %s" % (str(shp[0]), str(shp[1]))

    n = kernels.new_tensor(shp[0])
    d = kernels.new_tensor(shp[2])

    # t1: 1/n^2      sum_a  sum_b  prod_i K_ab^i
    # t2: -2/n^(d+1) sum_a  prod_i sum_b  K_ab^i
    # t3: 1/n^(2d)   prod_i sum_a  sum_b  K_ab^i

    if not logspace:
        sum_b = torch.sum(kernels, dim=1)

        t1 = torch.mean(torch.prod(kernels, dim=2))
        t2 = torch.sum(torch.prod(sum_b, dim=1)) * (-2 / (n ** (d + 1)))
        t3 = torch.prod(torch.sum(sum_b, dim=0)) / (n ** (2 * d))
        return t1 + t2 + t3
    else:
        log_n = torch.log(n)
        log_2 = torch.log(kernels.new_tensor(2))
        log_kernels = kernels.log()
        log_sum_b = log_kernels.logsumexp(dim=1)

        l1 = log_kernels.sum(dim=2).logsumexp(dim=1).logsumexp(dim=0) - 2 * log_n
        l2 = log_sum_b.sum(dim=1).logsumexp(dim=0) + log_2 - (d + 1) * log_n
        l3 = log_sum_b.logsumexp(dim=0).sum() - 2 * d * log_n

        # total_hsic = exp(l1) - exp(l2) + exp(l3)
        #   = exp(-a) (exp(l1 + a) - exp(l2 + a) + exp(l3 + a)) for any a
        # can't use logsumexp for this directly because we subtract the l2 term
        a = torch.max(kernels.new_tensor([l1, l2, l3]))
        return a.exp() * ((l1 - a).exp() - (l2 - a).exp() + (l3 - a).exp())



def total_hsic_parallel(kernels, return_log=False):
    """
    kernels should be shape (m, n, n, d) to test for total
    independence among d variables with n paired samples for m sets.

    output is of shape (m,)
    """
    assert kernels.ndimension() == 4
    shp = kernels.shape
    assert shp[1] == shp[2], "%s == %s" % (str(shp[0]), str(shp[1]))

    m = shp[0]
    n = kernels.new_tensor(shp[1])
    d = kernels.new_tensor(shp[3])

    log_n = torch.log(n)
    log_2 = torch.log(kernels.new_tensor(2))
    log_kernels = kernels.log()
    log_sum_b = log_kernels.logsumexp(dim=2)

    l1 = log_kernels.sum(dim=3).logsumexp(dim=2).logsumexp(dim=1) - 2 * log_n
    l2 = log_sum_b.sum(dim=2).logsumexp(dim=1) + log_2 - (d + 1) * log_n
    l3 = log_sum_b.logsumexp(dim=1).sum(dim=-1) - 2 * d * log_n

    assert l1.ndimension() == 1
    assert l2.ndimension() == 1
    assert l3.ndimension() == 1

    l = torch.stack([l1, l2, l3], dim=1)
    assert l.shape[1] == 3
    a = torch.max(l, dim=1)[0]

    if return_log:
        return a + ((l1 - a).exp() - (l2 - a).exp() + (l3 - a).exp()).log()
    else:
        return a.exp() * ((l1 - a).exp() - (l2 - a).exp() + (l3 - a).exp())


def sum_pairwise_hsic(kernels):
    """Sum of (biased) estimators of pairwise independence.

    kernels should be shape (n, n, d) when testing
    among d variables with n paired samples.
    """
    shp = kernels.shape
    assert len(shp) == 3
    assert shp[0] == shp[1], "%s == %s" % (str(shp[0]), str(shp[1]))

    n = torch.tensor(shp[0], dtype=kernels.dtype)

    # Centered kernel matrix is given by:
    # (I - 1/n 1 1^T) K (I - 1/n 1 1^T)
    #  = K - 1/n 1 1^T K - 1/n K 1 1^T + 1/n^2 1 1^T K 1 1^T
    row_means = torch.mean(kernels, dim=0, keepdims=True)
    col_means = torch.mean(kernels, dim=1, keepdims=True)
    grand_mean = torch.mean(row_means, dim=1, keepdims=True)
    centered_kernels = kernels - row_means - col_means + grand_mean

    # HSIC on dims (i, j) is  1/n^2 sum_{a, b} K[a, b, i] K[a, b, j]
    # sum over all dimensions is 1/n^2 sum_{i, j, a, b} K[a, b, i] K[a, b, j]
    return torch.einsum("abi,abj->", (centered_kernels, centered_kernels)) / n ** 2

def precompute_batch_hsic_stats(
    preds: Union[torch.Tensor, Sequence[torch.Tensor]],
    batch: Optional[Sequence[int]] = None,
    kernel: Callable = dimwise_mixrq_kernels,
) -> Sequence[torch.Tensor]:
    if not isinstance(preds, Sequence):
        preds = [preds]

    if batch:
        preds = [pred[:, batch] for pred in preds]

    batch_kernels = [kernel(pred) for pred in preds]
    batch_kernels = torch.cat(batch_kernels, dim=-1)
    batch_kernels.log()
    batch_kernels = batch_kernels.unsqueeze(-1)

    batch_sum_b = batch_kernels.logsumexp(dim=1)
    batch_l1_sum = batch_kernels.sum(dim=2)
    batch_l2_sum = batch_sum_b.sum(dim=1)
    batch_l3_sum = batch_sum_b.logsumexp(dim=0).sum(dim=0)
    return batch_sum_b, batch_l1_sum, batch_l2_sum, batch_l3_sum


def compute_point_hsics(
    preds: Union[torch.Tensor, Sequence[torch.Tensor]],
    next_points: Sequence[int],
    batch_sum_b: torch.Tensor,
    batch_l1_sum: torch.Tensor,
    batch_l2_sum: torch.Tensor,
    batch_l3_sum: torch.Tensor,
    kernel: Callable = dimwise_mixrq_kernels,
) -> torch.Tensor:
    if not isinstance(preds, Sequence):
        preds = [preds]

    log_n = preds[0].new_tensor(len(batch_sum_b)).log()
    d = preds[0].new_tensor(batch_sum_b.shape[1] + 1)
    log_2 = preds[0].new_tensor(2).log()

    point_kernels = [kernel(pred[:, next_points]) for pred in preds]
    point_kernels = torch.cat(point_kernels, dim=-1)
    point_kernels.log()
    point_kernels = point_kernels.unsqueeze(2)

    point_sum_b = point_kernels.logsumexp(dim=1)
    point_l1_sum = point_kernels.sum(dim=2)
    point_l2_sum = point_sum_b.sum(dim=1)
    point_l3_sum = point_sum_b.logsumexp(dim=0).sum(dim=0)

    l1 = (batch_l1_sum + point_l1_sum).logsumexp(dim=1).logsumexp(dim=0) - 2 * log_n
    l2 = (batch_l2_sum + point_l2_sum).logsumexp(dim=0) + log_2 - (d + 1) * log_n
    l3 = batch_l3_sum + point_l3_sum - 2 * d * log_n

    a = -torch.stack((l1, l2, l3)).max(dim=0)[0]

    hsics = torch.exp(-a) * (torch.exp(l1 + a) - torch.exp(l2 + a) + torch.exp(l3 + a))
    return hsics


def center_kernel(K):
    """Center a kernel matrix of shape (n, n, ...).
    If there are trailing dimensions, assumed to be a list of kernels and
    centers each one separately.
    """
    # (I - 1/n 1 1^T) K (I - 1/n 1 1^T)
    #  = K - 1/n 1 1^T K - 1/n K 1 1^T + 1/n^2 1 1^T K 1 1^T
    row_means = K.mean(dim=0, keepdim=True)
    col_means = row_means.transpose(0, 1)
    grand_mean = row_means.mean(dim=1, keepdim=True)
    return K - row_means - col_means + grand_mean

# Not n-way unlike total_hsic !!!!
def hsic_xy(
    kernel_x, 
    kernel_y, 
    normalized,
    biased=False, 
    center_x=False, 
    eps=ops._eps
):
    """Independence estimator.
    By default, uses the unbiased estimator.
    If normalized, returns HSIC(X, Y) / sqrt(HSIC(X, X) HSIC(Y, Y)). Uses
    eps if the denominator is too close to zero.
    If biased=True, center_x determines which kernel will be centered; might
    result in slightly simpler gradients for the kernel which is not centered.
    Ignored if biased=False.
    """
    n = kernel_x.shape[0]

    if biased:
        if normalized:
            kernel_x = center_kernel(kernel_x)
            kernel_y = center_kernel(kernel_y)
        elif center_x:
            kernel_x = center_kernel(kernel_x)
        else:
            kernel_y = center_kernel(kernel_y)

        tr_xy = torch.einsum('ij,ij->', (kernel_x, kernel_y))

        if normalized:
            tr_xx = torch.max(torch.einsum('ij,ij->', (kernel_x, kernel_x)), kernel_x.new_tensor(eps))
            tr_yy = torch.max(torch.einsum('ij,ij->', (kernel_y, kernel_y)), kernel_x.new_tensor(eps))
            return tr_xy / torch.sqrt(tr_xx * tr_yy)
        else:
            return tr_xy / (n - 1)**2

    # HSIC_1 from http://www.jmlr.org/papers/volume13/song12a/song12a.pdf
    z = torch.zeros(n).type(kernel_x.type())
    Kt = ops.set_diagonal(kernel_x, z)
    Lt = ops.set_diagonal(kernel_y, z)

    Kt_sums = torch.sum(Kt, dim=0)
    Lt_sums = torch.sum(Lt, dim=0)

    Kt_grand_sum = torch.sum(Kt_sums)
    Lt_grand_sum = torch.sum(Lt_sums)

    n1_n2 = (n - 1) * (n - 2)

    main_xy = (
        torch.einsum('ij,ij->', (Kt, Lt))
        + Kt_grand_sum * Lt_grand_sum / n1_n2
        - 2 / (n - 2) * torch.dot(Kt_sums, Lt_sums))

    if normalized:
        main_xx = torch.max(
            kernel_x.new_tensor(eps),
            torch.einsum('ij,ij->', (Kt, Kt))
            + Kt_grand_sum * Kt_grand_sum / n1_n2
            - 2 / (n - 2) * torch.dot(Kt_sums, Kt_sums))
        main_yy = torch.max(
            kernel_x.new_tensor(eps),
            torch.einsum('ij,ij->', (Lt, Lt))
            + Lt_grand_sum * Lt_grand_sum / n1_n2
            - 2 / (n - 2) * torch.dot(Lt_sums, Lt_sums))
        return main_xy / torch.sqrt(main_xx * main_yy)
    else:
        return main_xy / (n * (n - 3))

def hsic_xy_parallel(kernel_x, kernel_y, biased=False, center_x=False, normalized=False,
         eps=ops._eps):
    # kernel size is (m, n, n)
    m = kernel_x.shape[0]
    n = kernel_x.shape[1]

    if biased:
        if normalized:
            kernel_x = center_kernel(kernel_x)
            kernel_y = center_kernel(kernel_y)
        elif center_x:
            kernel_x = center_kernel(kernel_x)
        else:
            kernel_y = center_kernel(kernel_y)

        tr_xy = torch.einsum('kij,kij->', (kernel_x, kernel_y))

        if normalized:
            tr_xx = torch.max(torch.einsum('kij,kij->', (kernel_x, kernel_x)), kernel_x.new_tensor(eps))
            tr_yy = torch.max(torch.einsum('kij,kij->', (kernel_y, kernel_y)), kernel_x.new_tensor(eps))
            return tr_xy / torch.sqrt(tr_xx * tr_yy)
        else:
            return tr_xy / (n - 1)**2

    # HSIC_1 from http://www.jmlr.org/papers/volume13/song12a/song12a.pdf
    z = torch.zeros(n).type(kernel_x.type())
    Kt = ops.set_diagonal(kernel_x, z)
    Lt = ops.set_diagonal(kernel_y, z)

    Kt_sums = torch.sum(Kt, dim=1)
    Lt_sums = torch.sum(Lt, dim=1)

    Kt_grand_sum = torch.sum(Kt_sums.view(m, -1), dim=-1)
    Lt_grand_sum = torch.sum(Lt_sums.view(m, -1), dim=-1)

    n1_n2 = (n - 1) * (n - 2)

    def dot(X, Y):
        Y = Y.transpose(0, 1)
        prod = ((X-Y)**2).sum(dim=-1)
        return prod

    main_xy = (
        torch.einsum('kij,kij->', (Kt, Lt))
        + Kt_grand_sum * Lt_grand_sum / n1_n2
        - 2 / (n - 2) * dot(Kt_sums, Lt_sums))

    if normalized:
        main_xx = torch.max(
            kernel_x.new_tensor(eps),
            torch.einsum('kij,kij->', (Kt, Kt))
            + Kt_grand_sum * Kt_grand_sum / n1_n2
            - 2 / (n - 2) * dot(Kt_sums, Kt_sums))
        main_yy = torch.max(
            kernel_x.new_tensor(eps),
            torch.einsum('kij,kij->', (Lt, Lt))
            + Lt_grand_sum * Lt_grand_sum / n1_n2
            - 2 / (n - 2) * dot(Lt_sums, Lt_sums))
        return main_xy / torch.sqrt(main_xx * main_yy)
    else:
        return main_xy / (n * (n - 3))

def pairwise_hsic(
    kernels, 
    include_self=False, 
    biased=True, 
    normalized=False,
):
    """Sum of estimators of pairwise independence.
    kernels should be shape (n, n, d) when testing
    among d variables with n paired samples.

    If include_self = False, don't include terms comparing a variable to itself.
    """
    shp = kernels.shape
    assert len(shp) == 3
    assert shp[0] == shp[1], "%s == %s" % (str(shp[0]), str(shp[1]))

    n = torch.tensor(shp[0], dtype=kernels.dtype)
    d = torch.tensor(shp[2], dtype=kernels.dtype)

    if not biased:
        raise NotImplementedError("haven't implemented unbiased here yet")

    cent = center_kernel(kernels)
    # HSIC^2 on dims (i, j) is  1/n^2 sum_{a, b} K[a, b, i] K[a, b, j]

    if not normalized:
        # sum over dimensions is 1/n^2 sum_{i, j, a, b} K[a, b, i] K[a, b, j]
        s = torch.einsum('abi,abj->ij', cent, cent)
        return s / n**2
    else:
        hsics = torch.einsum('abi,abj->ij', cent, cent)
        norms = torch.sqrt(torch.max(torch.diag(hsics), torch.tensor(1e-8, device=hsics.device)))
        # numerator is n^2 * hsics, each denom term is n * the norms; cancels
        normed = hsics / torch.unsqueeze(norms, 0) / torch.unsqueeze(norms, 1)
        diag = torch.ones_like(torch.diag(normed))
        ops.set_diagonal(normed, diag)

        return normed
