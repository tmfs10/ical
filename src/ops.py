
"""ops.py"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
import inspect

_eps = 1.0e-5


def recon_loss(x, x_recon):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

def dot(x, y):
    return torch.squeeze(torch.matmul(tf.unsqueeze(x, 0), tf.unsqueeze(y, 1)))

def sq_sum(t):
    "The squared Frobenius-type norm of a tensor, sum(t ** 2)."
    return 2 * torch.sum(t**2)

def tensor_all(x):
    return (1-(x.long())).sum().item() < 1e-5

def tensor_any(x):
    return x.long().sum().item() > 1e-5

def is_inf(x):
    return tensor_any(x == float('inf')) and tensor_any(x == -float('inf'))

def isinf(x):
    return is_inf(x)

def is_nan(x):
    return tensor_any(x != x)

def isnan(x):
    return is_nan(x)

def is_finite(x):
    return tensor_all((x != float('inf')) * (x == x))

def isfinite(x):
    return is_finite(x)

def diag_part(X):
    if len(X.shape) == 2:
        return torch.diag(X)
    else:
        assert len(X.shape) > 2, X.shape
        assert X.shape[-2] == X.shape[-1], X.shape
        shp = X.shape
        diag_indices = torch.arange(shp[-2]) * shp[-1] + torch.arange(shp[-1])
        X = torch.index_select(X.view(shp[:-2], -1), diag_indices).view(shp)
        return X

def range_complement(n, idx):
    idx_select = torch.zeros(n)
    if isinstance(idx, set):
        idx_select[list(idx)] = 1
    else:
        idx_select[idx] = 1
    idx_select = (1-idx_select).byte()
    idx = torch.arange(n)[idx_select]
    return idx

def set_diagonal(x, v):
    assert len(x.shape) >= 2
    assert len(v.shape) == 1
    assert x.shape[-2] == v.shape[0], "%s[-2] == %s[0]" % (str(x.shape), str(v.shape))
    assert x.shape[-1] == v.shape[0], "%s[-1] == %s[0]" % (str(x.shape), str(v.shape))
    mask = torch.diag(torch.ones_like(v))
    for i in range(len(x.shape)-2):
        mask = mask.unsqueeze(1)
    x = mask*torch.diag(v) + (1-mask)*x
    return x

def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def cov(x):
    # calculate covariance matrix of rows
    # (n, p) -> (n, n)
    mean_x = x.mean(1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    return c

def corrcoef(x):
    """
    Mimics `np.corrcoef`

    Arguments
    ---------
    x : 2D torch.Tensor
    
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)

    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref: 
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013

    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    c = cov(x)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c

def cross_cov(x, y):
    # x is (n, p), y is (n, q) -> (p, q)
    # calculate covariance matrix of rows

    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    ym = y.sub(mean_y.expand_as(y))
    c = xm.t().mm(ym)
    c = c / (x.size(0) - 1)

    return c

def parallel_cross_cov(x, y):
    # calculate covariance vector
    # (p, n, a), (p, n, b) -> (p, a, b), n is the number of samples

    mean_x = x.mean(1, keepdim=True)
    mean_y = y.mean(1, keepdim=True)

    xm = x-mean_x.expand_as(x)
    ym = y-mean_y.expand_as(y)

    xm = xm.permute([0, 2, 1]) # becomes (p, a, n)
    c = torch.einsum('pan,pnb->pab', xm, ym)

    c /= (x.shape[1] - 1)

    return c # pab

def parallel_cross_corrcoef(x, y):
    # calculate correlation vector
    # (p, n, a), (p, n, b) -> (p, a, b), n is the number of samples

    c = parallel_cross_cov(x, y)
    stddev_x = x.std(1, keepdim=True).permute([0, 2, 1]) # (p, a, 1)
    stddev_y = y.std(1, keepdim=True) # (p, 1, b)
    norm = stddev_x*stddev_y # (p, a, b)

    c /= norm

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c

def cross_corrcoef(x, y):
    # x is (n, p), y is (n, q) -> (p, q)
    # calculate correlation matrix of rows

    c = cross_cov(x, y)

    # normalize
    x_std = x.std(0, keepdim=True).transpose(0, 1)
    y_std = y.std(0, keepdim=True)
    norm = x_std*y_std

    c  = c/norm

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c

def get_rng_state():
    return (copy.deepcopy(torch.get_rng_state()), copy.deepcopy(np.random.get_state()))

def set_rng_state(state):
    torch.set_rng_state(state[0])
    np.random.set_state(state[1])

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


def get_debug_loc(level=1):
    pf = inspect.currentframe()
    for _ in range(level):
        pf = pf.f_back
    (filename, line_number, function_name, lines, index) = inspect.getframeinfo(pf)
    filename = filename.split('/')[-1]
    return filename, line_number

def debug_print(*args, **kwargs):
    filename, line_number = get_debug_loc(2)
    print("(%s, %d) >> " % (filename, line_number), *args, **kwargs)

def shape_assert(s, t, prefix=""):
    if isinstance(s, torch.Tensor):
        s = s.shape
    filename, line_number = get_debug_loc(2)
    error_str = "(%s, %d) >> %s: %s == %s" % (filename, line_number, prefix, s, t)
    assert len(s) == len(t), error_str
    
    for i in range(len(s)):
        if t[i] == -1:
            continue
        assert s[i] == t[i], "(%s, %d) >> %s: %s[%d] == %s[%d]" % (filename, line_number, prefix, s, i, t, i)

def index_tensor_by_vector(t, v, keepdim=True):
    nr, nc = t.shape[:2]
    rest_shape = list(t.shape[2:])
    assert v.shape[0] == nr
    assert v.max() <= nc
    mask = torch.eye(nc, device=v.device)[v].bool().view(-1)
    ret = t.view([-1]+rest_shape)[mask]
    shape_assert(ret.shape, [nr]+rest_shape)
    if keepdim:
        return ret.unsqueeze(1)
    else:
        return ret

def scatter_tensor_by_vector(t, v, s):
    nr, nc = t.shape[:2]
    rest_shape = list(t.shape[2:])
    assert v.shape[0] == nr
    assert v.max() <= nc
    mask = torch.eye(nc, device=v.device)[v].bool().view(-1)
    t.view([-1]+rest_shape)[mask] = s
