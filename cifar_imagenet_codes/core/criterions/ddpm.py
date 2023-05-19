r""" for n>=1, betas[n] is the variance of q(x_n|x_{n-1})
     for n=0,  betas[0]=0
"""

__all__ = ["DDPMDSM"]


import torch
import numpy as np
from .base import NaiveCriterion
import core.utils.managers as managers
import core.func as func
import torch.nn as nn
import logging


def _rescale_timesteps(n, N, flag):
    if flag:
        return n * 1000.0 / float(N)
    return n


def _bipartition(ts):
    if ts.dim() == 4:  # bs * 2c * w * w
        assert ts.size(1) % 2 == 0
        c = ts.size(1) // 2
        return ts.split(c, dim=1)
    else:
        raise NotImplementedError


def _make_coeff(betas):
    assert betas[0] == 0  # betas[0] = 0 for convenience
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas
    return alphas, cum_alphas, cum_betas


def _sample(x_0, cum_alphas, cum_betas):
    N = len(cum_alphas) - 1
    n = np.random.choice(list(range(1, N + 1)), (len(x_0),))
    eps = torch.randn_like(x_0)
    x_n = func.stp(cum_alphas[n] ** 0.5, x_0) + func.stp(cum_betas[n] ** 0.5, eps)
    return N, n, eps, x_n

def _sample_t(x_0, cum_alphas, cum_betas, t):
    eps = torch.randn_like(x_0)
    x_t = func.stp(cum_alphas[t] ** 0.5, x_0) + func.stp(cum_betas[t] ** 0.5, eps)
    return eps, x_t


def _ddpm_dsm(x_0, eps_model, cum_alphas, cum_betas, rescale_timesteps):
    N, n, eps, x_n = _sample(x_0, cum_alphas, cum_betas)
    eps_pred = eps_model(x_n, _rescale_timesteps(torch.from_numpy(n).float().to(x_0.device), N, rescale_timesteps))
    return func.sos(eps - eps_pred)

def _ddpm_dsm_2steps(x_0, eps_model, cum_alphas, cum_betas, rescale_timesteps):
    T = len(cum_alphas) - 1
    t = np.random.choice(list(range(1, T + 1)), (len(x_0),))
    eps, x_t = _sample_t(x_0, cum_alphas, cum_betas, t)
    eps_pred = eps_model(x_t, _rescale_timesteps(torch.from_numpy(t).float().to(x_0.device), T, rescale_timesteps))
    x_0_pred = func.stp(cum_alphas[t] ** -0.5,  x_t) - func.stp((1. / cum_alphas[t] - 1.) ** 0.5, eps_pred)
    x_0_pred = x_0_pred.clamp(-1., 1.)
    mse = func.sos(eps - eps_pred)
    #First residual
    eps_first, x_t_1_first = _sample_t(x_0, cum_alphas, cum_betas, t-1)
    eps_pred_first = eps_model(x_t_1_first, _rescale_timesteps(torch.from_numpy(t-1).float().to(x_0.device), T, rescale_timesteps))
    #Second residual
    eps_second, x_t_1_second = _sample_t(x_0_pred, cum_alphas, cum_betas, t-1)
    eps_pred_second = eps_model(x_t_1_second, _rescale_timesteps(torch.from_numpy(t-1).float().to(x_0.device), T, rescale_timesteps))
    mse += (func.sos(eps_first - eps_pred_first) + func.sos(eps_second - eps_pred_second)) / 2
    return mse

def _ddpm_dsm_zero(x_0, d_model, cum_alphas, cum_betas, rescale_timesteps):
    N, n, eps, x_n = _sample(x_0, cum_alphas, cum_betas)
    d_pred = d_model(x_n, _rescale_timesteps(torch.from_numpy(n).float().to(x_0.device), N, rescale_timesteps))
    return func.sos(x_0 - d_pred)


def _ddpm_ddm(x_0, tau_model, cum_alphas, cum_betas, rescale_timesteps):
    N, n, eps, x_n = _sample(x_0, cum_alphas, cum_betas)
    tau_pred = tau_model(x_n, _rescale_timesteps(torch.from_numpy(n).float().to(x_0.device), N, rescale_timesteps))
    return func.sos(eps.pow(2) - tau_pred)


def _ddpm_ddm_zero(x_0, kappa_model, cum_alphas, cum_betas, rescale_timesteps):
    N, n, eps, x_n = _sample(x_0, cum_alphas, cum_betas)
    kappa_pred = kappa_model(x_n, _rescale_timesteps(torch.from_numpy(n).float().to(x_0.device), N, rescale_timesteps))
    return func.sos(x_0.pow(2) - kappa_pred)


def _ddpm_dsdm(x_0, eps_tau_model, cum_alphas, cum_betas, rescale_timesteps):
    N, n, eps, x_n = _sample(x_0, cum_alphas, cum_betas)
    eps_tau_pred = eps_tau_model(x_n, _rescale_timesteps(torch.from_numpy(n).float().to(x_0.device), N, rescale_timesteps))
    eps_pred, tau_pred = _bipartition(eps_tau_pred)
    return func.sos(eps - eps_pred), func.sos(eps.pow(2) - tau_pred)


class DDPMDSM(NaiveCriterion):
    def __init__(self,
                 betas,
                 rescale_timesteps,  # todo: remove this argument
                 two_steps,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 ):
        r""" Estimating the mean of optimal Gaussian reverse in DDPM = Denoising score matching (DSM)
        """
        assert isinstance(betas, np.ndarray) and betas[0] == 0
        super().__init__(models, optimizers, lr_schedulers)
        self.two_steps = two_steps
        self.eps_model = nn.DataParallel(models.eps_model)  # predict noise
        self.betas = betas
        self.alphas, self.cum_alphas, self.cum_betas = _make_coeff(self.betas)
        self.rescale_timesteps = rescale_timesteps
        logging.info("DDPMDSM with rescale_timesteps={}".format(self.rescale_timesteps))

    def objective(self, v, **kwargs):
        if self.two_steps:
            return _ddpm_dsm_2steps(v, self.eps_model, self.cum_alphas, self.cum_betas, self.rescale_timesteps)
        else:
            return _ddpm_dsm(v, self.eps_model, self.cum_alphas, self.cum_betas, self.rescale_timesteps)
