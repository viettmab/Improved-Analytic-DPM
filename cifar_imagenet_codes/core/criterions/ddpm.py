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

def _ddpm_dsm_2steps(x_0, eps_model, res_model, cum_alphas, cum_betas, alphas, rescale_timesteps):
    T = len(cum_alphas) - 1
    t = np.random.choice(list(range(1, T + 1)), (len(x_0),))
    recip_noise_coef =  np.sqrt(1.0 - cum_alphas) * np.sqrt(alphas) / (1-alphas)
    eps, x_t = _sample_t(x_0, cum_alphas, cum_betas, t)
    mean_prediction, x_0_pred = p_mean_variance(x_t, eps_model, res_model, None, t, T, cum_alphas, alphas, rescale_timesteps)
    true_mean = q_posterior_mean_variance(x_0, x_t, t, cum_alphas, alphas)
    
    mse = func.mos(func.stp(recip_noise_coef[t], (true_mean-mean_prediction)))

    #First residual
    eps_first, x_t_1_first = _sample_t(x_0, cum_alphas, cum_betas, t-1)
    mean_prediction_1, _ = p_mean_variance(x_t_1_first, eps_model, res_model, x_0_pred, t-1, T, cum_alphas, alphas, rescale_timesteps)
    true_mean_1 = q_posterior_mean_variance(x_0, x_t_1_first, t-1, cum_alphas, alphas)
    mse_first = func.mos(func.stp(recip_noise_coef[t-1], (true_mean_1-mean_prediction_1)))

    #Second residual
    eps_second, x_t_1_second = _sample_t(x_0_pred, cum_alphas, cum_betas, t-1)
    mean_prediction_2, _ = p_mean_variance(x_t_1_second, eps_model, res_model, x_0_pred, t-1, T, cum_alphas, alphas, rescale_timesteps)
    true_mean_2 = q_posterior_mean_variance(x_0, x_t_1_second, t-1, cum_alphas, alphas)
    mse_second = func.mos(func.stp(recip_noise_coef[t-1], (true_mean_2-mean_prediction_2)))
    mse += (mse_first+mse_second)/2
    return mse

def q_posterior_mean_variance(x_start, x_t, t, cum_alphas, alphas):
    cum_alphas_prev = np.append(1.0, cum_alphas[:-1])
    posterior_mean_coef1 = (
            (1. - alphas) * np.sqrt(cum_alphas_prev) / (1.0 - cum_alphas)
        )
    posterior_mean_coef2 = (
        (1.0 - cum_alphas_prev) * np.sqrt(alphas) / (1.0 - cum_alphas)
    )
    posterior_mean = func.stp(posterior_mean_coef1[t],  x_start) + func.stp(posterior_mean_coef2[t],  x_t)
    return posterior_mean

def p_mean_variance(x_t, eps_model, res_model, residual_x_start, t, T, cum_alphas, alphas, rescale_timesteps):
    eps_pred = eps_model(x_t, _rescale_timesteps(torch.from_numpy(t).float().to(x_t.device), T, rescale_timesteps))
    x_0_pred = func.stp(cum_alphas[t] ** -0.5,  x_t) - func.stp((1. / cum_alphas[t] - 1.) ** 0.5, eps_pred)
    x_0_pred = x_0_pred.clamp(-1., 1.)
    if residual_x_start is not None:
        if res_model is not None:
            tensor_t = torch.Tensor(t)
            residual_val_raw = res_model(residual_x_start, tensor_t).squeeze()
            while len(residual_val_raw.shape) < len(x_0_pred.shape):
                residual_val_raw = residual_val_raw[..., None]
            residual_val = residual_val_raw.expand(x_0_pred.shape)
            x_0_pred = (1.0 - residual_val) * x_0_pred + residual_val * residual_x_start
    
    posterior_mean = q_posterior_mean_variance(x_0_pred, x_t, t, cum_alphas, alphas)

    return posterior_mean, x_0_pred

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
        self.res_model = nn.DataParallel(models.res_model)
        self.betas = betas
        self.alphas, self.cum_alphas, self.cum_betas = _make_coeff(self.betas)
        self.alphas, self.cum_alphas, self.cum_betas = self.alphas[1:], self.cum_alphas[1:], self.cum_betas[1:]
        self.rescale_timesteps = rescale_timesteps
        logging.info("DDPMDSM with rescale_timesteps={}".format(self.rescale_timesteps))

    def objective(self, v, **kwargs):
        if self.two_steps:
            return _ddpm_dsm_2steps(v, self.eps_model, self.res_model, self.cum_alphas, self.cum_betas, self.alphas, self.rescale_timesteps)
        else:
            return _ddpm_dsm(v, self.eps_model, self.cum_alphas, self.cum_betas, self.rescale_timesteps)
