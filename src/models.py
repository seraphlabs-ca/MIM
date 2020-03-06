"""
A collection of toy models.
"""
import numbers

import numpy as np
import matplotlib.pyplot as plt
import os
import wrapt

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from auxiliary import Swish, log_prob, build_layers
import auxiliary as aux


#=============================================================================#
# MIM
#=============================================================================#


class MIM(nn.Module):
    """
    A base class for MIM
    """

    def __init__(self, p_z, p_x_given_z, q_x, q_z_given_x):
        super().__init__()

        self.p_z = p_z
        self.p_x_given_z = p_x_given_z
        self.q_x = q_x
        self.q_z_given_x = q_z_given_x

    def sample_p(self, batch_size=1):
        if self.p_z is None:
            pass
        else:
            p_z = self.p_z(batch_size)

        z = p_z.rsample()
        p_x_given_z = self.p_x_given_z(z)
        x = p_x_given_z.rsample()

        return z, x

    def sample_q(self, batch_size=1):
        q_x = self.q_x(batch_size)
        x = q_x.rsample()
        q_z_given_x = self.q_z_given_x(x)
        z = q_z_given_x.rsample()

        return z, x

    def sample(self, batch_size=1):
        return self.sample_p(batch_size=batch_size)

    def forward(self, x_obs=None, z_obs=None):
        if (x_obs is not None) and (z_obs is not None):
            raise ValueError("x and z cannot be given together")
        if (x_obs is None) and (z_obs is None):
            raise ValueError("x or z must be given together")

        if x_obs is not None:
            q_z_given_x = self.q_z_given_x(x_obs)
            z = q_z_given_x.rsample()

            p_x_given_z = self.p_x_given_z(z)
            x = p_x_given_z.rsample()

            if self.p_z is None:
                p_z = q_z_given_x
            else:
                p_z = self.p_z(z.shape[0])

            if self.q_x is None:
                q_x = self.p_x_given_z(z)
                q_x._log_prob = q_x.log_prob
                q_x.log_prob = lambda value: q_x._log_prob(value) + p_z.log_prob(z) - q_z_given_x.log_prob(z)
            else:
                q_x = self.q_x(x.shape[0])
        else:
            p_x_given_z = self.p_x_given_z(z_obs)
            x = p_x_given_z.rsample()

            q_z_given_x = self.q_z_given_x(x)
            z = q_z_given_x.rsample()

            if self.q_x is None:
                # q_x = p_x_given_z
                q_x = self.p_x_given_z(z)
                q_x._log_prob = q_x.log_prob
                q_x.log_prob = lambda value: q_x._log_prob(
                    value) + self.p_z(z.shape[0]).log_prob(z) - q_z_given_x.log_prob(z)
            else:
                q_x = self.q_x(x.shape[0])

            if self.p_z is None:
                p_z = self.q_z_given_x(x)
                p_z._log_prob = p_z.log_prob
                p_z.log_prob = lambda value: p_z._log_prob(value) + q_x.log_prob(x) - p_x_given_z.log_prob(x)
            else:
                p_z = self.p_z(z.shape[0])

        return z, x, q_x, q_z_given_x, p_z, p_x_given_z
