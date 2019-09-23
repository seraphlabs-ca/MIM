import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
import math

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):

    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel))
        self.scale = nn.Parameter(torch.ones(1, in_channel))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            mean = input.mean(0, keepdim=True)
            std = input.std(0, keepdim=True)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvMat(nn.Module):

    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        out = input @ self.weight
        logdet = (
            torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return output @ self.weight.inverse()


class InvMatLU(nn.Module):

    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.tensor(w_p)
        w_l = torch.tensor(w_l)
        w_s = torch.tensor(w_s)
        w_u = torch.tensor(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.tensor(u_mask))
        self.register_buffer('l_mask', torch.tensor(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        weight = self.calc_weight()

        out = input @ weight
        logdet = sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight

    def reverse(self, output):
        weight = self.calc_weight()
        return output @ weight.inverse()


class ZeroLinear(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.scale = nn.Parameter(torch.zeros(1, out_channel))

    def forward(self, input):
        out = self.linear(input)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):

    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Linear(in_channel // 2, filter_size),
            nn.ReLU(inplace=True),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(inplace=True),
            ZeroLinear(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):

    def __init__(self, in_channel, affine=True, mat_lu=True, filter_size=512):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if mat_lu:
            self.invconv = InvMatLU(in_channel)

        else:
            self.invconv = InvMat(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine, filter_size=filter_size)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def reshape(x, n_class):
    s = x.size()
    return x.view(s[0], n_class, s[1] // n_class)


def gaussian_mixture_log_p(x, mean, log_sd, n_class):
    logpx_all = gaussian_log_p(reshape(x, 1), reshape(mean, n_class), reshape(log_sd, n_class))
    return logpx_all


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


def gaussian_mixture_sample(eps, mean, log_sd, n_class):
    sample = gaussian_sample(reshape(eps, 1), reshape(mean, n_class), reshape(log_sd, n_class))
    return sample


class Block(nn.Module):

    def __init__(self, in_channel, n_flow, split=True, affine=True, mat_lu=True, n_class=None, filter_size=512):
        super().__init__()

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(in_channel, affine=affine, mat_lu=mat_lu, filter_size=filter_size))

        self.split = split

        # multiples the number of output channels if class-conditional
        self.n_class = n_class
        mult = 1 if n_class is None else n_class
        if split:
            self.prior = ZeroLinear(in_channel // 2, in_channel * mult)

        else:
            self.prior = ZeroLinear(in_channel, in_channel * 2 * mult)

    def forward(self, input, label=None):
        b_size, n_dim = input.shape
        out = input

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            if self.n_class is None:
                log_p = gaussian_log_p(z_new, mean, log_sd)
                log_p = log_p.view(b_size, -1).sum(1)
                logits = None
            else:
                log_p = gaussian_mixture_log_p(z_new, mean, log_sd, self.n_class)
                log_p = log_p.view(b_size, self.n_class, -1).sum(2)
                logits = log_p
                if label is None:
                    log_p = log_p.logsumexp(dim=1) - np.log(self.n_class)
                else:
                    log_p = torch.gather(log_p, 1, label[:, None])[:, 0]
        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            if self.n_class is None:
                log_p = gaussian_log_p(out, mean, log_sd)
                log_p = log_p.view(b_size, -1).sum(1)
                logits = None
            else:
                log_p = gaussian_mixture_log_p(out, mean, log_sd, self.n_class)
                log_p = log_p.view(b_size, self.n_class, -1).sum(2)
                logits = log_p
                if label is None:
                    log_p = log_p.logsumexp(dim=1) - np.log(self.n_class)
                else:
                    log_p = torch.gather(log_p, 1, label[:, None])[:, 0]
            z_new = out

        return out, logdet, log_p, z_new, logits

    def reverse(self, output, eps=None, reconstruct=False, label=None):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                if self.n_class is None:
                    z = gaussian_sample(eps, mean, log_sd)
                else:
                    z = gaussian_mixture_sample(eps, mean, log_sd, n_class=self.n_class)
                    if label is None:
                        label = torch.randint(0, self.n_class, (z.size(0),)).to(z.device)
                    s = z.size()
                    z = torch.gather(z, 1, label[:, None, None].repeat(1, 1, s[2]))
                    z = z[:, 0, :]
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                if self.n_class is None:
                    z = gaussian_sample(eps, mean, log_sd)
                else:
                    z = gaussian_mixture_sample(eps, mean, log_sd, n_class=self.n_class)
                    if label is None:
                        label = torch.randint(0, self.n_class, (z.size(0),)).to(z.device)
                    s = z.size()
                    z = torch.gather(z, 1, label[:, None, None].repeat(1, 1, s[2]))
                    z = z[:, 0, :]
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        return input


class ConditionalBlock(nn.Module):

    def __init__(self, in_channel, n_flow, n_class, affine=True, mat_lu=True, filter_size=512):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.n_class = n_class
        for i in range(n_class):
            self.blocks.append(Block(in_channel, n_flow,
                                     split=False, affine=affine, mat_lu=mat_lu, filter_size=filter_size))

    def forward(self, input, label=None):
        outputs = [block(input, label=label) for block in self.blocks]
        outs, logdets, log_ps, z_news, _ = zip(*outputs)
        outs = torch.cat([out[:, None, :] for out in outs], dim=1)
        z_news = outs
        logdets = torch.cat([logdet[:, None] for logdet in logdets], dim=1)
        log_ps = torch.cat([log_p[:, None] for log_p in log_ps], dim=1)
        log_ps = log_ps + logdets  # fold logdet into log_p
        logits = log_ps
        if label is None:
            log_ps = log_ps.logsumexp(dim=1) - np.log(self.n_class)
        else:
            log_ps = torch.gather(log_ps, 1, label[:, None])[:, 0]
        logdets = torch.zeros_like(log_ps)
        return outs, logdets, log_ps, z_news, logits

    def reverse(self, output, eps=None, reconstruct=False, label=None):
        zs = [block.reverse(output, eps=eps, reconstruct=reconstruct, label=None) for block in self.blocks]
        zs = torch.cat([z[:, None, :] for z in zs], dim=1)
        if label is None:
            label = torch.randint(0, self.n_class, (zs.size(0),)).to(zs.device)
        s = zs.size()
        z = torch.gather(zs, 1, label[:, None, None].repeat(1, 1, s[2]))
        z = z[:, 0, :]
        return z


class Glow(nn.Module):

    def __init__(self, in_channel, n_flow, n_block,
                 affine=True, mat_lu=True, n_class=None, n_cond_flow=-1, filter_size=512, alpha=None):
        super().__init__()
        self.alpha = alpha
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            print(in_channel)
            self.blocks.append(Block(n_channel, n_flow, affine=affine, mat_lu=mat_lu, filter_size=filter_size))
            n_channel //= 2
        if n_cond_flow != -1:
            assert n_class is not None, "need to give class number if using conditional flows"
            self.blocks.append(ConditionalBlock(n_channel, n_cond_flow, n_class,
                                                affine=affine, mat_lu=mat_lu, filter_size=filter_size))
        else:
            self.blocks.append(Block(n_channel, n_flow,
                                     split=False, affine=affine, mat_lu=mat_lu,
                                     n_class=n_class, filter_size=filter_size))

    def forward(self, input, label=None):
        if self.alpha is not None:
            input, logdet = logit_transform(input, self.alpha)
        else:
            logdet = 0

        log_p_sum = 0
        out = input
        z_outs = []

        for i, block in enumerate(self.blocks):
            out, det, log_p, z_new, logits = block(out, label=label if i == len(self.blocks) - 1 else None)
            z_outs.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs, logits

    def reverse(self, z_list, reconstruct=False, label=None):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct, label=label)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct, label=label)

        if self.alpha is not None:
            input = (torch.sigmoid(input) - self.alpha) / (1. - 2. * self.alpha)
        return input


def logit_logdetgrad(x, alpha):
    s = alpha + (1 - 2 * alpha) * x
    logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * alpha)
    return logdetgrad


def logit_transform(x, alpha):
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log(1 - s)
    return y, logit_logdetgrad(x, alpha).view(x.size(0), -1).sum(1)

if __name__ == "__main__":
    model_single = Glow(16, 4, 1, affine=True, mat_lu=True, n_class=10, n_cond_flow=-1, filter_size=64)

    x = torch.randn(1000, 16)
    logp, logdet, z, logits = model_single(x)
    logp, logdet, z, logits = model_single(x)
    print(logp.size(), logdet.size())
    print(z[0].size())

    # print(logits.size())
    x_re = model_single.reverse(z, label=torch.randint(0, 10, (1000,)))
    print(x_re.size())

    print((x - x_re).mean())
