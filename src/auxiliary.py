import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import csv
import itertools

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision
from torchvision.utils import save_image


from cond_pixelcnn.utils.knnie import kraskov_mi as ksg_mi

#=============================================================================#
# Auxiliary functions
#=============================================================================#


def log_prob(p, c=1e-5):
    """
    Truncated log_prob for numerical stability.
    """
    return torch.log((1.0 - c) * p + c / 2)


def swish(input, beta=1.0):
    r"""
    Applies element-wise :math:`\text{Swish}(x) = x*\sigmoid(x)`
    """
    return input * torch.sigmoid(beta * input)


#=============================================================================#
# Modules
#=============================================================================#


class Swish(torch.nn.Module):
    """
    A swish Module wrapper
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)


class ResBlock(torch.nn.Module):
    """
    A residual block
    """

    def __init__(self, *args):
        super().__init__()

        self.act = torch.nn.Sequential(*args)

    def forward(self, input):
        return self.act(input) + input


class ActNorm(torch.nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_inputs))
        self.bias = torch.nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-0)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        outputs = (inputs - self.bias) * torch.exp(self.weight)

        return outputs

#=============================================================================#
# Build layers
#=============================================================================#


def expand_layers_dim(input_dim, mid_dim, output_dim, layers):
    """
    Returns a list [(in dim, out dim), ...] with layers count of elements.
    """
    if layers < 0:
        raise ValueError("Expecting layers >= 0 but received layers = {layers}".format(layers=layers))

    if layers == 0:
        layers_dim = []
    elif layers == 1:
        layers_dim = [(input_dim, output_dim)]
    else:
        layers_dim = (
            [(input_dim, mid_dim)] +
            [(mid_dim, mid_dim)] * (layers - 2) +
            [(mid_dim, output_dim)]
        )

    return layers_dim


def build_layers(input_dim, mid_dim, output_dim, layers,
                 input_layer_gen=lambda i, o: torch.nn.Linear(i, o),
                 mid_layer_gen=lambda i, o: torch.nn.Linear(i, o),
                 output_layer_gen=lambda i, o: torch.nn.Linear(i, o),
                 act=Swish(),
                 ):
    layers_dim = expand_layers_dim(input_dim, mid_dim, output_dim, layers)
    layers_list = [
        gen(i, o) for ((i, o), gen) in zip(
            layers_dim,
            [input_layer_gen] * (layers > 1) +
            [mid_layer_gen] * max(0, layers - 2) +
            [output_layer_gen] * min(1, layers)
        )
    ]

    layers_list = list(itertools.chain(*zip(layers_list, [act] * len(layers_list))))
    layers_list.pop()

    return torch.nn.Sequential(*layers_list)

#=============================================================================#
# Distributions
#=============================================================================#


class DiagGaussianDist(torch.distributions.Normal):
    """
    A multivariate Gaussian distribution with a diagonal covariance matrix.
    """

    def __init__(self, loc, scale):
        super().__init__(
            loc=loc,
            scale=scale,
        )

    def log_prob(self, value):
        return super().log_prob(value=value).sum(-1)

    def plot(self, N=None, dims=[0, 1], label=None, **kwargs):
        """
        Plot a representation of the distribution.

        N - number of samples to return.
        dims - dimensions to visualize
        """
        defaults = dict(
            edgecolor="k",
            facecolor='none',
            ls="--",
            lw=5.0,
            alpha=1.0,
        )

        defaults.update(kwargs)

        N = min(N, self.scale.shape[0])
        scale = self.scale[:N]
        loc = self.loc[:N]

        b = loc.shape[0]
        d = loc.shape[1]

        dims = [min(d, cur_dim) for cur_dim in dims]

        ax = plt.gca()
        all_h = []
        for i in range(b):
            w, h = (max(v, 0.01) for v in scale[i, dims])
            ell = Ellipse(
                xy=loc[i, dims],
                width=w * 2,
                height=h * 2,
                **defaults
            )

            ax.add_patch(ell)
            all_h.append(ell)

        if label is not None:
            ell.set(label=label)

        return all_h


class BernoulliDist(torch.distributions.Bernoulli):
    """
    An IID Bernoulli distribution.
    """

    def __init__(self, probs):
        super().__init__(
            probs=probs,
        )

    def log_prob(self, value):
        return super().log_prob(value=value).sum(-1)

    def rsample(self, sample_shape=torch.Size([])):
        return self.probs

    def plot(self, N=None, dims=[0, 1], label=None, **kwargs):
        """
        Plot a representation of the distribution.

        N - number of samples to return.
        dims - dimensions to visualize
        """
        # TODO: FINISH ME


class GMMDist(torch.distributions.Distribution):
    """
    A Gaussian Mixture Model distribution.
    """

    def __init__(self, comps, logcoefs, truncate_comps=False):
        super().__init__()

        # store components
        self.comps = comps

        # truncate low probability components
        if truncate_comps:
            logcoefs = logcoefs * 1
            logcoefs[logcoefs < logcoefs.mean(dim=-1, keepdim=True) - logcoefs.std(dim=-1, keepdim=True)] -= 1e5

        # store normalized coefficients
        self.logcoefs = logcoefs - torch.logsumexp(logcoefs, dim=-1, keepdim=True)
        self.coefs = self.logcoefs.exp()

        self.k = self.logcoefs.shape[-1]

    def sample(self, batch_size=None, with_index=False):
        with torch.no_grad():
            return self.rsample(batch_size=batch_size, with_index=with_index)

    def rsample(self, batch_size=None, with_index=False):
        b = self.coefs.shape[0]
        if (batch_size is not None):
            if (b == 1) or (batch_size == b):
                if batch_size == b:
                    sample_shape = ()
                else:
                    sample_shape = ()
            else:
                raise ValueError("Mismatch between batch_size = {batch_size} and shape = {shape}".format(
                    batch_size=batch_size,
                    shape=self.coefs.shape,
                ))
        else:
            sample_shape = ()

        # randomly select mixture component
        coefs = self.coefs
        index = torch.multinomial(coefs, num_samples=1, replacement=True).flatten()
        samp = None
        for i in range(self.k):
            mask = index.eq(i)

            if mask.any():
                cur_samp = self.comps[i].rsample(sample_shape=sample_shape)
                if samp is None:
                    samp = torch.zeros_like(cur_samp)

                samp[mask, ...] = cur_samp[mask, ...]

        if with_index:
            return samp, index
        else:
            return samp

    def log_prob(self, value):
        logcoefs = self.logcoefs
        all_logp = []
        for i in range(self.k):
            all_logp.append(logcoefs[:, i] + self.comps[i].log_prob(value))

        log_prob = torch.logsumexp(torch.stack(all_logp, dim=0), dim=0)

        return log_prob

    def plot(self, N=None, dims=[0, 1], label=None, **kwargs):
        """
        Plot a representation of the distribution.

        N - number of samples to return.
        dims - dimensions to visualize
        """
        all_h = []
        if "alpha" in kwargs:
            alpha = kwargs.pop("alpha")
        else:
            alpha = 1.0

        alpha_scale = self.coefs.max()
        for i, dist in enumerate(self.comps):
            all_h.extend(dist.plot(
                N=N,
                dims=dims,
                label=label,
                alpha=alpha * self.coefs[0, i] / alpha_scale,
                ** kwargs))
            # label only first component
            label = None

    @classmethod
    def factory(cls, k, d=2):
        """
        Builds a GMM from int or str.

        k (int) - k is number of components for k >= 0,
                  else k+1 (with additional component in origin)
        k (str) - Extract GMM from string.
        """
       # if k is not integer treat as string
        try:
            # if k is integer treat
            k = int(k)

            i = np.arange(abs(k), dtype=np.float)

            # mixture components (include origin if non-negative)
            k0 = (k >= 0)
            k = abs(k) + int(k0)

            # build mixture components
            scale = 0.25 * 5 / k

            locs = torch.linspace(-10 * k, 10 * k, k * d).view((k, d))

            if k0:
                locs[:k0, :min(2, d)] = 0

            locs[k0:, :min(2, d)] = torch.tensor([
                np.cos(i / k * 2 * np.pi),
                np.sin(i / k * 2 * np.pi),
            ], dtype=torch.float).t()[:, :min(2, d)]

            scales = scale * torch.ones_like(locs)
        except:
            # else create a binary mask
            msg = str(k)

            font = ImageFont.load_default()
            im = Image.new("I", font.getsize(msg))
            ImageDraw.Draw(im).text((0, 0), msg, fill='white', font=font)

            # mask = np.fliplr(np.array(im.resize((im.size[0] * 2, im.size[1] * 2), Image.BICUBIC)).T)
            mask = np.fliplr(np.array(im).T)
            shape = np.array(mask.shape[0:2]).astype(np.float)
            locs = torch.tensor(np.vstack(np.where(mask)).T / shape - [0.45, 0.45],
                                dtype=torch.float) * torch.tensor([2.0, 2.0]) * 1.8
            scale = torch.tensor([4.0, 4.0]) / torch.tensor(shape, dtype=torch.float) / 4
            scales = scale * torch.ones_like(locs)

            k = scales.shape[0]

        logcoefs = (torch.ones(1, k) / k).log()
        comps = []
        for i in range(k):
            dist = DiagGaussianDist(
                loc=locs[i:(i + 1)],
                scale=scales[i:(i + 1)],
            )

            comps.append(dist)

        return cls(
            comps=comps,
            logcoefs=logcoefs,
        )


#=============================================================================#
# Build distribution layers
#=============================================================================#


class GaussianPriorLayer(torch.nn.Module):
    """
    A layer that returns a Gaussian distribution.
    """

    def __init__(self, output_dim,
                 min_scale=1e-10, active=True, loc=None, logscale=None):
        super().__init__()

        self.output_dim = output_dim
        self.min_scale = min_scale
        self.active = active

        if active:
            self.loc = torch.nn.Parameter(torch.randn(1, self.output_dim))
            self.logscale = torch.nn.Parameter(torch.randn(1, self.output_dim) * 1e-3)
        else:
            self.loc = torch.zeros(1, self.output_dim)
            self.logscale = torch.zeros(1, self.output_dim)

        if loc is not None:
            self.loc.data.set_(loc)
        if logscale is not None:
            self.logscale.data.set_(logscale)

    def forward(self, input):
        if torch.is_tensor(input):
            b = input.shape[0]
        else:
            b = int(input)

        loc = self.loc.expand((b, self.output_dim))
        scale = torch.exp(self.logscale).expand((b, self.output_dim)) + self.min_scale

        return DiagGaussianDist(
            loc=loc,
            scale=scale,
        )


class GaussianCondLayer(torch.nn.Module):
    """
    A layer that returns a conditional Gaussian distribution.
    """

    def __init__(self, input_dim, output_dim, min_scale=1e-10):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_scale = min_scale

        self.fc_loc = torch.nn.Linear(self.input_dim, self.output_dim)
        self.fc_logscale = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, input):
        loc = self.fc_loc(input)
        scale = torch.exp(self.fc_logscale(input)) + self.min_scale

        return DiagGaussianDist(
            loc=loc,
            scale=scale,
        )


class BernoulliPriorLayer(torch.nn.Module):
    """
    A layer that returns a Bernoulli distribution.
    """

    def __init__(self, output_dim, min_scale=1e-3):
        super().__init__()

        self.output_dim = output_dim
        self.min_scale = min_scale

        self.logprobs = torch.nn.Parameter(torch.randn(1, self.output_dim) * 1e-3)

    def forward(self, input):
        probs = torch.sigmoid(self.logprobs).clamp(self.min_scale, 1 - self.min_scale)

        return BernoulliDist(probs=probs)


class BernoulliCondLayer(torch.nn.Module):
    """
    A layer that returns a Bernoulli distribution.
    """

    def __init__(self, input_dim, output_dim, min_scale=1e-3):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_scale = min_scale

        self.fc_logprobs = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, input):
        probs = torch.sigmoid(self.fc_logprobs(input)).clamp(self.min_scale, 1 - self.min_scale)

        return BernoulliDist(probs=probs)


class GMMPriorLayer(torch.nn.Module):
    """
    A layer that returns a Gaussian Mixture Model distribution.
    """

    def __init__(self, output_dim, components_num,
                 min_scale=1e-10, active=True,
                 logcoefs=None, locs=None, logscales=None):
        super().__init__()

        self.output_dim = output_dim
        self.components_num = components_num
        self.min_scale = min_scale
        self.active = active

        if logscales is None:
            logscales = [None] * components_num
            # logscales = [(torch.ones(1, output_dim) * 0.1).log()] * components_num
        if locs is None:
            locs = [None] * components_num

        self.comps = torch.nn.ModuleList([GaussianPriorLayer(
            output_dim=self.output_dim,
            min_scale=self.min_scale,
            active=self.active,
            loc=locs[i],
            logscale=logscales[i],
        ) for i in range(self.components_num)])

        if active:
            self.logcoefs = torch.nn.Parameter(torch.randn(1, self.components_num))
        else:
            self.logcoefs = (torch.ones((1, self.components_num)) / self.components_num).log()

        if logcoefs is not None:
            self.logcoefs.data.set_(logcoefs)

    def forward(self, input):
        if torch.is_tensor(input):
            b = input.shape[0]
        else:
            b = int(input)

        comps = [c(input) for c in self.comps]
        logcoefs = self.logcoefs.expand((b, self.components_num))

        return GMMDist(
            comps=comps,
            logcoefs=logcoefs,
            truncate_comps=True,
        )


class GMMCondLayer(torch.nn.Module):
    """
    A layer that returns a conditional Gaussian Mixture Model distribution.
    """

    def __init__(self, input_dim, output_dim, components_num, min_scale=1e-10):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.components_num = components_num
        self.min_scale = min_scale

        self.comps = torch.nn.ModuleList([GaussianCondLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            min_scale=self.min_scale,
        ) for i in range(self.components_num)])

    def forward(self, input):
        comps = [c(input) for c in self.comps]
        logcoefs = self.fc_logcoefs(input)

        return GMMDist(comps=comps, logcoefs=logcoefs)


class GMMPseudoPriorLayer(torch.nn.Module):
    """
    A layer that returns a Gaussian Mixture Model distribution by learning
    pseudo conditional samples.
    """

    def __init__(self, cond_comp, input_dim, output_dim, components_num, min_scale=1e-10,
                 active=False, pseudo_input=None):
        super().__init__()

        self.cond_comp = cond_comp
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.components_num = components_num
        self.min_scale = min_scale
        self.active = active

        if active:
            self.logcoefs = torch.nn.Parameter(torch.randn(1, self.components_num))
        else:
            self.logcoefs = (torch.ones((1, self.components_num)) / self.components_num).log()

        self.pseudo_input = torch.nn.Parameter(torch.randn(components_num, input_dim))
        if pseudo_input is not None:
            self.pseudo_input.data.set_(pseudo_input)

    def forward(self, input):
        if torch.is_tensor(input):
            b = input.shape[0]
        else:
            b = int(input)

        inputs = [
            i.expand((b, self.input_dim))
            for i in torch.split(self.pseudo_input, 1, dim=0)
        ]
        comps = [self.cond_comp(i) for i in inputs]
        logcoefs = self.logcoefs.expand((b, self.components_num))

        return GMMDist(
            comps=comps,
            logcoefs=logcoefs,
            # truncate_comps=True,
        )


#=============================================================================#
# Scheduler
#=============================================================================#


class NoamLRScheduler(object):
    """
    A Noam LR decay scheduler with warm up.
    Grows linearly for warmup_steps to init_lr (taken from param_group["lr"],
    and than decays max(init_lr*step**-0.5, min_lr).

    Should be called before optimizer.step().
    """

    def __init__(self, optimizer, min_lr=None, warmup_steps=None, enable_decay=None, step=None, params={}):
        super().__init__()

        kwargs = {
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "enable_decay": enable_decay,
            "step": step,
        }

        params = {
            "min_lr": 0.0,
            "warmup_steps": 4000,
            # if 0 will not manipulate lr after warmup
            "enable_decay": 1,
            "step": -1,
        }

        for k, v in kwargs.items():
            if v is not None:
                params[k] = v

        self.params = params

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer

    def step(self, step=None):
        if step is None:
            step = self.params["step"] + 1
        self.params["step"] = step

        min_lr = self.params["min_lr"]
        warmup_steps = self.params["warmup_steps"]

        if self.params["enable_decay"] or (step <= warmup_steps):
            for param_group in self.optimizer.param_groups:
                init_lr = param_group.get("init_lr", param_group["lr"])
                lr = max(
                    init_lr * warmup_steps**0.5 * min(
                        step * warmup_steps**-1.5,
                        max(step, 1)**-0.5,
                    ),
                    min_lr,
                )
                param_group['lr'] = lr
                param_group['init_lr'] = init_lr

#=============================================================================#
# Datasets
#=============================================================================#


def load_dataset(dataset_name,
                 z_dim,
                 batch_size=128,
                 cuda=torch.cuda.is_available(), data_path=None,
                 anchors={},
                 ):
    from datasets import DistDataset, PCAFashionMNIST

    if cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # z anchor
    shape = (1, z_dim)
    P_z = DiagGaussianDist(loc=torch.zeros(shape), scale=torch.ones(shape))

    # store global access to anchors
    anchors.update({
        "P_z": P_z,
        "P_x": None,
    })

    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__),
            "../data",
            "assets/datasets",
            dataset_name,
        )

    kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}
    kwargs["drop_last"] = True

    if dataset_name == "mnist":
        os.makedirs(data_path, exist_ok=True)
        # MNIST
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomApply([transforms.RandomAffine(
                degrees=10.0,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                shear=5.0,
                resample=False
            )], p=0.5),
            transforms.ToTensor(),
            # dequantize
            transforms.Lambda(lambda im: ((255.0 * im + torch.rand_like(im)) / 256.0).clamp(1e-3, 1 - 1e-3)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=True, download=True,
                           transform=transform),
            batch_size=batch_size, shuffle=True, **kwargs)
        transform = torchvision.transforms.Compose([
            transforms.ToTensor(),
            # dequantize
            transforms.Lambda(lambda im: ((255.0 * im) / 256.0).clamp(1e-3, 1 - 1e-3)),
        ])
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, download=False,
                           transform=transform),
            batch_size=batch_size, shuffle=True, **kwargs)

        im_dim = 28
        ch_dim = 1
        x_dim = im_dim**2 * ch_dim

        binary_x = True
    elif dataset_name.startswith("pca-fashion-mnist"):
        base_path = os.sep.join(data_path.split(os.sep)[:-1])
        data_path = os.path.join(base_path, "fashion-mnist")
        os.makedirs(data_path, exist_ok=True)

        # PCA MNIST
        k = int(dataset_name[17:])

        train_loader = torch.utils.data.DataLoader(
            PCAFashionMNIST(k, data_path, train=True, download=True),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            PCAFashionMNIST(k, data_path, train=False),
            batch_size=batch_size, shuffle=True, **kwargs)

        im_dim = 1
        ch_dim = k
        x_dim = im_dim**2 * ch_dim

        binary_x = False

    elif dataset_name.startswith("toy"):
        # 2D toy example: toy4 - 4+1 (in center) GMM, toy-4 - 4 GMM (no center),
        # toy4_20 - 4+1 GMM with 20D x. toyMIM - GMM that spells MIM
        k_modes = dataset_name[3:]
        if "_" in k_modes:
            k_modes, ch_dim = k_modes.split("_")
            ch_dim = int(ch_dim)
        else:
            ch_dim = 2

        im_dim = 1
        x_dim = im_dim**2 * ch_dim

        P_x = GMMDist.factory(k=k_modes, d=x_dim)
        anchors["P_x"] = P_x

        train_loader = torch.utils.data.DataLoader(
            DistDataset(dist=P_x, with_index=True),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            DistDataset(dist=P_x, with_index=True),
            batch_size=batch_size, shuffle=True, **kwargs)

        binary_x = False
    else:
        raise ValueError("Unknown dataset = {dataset}".format(dataset=dataset_name))

    return (
        train_loader, test_loader,
        im_dim, ch_dim, x_dim, binary_x, anchors
    )


#=============================================================================#
# Testing and visualization
#=============================================================================#


def calc_dist_stats(x, z, q_x, q_z_given_x, p_z, p_x_given_z,
                    P_z=None, P_x=None,
                    x_recon=None, z_recon=None,
                    N=3000,
                    ):
    """
    Computes various statistics for encoder-decoder over joint sample (x,z)
    """
    log_prob_p_x_given_z = p_x_given_z.log_prob(x)
    log_prob_p_z = p_z.log_prob(z)
    log_prob_q_x = q_x.log_prob(x)
    log_prob_q_z_given_x = q_z_given_x.log_prob(z)

    # MI
    MI = (log_prob_q_z_given_x - log_prob_p_z).mean()

    # KSG MI estimator
    N = min(x.shape[0], N)
    MI_ksg = ksg_mi(x[:N].detach().cpu().numpy(), z[:N].detach().cpu().numpy())
    dx = x[0].numel()
    dz = z[0].numel()
    MI_ksg_l2_err = 1.0 / N**0.5 + N**(-1 / (dx + dz))

    # H
    H_q_x = -log_prob_q_x.mean()
    H_p_z = -log_prob_p_z.mean()

    H_enc = -(log_prob_q_x + log_prob_q_z_given_x).mean()
    H_dec = -(log_prob_p_z + log_prob_p_x_given_z).mean()

    H_Msamp = - (torch.logsumexp(torch.stack([
        (log_prob_q_x + log_prob_q_z_given_x) + torch.log(torch.tensor(0.5)),
        (log_prob_p_z + log_prob_p_x_given_z) + torch.log(torch.tensor(0.5)),
    ], dim=0),
        dim=0
    )).mean()

    H_mean = 0.5 * (H_enc + H_dec)

    Rtheta = H_mean - H_Msamp
    Rtheta_norm = (Rtheta / H_Msamp).abs()

    # Dkl
    z_q = q_z_given_x.sample()
    Dkl_q_p = (q_z_given_x.log_prob(z_q) - p_z.log_prob(z_q)).mean()
    z_p = p_z.sample()
    Dkl_p_q = (p_z.log_prob(z_p) - q_z_given_x.log_prob(z_p)).mean()

    Dkl = (Dkl_q_p + Dkl_p_q) / 2

    stats = dict(
        Dkl_q_p=Dkl_q_p.mean().detach().cpu().numpy(),
        Dkl_p_q=Dkl_p_q.mean().detach().cpu().numpy(),
        Dkl=Dkl.mean().detach().cpu().numpy(),
        MI=MI.mean().detach().cpu().numpy(),
        H_q_x=H_q_x.mean().detach().cpu().numpy(),
        H_p_z=H_p_z.mean().detach().cpu().numpy(),
        H_enc=H_enc.mean().detach().cpu().numpy(),
        H_dec=H_dec.mean().detach().cpu().numpy(),
        H_Msamp=H_Msamp.mean().detach().cpu().numpy(),
        H_mean=H_mean.mean().detach().cpu().numpy(),
        MI_ksg=MI_ksg,
        MI_ksg_l2_err=MI_ksg_l2_err,
        Rtheta=Rtheta.mean().detach().cpu().numpy(),
        Rtheta_norm=Rtheta_norm.mean().detach().cpu().numpy(),
    )

    if P_x is not None:
        stats["H_P_x"] = -P_x.log_prob(x).mean().detach().cpu().numpy()

    if P_z is not None:
        stats["H_P_z"] = -P_z.log_prob(z).mean().detach().cpu().numpy()

    if x_recon is not None:
        stats["x_recon_err"] = torch.norm(x - x_recon, dim=-1).mean().detach().cpu().numpy()

    if z_recon is not None:
        stats["z_recon_err"] = torch.norm(z - z_recon, dim=-1).mean().detach().cpu().numpy()

    return stats


def plot_dist_contour(dist, x, y, dim, levels, label=None, **kwargs):
    """
    Plots contour for a distribution
    """
    X, Y = torch.meshgrid(x, y)
    XY = torch.stack([X.flatten(), Y.flatten()])
    if dim > 2:
        XY = torch.cat([XY, torch.zeros((dim - 2, XY.shape[1]))])

    Z = dist.log_prob(XY.t()).view_as(X).exp()

    contour = plt.contour(
        X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy(),
        levels, **kwargs)
    if label:
        contour.collections[0].set_label(label)

    return contour


def plot_model_dist(
    x, z,
    q_x, q_z_given_x, p_z, p_x_given_z,
    N=200,
    P_x=None, P_z=None,
    x_recon=None, z_recon=None,
    x_name="x", z_name="z", title="",
    show_detailed=False,
    plot_x_recon_err=False,
    x_im_dim=None,
    stats=None,
):
    """
    Returns a plot of data.
    """
    if stats is None:
        stats = calc_dist_stats(
            x=x,
            z=z,
            q_x=q_x,
            q_z_given_x=q_z_given_x,
            p_z=p_z,
            p_x_given_z=p_x_given_z,
            P_x=P_x,
            P_z=P_z,
            x_recon=x_recon,
            z_recon=z_recon,
        )

    x_recon_err = stats.get("x_recon_err", None)
    contour_kargs = dict(
        colors='k',
        alpha=0.75,
    )
    if show_detailed:
        contour_kargs.update(dict(
            levels=10,
            linewidths=0.5,
        ))
    else:
        contour_kargs.update(dict(
            levels=5,
            linewidths=2.0,
        ))

    fig = plt.figure(dpi=300, figsize=(3, 6))

    #=============================================================================#
    # Visualize x
    #=============================================================================#
    ax = plt.subplot(2, 1, 1)
    legend_kwargs = {
        # "markerscale": 10.0,
        "markerscale": 3.0,
        "loc": "lower right",
    }

    if x_im_dim is None:
        # plot x as a distribution

        if P_x is not None:
            try:
                contour = plot_dist_contour(
                    dist=P_x,
                    x=torch.arange(-2, 2, 0.1),
                    y=torch.arange(-2, 2, 0.1),
                    dim=x.shape[-1],
                    label="P(x)",
                    **contour_kargs)
            except:
                pass

        if x_recon is not None:
            x_recon_coords = x_recon.detach().cpu().t().numpy()

            ax.plot(x_recon_coords[0][:N], x_recon_coords[1][:N], 'r.',
                    # ms=0.5, alpha=0.75, label=", ".join([x_name, z_name]))
                    ms=6.0, alpha=0.75, label=", ".join([x_name, z_name]))

            x_coords = x.detach().cpu().t().numpy()

            if plot_x_recon_err:
                data = []
                for i in range(N):
                    data.extend(
                        ((x_recon_coords[0][i], x_coords[0][i]), (x_recon_coords[1][i], x_coords[1][i]), 'r')
                    )

                ax.plot(*data, lw=0.5, alpha=0.75)

        ax.set_aspect("equal")
        ax.axis((-2, 2, -2, 2))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        # plot x as an image
        x_im_shape = (-1, ) + tuple(x_im_dim)

        n = int((N * 2)**0.5)

        x_im = x[:n**2 // 2].detach().cpu().view(x_im_shape)
        x_recon_im = x_recon[:n**2 // 2].detach().cpu().view(x_im_shape)

        comparison = torch.cat(list(itertools.chain(*zip(
            torch.split(x_im, n),
            torch.split(x_recon_im, n),
        )))).clamp(0.0, 1.0)

        recon_im = np.transpose(torchvision.utils.make_grid(
            comparison,
            nrow=n,
        ).numpy(), [1, 2, 0])
        plt.imshow(recon_im)
        plt.gca().axis("off")

    legend = [
        "MI(x;z) = {MI_ksg:.2e}",
        "H(q(x)) = {H_q_x:.2e}",
    ]
    if P_x is not None:
        legend.append("H(P(x)) = {H_P_x:.2e}")

    if x_recon_err is not None:
        legend.append("Recon. MSE = {x_recon_err:.2e}")

    legend_kwargs["title"] = "\n".join([l.format(**stats) for l in legend])
    ax.set_aspect("equal")
    if show_detailed:
        plt.legend(**legend_kwargs)
        ax.grid(linestyle=':')
        ax.set_title("observations")
    else:
        if x_recon_err is not None:
            ax.text(0.03, 0.03, "RMSE = {x_recon_err:.2g}".format(x_recon_err=x_recon_err),
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes,
                    color='black', fontsize=17, fontweight="bold")
            ax.text(0.03, 0.97, "MI = {MI_ksg:.2g}".format(MI_ksg=stats["MI_ksg"]),
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes,
                    color='black', fontsize=17, fontweight="bold")

    #=============================================================================#
    # Visualize z
    #=============================================================================#
    # plot z
    ax = plt.subplot(2, 1, 2)

    if P_z is not None:
        try:
            contour = plot_dist_contour(
                dist=P_z,
                x=torch.arange(-3, 3, 0.1),
                y=torch.arange(-3, 3, 0.1),
                dim=z.shape[-1],
                label="P(z)",
                **contour_kargs)
        except:
            pass

    q_z_given_x.plot(
        N=N,
        label="q(z|x)",
        edgecolor="g",
        facecolor='none',
        ls="-",
        # lw=0.5,
        lw=1.0,
        # alpha=0.5,
        alpha=1.0,
    )

    p_z.plot(
        N=1,
        label="p(z)",
        edgecolor="k",
        facecolor='none',
        ls="--",
        lw=5.0,
        alpha=1.0,
    )

    legend_kwargs = {
        "markerscale": 3.0,
        "loc": "lower right",
        "title": "Dkl = {Dkl:.2e}\nDkl(q||p) = {Dkl_q_p:.2e}\nDkl(p||q) = {Dkl_p_q:.2e}\n".format(
            **stats
        )
    }
    ax.set_aspect("equal")
    ax.axis((-3, 3, -3, 3))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if show_detailed:
        plt.legend(**legend_kwargs)
        # ax.autoscale()
        ax.grid(linestyle=':')
        ax.set_title("latent state")

        plt.suptitle(title + "  ({x_dim}D x, {z_dim}D z)".format(z_dim=z.shape[1], x_dim=x.shape[1]))

    plt.tight_layout()
    # plt.subplots_adjust(top=0.95, bottom=0.0, hspace=0.0)

    return fig


def vis_model_toy2d(model, P_x, P_z, device, results_path, tag, z_dim, x_dim, ch_dim, im_dim,
                    loss_name="", show_detailed=False, plot_x_recon_err=False,
                    stats=None):
    """
    Visualize model that is trained on toy 2D data.
    """
    #=============================================================================#
    # reconstruction
    #=============================================================================#
    vis_samples = 10000
    data = torch.cat([P_x.sample() for i in range(vis_samples)])
    x_obs = data.to(device).view(-1, x_dim)
    z, x_recon, q_x, q_z_given_x, p_z, p_x_given_z = model(x_obs)

    data_title = "loss = {loss}".format(loss=loss_name)
    fig = plot_model_dist(
        x=x_obs,
        z=z,
        q_x=q_x,
        q_z_given_x=q_z_given_x,
        p_z=p_z,
        p_x_given_z=p_x_given_z,
        N=200,
        P_x=P_x,
        # P_z=P_z,
        x_recon=x_recon,
        z_recon=None,
        x_name="x ~ p(x|z)",
        z_name="z ~ q(z|x)",
        title="{loss}".format(loss=loss_name),
        show_detailed=show_detailed,
        plot_x_recon_err=plot_x_recon_err,
        stats=stats,
    )

    fig.savefig(os.path.join(results_path, 'reconstruction_{tag}.png'.format(tag=tag)),)
    plt.close(fig)

    #=============================================================================#
    # random samples
    #=============================================================================#
    z, x_obs = model.sample(vis_samples)

    z_recon, x_recon_, q_x, q_z_given_x, p_z, p_x_given_z = model(z_obs=z)

    fig = plot_model_dist(
        x=x_obs,
        z=z,
        q_x=q_x,
        q_z_given_x=q_z_given_x,
        p_z=p_z,
        p_x_given_z=p_x_given_z,
        N=200,
        P_x=P_x,
        P_z=P_z,
        x_recon=x_obs,
        z_recon=z_recon,
        x_name="x ~ p(x|z)",
        z_name="z ~ p(z)",
        title="{loss}".format(loss=loss_name),
        show_detailed=show_detailed,
        plot_x_recon_err=False,
        stats=stats,
    )
    fig.savefig(os.path.join(results_path, 'grid_{tag}.png'.format(tag=tag)))
    plt.close(fig)


def vis_model_image(model, train_loader, test_loader, P_z, device, results_path, tag,
                    z_dim, x_dim, ch_dim, im_dim,
                    loss_name="",
                    show_detailed=False,
                    plot_x_recon_err=False,
                    stats=None,
                    ):
    """
    Visualize model that is trained on image data.
    """
    with torch.no_grad():
        #=============================================================================#
        # reconstruction
        #=============================================================================#
        vis_samples = 10000
        data, labels = list(zip(*[test_loader.dataset[i] for i in range(vis_samples)]))
        data = torch.stack(data)
        labels = torch.stack(labels)
        x_obs = data.to(device).view(-1, x_dim).clamp(1e-3, 1 - 1e-3)
        z, x_recon, q_x, q_z_given_x, p_z, p_x_given_z = model(x_obs)

        data_title = "loss = {loss}".format(loss=loss_name)
        fig = plot_model_dist(
            x=x_obs,
            z=z,
            q_x=q_x,
            q_z_given_x=q_z_given_x,
            p_z=p_z,
            p_x_given_z=p_x_given_z,
            N=50,
            P_x=None,
            P_z=P_z,
            x_recon=x_recon,
            z_recon=None,
            x_name="x ~ p(x|z)",
            z_name="z ~ q(z|x)",
            title="{loss}".format(loss=loss_name),
            show_detailed=show_detailed,
            plot_x_recon_err=False,
            x_im_dim=(ch_dim, im_dim, im_dim),
            stats=stats,
        )

        fig.savefig(os.path.join(results_path, 'reconstruction_{tag}.png'.format(tag=tag)),)
        plt.close(fig)

        #=============================================================================#
        # latent representation
        #=============================================================================#

        z = z.cpu()
        fig = plt.figure()
        ax = fig.gca()

        if z_dim == 1:
            i = j = 0
        else:
            i = 0
            j = 1

        cmap = plt.cm.tab10
        for l in range(int(labels.max().item())):
            I = labels == l
            c = cmap(l / 9)
            ax.scatter(z[I, i], z[I, j], c=c, label="'{l}'".format(l=int(l)), s=2)

        p_z.plot(
            N=1,
            label="p(z)",
            edgecolor="k",
            facecolor='none',
            ls="--",
            lw=5.0,
            alpha=1.0,
        )

        ax.grid()
        plt.tight_layout()

        fig.savefig(os.path.join(results_path, 'latent_{tag}.png'.format(tag=tag)))
        plt.close(fig)

        #=============================================================================#
        # interpolation
        #=============================================================================#
        steps = 10
        I = torch.linspace(0.0, 1.0, steps)
        I = I.view(-1, 1).to(device)

        z = z[0:4].detach().to(device)

        interp_z = (
            ((1.0 - I) * (1.0 - I).t()).view(steps, steps, 1) * z[0].view(1, 1, -1) +
            ((1.0 - I) * (I).t()).view(steps, steps, 1) * z[1].view(1, 1, -1) +
            ((I) * (1.0 - I).t()).view(steps, steps, 1) * z[2].view(1, 1, -1) +
            ((I) * (I).t()).view(steps, steps, 1) * z[3].view(1, 1, -1)
        ).reshape(*((-1, ) + z.shape[1:]))

        interp_x = model.p_x_given_z(interp_z).rsample().view(-1, ch_dim, im_dim, im_dim).clamp(0.0, 1.0)
        save_image(interp_x, os.path.join(results_path, 'interp_{tag}.png'.format(tag=tag)), nrow=steps)

        #=============================================================================#
        # random samples from prior p(z)
        #=============================================================================#
        sample_num = 100
        z_sample, x_sample = model.sample(sample_num)
        x_sample = x_sample.cpu().clamp(0.0, 1.0).view(-1, ch_dim, im_dim, im_dim)
        save_image(
            x_sample,
            os.path.join(results_path, 'sample_{tag}.png'.format(tag=tag)),
            nrow=int(sample_num**0.5),
        )

        #=============================================================================#
        # grid samples
        #=============================================================================#
        # first 2 dim
        N = 30
        Z = 3.0
        y = x = torch.linspace(-Z, Z, N)
        X, Y = torch.meshgrid(x, y)
        if z_dim == 1:
            z = torch.linspace(-Z, Z, N * N).view(N * N, 1)
        else:
            z = torch.zeros(N, N, z_dim)
            z[:, :, 0] = X
            z[:, :, 1] = Y
            z = z.view(N * N, z_dim)

        x = model.p_x_given_z(z.to(device)).rsample().detach().cpu().clamp(0.0, 1.0)
        im = torchvision.utils.make_grid(x.view((-1, ch_dim, im_dim, im_dim)), nrow=N, padding=0)
        im_np = np.transpose(im.cpu().detach().numpy(), [1, 2, 0])
        # mark standard deviation
        X, Y = torch.meshgrid(torch.linspace(-Z, Z, N * im_dim), torch.linspace(-Z, Z, N * im_dim))
        X = X.detach().cpu()
        Y = Y.detach().cpu()
        for r in range(1, int(Z) + 1):
            R = np.abs(X**2 + Y**2 - r**2) < 1 / N
            im_np[:, :, 1][np.where(R)] = 0.5 + 0.5 * im_np[:, :, 0][np.where(R)]

        plt.imsave(os.path.join(results_path, 'grid_{tag}.png'.format(tag=tag)), im_np)


def test_model(model, train_loader, test_loader, device, results_fname,
               z_dim, x_dim, ch_dim, im_dim,
               P_z=None, P_x=None, epoch=-1, loss=0.0, tag=""):
    """
    Tests latent representation of model.
    """
    print("Testing latent representation")
    #=============================================================================#
    # explore latent representation z
    #=============================================================================#
    # test classification over z

    # collect train/test data

    train_z = []
    train_label = []
    for data, label in iter(train_loader):
        train_label.extend(label.cpu().tolist())
        x_obs = data.to(device).view(-1, x_dim)
        z, x_recon, q_x, q_z_given_x, p_z, p_x_given_z = model(x_obs)

        train_z.append(z)

    test_x = []
    test_x_recon = []
    test_z = []
    test_label = []
    for data, label in iter(test_loader):
        test_label.extend(label.cpu().tolist())
        x_obs = data.to(device).view(-1, x_dim)
        z, x_recon, q_x, q_z_given_x, p_z, p_x_given_z = model(x_obs)

        test_x.append(x_obs)
        test_x_recon.append(x_recon)
        test_z.append(z)

    train_z = torch.cat(train_z)

    test_x = torch.cat(test_x)
    test_x_recon = torch.cat(test_x_recon)
    test_z = torch.cat(test_z)
    z, x, q_x, q_z_given_x, p_z, p_x_given_z = model(test_x)

    stats = calc_dist_stats(
        x=test_x,
        z=test_z,
        q_x=q_x,
        q_z_given_x=q_z_given_x,
        p_z=p_z,
        p_x_given_z=p_x_given_z,
        P_z=P_z,
        P_x=P_x,
        x_recon=test_x_recon,
        z_recon=None,
        N=10000,
    )

    stats["epoch"] = epoch
    stats["loss"] = loss

    # test KNN classification
    train_z = train_z.detach().cpu().numpy()
    test_z = test_z.detach().cpu().numpy()
    train_label = np.array(train_label).flatten()
    test_label = np.array(test_label).flatten()

    # train classifiers
    for clf_name, clf in [
        ("KNN1", KNeighborsClassifier(n_neighbors=1)),
        ("KNN3", KNeighborsClassifier(n_neighbors=3)),
        ("KNN5", KNeighborsClassifier(n_neighbors=5)),
        ("KNN10", KNeighborsClassifier(n_neighbors=10)),
    ]:
        print("Training {clf_name}".format(clf_name=clf_name))
        clf.fit(train_z, train_label)
        clf_score = clf.score(test_z, test_label)
        stats["clf_acc_" + clf_name] = clf_score

    print("=====\n{stats}\n=====\n".format(stats="\n".join(["{k} = {v}".format(k=k, v=v) for k, v in stats.items()])))

    # save stats
    with open(results_fname, 'w') as fh:
        writer = csv.writer(fh)
        for k, v in stats.items():
            writer.writerow([k, v])

    return stats


def visualize_model(model, dataset_name, train_loader, test_loader, anchors,
                    device, results_path, tag,
                    z_dim, x_dim, ch_dim, im_dim,
                    loss_name="",
                    show_detailed=False,
                    plot_x_recon_err=False,
                    stats=None,
                    ):
    with torch.no_grad():
        if dataset_name in ["mnist"]:
            vis_model_image(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                P_z=anchors["P_z"],
                device=device,
                results_path=results_path,
                tag=tag,
                z_dim=z_dim,
                x_dim=x_dim,
                ch_dim=ch_dim,
                im_dim=im_dim,
                show_detailed=show_detailed,
            )
        elif dataset_name.startswith("toy"):
            vis_model_toy2d(
                model=model,
                P_x=anchors["P_x"],
                P_z=anchors["P_z"],
                device=device,
                results_path=results_path,
                tag=tag,
                z_dim=z_dim,
                x_dim=x_dim,
                ch_dim=ch_dim,
                im_dim=im_dim,
                loss_name=loss_name,
                show_detailed=show_detailed,
                plot_x_recon_err=plot_x_recon_err,
            )


def load_test_vis_model(model_fname):
    _ = model_fname.split(os.sep)
    dataset_name, z_dim, model_name, seed_name = _[-4:]
    z_dim = int(z_dim[1:])
    seed = int(seed_name.split(".")[1])

    model = torch.load(model_fname)

    try:
        args = model.metadata["args"]
        batch_size = args.batch_size
        z_dim = args.z_dim
        cuda = not args.no_cuda and torch.cuda.is_available()

        loss_name = model.metadata["loss_name"]
        best_test_epoch = model.metadata["best_test_epoch"]
        best_test_loss = model.metadata["best_test_loss"]
    except:
        batch_size = 128
        cuda = torch.cuda.is_available()
        loss_name = model_name.split("-")[0].upper()
        best_test_epoch = -1
        best_test_loss = 0.0

    if cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    (
        train_loader, test_loader,
        im_dim, ch_dim, x_dim, binary_x, anchors
    ) = load_dataset(
        dataset_name=dataset_name,
        z_dim=z_dim,
        batch_size=batch_size,
        cuda=cuda,
    )

    device = torch.device("cuda" if cuda else "cpu")
    results_path = os.path.dirname(model_fname)
    results_fname = os.path.join(results_path, "results.{seed}.csv".format(seed=seed))

    model = model.to(device)

    with torch.no_grad():
        stats = test_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            results_fname=results_fname,
            z_dim=z_dim,
            x_dim=x_dim,
            ch_dim=ch_dim,
            im_dim=im_dim,
            P_z=anchors["P_z"],
            P_x=anchors["P_x"],
            epoch=best_test_epoch,
            loss=best_test_loss,
        )
        visualize_model(
            model=model,
            dataset_name=dataset_name,
            train_loader=train_loader,
            test_loader=test_loader,
            anchors=anchors,
            device=device,
            results_path=results_path,
            tag="best",
            z_dim=z_dim,
            x_dim=x_dim,
            ch_dim=ch_dim,
            im_dim=im_dim,
            loss_name=loss_name,
            show_detailed=False,
            plot_x_recon_err=False,
            stats=stats,
        )
