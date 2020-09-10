#!/usr/bin/env python
"""
Comparison (ablation study) of training a given VAE model with VAE learning and with MIM learning.

Code is based on https://github.com/pytorch/examples/tree/master/vae
"""

import argparse

import math
import os
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch import optim
import torchvision

import circlify as circ

import auxiliary as aux
from models import MIM

from cond_pixelcnn.utils.knnie import kraskov_mi as ksg_mi


#=============================================================================#
# Parse command line arguments
#=============================================================================#

parser = argparse.ArgumentParser(description='VAE with VI vs. MIM Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--warmup-steps', type=int, default=10,
                    help='warmup steps in NOAM scheduler (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--show-detailed', action='store_true', default=False,
                    help='Visualization includes detailed information.')
parser.add_argument('--plot-x-recon-err', action='store_true', default=False,
                    help='Visualization of x reconstruction error. Draws lines between sample and reconstruction for X')
parser.add_argument('--vis-progress', action='store_true', default=False,
                    help='Visualizes the convergence of model')

# experiment
parser.add_argument('--mim-loss', action='store_true', default=False,
                    help='MIM loss.')
parser.add_argument('--mim-samp', action='store_true', default=False,
                    help='MIM sampling (add an unsupervised sampling step).')
parser.add_argument('--inv-H-loss', action='store_true', default=False,
                    help='If used, will add H regularizer to VAE or remove it for MIM.')
parser.add_argument('--act', type=str, default="tanh",
                    help='Activation function: swish, tanh, relu. (default: tanh)')
parser.add_argument('--learn-P-z', action='store_true', default=False,
                    help='If used, will use prior as anchor P(z) = p(z)')
parser.add_argument('--p-z', type=str, default="anchor",
                    help='Defines p(z). prior: p(z) is parameterized. anchor: p(z) = P(z) = Normal. marginal: p(z) = E_{x ~ q(x)}[q(z|x)]. gmm-marginal: VampPrior. (default: anchor)')
parser.add_argument('--q-x', type=str, default="marginal",
                    help='Defines q(x). prior: q(x) is parameterized. anchor: q(x) = P(x). marginal: q(x) = E_{z ~ p(z)}[p(x|z)].  gmm-marginal: VampPrior. (default: marginal)')
parser.add_argument('--layers', type=int, default=2,
                    help='Number of linear layers (default: 2)')
parser.add_argument('--z-dim', type=int, default=2, metavar='Z',
                    help='dimensionality of latent state z (default: 2)')
parser.add_argument('--mid-dim', type=int, default=400, metavar='M',
                    help='mid layes dimensionality (default: 400)')
parser.add_argument('--min-logvar', type=float, default=10,
                    help='Minimal value for logvar (default: 10)')
parser.add_argument('--q-x-gmm', type=int, default=0,
                    help='Use GMM with N components if N > 0, Normal distribution if N = 0. (default: 0)')
parser.add_argument('--q-zx-gmm', type=int, default=0,
                    help='Use GMM with N components if N > 0, Normal distribution if N = 0. (default: 0)')
parser.add_argument('--p-z-gmm', type=int, default=0,
                    help='Use GMM with N components if N > 0, Normal distribution if N = 0. (default: 0)')
parser.add_argument('--p-xz-gmm', type=int, default=0,
                    help='Use GMM with N components if N > 0, Normal distribution if N = 0. (default: 0)')
parser.add_argument('--dataset', type=str, default="toyMIM",
                    help='dataset to use (N=int,S=string): mnist. fashion-mnist, pca-fashion-mnist<N>, fashion-mnist, toy<N|S>. (default: toyMIM)')
parser.add_argument('--tag', type=str, default="",
                    help='A tag to add to results path. (default: ""')
args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)

if args.seed > 0:
    torch.manual_seed(args.seed)


if args.q_x not in ["prior", "anchor", "marginal", "gmm-marginal"]:
    raise ValueError("Unknown q-x = {q_x}".format(q_x=args.q_x))

if args.p_z not in ["prior", "anchor", "marginal", "gmm-marginal"]:
    raise ValueError("Unknown p-z = {p_z}".format(p_z=args.p_z))

self_supervision = True if args.dataset.startswith("T") else False
args.self_supervision = self_supervision


device = torch.device("cuda" if args.cuda else "cpu")

base_path = os.path.join(os.path.dirname(__file__), "../data")

results_path = os.path.join(base_path, "torch-generated/",
                            os.path.splitext(os.path.basename(__file__))[0],
                            args.dataset,
                            "z{z_dim}".format(z_dim=args.z_dim),
                            )
if args.mim_loss:
    results_path = os.path.join(results_path, "mim")
    if args.mim_samp:
        results_path = "{results_path}-samp".format(results_path=results_path)
else:
    results_path = os.path.join(results_path, "vae")

results_path = "{results_path}_logvar{min_logvar:g}_mid-dim{mid_dim}_layers{layers}_q-x{q_x_gmm}{q_x}_q-zx{q_zx_gmm}_p-z{p_z_gmm}{p_z}_p-xz{p_xz_gmm}".format(
    results_path=results_path,
    min_logvar=args.min_logvar,
    mid_dim=args.mid_dim,
    layers=args.layers,
    q_x=args.q_x,
    q_x_gmm=args.q_x_gmm,
    q_zx_gmm=args.q_zx_gmm,
    p_z=args.p_z,
    p_z_gmm=args.p_z_gmm,
    p_xz_gmm=args.p_xz_gmm,
)


if args.inv_H_loss:
    results_path = "{results_path}-inv_H".format(results_path=results_path)

if args.learn_P_z:
    results_path = "{results_path}-learn_P_z".format(results_path=results_path)

if args.self_supervision:
    results_path = "{results_path}-self".format(results_path=results_path)


if args.tag:
    results_path = "{results_path}-{tag}".format(results_path=results_path, tag=args.tag)

progress_results_path = os.path.join(results_path, "progress")
os.makedirs(results_path, exist_ok=True)
os.makedirs(progress_results_path, exist_ok=True)

# store results per seed to support multiple runs
results_fname = os.path.join(results_path, "results.{seed}.csv".format(seed=args.seed))
model_fname = os.path.join(results_path, "best.{seed}.model".format(seed=args.seed))
best_test_loss = torch.tensor(float("inf"))
best_test_epoch = -1

print("\nresults_path = {results_path}".format(results_path=results_path))

#=============================================================================#
# Build dataset and loss functions
#=============================================================================#
os.makedirs(os.path.join(base_path, "assets/datasets"), exist_ok=True)
data_path = os.path.join(base_path, "assets/datasets", args.dataset)

z_dim = args.z_dim
mid_dim = args.mid_dim
min_scale = 10**(-args.min_logvar)

if args.act == "swish":
    act = aux.Swish()
elif args.act == "tanh":
    act = torch.nn.Tanh()
elif args.act == "relu":
    act = torch.nn.ReLU()
else:
    raise ValueError("Unknown act = {act}".format(act=act))

# load datasets
(
    train_loader, test_loader,
    im_dim, ch_dim, x_dim, binary_x, anchors
) = aux.load_dataset(
    dataset_name=args.dataset,
    z_dim=args.z_dim,
    batch_size=args.batch_size,
    cuda=args.cuda,
)
P_z = anchors["P_z"]

# x conditional likelihood and prior
if binary_x:
    p_x_given_z = aux.build_layers(
        input_dim=z_dim,
        mid_dim=mid_dim,
        output_dim=x_dim,
        layers=args.layers,
        output_layer_gen=lambda i, o: aux.BernoulliCondLayer(
            input_dim=i,
            output_dim=o,
            min_scale=min_scale,
        ),
        act=act,
    )

    if args.q_x == "anchor":
        raise NotImplementedError("Not implemented yet")
    elif args.q_x == "prior":
        q_x = aux.BernoulliPriorLayer(
            output_dim=x_dim,
            min_scale=min_scale,
        )
    elif args.q_x == "marginal":
        q_x = None
    elif args.q_x == "gmm-marginal":
        q_x = aux.GMMPseudoPriorLayer(
            cond_comp=p_x_given_z,
            input_dim=z_dim,
            output_dim=x_dim,
            components_num=args.q_x_gmm,
            min_scale=min_scale,
            pseudo_input=torch.cat([P_z.sample().to(device) for i in range(args.q_x_gmm)]),
        )
    else:
        raise NotImplementedError("Not implemented yet")
else:
    if args.p_xz_gmm <= 0:
        p_x_given_z = aux.build_layers(
            input_dim=z_dim,
            mid_dim=mid_dim,
            output_dim=x_dim,
            layers=args.layers,
            output_layer_gen=lambda i, o: aux.GaussianCondLayer(
                input_dim=i,
                output_dim=o,
                min_scale=min_scale,
            ),
            act=act,
        )
    else:
        p_x_given_z = aux.build_layers(
            input_dim=z_dim,
            mid_dim=mid_dim,
            output_dim=x_dim,
            layers=args.layers,
            output_layer_gen=lambda i, o: aux.GMMCondLayer(
                input_dim=i,
                output_dim=o,
                components_num=args.p_xz_gmm,
                min_scale=min_scale,
            ),
            act=act,
        )

    if args.q_x == "anchor":
        q_x = aux.GMMPriorLayer(
            output_dim=x_dim,
            components_num=P_x.k,
            min_scale=min_scale,
            active=False,
            logcoefs=P_x.logcoefs,
            locs=[c.loc for c in P_x.comps],
            logscales=[c.scale.log() for c in P_x.comps],
        )
    elif args.q_x == "prior":
        if args.q_x_gmm <= 0:
            q_x = aux.GaussianPriorLayer(
                output_dim=x_dim,
                min_scale=min_scale,
                active=True,
            )
        else:
            q_x = aux.GMMPriorLayer(
                output_dim=x_dim,
                components_num=args.q_x_gmm,
                min_scale=min_scale,
                active=True,
            )
    elif args.q_x == "marginal":
        q_x = None
    elif args.q_x == "gmm-marginal":
        q_x = aux.GMMPseudoPriorLayer(
            cond_comp=p_x_given_z,
            input_dim=z_dim,
            output_dim=x_dim,
            components_num=args.q_x_gmm,
            min_scale=min_scale,
            pseudo_input=torch.cat([P_z.sample().to(device) for i in range(args.q_x_gmm)]),
        )
    else:
        raise NotImplementedError("Not implemented yet")

# z conditional likelihood
if args.q_zx_gmm <= 0:
    q_z_given_x = aux.build_layers(
        input_dim=x_dim,
        mid_dim=mid_dim,
        output_dim=z_dim,
        layers=args.layers,
        output_layer_gen=lambda i, o: aux.GaussianCondLayer(
            input_dim=i,
            output_dim=o,
            min_scale=min_scale,
        ),
        act=act,
    )
else:
    q_z_given_x = aux.build_layers(
        input_dim=x_dim,
        mid_dim=mid_dim,
        output_dim=z_dim,
        layers=args.layers,
        output_layer_gen=lambda i, o: aux.GMMCondLayer(
            input_dim=i,
            output_dim=o,
            components_num=args.q_zx_gmm,
            min_scale=min_scale,
        ),
        act=act,
    )

# z prior
if args.p_z == "anchor":
    p_z = aux.GaussianPriorLayer(
        output_dim=z_dim,
        min_scale=min_scale,
        active=False,
    )
elif args.p_z == "prior":
    if args.p_z_gmm <= 0:
        p_z = aux.GaussianPriorLayer(
            output_dim=z_dim,
            min_scale=min_scale,
            active=True,
        )
    else:
        # equally distributed circles over 2D plane
        circles = circ.circlify([1] * args.p_z_gmm)

        if z_dim == 1:
            locs = [torch.tensor([c.x]).view(1, -1) for c in circles]
        else:
            locs = [torch.tensor([c.x, c.y] + [0.0] * (z_dim - 2)).view(1, -1)
                    for c in circles]

        p_z = aux.GMMPriorLayer(
            output_dim=z_dim,
            components_num=args.p_z_gmm,
            min_scale=min_scale,
            active=True,
            logcoefs=(torch.ones(1, args.p_z_gmm) / args.p_z_gmm).log(),
            locs=locs,
            logscales=[torch.tensor([c.r] * z_dim).view(1, -1).log() for c in circles],
        )
elif args.p_z == "marginal":
    p_z = None
elif args.p_z == "gmm-marginal":
    p_z = aux.GMMPseudoPriorLayer(
        cond_comp=q_z_given_x,
        input_dim=x_dim,
        output_dim=z_dim,
        components_num=args.p_z_gmm,
        min_scale=min_scale,
        pseudo_input=torch.cat([
            train_loader.dataset[i][0].view(-1, x_dim).to(device)
            for i in range(args.p_z_gmm)
        ]),
    )


# build model from distributions
model = MIM(
    p_z=p_z,
    p_x_given_z=p_x_given_z,
    q_x=q_x,
    q_z_given_x=q_z_given_x,
).to(device)

if not hasattr(model, "metadata"):
    model.metadata = {}

model.metadata["args"] = args


#=============================================================================#
# Auxiliary loss functions
#=============================================================================#


def loss_name():
    if args.mim_loss:
        if args.mim_samp:
            name = "MIM"
        else:
            name = "MIM + VAE Sampling"

        if args.inv_H_loss:
            name = "{name} - R_H".format(name=name)

    else:
        name = "VAE"
        if args.inv_H_loss:
            name = "{name} + R_H".format(name=name)

    return name


def loss_function_vae(x_obs, z_obs, q_x, q_z_given_x, p_z, p_x_given_z,
                      samp_p=False, binary_x=False, beta=1.0):
    # vae loss
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # cross entropy (binary or continuous)
    CE = -p_x_given_z.log_prob(x_obs)

    # KL divergence
    # KLD = torch.distributions.kl.kl_divergence(q_z_given_x, p_z).sum(-1)
    KLD = q_z_given_x.log_prob(z_obs) - p_z.log_prob(z_obs)

    loss = (CE + beta * KLD).mean()

    return loss


def loss_function_vae_with_H(x_obs, z_obs, q_x, q_z_given_x, p_z, p_x_given_z,
                             samp_p=False, binary_x=False, beta=1.0):
    # vae loss
    loss = loss_function_vae(
        x_obs=x_obs,
        z_obs=z_obs,
        q_x=q_x,
        q_z_given_x=q_z_given_x,
        p_z=p_z,
        p_x_given_z=p_x_given_z,
        samp_p=samp_p,
        binary_x=binary_x,
        beta=beta,
    )

    # add entropy regularizer
    log_prob_x = q_x.log_prob(x_obs)
    log_prob_z_given_x = q_z_given_x.log_prob(z_obs)
    loss += -(log_prob_z_given_x + log_prob_x).mean()

    return loss


def loss_function_mim(x_obs, z_obs, q_x, q_z_given_x, p_z, p_x_given_z,
                      samp_p=False, binary_x=False, beta=1.0):

    # MIM loss
    log_prob_x_given_z = p_x_given_z.log_prob(x_obs)
    log_prob_x = q_x.log_prob(x_obs)

    log_prob_z = p_z.log_prob(z_obs)
    log_prob_z_given_x = q_z_given_x.log_prob(z_obs)

    loss = -0.5 * (log_prob_x_given_z + log_prob_z + beta * (log_prob_z_given_x + log_prob_x))

    # REINFORCE
    if binary_x and samp_p:
        loss = loss + loss.detach() * log_prob_x_given_z - (loss * log_prob_x_given_z).detach()

    loss = loss.mean()

    return loss


def loss_function_mim_without_H(x_obs, z_obs, q_x, q_z_given_x, p_z, p_x_given_z,
                                samp_p=False, binary_x=False, beta=1.0):
    # MIM loss

    log_prob_x_given_z = p_x_given_z.log_prob(x_obs)

    log_prob_z = p_z.log_prob(z_obs)
    log_prob_z_given_x = q_z_given_x.log_prob(z_obs)

    log_prob_x = q_x.log_prob(x_obs)

    loss = -0.5 * (log_prob_x_given_z + log_prob_z + beta * (log_prob_z_given_x + log_prob_x))

    # subtract entropy regularizer
    entropy_reg = - (torch.logsumexp(torch.stack([
        (log_prob_x_given_z + log_prob_z) + torch.log(torch.tensor(0.5)),
        (log_prob_z_given_x + log_prob_x) + torch.log(torch.tensor(0.5)),
    ], dim=0),
        dim=0
    ))
    loss -= entropy_reg

    # REINFORCE
    if binary_x and samp_p:
        loss = loss + loss.detach() * log_prob_x_given_z - (loss * log_prob_x_given_z).detach()

    loss = loss.mean()

    return loss


#=============================================================================#
# Build  train/test functions
#=============================================================================#

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = aux.NoamLRScheduler(optimizer, warmup_steps=args.warmup_steps)
scheduler.step()

# select loss
if args.mim_loss:
    if args.inv_H_loss:
        loss_function = loss_function_mim_without_H
    else:
        loss_function = loss_function_mim
else:
    if args.inv_H_loss:
        loss_function = loss_function_vae_with_H
    else:
        loss_function = loss_function_vae


def train(epoch):
    if args.learn_P_z:
        anchors["P_z"] = model.p_z(1)

    scheduler.step()
    beta = min(1.0, max(epoch / args.warmup_steps, 0.0))

    model.train()
    train_loss = 0
    train_loss_count = 0
    cur_train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        train_loss_count += 1
        optimizer.zero_grad()

        if not self_supervision:
            x_obs = data.to(device).view(-1, x_dim)
            x_in = x_obs
        else:
            x_obs = data[1].to(device).view(-1, x_dim)
            x_in = data[0].to(device).view(-1, x_dim)

        if binary_x:
            x_obs = x_obs.clamp(1e-3, 1 - 1e-3)

        z_obs, x_recon, q_x, q_z_given_x, p_z, p_x_given_z = model(x_obs=x_in)

        loss = loss_function(
            x_obs=x_obs,
            z_obs=z_obs,
            q_x=q_x,
            q_z_given_x=q_z_given_x,
            p_z=p_z,
            p_x_given_z=p_x_given_z,
            samp_p=False,
            binary_x=binary_x,
            beta=beta,
        )
        if not torch.isfinite(loss).all():
            raise ValueError("NaN/Inf was detected")

        loss.backward()
        train_loss += loss.cpu().detach().item()
        cur_train_loss += loss.cpu().detach().item()

        if args.mim_loss and args.mim_samp and (beta >= 1.0):
            P_z = anchors["P_z"]

            # generate fair sample
            z_obs = torch.cat([P_z.sample().to(device) for i in range(args.batch_size)])
            z_recon, x_obs, q_x, q_z_given_x, p_z, p_x_given_z = model(z_obs=z_obs)

            if binary_x:
                # binary samples should have no gradient
                x_obs = x_obs.detach()

            loss = loss_function(
                x_obs=x_obs,
                z_obs=z_obs,
                q_x=q_x,
                q_z_given_x=q_z_given_x,
                p_z=p_z,
                p_x_given_z=p_x_given_z,
                samp_p=True,
                binary_x=binary_x,
                beta=beta,
            )

            if not torch.isfinite(loss).all():
                raise ValueError("NaN/Inf was detected")

            (beta * loss).backward()

            train_loss += loss.cpu().detach().item()
            cur_train_loss += loss.cpu().detach().item()

        if (batch_idx + 1) % args.log_interval == 0:
            cur_train_loss /= args.log_interval
            if args.mim_loss and args.mim_samp:
                cur_train_loss /= 2

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x_recon), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                cur_train_loss))

            cur_train_loss = 0

        optimizer.step()

        if args.learn_P_z:
            anchors["P_z"] = model.p_z(1)

    train_loss /= train_loss_count
    if args.mim_loss and args.mim_samp:
        train_loss /= 2

    print('====> Epoch: {} beta: {} lr: {:.4g} Average loss: {:.4g}'.format(
        epoch, beta, optimizer.param_groups[0]["lr"], train_loss
    ))

    return train_loss


def test(epoch):
    if args.learn_P_z:
        anchors["P_z"] = model.p_z(1)

    model.eval()
    test_loss = 0
    test_loss_count = 0
    nllbpd_x = 0
    nllbpd_z = 0

    all_x = []
    all_z = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            test_loss_count += 1

            if not self_supervision:
                x_obs = data.to(device).view(-1, x_dim)
            else:
                x_obs = data[1].to(device).view(-1, x_dim)

            if binary_x:
                x_obs = x_obs.clamp(1e-3, 1 - 1e-3)

            z_obs, x_recon, q_x, q_z_given_x, p_z, p_x_given_z = model(x_obs=x_obs)

            all_x.append(x_obs)
            all_z.append(z_obs)

            test_loss += loss_function(
                x_obs=x_obs,
                z_obs=z_obs,
                q_x=q_x,
                q_z_given_x=q_z_given_x,
                p_z=p_z,
                p_x_given_z=p_x_given_z,
                samp_p=False,
                binary_x=binary_x,
                beta=1.0,
            )
            # negative log-likelihood (bits per dimension)
            nll_x_given_z = -p_x_given_z.log_prob(x_obs).mean()
            nll_z = -p_z.log_prob(z_obs).mean()

            nllbpd_x += nll_x_given_z.item()
            nllbpd_z += nll_z.item()

            if args.mim_loss and args.mim_samp:
                P_z = anchors["P_z"]

                # generate fair sample
                z_obs = torch.cat([P_z.sample().to(device) for i in range(args.batch_size)])
                z_recon, x_obs, q_x, q_z_given_x, p_z, p_x_given_z = model(z_obs=z_obs)

                if binary_x:
                    # binary samples should have no gradient
                    x_obs = x_obs.detach()

                loss = loss_function(
                    x_obs=x_obs,
                    z_obs=z_obs,
                    q_x=q_x,
                    q_z_given_x=q_z_given_x,
                    p_z=p_z,
                    p_x_given_z=p_x_given_z,
                    samp_p=True,
                    binary_x=binary_x,
                    beta=1.0,
                )

                test_loss += loss.cpu().detach().item()

    all_x = torch.cat(all_x, dim=0)
    all_z = torch.cat(all_z, dim=0)
    N = min(1000, all_z.shape[0])
    MI = ksg_mi(all_x.detach().cpu().numpy(), all_z.detach().cpu().numpy())
    dx = all_x.shape[1]
    dz = all_z.shape[1]
    MI_l2_err = 1.0 / N**0.5 + N**(-1 / (dx + dz))

    test_loss /= test_loss_count
    if args.mim_loss and args.mim_samp:
        test_loss /= 2

    print('====> Test set loss: {test_loss:.4g} nllbpd_x: {nllbpd_x:.4g} nllbpd_z: {nllbpd_z:.4g} MI: {MI:.4g} MI_l2_err: {MI_l2_err:.4g}'.format(
        test_loss=test_loss,
        nllbpd_x=nllbpd_x,
        nllbpd_z=nllbpd_z,
        MI=MI,
        MI_l2_err=MI_l2_err,
    ))

    return test_loss

#=============================================================================#
# Main script
#=============================================================================#
if __name__ == "__main__":
    print(args)

    print("\n\npress CTRL-C for early stopping...\n")

    model.metadata["loss_name"] = loss_name()

    epoch = 1
    epoch0 = epoch
    epoch1 = args.epochs + epoch

    try:
        for epoch in range(epoch0, epoch1):
            cur_train_loss = train(epoch)

            cur_test_loss = test(epoch)
            import pudb; pudb.set_trace()
            if (cur_test_loss < best_test_loss) and (epoch >= args.warmup_steps):
                print("===> Saving best model <===")
                best_test_loss = cur_test_loss
                best_test_epoch = epoch

                model.metadata["best_test_epoch"] = best_test_epoch
                model.metadata["best_test_loss"] = best_test_loss
                model.metadata["best_train_loss"] = cur_train_loss
                torch.save(model, model_fname)

            if args.vis_progress:
                aux.visualize_model(
                    model=model,
                    dataset_name=args.dataset,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    anchors=anchors,
                    device=device,
                    results_path=progress_results_path,
                    tag="{epoch:03d}".format(epoch=epoch),
                    z_dim=args.z_dim,
                    x_dim=x_dim,
                    ch_dim=ch_dim,
                    im_dim=im_dim,
                    loss_name=loss_name(),
                    show_detailed=args.show_detailed,
                    plot_x_recon_err=args.plot_x_recon_err,
                    self_supervision=self_supervision,
                )

            # early stopping
            if ((epoch - best_test_epoch) > args.warmup_steps) and (epoch >= args.warmup_steps):
                print("*****  Early stopping after steps = {steps} *****".format(
                    steps=args.warmup_steps))
                break
            else:
                if (epoch >= args.warmup_steps):
                    early_stop_steps = min(args.warmup_steps - (epoch - best_test_epoch), args.warmup_steps)
                else:
                    early_stop_steps = args.warmup_steps

                print("====> Best loss = {loss:.4g} epoch = {epoch} early stopping in steps = {steps}".format(
                    loss=best_test_loss,
                    epoch=best_test_epoch,
                    steps=early_stop_steps,
                ))
    except KeyboardInterrupt as e:
        pass

    # load best model
    print("Loading best model epoch = {epoch}".format(epoch=best_test_epoch))
    try:
        model = torch.load(model_fname)
    except:
        print("ERROR: Failed loading best model, using existing model instead")

    with torch.no_grad():
        stats = aux.test_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            results_fname=results_fname,
            z_dim=args.z_dim,
            x_dim=x_dim,
            ch_dim=ch_dim,
            im_dim=im_dim,
            P_z=anchors["P_z"],
            P_x=anchors["P_x"],
            epoch=best_test_epoch,
            loss=best_test_loss.cpu().detach().numpy(),
        )
        aux.visualize_model(
            model=model,
            dataset_name=args.dataset,
            train_loader=train_loader,
            test_loader=test_loader,
            anchors=anchors,
            device=device,
            results_path=results_path,
            tag="best",
            z_dim=args.z_dim,
            x_dim=x_dim,
            ch_dim=ch_dim,
            im_dim=im_dim,
            loss_name=loss_name(),
            show_detailed=args.show_detailed,
            plot_x_recon_err=args.plot_x_recon_err,
            self_supervision=self_supervision,
            stats=stats,
        )
