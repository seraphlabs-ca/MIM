import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

from cond_pixelcnn.utils.load_data import load_dataset


def plot_images(args, x_sample, size_x=3, size_y=3):

    fig = plt.figure(figsize=(size_y, size_x))
    # fig = plt.figure(1)
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x_sample[:size_x * size_y]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample = sample.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        if args.input_type == 'binary' or args.input_type == 'gray':
            sample = sample[:, :, 0]
            plt.imshow(sample, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(sample, vmin=0, vmax=1)

    return fig


aaa

model = torch.load(
    # "../../data/vector/snapshots/2019-08-07 15:50:58_dynamic_mnist_pixelhvae_2level_vampprior(K_500)_wu(100)_z1_40_z2_40/pixelhvae_2level.model",
    "../../data/vector/snapshots/2019-08-07 15:51:19_dynamic_mnist_pixelhvae_2level-mim_vampprior(K_500)_wu(100)_z1_40_z2_40/pixelhvae_2level-mim.model",
    map_location="cpu",
)
model.args.cuda = False
print("sampling...")
x_sample = model.generate_x(100)
plot_images(
    model.args,
    x_sample=x_sample.detach().cpu().numpy(),
    size_x=10,
    size_y=10,
)

#=============================================================================#
#
#=============================================================================#

from utils.evaluation import visualize_vae as visualize

print("MIM")
model = torch.load(
    # "../../data/torch-generated/snapshots/2019-08-07 19:54:26_cifar10_pixelhvae_2level-mim_vampprior(K_500)_wu(100)_z1_200_z2_200/pixelhvae_2level-mim.model"
    "../../data/torch-generated/snapshots/2019-08-07 18:49:26_dynamic_fashion_mnist_pixelhvae_2level-mim_vampprior(K_500)_wu(100)_z1_40_z2_40/pixelhvae_2level-mim.model"
)

os.makedirs("snapshots/debug/mim/", exist_ok=True)
visualize(
    args=model.args,
    model=model,
    dir="snapshots/debug/mim/",
    N=10,
)

print("VAE")
model = torch.load(
    # "../../data/torch-generated/snapshots/2019-08-07 19:53:46_cifar10_pixelhvae_2level_vampprior(K_500)_wu(100)_z1_200_z2_200/pixelhvae_2level.model"
    "../../data/torch-generated/snapshots/2019-08-07 18:50:06_dynamic_fashion_mnist_pixelhvae_2level_vampprior(K_500)_wu(100)_z1_40_z2_40/pixelhvae_2level.model"
)

os.makedirs("snapshots/debug/vae/", exist_ok=True)
visualize(
    args=model.args,
    model=model,
    dir="snapshots/debug/vae/",
    N=10,
)

#=============================================================================#
# Latent representation
#=============================================================================#

model = torch.load(
    # MNIST
    # "../data/vector/vae-as-mim-image/2019-08-09_07-07-28_dynamic_mnist_pixelhvae_2level-mim_vampprior__K_100__wu_100__z1_40_z2_40/pixelhvae_2level-mim.model",
    # "../data/vector/vae-as-mim-image/2019-08-09_07-07-31_dynamic_mnist_pixelhvae_2level_vampprior__K_100__wu_100__z1_40_z2_40/pixelhvae_2level.model",
    # Omniglot
    # "../data/vector/vae-as-mim-image/2019-08-09_02-24-23_omniglot_pixelhvae_2level-mim_vampprior__K_1000__wu_100__z1_40_z2_40/pixelhvae_2level-mim.model",
    # "../data/vector/vae-as-mim-image/2019-08-09_02-34-22_omniglot_pixelhvae_2level_vampprior__K_1000__wu_100__z1_40_z2_40/pixelhvae_2level.model",
    # Fashion-MNIST
    # "../data/vector/vae-as-mim-image/2019-08-08_20-41-01_dynamic_fashion_mnist_pixelhvae_2level-mim_vampprior__K_500__wu_100__z1_40_z2_40/pixelhvae_2level-mim.model",
    # "../data/vector/vae-as-mim-image/2019-08-08_20-42-01_dynamic_fashion_mnist_pixelhvae_2level_vampprior__K_500__wu_100__z1_40_z2_40/pixelhvae_2level.model",
    # Caltech 101
    # "../data/vector/vae-as-mim-image/2019-08-08_17-23-09_caltech101silhouettes_pixelhvae_2level-mim_vampprior__K_500__wu_100__z1_40_z2_40/pixelhvae_2level-mim.model",
    # "../data/vector/vae-as-mim-image/2019-08-08_17-23-39_caltech101silhouettes_pixelhvae_2level_vampprior__K_500__wu_100__z1_40_z2_40/pixelhvae_2level.model",
    # "../data/vector/vae-as-mim-image/2019-08-09_20-20-59_caltech101silhouettes_pixelhvae_2level-mim_vampprior__K_1000__wu_100__z1_100_z2_100/pixelhvae_2level-mim.model",
    # "../data/vector/vae-as-mim-image/2019-08-09_20-21-24_caltech101silhouettes_pixelhvae_2level_vampprior__K_1000__wu_100__z1_100_z2_40/pixelhvae_2level.model",
    # q(x) marginal
    # "../data/vector/vae-as-mim-image/2019-08-12_18-23-04_caltech101silhouettes_pixelhvae_2level-mim_vampprior__K_500__wu_100__z1_40_z2_40/pixelhvae_2level-mim.model",
    # "../data/vector/vae-as-mim-image/2019-08-12_18-02-44_histopathologyGray_pixelhvae_2level-mim_vampprior__K_500__wu_100__z1_40_z2_40/pixelhvae_2level-mim.model",
    # "../data/vector/vae-as-mim-image/2019-08-12_18-36-36_omniglot_pixelhvae_2level-mim_vampprior__K_500__wu_100__z1_40_z2_40/pixelhvae_2level-mim.model",
    # "../data/vector/vae-as-mim-image/2019-08-12_20-25-37_dynamic_fashion_mnist_pixelhvae_2level-mim_vampprior__K_500__wu_100__z1_40_z2_40/pixelhvae_2level-mim.model",
    # Fashion MNIST
    "../data/vector/vae-as-mim-image/2019-08-15_12-46-30_dynamic_fashion_mnist_pixelhvae_2level-mim_vampprior__K_500__wu_100__z1_20_z2_20/pixelhvae_2level-mim.model",
    # "../data/vector/vae-as-mim-image/2019-08-15_12-47-19_dynamic_fashion_mnist_pixelhvae_2level_vampprior__K_500__wu_100__z1_10_z2_10/pixelhvae_2level.model",
    # "../data/vector/vae-as-mim-image/2019-08-15_12-47-19_dynamic_fashion_mnist_pixelhvae_2level-mim_vampprior__K_500__wu_100__z1_10_z2_10/pixelhvae_2level-mim.model",
    # "../data/vector/vae-as-mim-image/2019-08-15_12-47-19_dynamic_fashion_mnist_pixelhvae_2level_vampprior__K_500__wu_100__z1_20_z2_20/pixelhvae_2level.model",
    # map_location='cpu')
)

model_name = model.args.model_name + "_" + model.args.dataset_name
dir = "../data/torch-generated/z_pert/"
os.makedirs(dir, exist_ok=True)

train_loader, val_loader, test_loader, args = load_dataset(model.args)

#=============================================================================#
# pseudo inputs compositionality
#=============================================================================#

train_loader, val_loader, test_loader, args = load_dataset(model.args)
x, y = next(iter(test_loader))

model.args.cuda = False
x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = model(x)
log_p_z2_a = model.log_p_z2(z2_q, return_a=True)
V, I = torch.topk(log_p_z2_a, 10, dim=1)
pseudoinputs = model.means(model.idle_input)

N = 10
all_x = []
for i in range(N):
    print(i)
    cur_x = x[i:(i + 1)]
    cur_pseudoinputs = pseudoinputs[I[i]]
    p = (V[i] - torch.logsumexp(V[i], dim=-1, keepdim=True)).exp().view((-1, 1))
    x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = model(
        cur_pseudoinputs)
    recon_z1 = (p * z1_q).sum(0, True)
    recon_z2 = (p * z2_q).sum(0, True)

    recon_pseudoinputs = (p * cur_pseudoinputs).sum(0, True)
    recon_x = (p * x_mean).sum(0, True)
    # samp_x = model.pixelcnn_generate(recon_z1, recon_z2).view((1, -1))
    # recon_x_mean, _ = model.p_x(recon_x.view(
    #     (-1, model.args.input_size[0], model.args.input_size[1], model.args.input_size[2])), recon_z1, recon_z2)

    # joint_x = torch.cat([cur_x, samp_x, recon_x, recon_pseudoinputs, cur_pseudoinputs])

    for j in range(cur_pseudoinputs.shape[0]):
        cur_pseudoinputs[j] = (1 - p[j]) + p[j] * cur_pseudoinputs[j]

    joint_x = torch.cat([cur_x, recon_x, recon_pseudoinputs, cur_pseudoinputs])
    all_x.append(joint_x)

all_x = torch.cat(all_x)
plot_images(model.args, all_x.detach().cpu().numpy(), size_x=N, size_y=joint_x.shape[0])

#=============================================================================#
# latent representation
#=============================================================================#
torch.set_default_tensor_type("torch.cuda.FloatTensor")
model = model.cuda()

x, y = next(iter(test_loader))

x = x.cuda()

# number of samples to visualize
N = 3
# number of dimensions to visualize
D = 10
# range of z1
Z1 = (-2, 2, 9)

x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = model(
    x[:N])


for i in range(N):
    cur_x = x[i:(i + 1)]
    all_x = []
    for d in range(D):
        print("N = {N} D = {D}".format(N=i, D=d))
        cur_z1 = z1_q[i:(i + 1)].expand(Z1[2], z1_q.shape[-1])
        cur_z2 = z2_q[i:(i + 1)].expand(Z1[2], z2_q.shape[-1])
        cur_z1[:, d] = torch.linspace(*Z1).type_as(x)
        cur_x_pert = model.pixelcnn_generate(cur_z1, cur_z2).view((-1, cur_x.shape[-1]))
        joint_x = torch.cat([cur_x, cur_x_pert])

        all_x.append(joint_x)

    all_x = torch.cat(all_x)
    fig = plot_images(model.args, all_x.detach().cpu().numpy(), size_x=(D + 1), size_y=joint_x.shape[0])

    file_name = "{model_name}_{i:03g}".format(model_name=model_name, i=i)
    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)

#=============================================================================#
# clustering
#=============================================================================#
from sklearn.manifold import TSNE

all_z1 = []
all_z2 = []
all_y = []

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

        x = data

        x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = model(
            x)

        all_z1.append(z1_q)
        all_z2.append(z2_q)
        all_y.append(target)

all_y = torch.cat(all_y).detach().cpu().numpy().flatten()
labels = np.unique(all_y)

all_z1 = torch.cat(all_z1)
all_z2 = torch.cat(all_z2)
all_z = torch.cat([all_z1, all_z2], dim=1)
print("Computing z embeddings...")
all_z_embed = TSNE(n_components=2).fit_transform(all_z.detach().cpu().numpy())
print("Done")


fig = plt.figure()
ax = plt.gca()

i = 0
j = 1

cmap = plt.cm.gist_rainbow
for ind, l in enumerate(labels):
    I = all_y == l
    c = cmap(ind / len(labels))
    ax.scatter(all_z_embed[I, i], all_z_embed[I, j], c=c, label="'{l}'".format(l=int(l)), s=2)

ax.grid()
plt.tight_layout()
# plt.legend()

file_name = "{model_name}_z_embed".format(model_name=model_name, i=i)
plt.savefig(dir + file_name + '.png', bbox_inches='tight')
plt.close(fig)
