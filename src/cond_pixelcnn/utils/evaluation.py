from __future__ import print_function

import torch
from torch.autograd import Variable

from .visual_evaluation import plot_images
from .load_data import load_dataset
from .distributions import log_Normal_diag
from .knnie import kraskov_mi as ksg_mi

import numpy as np
import time
import os

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import normalized_mutual_info_score
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================


def test_vae(args, model, train_loader, data_loader, dir, vis_only=False, with_z_embedding=True):
    # load all data
    # grab the test data by iterating over the loader
    # there is no standardized tensor_dataset member across pytorch datasets
    test_data, test_target = [], []
    for data, lbls in data_loader:
        test_data.append(data)
        test_target.append(lbls)

    test_data, test_target = [torch.cat(test_data, 0), torch.cat(test_target, 0).squeeze()]

    # grab the train data by iterating over the loader
    # there is no standardized tensor_dataset member across pytorch datasets
    full_data = []
    full_target = []
    for data, lbls in train_loader:
        full_data.append(data)
        full_target.append(lbls)

    full_data, full_target = [torch.cat(full_data, 0), torch.cat(full_target, 0).squeeze()]

    if args.cuda:
        test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

    if args.dynamic_binarization:
        full_data = torch.bernoulli(full_data)

    # print(model.means(model.idle_input))

    # CALCULATE clustering and MI
    def embed_z(x, batch=model.args.MB, with_MI=False):
        with torch.no_grad():
            # MI = 0
            embed_z = []
            for data in torch.split(x, batch):
                x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = model(
                    data)
                embed_z.append(torch.cat([z1_q, z2_q], dim=1))

                # # p(z)
                # log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
                # log_p_z2 = model.log_p_z2(z2_q)
                # log_p_z = log_p_z1 + log_p_z2

                # # q(z|x)
                # log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
                # log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
                # log_q_z_given_x = log_q_z1 + log_q_z2

                # MI = MI + (log_q_z_given_x - log_p_z).mean()

            embed_z = torch.cat(embed_z, dim=0)

        if with_MI:
            N = min(x.shape[0], 3000)
            MI = ksg_mi(x[:N].detach().cpu().numpy(), embed_z[:N].detach().cpu().numpy())
            dx = x[0].numel()
            dz = embed_z[0].numel()
            MI_l2_err = 1.0 / N**0.5 + N**(-1 / (dx + dz))
            return embed_z, MI, MI_l2_err
        else:
            return embed_z

    # collect all additional stats
    stats = {}
    if with_z_embedding:
        # project train/test data to latent representation
        train_target = full_target.detach().cpu().numpy()
        train_z = embed_z(full_data)
        train_z = train_z.detach().cpu().numpy()

        test_target = test_target.detach().cpu().numpy()
        test_labels = np.unique(test_target)
        if vis_only:
            test_z = embed_z(test_data, with_MI=False)
            test_z = test_z.detach().cpu().numpy()
        else:
            test_z, MI, MI_l2_err = embed_z(test_data, with_MI=True)
            test_z = test_z.detach().cpu().numpy()
            stats["MI"] = MI
            stats["MI_l2_err"] = MI_l2_err
            print("MI = {MI} MI_l2_err = {MI_l2_err}".format(MI=MI, MI_l2_err=MI_l2_err))

            test_pred = KMeans(n_clusters=len(test_labels), random_state=177).fit_predict(test_z)
            NMI = normalized_mutual_info_score(test_target, test_pred, "max")
            stats["NMI"] = NMI
            print("NMI = {NMI}".format(NMI=NMI))

            # CALCULATE classification
            # train classifiers
            for clf_name, clf in [
                # ("MLP", MLPClassifier(hidden_layer_sizes=(200,), activation="tanh", max_iter=1000)),
                # ("KNN", KNeighborsClassifier(n_neighbors=5)),
                # ("SVC", SVC(kernel='linear')),
                ("KNN1", KNeighborsClassifier(n_neighbors=1)),
                ("KNN3", KNeighborsClassifier(n_neighbors=3)),
                ("KNN5", KNeighborsClassifier(n_neighbors=5)),
                ("KNN10", KNeighborsClassifier(n_neighbors=10)),
            ]:
                print("Training {clf_name}".format(clf_name=clf_name))
                clf.fit(train_z, train_target)
                clf_score = clf.score(test_z, test_target)
                stats["clf_acc_" + clf_name] = clf_score

                print("{clf_name} = {clf_score}".format(clf_name=clf_name, clf_score=clf_score))

        # VISUALIZE z embedding
        print("Computing z embeddings...")
        test_z_embed = TSNE(n_components=2).fit_transform(test_z)
        print("Done")

        fig = plt.figure()
        ax = plt.gca()

        i = 0
        j = 1

        cmap = plt.cm.tab20
        for ind, l in enumerate(test_labels):
            I = (test_target == l)
            c = cmap(ind / len(test_labels))
            ax.scatter(test_z_embed[I, i], test_z_embed[I, j],
                       c=c, label="'{l}'".format(l=int(l)), s=2)

        # ax.grid()
        # plt.legend()
        plt.tight_layout()

        file_name = "z_embed"
        plt.savefig(dir + file_name + '.png', bbox_inches='tight')
        plt.close(fig)

    if model.args.number_components >= 64:
        # M = N = int(model.args.number_components**0.5)
        M = N = 8
    else:
        N = min(8, model.args.number_components)
        M = max(1, model.args.number_components // N)

    # VISUALIZATION: plot real images
    plot_images(args, test_data.data.cpu().numpy()[0:N * M], dir, 'real', size_x=N, size_y=M)
    plot_images(args, test_data.data.cpu().numpy()[0:int(N * 3 // 2)],
                dir, 'real_flat', size_x=1, size_y=int(N * 3 // 2))

    # VISUALIZATION: plot reconstructions
    samples = model.reconstruct_x(test_data[0:N * M])
    plot_images(args, samples.data.cpu().numpy(), dir, 'reconstructions', size_x=N, size_y=M)

    samples_flat = model.reconstruct_x(test_data[0:int(N * 3 // 2)]).data.cpu().numpy()
    plot_images(args, samples_flat, dir, 'reconstructions_flat', size_x=1, size_y=int(N * 3 // 2))

    # VISUALIZATION: plot real images + reconstructions
    test_data_recon = []
    for i in range(max(1, M // 2)):
        test_data_recon.append(test_data[(i * N):((i + 1) * N)])
        test_data_recon.append(samples[(i * N):((i + 1) * N)].view_as(test_data_recon[-1]))

    test_data_recon = torch.cat(test_data_recon)
    plot_images(args, test_data_recon.data.cpu().numpy(), dir, 'real_recon', size_x=N, size_y=(max(1, M // 2 * 2)))

    # VISUALIZATION: plot generations
    samples_rand = model.generate_x(N * M)

    plot_images(args, samples_rand.data.cpu().numpy(), dir, 'generations', size_x=N, size_y=M)

    if args.prior == 'vampprior':
        # VISUALIZE pseudoinputs
        pseudoinputs = model.means(model.idle_input).cpu().data.numpy()

        plot_images(args, pseudoinputs[0:N * M], dir, 'pseudoinputs', size_x=N, size_y=M)

    if not vis_only:
        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = model.calculate_lower_bound(test_data, MB=args.MB)
        t_ll_e = time.time()
        print('Test lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_test, t_ll_e - t_ll_s))

        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_train = model.calculate_lower_bound(full_data, MB=args.MB)
        t_ll_e = time.time()
        print('Train lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_train, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_test = model.calculate_likelihood(test_data, dir, mode='test', S=args.S, MB=args.MB)
        t_ll_e = time.time()
        print('Test log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        # model.calculate_likelihood(full_data, dir, mode='train', S=args.S,
        # MB=args.MB)) #commented because it takes too much time
        log_likelihood_train = 0.
        t_ll_e = time.time()
        print('Train log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_train, t_ll_e - t_ll_s))

        stats["LL"] = log_likelihood_test

        torch.save(stats, dir + 'stats.pkl')

        return log_likelihood_test, log_likelihood_train, elbo_test, elbo_train, stats


def evaluate_vae(args, model, train_loader, data_loader, epoch, dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0
    # set model to evaluation mode
    model.eval()

    # evaluate
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            x = data

            # calculate loss function
            loss, RE, KL = model.calculate_loss(x, average=True)

            evaluate_loss += loss.item()
            evaluate_re += -RE.item()
            evaluate_kl += KL.item()

            # print N digits
            if batch_idx == 1 and mode == 'validation':
                if epoch == 1:
                    if not os.path.exists(dir + 'reconstruction/'):
                        os.makedirs(dir + 'reconstruction/')
                    # VISUALIZATION: plot real images
                    plot_images(args, data.data.cpu().numpy()[0:9], dir + 'reconstruction/', 'real', size_x=3, size_y=3)
                x_mean = model.reconstruct_x(x)
                plot_images(args, x_mean.data.cpu().numpy()[0:9], dir +
                            'reconstruction/', str(epoch), size_x=3, size_y=3)

    if mode == 'test':
        log_likelihood_test, log_likelihood_train, elbo_test, elbo_train, stats = test_vae(
            args, model, train_loader, data_loader, dir)

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size
    if mode == 'test':
        return evaluate_loss, evaluate_re, evaluate_kl, log_likelihood_test, log_likelihood_train, elbo_test, elbo_train, stats
    else:
        return evaluate_loss, evaluate_re, evaluate_kl


def load_visualize_vae(model_fname):
    model = torch.load(model_fname)

    dir = os.path.dirname(model_fname) + os.sep

    train_loader, val_loader, test_loader, args = load_dataset(model.args)

    test_vae(
        args, model, train_loader, test_loader, dir,
        vis_only=True,
        with_z_embedding=True,
    )


def load_test_vae(model_fname):
    model = torch.load(model_fname)

    dir = os.path.dirname(model_fname) + os.sep

    train_loader, val_loader, test_loader, args = load_dataset(model.args)

    test_log_likelihood, train_log_likelihood, test_elbo, train_elbo, stats = test_vae(
        args, model, train_loader, test_loader, dir, vis_only=False)

    print('FINAL EVALUATION ON TEST SET\n'
          'LogL (TEST): {:.2f}\n'
          'LogL (TRAIN): {:.2f}\n'
          'ELBO (TEST): {:.2f}\n'
          'ELBO (TRAIN): {:.2f}\n'
          'stats: {}\n'.format(
              test_log_likelihood,
              train_log_likelihood,
              test_elbo,
              train_elbo,
              str(stats)
          ))

    with open(dir + 'vae_experiment_log.txt', 'a') as f:
        print('FINAL EVALUATION ON TEST SET\n'
              'LogL (TEST): {:.2f}\n'
              'LogL (TRAIN): {:.2f}\n'
              'ELBO (TEST): {:.2f}\n'
              'ELBO (TRAIN): {:.2f}\n'
              'stats: {}\n'.format(
                  test_log_likelihood,
                  train_log_likelihood,
                  test_elbo,
                  train_elbo,
                  str(stats)
              ), file=f)
