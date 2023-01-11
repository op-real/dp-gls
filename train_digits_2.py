# Copyright(c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import numpy as np
import os
import pickle
import scipy.stats
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import loss as loss_func
import network
from data_list import build_uspsmnist, sample_ratios, subsampling


def write_list(f, l, comments=''):
    f.write(comments)
    f.write(",".join(map(str, l)) + "\n")
    f.flush()
    sys.stdout.flush()

def train_wn(args, model, source_samples, target_samples, wn_disc, wn_gen, optimizer_wnd, optimizer_wng, criterion, criterion_D): 
    model.eval()
    
    len_source = source_samples.shape[0]
    len_target = target_samples.shape[0]

    size = max(len_source, len_target)
    num_iter = int(size / 256)

    for _ in range(5):
        source_idx = np.random.choice(len_source, args.batch_size)
        target_idx = np.random.choice(len_target, args.batch_size)
        data_source = source_samples[source_idx]
        data_target = target_samples[target_idx]

        with torch.no_grad():
            feature_source, _ = model(data_source)
            feature_target, _ = model(data_target)

        # train disc against source
        optimizer_wnd.zero_grad()
 
        wn_disc.train()
        wn_gen.eval()

        outputs_source = wn_disc(feature_source)            
        weights_source = wn_gen(feature_target) 

        loss_source = criterion_D(outputs_source, torch.zeros(feature_source.shape[0]).to(args.device)) 
        loss_source = torch.dot(loss_source, weights_source.detach()) / torch.sum(weights_source.detach())

        # train disc against target
        outputs_target = wn_disc(feature_target)
        loss_target = criterion(outputs_target, torch.ones(feature_target.shape[0]).to(args.device))          
        
        # calculate disc loss and update disc params
        loss_disc = loss_source + loss_target
        loss_disc.backward()
        optimizer_wnd.step()
                
        # train gen against source 
        optimizer_wng.zero_grad()
        wn_disc.eval()
        wn_gen.train()

        weights_gen = wn_gen(feature_source)  
        outputs_gen = wn_disc(feature_source)  
        
        # times -1 for generator loss, and update gen params
        loss_gen = -1 * criterion_D(outputs_gen.detach(), torch.zeros(feature_source.shape[0]).to(args.device))
        loss_gen = torch.dot(loss_gen, weights_gen) / torch.sum(weights_gen)     
        
        loss_gen.backward()
        optimizer_wng.step()  

def train(args, model, ad_net, source_samples, source_labels, target_samples, target_labels, 
          optimizer, optimizer_ad, epoch, start_epoch, method, source_label_distribution, 
          out_wei_file, cov_mat, pseudo_target_label, class_weights, true_weights, wn_disc, wn_gen, optimizer_wnd, optimizer_wng):

    cov_mat[:] = 0.0
    pseudo_target_label[:] = 0.0

    len_source = source_labels.shape[0]
    len_target = target_labels.shape[0]

    size = max(len_source, len_target)
    num_iter = int(size / args.batch_size)

    for batch_idx in range(num_iter):
        # learn/update weights for WNDANN
        if 'WN' in method and epoch > start_epoch:
            train_wn(args, model, source_samples, target_samples, wn_disc, wn_gen, optimizer_wnd, optimizer_wng, nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss(reduction='none'))

        wn_gen.eval()
        model.train()

        source_idx = np.random.choice(len_source, args.batch_size)
        target_idx = np.random.choice(len_target, args.batch_size)
        data_source, label_source = source_samples[source_idx], source_labels[source_idx]
        data_target, _ = target_samples[target_idx], target_labels[target_idx]

        optimizer.zero_grad()
        optimizer_ad.zero_grad()

        # source and target feature and classifier output
        feature, output = model(torch.cat((data_source, data_target), 0))

        # source classifier loss
        if 'IW' in method:
            ys_onehot = torch.zeros(args.batch_size, 10).to(args.device)
            ys_onehot.scatter_(1, label_source.view(-1, 1), 1)
            # Compute weights on source data.
            if 'ORACLE' in method:
                weights = torch.mm(ys_onehot, true_weights)
            else:
                weights = torch.mm(ys_onehot, model.im_weights)

            source_preds, target_preds = output[:args.batch_size], output[args.batch_size:]
            # Compute the aggregated distribution of pseudo-label on the target domain.
            pseudo_target_label += torch.sum(F.softmax(target_preds, dim=1), dim=0).view(-1, 1).detach()
            # Update the covariance matrix on the source domain as well.
            cov_mat += torch.mm(F.softmax(source_preds, dim=1).transpose(1, 0), ys_onehot).detach()

            loss = torch.mean(nn.CrossEntropyLoss(weight=class_weights, reduction='none')
                (output.narrow(0, 0, data_source.size(0)), label_source) * weights) / 10.0
        elif 'WN' in method and epoch > start_epoch:
            # Compute weights on source data.
            if 'ORACLE' in method:
                ys_onehot = torch.zeros(args.batch_size, 10).to(args.device)
                ys_onehot.scatter_(1, label_source.view(-1, 1), 1)
                weights = torch.mm(ys_onehot, true_weights)
            else:
                weights = wn_gen(feature[:data_source.size(0)])
                # weights = torch.ones(data_source.size(0)).to('cuda')
                loss = torch.dot(
                    nn.CrossEntropyLoss(reduction='none')(output.narrow(0, 0, data_source.size(0)), label_source), 
                    weights.detach()
                ) / torch.sum(weights.detach())
        else:
            loss = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)

        # source and target domain classifier loss (weights on source only)
        if epoch > start_epoch:
            if method == 'CDAN-E':
                softmax_output = nn.Softmax(dim=1)(output)
                entropy = loss_func.Entropy(softmax_output)
                loss += loss_func.CDAN([feature, softmax_output], ad_net, entropy, network.calc_coeff(
                    num_iter*(epoch-start_epoch)+batch_idx), None, device=args.device)

            elif 'IWCDAN-E' in method:
                softmax_output = nn.Softmax(dim=1)(output)
                entropy = loss_func.Entropy(softmax_output)
                loss += loss_func.CDAN([feature, softmax_output], ad_net, entropy, network.calc_coeff(
                    num_iter*(epoch-start_epoch)+batch_idx), None, weights=weights, device=args.device)

            elif method == 'CDAN':
                softmax_output = nn.Softmax(dim=1)(output)
                loss += loss_func.CDAN([feature, softmax_output],
                                    ad_net, None, None, None, device=args.device)

            elif 'IWCDAN' in method:
                softmax_output = nn.Softmax(dim=1)(output)
                loss += loss_func.CDAN([feature, softmax_output],
                                    ad_net, None, None, None, weights=weights, device=args.device)

            elif method == 'DANN':
                loss += loss_func.DANN(feature, ad_net, args.device)

            elif 'IWDAN' in method:
                dloss = loss_func.IWDAN(feature, ad_net, weights)
                loss += args.mu * dloss

            elif 'WNDANN' in method:
                dloss = loss_func.IWDAN(feature, ad_net, weights)
                loss += args.mu * dloss

            elif method == 'NANN':
                pass

            else:
                raise ValueError('Method cannot be recognized.')

        loss.backward()
        optimizer.step()

        if epoch > start_epoch and method != 'NANN':
            optimizer_ad.step()

    if 'IW' in method  and epoch > start_epoch:
        pseudo_target_label /= args.batch_size * num_iter
        cov_mat /= args.batch_size * num_iter
        # Recompute the importance weight by solving a QP.
        model.im_weights_update(source_label_distribution,
                                pseudo_target_label.cpu().detach().numpy(),
                                cov_mat.cpu().detach().numpy(),
                                args.device)
        current_weights = [round(x, 4) for x in model.im_weights.data.cpu().numpy().flatten()]
        write_list(out_wei_file, [np.linalg.norm(
            current_weights - true_weights.cpu().numpy().flatten())] + current_weights)
        print(np.linalg.norm(current_weights - true_weights.cpu().numpy().flatten()), current_weights)


def test(args, epoch, model, test_samples, test_labels, start_time_test, out_log_file, name=''):
    model.eval()
    test_loss = 0
    correct = 0
    len_test = test_labels.shape[0]

    for i in range(len_test):
        data, target = test_samples[i].unsqueeze(0), test_labels[i].unsqueeze(0)
        _, output = model(data)
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = output.data.cpu().max(1, keepdim=True)[1]
        correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len_test
    temp_acc = 100. * correct / len_test
    log_str = "  {}, epoch: {:05d}, sec: {:.0f}, loss: {:.5f}, accuracy: {}/{}, precision: {:.5f}".format(name, epoch, time.time() - start_time_test, test_loss, correct, len_test, temp_acc)
    print(log_str)
    sys.stdout.flush()
    out_log_file.write(log_str+"\n")
    out_log_file.flush()


def main():
    # Training settings
    def parse_args(): 
        parser = argparse.ArgumentParser(description='CDAN USPS MNIST')
        parser.add_argument('method', type=str, default='CDAN-E',
                            choices=['WNDANN', 'CDAN', 'CDAN-E', 'DANN', 'IWDAN', 'NANN', 'IWDANORACLE', 'IWCDAN', 'IWCDANORACLE', 'IWCDAN-E', 'IWCDAN-EORACLE'])
        parser.add_argument('--task', default='mnist2usps', help='task to perform', choices=['usps2mnist', 'mnist2usps'])
        parser.add_argument('--batch_size', type=int, default=64,
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test_batch_size', type=int, default=1000,
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=70, metavar='N',
                            help='number of epochs to train (default: 70)')
        parser.add_argument('--lr', type=float, default=0.0, metavar='LR',
                            help='learning rate (default: 0.02)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--seed', type=int, default=42, metavar='S',
                            help='random seed (default: 42)')
        parser.add_argument('--log_interval', type=int, default=50,
                            help='how many batches to wait before logging training status')
        parser.add_argument('--root_folder', type=str, default='data/usps2mnist/', help="The folder containing the datasets and the lists")
        parser.add_argument('--output_dir', type=str, default='', help="output directory")
        parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient of the domain adversarial loss", type=float, default=1.0)
        parser.add_argument('--ratio', type =float, default=0, help='ratio option')
        parser.add_argument('--ma', type=float, default=0.5, help='weight for the moving average of iw')
        args = parser.parse_args()
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args.output_dir = 'results_' + args.output_dir
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        # Running the JSD experiment on fewer epochs for efficiency
        if args.ratio >= 100:
            args.epochs = 25

        print('Running {} on {} for {} epochs on task {}'.format(
            args.method, args.device, args.epochs, args.task))

        # Set random number seed.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        return args

    args = parse_args()
    out_log_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    out_log_file_train = open(os.path.join(args.output_dir, "log_train.txt"), "w")
    class_num = 10
    start_epoch = 1

    def load_data():
        if args.task == 'usps2mnist':
            # CDAN parameters
            decay_epoch = 6
            decay_frac = 0.5
            lr = 0.02
            model = network.LeNet(args.ma)
            build_dataset = build_uspsmnist

            source_list = os.path.join(args.root_folder, 'usps_train.txt')
            source_path = os.path.join(args.root_folder, 'usps_train_dataset.pkl')
            target_list = os.path.join(args.root_folder, 'mnist_train.txt')
            target_path = os.path.join(args.root_folder, 'mnist_train_dataset.pkl')
            test_list   = os.path.join(args.root_folder, 'mnist_test.txt')
            test_path   = os.path.join(args.root_folder, 'mnist_test_dataset.pkl')

        elif args.task == 'mnist2usps':

            decay_epoch = 5
            decay_frac = 0.5
            lr = 0.02
            model = network.LeNet(args.ma)
            build_dataset = build_uspsmnist

            source_list = os.path.join(args.root_folder, 'mnist_train.txt')
            source_path = os.path.join(args.root_folder, 'mnist_train_dataset.pkl')
            target_list = os.path.join(args.root_folder, 'usps_train.txt')
            target_path = os.path.join(args.root_folder, 'usps_train_dataset.pkl')
            test_list   = os.path.join(args.root_folder, 'usps_test.txt')
            test_path   = os.path.join(args.root_folder, 'usps_test_dataset.pkl')

        else:
            raise Exception('Task cannot be recognized!')

        model = model.to(args.device)

        if args.lr > 0:
            lr = args.lr

        print('Starting loading data')
        sys.stdout.flush()
        t_data = time.time()
        if os.path.exists(source_path):
            print('Found existing dataset for source')
            with open(source_path, 'rb') as f:
                [source_samples, source_labels] = pickle.load(f)
                source_samples, source_labels = torch.Tensor(source_samples).to(
                    args.device), torch.LongTensor(source_labels).to(args.device)
        else:
            print('Building dataset for source and writing to {}'.format(source_path))
            source_samples, source_labels = build_dataset(
                source_list, source_path, args.root_folder, args.device)

        if os.path.exists(target_path):
            print('Found existing dataset for target')
            with open(target_path, 'rb') as f:
                [target_samples, target_labels] = pickle.load(f)
                target_samples, target_labels = torch.Tensor(
                    target_samples).to(args.device), torch.LongTensor(target_labels).to(args.device)
        else:
            print('Building dataset for target and writing to {}'.format(target_path))
            target_samples, target_labels = build_dataset(
                target_list, target_path, args.root_folder, args.device)

        if os.path.exists(test_path):
            print('Found existing dataset for test')
            with open(test_path, 'rb') as f:
                [test_samples, test_labels] = pickle.load(f)
                test_samples, test_labels = torch.Tensor(
                    test_samples).to(args.device), torch.LongTensor(test_labels).to(args.device)
        else:
            print('Building dataset for test and writing to {}'.format(test_path))
            test_samples, test_labels = build_dataset(
                test_list, test_path, args.root_folder, args.device)

        print('Data loaded in {}'.format(time.time() - t_data))

        if args.ratio == 1:
            # RATIO OPTION 1
            # 30% of the samples from the first 5 classes
            print('Using option 1, ie [0.3] * 5 + [1] * 5')
            ratios_source = [0.3] * 5 + [1] * 5
            ratios_target = [1] * 10
        elif args.ratio >= 200:
            s_ = subsampling[int(args.ratio) % 100]
            ratios_source = s_[0]
            ratios_target = [1] * 10
            print('Using random subset ratio {} of the source, with theoretical jsd {}'.format(args.ratio, s_[1]))
        elif 200 > args.ratio >= 100:
            s_ = subsampling[int(args.ratio) % 100]
            ratios_source = [1] * 10
            ratios_target = s_[0]
            print('Using random subset ratio {} of the target, with theoretical jsd {}'.format(args.ratio, s_[1]))
        else:
            # ORIGINAL DATASETS
            print('Using original datasets')
            ratios_source = [1] * 10
            ratios_target = [1] * 10
        ratios_test = ratios_target

        # Subsample dataset if need be
        source_samples, source_labels = sample_ratios(
            source_samples, source_labels, ratios_source)
        target_samples, target_labels = sample_ratios(
            target_samples, target_labels, ratios_target)
        test_samples, test_labels = sample_ratios(
            test_samples, test_labels, ratios_test)

        # compute labels distribution on the source and target domain
        source_label_distribution = np.zeros((class_num))
        for img in source_labels:
            source_label_distribution[int(img.item())] += 1
        print("Total source samples: {}".format(
            np.sum(source_label_distribution)), flush=True)
        print("Source samples per class: {}".format(source_label_distribution))
        source_label_distribution /= np.sum(source_label_distribution)
        write_list(out_log_file_train, source_label_distribution, 'source_label_distribution:')
        print("Source label distribution: {}".format(source_label_distribution))
        target_label_distribution = np.zeros((class_num))
        for img in target_labels:
            target_label_distribution[int(img.item())] += 1
        print("Total target samples: {}".format(
            np.sum(target_label_distribution)), flush=True)
        print("Target samples per class: {}".format(target_label_distribution))
        target_label_distribution /= np.sum(target_label_distribution)
        write_list(out_log_file_train, target_label_distribution, 'target_label_distribution:')
        print("Target label distribution: {}".format(target_label_distribution))
        test_label_distribution = np.zeros((class_num))
        for img in test_labels:
            test_label_distribution[int(img.item())] += 1
        print("Test samples per class: {}".format(test_label_distribution))
        test_label_distribution /= np.sum(test_label_distribution)
        write_list(out_log_file_train, test_label_distribution, 'test_label_distribution:')
        print("Test label distribution: {}".format(test_label_distribution))
        mixture = (source_label_distribution + target_label_distribution) / 2
        jsd = (scipy.stats.entropy(source_label_distribution, qk=mixture)
            + scipy.stats.entropy(target_label_distribution, qk=mixture)) / 2
        print("JSD source to target : {}".format(jsd))
        mixture_2 = (test_label_distribution + target_label_distribution) / 2
        jsd_2 = (scipy.stats.entropy(test_label_distribution, qk=mixture_2)
            + scipy.stats.entropy(target_label_distribution, qk=mixture_2)) / 2
        print("JSD test to target : {}".format(jsd_2))
        out_wei_file = open(os.path.join(args.output_dir, "log_weights_{}.txt".format(jsd)), "w")
        write_list(out_wei_file, [round(x, 4) for x in source_label_distribution], "JSD source_label_distribution to target_label_distribution:")
        write_list(out_wei_file, [round(x, 4) for x in target_label_distribution], "JSD test_label_distribution to target_label_distribution:")
        out_wei_file.write(str(jsd) + "\n")
        true_weights = torch.tensor(
            target_label_distribution / source_label_distribution, dtype=torch.float, requires_grad=False)[:, None].to(args.device)
        print("True weights : {}".format(true_weights[:, 0].cpu().numpy()))

        return model, lr, decay_frac, decay_epoch, source_samples, source_labels, target_samples, target_labels, \
            test_samples, test_labels, source_label_distribution, true_weights, out_wei_file

    model, lr, decay_frac, decay_epoch, \
    source_samples, source_labels, \
    target_samples, target_labels, \
    test_samples, test_labels,\
    source_label_distribution,\
    true_weights, out_wei_file = load_data()    

    print(f'learning rate {lr}')

    # adversarial net 
    if 'CDAN' in args.method:
        ad_net = network.AdversarialNetwork(model.output_num() * class_num, 500, True)
        # wn_disc = network.WNDisc()
    else:
        ad_net = network.AdversarialNetwork(model.output_num(), 500, True)
        wn_disc = network.WNDisc(model.output_num(), 500, True)
        wn_gen = network.WNGen(model.output_num(), 500)

    ad_net = ad_net.to(args.device)
    wn_disc = wn_disc.to(args.device)
    wn_gen = wn_gen.to(args.device)

    out_log_file_train.write(str(wn_gen) + "\n")
    out_log_file_train.write(str(wn_disc) + "\n")
    out_log_file.flush()
    
    # optimizers
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
    optimizer_ad = optim.SGD(ad_net.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
    optimizer_wnd = optim.SGD(wn_disc.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
    optimizer_wng = optim.SGD(wn_gen.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)

    # Maintain two quantities for the QP.
    cov_mat = torch.tensor(np.zeros((class_num, class_num), dtype=np.float32),
                           requires_grad=False).to(args.device)
    pseudo_target_label = torch.tensor(np.zeros((class_num, 1), dtype=np.float32),
                                       requires_grad=False).to(args.device)
    # Maintain one weight vector for BER.
    class_weights = torch.tensor( 1.0 / source_label_distribution, dtype=torch.float, requires_grad=False).to(args.device)

    for epoch in range(1, args.epochs + 1):
        start_time_test = time.time()
        if epoch % decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * decay_frac
        test(args, epoch, model, test_samples, test_labels, start_time_test, out_log_file, name='Target test')
        train(args, model, ad_net, source_samples, source_labels, target_samples, target_labels,
              optimizer, optimizer_ad, epoch, start_epoch, args.method, source_label_distribution, 
              out_wei_file, cov_mat, pseudo_target_label, class_weights, true_weights, wn_disc, wn_gen, optimizer_wnd, optimizer_wng)
    test(args, epoch+1, model, test_samples, test_labels, start_time_test, out_log_file, name='Target test')
    test(args, epoch+1, model, source_samples, source_labels, start_time_test, out_log_file_train, name='Source train')


if __name__ == '__main__':
    main()
