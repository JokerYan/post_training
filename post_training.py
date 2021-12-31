import copy
import random
import logging

import torch
import torch.nn as nn

from attacks import attack_pgd_targeted, attack_pgd


def cal_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, 1)
    # collect the correct predictions for each class
    correct = 0
    total = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
        total += 1
    return correct / total


def post_train(model, images, train_loader, train_loaders_by_class, args):
    logger = logging.getLogger("eval")

    mu = torch.tensor(args.mean).view(3, 1, 1).cuda()
    std = torch.tensor(args.std).view(3, 1, 1).cuda()

    lower_limit = ((0 - mu)/ std)
    upper_limit = ((1 - mu)/ std)

    alpha = (10 / 255) / std
    epsilon = (8 / 255) / std
    loss_func = nn.CrossEntropyLoss()
    device = torch.device('cuda')
    model = copy.deepcopy(model)
    # model.train()
    fix_model = copy.deepcopy(model)
    # attack_model = torchattacks.PGD(model, eps=(8/255)/std, alpha=(2/255)/std, steps=20)
    optimizer = torch.optim.SGD(lr=args.pt_lr,
                                params=model.parameters(),
                                momentum=0.9,
                                nesterov=True)
    images = images.detach()
    with torch.enable_grad():
        # find neighbour
        original_output = fix_model(images)
        original_class = torch.argmax(original_output).reshape(1)

        if args.neigh_method == 'targeted':
            # targeted attack to find neighbour
            min_target_loss = float('inf')
            max_target_loss = float('-inf')
            neighbour_delta = None
            for target_idx in range(10):
                if target_idx == original_class:
                    continue
                target = torch.ones_like(original_class) * target_idx
                neighbour_delta_targeted = attack_pgd_targeted(model, images, original_class, target, epsilon, alpha,
                                                               attack_iters=20, restarts=1,
                                                               lower_limit=lower_limit, upper_limit=upper_limit,
                                                               random_start=False).detach()
                target_output = fix_model(images + neighbour_delta_targeted)
                target_loss = loss_func(target_output, target)
                if target_loss < min_target_loss:
                    min_target_loss = target_loss
                    neighbour_delta = neighbour_delta_targeted
                # print(int(target), float(target_loss))
        elif args.neigh_method == 'untargeted':
            # neighbour_images = attack_model(images, original_class)
            neighbour_delta = attack_pgd(model, images, original_class, epsilon, alpha, attack_iters=20, restarts=1,
                                         lower_limit=lower_limit, upper_limit=upper_limit,
                                         random_start=False).detach()

        neighbour_images = neighbour_delta + images
        neighbour_output = fix_model(neighbour_images)
        neighbour_class = torch.argmax(neighbour_output).reshape(1)

        if original_class == neighbour_class:
            logger.info('original class == neighbour class')
            if args.pt_data == 'ori_neigh':
                return model, original_class, neighbour_class, None, None, neighbour_delta

        loss_list = []
        acc_list = []
        for _ in range(args.pt_iter):
            if args.pt_data == 'ori_neigh':
                original_data, original_label = next(iter(train_loaders_by_class[original_class]))
                neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))
            elif args.pt_data == 'ori_rand':
                original_data, original_label = next(iter(train_loaders_by_class[original_class]))
                neighbour_class = (original_class + random.randint(1, 10)) % 10
                neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))
            elif args.pt_data == 'train':
                original_data, original_label = next(iter(train_loader))
                neighbour_data, neighbour_label = next(iter(train_loader))
            else:
                raise NotImplementedError

            data = torch.vstack([original_data, neighbour_data]).to(device)
            label = torch.hstack([original_label, neighbour_label]).to(device)
            # label_mixup = torch.hstack([original_label_mixup, neighbour_label_mixup]).to(device)

            if args.pt_method == 'adv':
                # generate fgsm adv examples
                delta = (torch.rand_like(data) * 2 - 1) * epsilon  # uniform rand from [-eps, eps]
                noise_input = data + delta
                noise_input.requires_grad = True
                noise_output = model(noise_input)
                loss = loss_func(noise_output, label)  # loss to be maximized
                input_grad = torch.autograd.grad(loss, noise_input)[0]
                delta = delta + alpha * torch.sign(input_grad)
                delta.clamp_(-epsilon, epsilon)
                adv_input = data + delta
            elif args.pt_method == 'dir_adv':
                # use fixed direction attack
                if args.adv_dir == 'pos':
                    adv_input = data + 1 * neighbour_delta
                elif args.adv_dir == 'neg':
                    adv_input = data + -1 * neighbour_delta
                elif args.adv_dir == 'both':
                    directed_delta = torch.vstack([torch.ones_like(original_data).to(device) * neighbour_delta,
                                                    torch.ones_like(neighbour_data).to(device) * -1 * neighbour_delta])
                    adv_input = data + directed_delta
            elif args.pt_method == 'normal':
                adv_input = data
            else:
                raise NotImplementedError

            adv_output = model(adv_input.detach())

            loss = loss_func(adv_output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            defense_acc = cal_accuracy(adv_output, label)
            loss_list.append(loss)
            acc_list.append(defense_acc)
    return model, original_class, neighbour_class