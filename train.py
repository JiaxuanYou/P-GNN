from sklearn.metrics import roc_auc_score

from model import *
from utils import *
from dataset import *

# for data augmentation
def pretrain(args, data_list, model, optimizer, writer_train, writer_val, writer_test, device,
          epoch_num=500, repeat=0, dataset_name='Cora'):
    loss_func = nn.BCEWithLogitsLoss()
    out_act = nn.Sigmoid()

    for epoch in range(epoch_num):
        model.train()
        optimizer.zero_grad()
        shuffle(data_list)
        effective_len = len(data_list) // args.batch_size * len(data_list)
        for id, data in enumerate(data_list[:effective_len]):
            out = model(data)

            # if original task is not link prediction, and do pretraining
            # use all edges to train
            if args.task != 'link':
                mask_link_positive = data.mask_link_positive
            # otherwise, either it is normal link prediction, or you are doing link pretraining for link prediction
            # only use train edges to train
            else:
                mask_link_positive = data.mask_link_positive_train

            mask_link_negative = get_edge_mask_link_negative(mask_link_positive,
                                                             num_nodes=data.num_nodes,
                                                             num_negtive_edges=mask_link_positive.shape[1])
            edge_mask_train = np.concatenate((mask_link_positive, mask_link_negative),
                                             axis=-1)


            nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
            nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
            nodes_first = nodes_first.view(nodes_first.shape[0], 1, nodes_first.shape[1])
            nodes_second = nodes_second.view(nodes_second.shape[0], nodes_second.shape[1], 1)
            pred = torch.matmul(nodes_first, nodes_second).squeeze()
            label_positive = torch.ones([mask_link_positive.shape[1], ], dtype=pred.dtype)
            label_negative = torch.zeros([mask_link_negative.shape[1], ], dtype=pred.dtype)
            label = torch.cat((label_positive, label_negative)).to(device)
            loss = loss_func(pred, label)

            # update
            loss.backward()
            if id % args.batch_size == args.batch_size - 1:
                if args.batch_size > 1:
                    # if this is slow, no need to do this normalization
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad /= args.batch_size
                optimizer.step()
                optimizer.zero_grad()
            time3 = time.time()
            # print(time1-time0, time2-time0, time3-time0)

        if epoch % args.epoch_log == 0:
            # evaluate
            model.eval()

            loss_train = 0
            loss_val = 0
            loss_test = 0
            auc_train = 0
            auc_val = 0
            auc_test = 0
            emb_norm_min = 0
            emb_norm_max = 0
            emb_norm_mean = 0
            for id, data in enumerate(data_list):
                out = model(data)
                emb_norm_min += torch.norm(out.data, dim=1).min().cpu().numpy()
                emb_norm_max += torch.norm(out.data, dim=1).max().cpu().numpy()
                emb_norm_mean += torch.norm(out.data, dim=1).mean().cpu().numpy()
                # pdb.set_trace()
                # train
                edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train),
                                                 axis=-1)
                nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
                nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
                nodes_first = nodes_first.view(nodes_first.shape[0], 1, nodes_first.shape[1])
                nodes_second = nodes_second.view(nodes_second.shape[0], nodes_second.shape[1], 1)
                pred = torch.matmul(nodes_first, nodes_second).squeeze()
                label_positive = torch.ones([data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                label_negative = torch.zeros([data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                label = torch.cat((label_positive, label_negative)).to(device)

                loss_train += loss_func(pred, label).cpu().data.numpy()
                auc_train += roc_auc_score(label.flatten().cpu().numpy(),
                                           out_act(pred).flatten().data.cpu().numpy())
                # val
                edge_mask_val = np.concatenate((data.mask_link_positive_val, data.mask_link_negative_val), axis=-1)
                nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[0, :]).long().to(device))
                nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[1, :]).long().to(device))
                nodes_first = nodes_first.view(nodes_first.shape[0], 1, nodes_first.shape[1])
                nodes_second = nodes_second.view(nodes_second.shape[0], nodes_second.shape[1], 1)
                pred = torch.matmul(nodes_first, nodes_second).squeeze()
                label_positive = torch.ones([data.mask_link_positive_val.shape[1], ], dtype=pred.dtype)
                label_negative = torch.zeros([data.mask_link_negative_val.shape[1], ], dtype=pred.dtype)
                label = torch.cat((label_positive, label_negative)).to(device)
                loss_val += loss_func(pred, label).cpu().data.numpy()
                auc_val += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                # test
                edge_mask_test = np.concatenate((data.mask_link_positive_test, data.mask_link_negative_test),
                                                axis=-1)
                nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[0, :]).long().to(device))
                nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[1, :]).long().to(device))
                nodes_first = nodes_first.view(nodes_first.shape[0], 1, nodes_first.shape[1])
                nodes_second = nodes_second.view(nodes_second.shape[0], nodes_second.shape[1], 1)
                pred = torch.matmul(nodes_first, nodes_second).squeeze()
                label_positive = torch.ones([data.mask_link_positive_test.shape[1], ], dtype=pred.dtype)
                label_negative = torch.zeros([data.mask_link_negative_test.shape[1], ], dtype=pred.dtype)
                label = torch.cat((label_positive, label_negative)).to(device)
                loss_test += loss_func(pred, label).cpu().data.numpy()
                auc_test += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                pdb.set_trace()

            loss_train /= id + 1
            loss_val /= id + 1
            loss_test /= id + 1
            emb_norm_min /= id + 1
            emb_norm_max /= id + 1
            emb_norm_mean /= id + 1
            auc_train /= id + 1
            auc_val /= id + 1
            auc_test /= id + 1

            print(epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                  'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test))
            writer_train.add_scalar('link_pretrain_repeat_' + str(repeat) + '/auc_' + dataset_name, auc_train, epoch)
            writer_train.add_scalar('link_pretrain_repeat_' + str(repeat) + '/loss_' + dataset_name, loss_train, epoch)
            writer_val.add_scalar('link_pretrain_repeat_' + str(repeat) + '/auc_' + dataset_name, auc_val, epoch)
            writer_train.add_scalar('link_pretrain_repeat_' + str(repeat) + '/loss_' + dataset_name, loss_val, epoch)
            writer_test.add_scalar('link_pretrain_repeat_' + str(repeat) + '/auc_' + dataset_name, auc_test, epoch)
            writer_test.add_scalar('link_pretrain_repeat_' + str(repeat) + '/loss_' + dataset_name, loss_test, epoch)
            writer_test.add_scalar('link_pretrain_repeat_' + str(repeat) + '/emb_min_' + dataset_name, emb_norm_min, epoch)
            writer_test.add_scalar('link_pretrain_repeat_' + str(repeat) + '/emb_max_' + dataset_name, emb_norm_max, epoch)
            writer_test.add_scalar('link_pretrain_repeat_' + str(repeat) + '/emb_mean_' + dataset_name, emb_norm_mean, epoch)

    return model







def train(args, data_list, model, optimizer, writer_train, writer_val, writer_test, device,
          epoch_num=500, repeat=0, dataset_name='Cora', augmentation=True, model_pretrained=None):
    if args.task == 'link':
        loss_func = nn.BCEWithLogitsLoss()
        out_act = nn.Sigmoid()

    for epoch in range(epoch_num):
        model.train()
        optimizer.zero_grad()
        # if args.task == 'graph':
        #     data_list = data_list_train
        shuffle(data_list)
        effective_len = len(data_list) // args.batch_size * len(data_list)
        for id, data in enumerate(data_list[:effective_len]):
            if augmentation:
                augment_adj(data, model_pretrained, args.perturb_ratio, args.perturb_prob)
            out = model(data)
            # node classification
            if args.task == 'node':
                out = F.log_softmax(out, dim=1)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            # link prediction
            elif args.task == 'link':
                resample_edge_mask_link_negative(data)  # resample negative links
                edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train),
                                                 axis=-1)
                nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
                nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
                nodes_first = nodes_first.view(nodes_first.shape[0], 1, nodes_first.shape[1])
                nodes_second = nodes_second.view(nodes_second.shape[0], nodes_second.shape[1], 1)
                pred = torch.matmul(nodes_first, nodes_second).squeeze()
                label_positive = torch.ones([data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                label_negative = torch.zeros([data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                label = torch.cat((label_positive, label_negative)).to(device)
                loss = loss_func(pred, label)

            # update
            loss.backward()
            if id % args.batch_size == args.batch_size - 1:
                if args.batch_size > 1:
                    # if this is slow, no need to do this normalization
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad /= args.batch_size
                optimizer.step()
                optimizer.zero_grad()

        if epoch % args.epoch_log == 0:
            # evaluate
            model.eval()

            # if args.task == 'graph':
            #     data_list = data_list_val
            loss_train = 0
            loss_val = 0
            loss_test = 0
            correct_train = 0
            all_train = 0
            correct_val = 0
            all_val = 0
            correct_test = 0
            all_test = 0
            auc_train = 0
            auc_val = 0
            auc_test = 0
            emb_norm_min = 0
            emb_norm_max = 0
            emb_norm_mean = 0
            for id, data in enumerate(data_list):
                if augmentation:
                    augment_adj(data, model_pretrained, args.perturb_ratio, 0) # make sure no augmentation
                out = model(data)
                emb_norm_min += torch.norm(out.data, dim=1).min().cpu().numpy()
                emb_norm_max += torch.norm(out.data, dim=1).max().cpu().numpy()
                emb_norm_mean += torch.norm(out.data, dim=1).mean().cpu().numpy()

                if args.task == 'node':
                    # classification
                    out = F.log_softmax(out, dim=1)
                    _, pred = out.max(dim=1)

                    # node classification
                    loss_train += F.nll_loss(out[data.train_mask], data.y[data.train_mask]).cpu().data.numpy()
                    loss_val += F.nll_loss(out[data.val_mask], data.y[data.val_mask]).cpu().data.numpy()
                    loss_test += F.nll_loss(out[data.test_mask], data.y[data.test_mask]).cpu().data.numpy()

                    correct_train += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
                    all_train += data.train_mask.sum().item()
                    correct_val += pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
                    all_val += data.val_mask.sum().item()
                    correct_test += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
                    all_test += data.test_mask.sum().item()

                    # pdb.set_trace()

                elif args.task == 'link' or args.task == 'link_pretrain':
                    # train
                    edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train),
                                                     axis=-1)
                    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
                    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
                    nodes_first = nodes_first.view(nodes_first.shape[0], 1, nodes_first.shape[1])
                    nodes_second = nodes_second.view(nodes_second.shape[0], nodes_second.shape[1], 1)
                    pred = torch.matmul(nodes_first, nodes_second).squeeze()
                    label_positive = torch.ones([data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                    label = torch.cat((label_positive, label_negative)).to(device)
                    loss_train += loss_func(pred, label).cpu().data.numpy()
                    auc_train += roc_auc_score(label.flatten().cpu().numpy(),
                                               out_act(pred).flatten().data.cpu().numpy())
                    # val
                    edge_mask_val = np.concatenate((data.mask_link_positive_val, data.mask_link_negative_val), axis=-1)
                    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[0, :]).long().to(device))
                    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[1, :]).long().to(device))
                    nodes_first = nodes_first.view(nodes_first.shape[0], 1, nodes_first.shape[1])
                    nodes_second = nodes_second.view(nodes_second.shape[0], nodes_second.shape[1], 1)
                    pred = torch.matmul(nodes_first, nodes_second).squeeze()
                    label_positive = torch.ones([data.mask_link_positive_val.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_link_negative_val.shape[1], ], dtype=pred.dtype)
                    label = torch.cat((label_positive, label_negative)).to(device)
                    loss_val += loss_func(pred, label).cpu().data.numpy()
                    auc_val += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                    # test
                    edge_mask_test = np.concatenate((data.mask_link_positive_test, data.mask_link_negative_test),
                                                    axis=-1)
                    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[0, :]).long().to(device))
                    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[1, :]).long().to(device))
                    nodes_first = nodes_first.view(nodes_first.shape[0], 1, nodes_first.shape[1])
                    nodes_second = nodes_second.view(nodes_second.shape[0], nodes_second.shape[1], 1)
                    pred = torch.matmul(nodes_first, nodes_second).squeeze()
                    label_positive = torch.ones([data.mask_link_positive_test.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_link_negative_test.shape[1], ], dtype=pred.dtype)
                    label = torch.cat((label_positive, label_negative)).to(device)
                    loss_test += loss_func(pred, label).cpu().data.numpy()
                    auc_test += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())

            loss_train /= id + 1
            loss_val /= id + 1
            loss_test /= id + 1
            emb_norm_min /= id + 1
            emb_norm_max /= id + 1
            emb_norm_mean /= id + 1
            if args.task == 'node':
                acc_train = correct_train / all_train
                acc_val = correct_val / all_val
                acc_test = correct_test / all_test

                print(epoch, 'Loss {:.4f}'.format(loss_train), 'Train Accuracy: {:.4f}'.format(acc_train),
                      'Val Accuracy: {:.4f}'.format(acc_val), 'Test Accuracy: {:.4f}'.format(acc_test))

                writer_train.add_scalar('repeat_' + str(repeat) + '/acc_' + dataset_name, acc_train, epoch)
                writer_train.add_scalar('repeat_' + str(repeat) + '/loss_' + dataset_name, loss_train, epoch)
                writer_val.add_scalar('repeat_' + str(repeat) + '/acc_' + dataset_name, acc_val, epoch)
                writer_train.add_scalar('repeat_' + str(repeat) + '/loss_' + dataset_name, loss_val, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/acc_' + dataset_name, acc_test, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/loss_' + dataset_name, loss_test, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/emb_min_' + dataset_name, emb_norm_min, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/emb_max_' + dataset_name, emb_norm_max, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/emb_mean_' + dataset_name, emb_norm_mean, epoch)

            if args.task == 'link':
                auc_train /= id + 1
                auc_val /= id + 1
                auc_test /= id + 1

                print(epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                      'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test))
                writer_train.add_scalar('repeat_' + str(repeat) + '/auc_' + dataset_name, auc_train, epoch)
                writer_train.add_scalar('repeat_' + str(repeat) + '/loss_' + dataset_name, loss_train, epoch)
                writer_val.add_scalar('repeat_' + str(repeat) + '/auc_' + dataset_name, auc_val, epoch)
                writer_train.add_scalar('repeat_' + str(repeat) + '/loss_' + dataset_name, loss_val, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/auc_' + dataset_name, auc_test, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/loss_' + dataset_name, loss_test, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/emb_min_' + dataset_name, emb_norm_min, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/emb_max_' + dataset_name, emb_norm_max, epoch)
                writer_test.add_scalar('repeat_' + str(repeat) + '/emb_mean_' + dataset_name, emb_norm_mean, epoch)
    return model



