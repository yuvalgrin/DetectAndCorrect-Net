from collections import Counter

import tools
import data_load
import argparse
from models import *
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer import transform_train, transform_test, transform_target
from torch.optim.lr_scheduler import MultiStepLR

from pretrain import estimate_initial_t, fit_gmm, plot_noise_hist

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--save_dir', type=str, help='dir to save model files', default='saves')
parser.add_argument('--dataset', type=str, help='azimuth, mnist, cifar10, or cifar100', default='cifar10')
parser.add_argument('--n_epoch_phase1', type=int, default=100)
parser.add_argument('--n_epoch_phase2', type=int, default=100)
parser.add_argument('--n_epoch_phase3', type=int, default=80)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--noise_type', type=str, default='symmetric')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--lam', type = float, default =0.0001)
parser.add_argument('--anchor', action='store_false')



args = parser.parse_args()
np.set_printoptions(precision=2,suppress=True)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# GPU
device = torch.device('cuda:'+ str(args.device))


if args.dataset == 'mnist':

    num_classes = 10
    milestones = None

    train_data = data_load.mnist_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                         noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type,anchor=args.anchor)
    val_data = data_load.mnist_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                       noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
    test_data = data_load.mnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    model = Lenet()

if args.dataset == 'cifar10':

    args.num_classes = 10
    milestones = [30,60]

    train_data = data_load.cifar10_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                         noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)
    val_data = data_load.cifar10_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                       noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
    test_data = data_load.cifar10_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    model = ResNet18(args.num_classes)

if args.dataset == 'cifar100':
    args.init = 4.5

    args.num_classes = 100

    milestones = [30, 60]

    train_data = data_load.cifar100_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                         noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)
    val_data = data_load.cifar100_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                       noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
    test_data = data_load.cifar100_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    model = ResNet34(args.num_classes)

if args.dataset == 'azimuth':

    args.num_classes = 31
    milestones = [30, 60]

    train_data = data_load.azimuth_scrna_pbmc_dataset(train=True, transform=transform_train(args.dataset),
                                              target_transform=transform_target,
                                              noise_rate=args.noise_rate, random_seed=args.seed,
                                              noise_type=args.noise_type, num_class=args.num_classes)
    val_data = data_load.azimuth_scrna_pbmc_dataset(train=False, transform=transform_train(args.dataset),
                                            target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed,
                                            noise_type=args.noise_type, num_class=args.num_classes)
    test_data = data_load.azimuth_scrna_pbmc_test_dataset(transform=transform_test(args.dataset),
                                                                target_transform=transform_target)
    model = FCNN(args.num_classes)

save_dir, model_dir, matrix_dir, logs = create_dir(args)

print(args, file=logs, flush=True)


# data_loader
train_loader = DataLoader(dataset=train_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=False)

pretrain_loader = DataLoader(dataset=train_data,
                             batch_size=16,
                             shuffle=True,
                             num_workers=4,
                             drop_last=False)

val_loader = DataLoader(dataset=val_data,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4,
                        drop_last=False)

test_loader = DataLoader(dataset=test_data,
                         batch_size=args.batch_size,
                         num_workers=4,
                         drop_last=False)


loss_func_ce = F.nll_loss


#cuda
if torch.cuda.is_available:
    model = model.to(device)

val_loss_list = []
val_acc_list = []
test_acc_list = []

print(train_data.t, file=logs, flush=True)

noise_or_not = np.transpose(list(train_data.clean_train_labels)) == np.transpose(list(train_data.train_labels))


def get_predicted_noise_rate(loss_1_sorted, ind_1_sorted):
    loss_sorted_norm = loss_1_sorted + abs(loss_1_sorted.min())
    gaus1, gaus2, threshold = fit_gmm(loss_sorted_norm)
    print(f'Intersection Threshold: {threshold:.2f}', file=logs, flush=True)
    predicted_noise_rate = 1 - np.sum(loss_sorted_norm < threshold) / len(loss_sorted_norm)
    plot_noise_hist(loss_sorted_norm, ind_1_sorted, gaus1, gaus2, threshold, predicted_noise_rate, matrix_dir, noise_or_not)
    print(f'Predicted Noise Rate: {predicted_noise_rate:.3f}, Real Noise Rate: {args.noise_rate}', file=logs, flush=True)
    return predicted_noise_rate


def pretrain_noise_gate():
    optimizer_es = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()
    moving_loss_dic = np.zeros_like(noise_or_not)
    predictions_counters = [Counter() for _ in range(len(train_data))]

    for epoch in range(1, args.n_epoch_phase2):
        globals_loss = 0
        model.train()
        example_loss = np.zeros_like(noise_or_not,dtype=float)
        tlr = (epoch % 10 + 1) / float(10)
        lr = (1 - tlr) * 0.01 + tlr * 0.001
        for param_group in optimizer_es.param_groups:
            param_group['lr'] = lr

        for batch_x, batch_y, batch_index in pretrain_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            clean = model(batch_x)
            ce_loss = criterion(clean.log(), batch_y.long())
            preds = torch.max(clean, 1)[1]
            for idx, sample_loss, sample_pred in zip(batch_index, ce_loss, preds):
                example_loss[idx] = sample_loss.cpu().data.item()
                predictions_counters[idx.item()].update([sample_pred.cpu().item()])

            globals_loss += ce_loss.sum().cpu().data.item()
            optimizer_es.zero_grad()
            ce_loss.mean().backward()
            optimizer_es.step()
        example_loss = example_loss - example_loss.mean()
        moving_loss_dic = moving_loss_dic+example_loss

        ind_1_sorted = np.argsort(moving_loss_dic)
        loss_1_sorted = moving_loss_dic[ind_1_sorted]

        num_remember = int((1 - args.noise_rate) * len(loss_1_sorted))

        noise_accuracy = 1 - np.sum(noise_or_not[ind_1_sorted[num_remember:]]) / float(len(loss_1_sorted) - num_remember)
        mask = np.ones_like(noise_or_not, dtype=np.float32)
        mask[ind_1_sorted[num_remember:]] = 0

        top_accuracy_rm = int(0.9 * len(loss_1_sorted))
        top_accuracy = 1-np.sum(noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(len(loss_1_sorted) - top_accuracy_rm)

        print("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / len(train_data),"noise_accuracy:%f"%noise_accuracy,"top 0.1 noise accuracy:%f"%top_accuracy, file=logs, flush=True)
    predicted_noise_rate = get_predicted_noise_rate(loss_1_sorted, ind_1_sorted)
    num_remember = int((1 - predicted_noise_rate) * len(loss_1_sorted))
    is_noisy_sample = np.zeros_like(noise_or_not, dtype=np.float32)
    is_noisy_sample[ind_1_sorted[num_remember:]] = 1
    initial_t = estimate_initial_t(train_data, predictions_counters, args.num_classes, is_noisy_sample)
    print("initial T")
    print(initial_t)
    return is_noisy_sample, initial_t


def adjust_learning_rate(optimizer, epoch, max_epoch):
    if epoch < 0.25 * max_epoch:
        lr = 0.01
    elif epoch < 0.5 * max_epoch:
        lr = 0.005
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def create_optimizer_trans(trans):
    if args.dataset in ('mnist', 'cifar100'):
        return optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)
    if args.dataset in ('azimuth', 'cifar10'):
        return optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)


def model_train(epochs, is_noisy_sample, initial_t):
    is_pretrain = is_noisy_sample is None
    _milestones = milestones if not is_pretrain else None

    # optimizer and StepLR
    optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler1 = MultiStepLR(optimizer_es, milestones=_milestones, gamma=0.1)
    trans = sig_t(device, args.num_classes, initial_t=initial_t)
    trans = trans.to(device)
    optimizer_trans = create_optimizer_trans(trans)
    scheduler2 = MultiStepLR(optimizer_trans, milestones=_milestones, gamma=0.1)

    t = trans()
    est_T = t.detach().cpu().numpy()
    print(est_T, file=logs, flush=True)
    estimate_error = tools.error(est_T, train_data.t)
    print('Estimation Error: {:.2f}'.format(estimate_error), file=logs, flush=True)

    boot_checkpoint = f"{args.dataset}_{args.noise_rate}_boot_checkpoint.pth"

    if is_pretrain:
        torch.save(model.state_dict(), boot_checkpoint)

    if not is_pretrain:
        model.load_state_dict(torch.load(boot_checkpoint))

    for epoch in range(epochs):

        print('epoch {}'.format(epoch + 1), file=logs,flush=True)
        model.train()
        trans.train()

        train_loss = 0.
        train_vol_loss =0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.

        for batch_x, batch_y, batch_index in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer_es.zero_grad()
            optimizer_trans.zero_grad()


            clean = model(batch_x)

            t = trans()
            out = clean
            if not is_pretrain:
                out = mm(clean, t, format_idx(batch_index, is_noisy_sample))
                vol_loss = t.slogdet().logabsdet

            ce_loss = loss_func_ce(out.log(), batch_y.long())
            loss = ce_loss
            train_loss += loss.item()

            lr = args.lr
            if not is_pretrain:
                loss += args.lam * vol_loss
                train_vol_loss += vol_loss.item()

            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()


            loss.backward()
            optimizer_es.step()
            optimizer_trans.step()

        print('Train Loss: {:.6f}, Vol_loss: {:.6f}  Acc: {:.6f}'.format(train_loss / (len(train_data))*args.batch_size, train_vol_loss / (len(train_data))*args.batch_size, train_acc / (len(train_data))),  file=logs, flush=True)

        scheduler1.step()
        scheduler2.step()

        with torch.no_grad():
            model.eval()
            trans.eval()
            for batch_x, batch_y, batch_index in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                clean = model(batch_x)
                t = trans()

                out = clean
                if not is_pretrain:
                    out = mm(clean, t, format_idx(batch_index, is_noisy_sample))
                loss = loss_func_ce(out.log(), batch_y.long())
                val_loss += loss.item()
                pred = torch.max(out, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()

                
        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*args.batch_size, val_acc / (len(val_data))),  file=logs,flush=True)

        with torch.no_grad():
            model.eval()
            trans.eval()

            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                clean = model(batch_x)

                loss = loss_func_ce(clean.log(), batch_y.long())
                eval_loss += loss.item()
                pred = torch.max(clean, 1)[1]
                eval_correct = (pred == batch_y).sum()
                eval_acc += eval_correct.item()

            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)) * args.batch_size,
                                                          eval_acc / (len(test_data))), file=logs, flush=True)


            est_T = t.detach().cpu().numpy()
            estimate_error = tools.error(est_T, train_data.t)

            if not is_pretrain:
                matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (epoch + 1)
                np.save(matrix_path, est_T)

            print('Estimation Error: {:.2f}'.format(estimate_error), file=logs, flush=True)
            print(est_T, file=logs, flush=True)

        val_loss_list.append(val_loss / (len(val_data)))
        val_acc_list.append(val_acc / (len(val_data)))
        test_acc_list.append(eval_acc / (len(test_data)))


    val_loss_array = np.array(val_loss_list)
    val_acc_array = np.array(val_acc_list)
    model_index = np.argmin(val_loss_array)
    model_index_acc = np.argmax(val_acc_array)

    print("Final test accuracy: %f" % test_acc_list[model_index], file=logs, flush=True)
    print("Final test accuracy acc: %f" % test_acc_list[model_index_acc], file=logs, flush=True)

    print("Best epoch: %d" % model_index, file=logs, flush=True)
    if not is_pretrain:
        matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (model_index + 1)
        final_est_T = np.load(matrix_path)
        final_estimate_error = tools.error(final_est_T, train_data.t)

        matrix_path_acc = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (model_index_acc)
        final_est_T_acc = np.load(matrix_path_acc)
        final_estimate_error_acc = tools.error(final_est_T_acc, train_data.t)

        print("Final estimation error loss: %f" % final_estimate_error, file=logs, flush=True)
        print("Final estimation error loss acc: %f" % final_estimate_error_acc, file=logs, flush=True)
        print(final_est_T, file=logs, flush=True)


def mm(clean, t, is_noisy_samples):
    is_noisy_samples = torch.tensor(is_noisy_samples)
    idx_noisy = (is_noisy_samples == 1).nonzero()
    idx_clean = (is_noisy_samples == 0).nonzero()
    noisy_labels = clean[idx_noisy]
    clean_labels = clean[idx_clean]

    transformed_confidence_1 = torch.matmul(noisy_labels, t)

    out = torch.zeros_like(clean)
    out[idx_noisy] = transformed_confidence_1
    out[idx_clean] = clean_labels
    return out


def format_idx(batch_indexes, is_noisy_sample):
    _data = []
    for idx in batch_indexes:
        _data.append(is_noisy_sample[idx])
    return _data


if __name__ == '__main__':
    model_train(args.n_epoch_phase1, None, None)
    is_noisy_sample, initial_t = pretrain_noise_gate()
    model_train(args.n_epoch_phase3, is_noisy_sample, initial_t)
    logs.close()

