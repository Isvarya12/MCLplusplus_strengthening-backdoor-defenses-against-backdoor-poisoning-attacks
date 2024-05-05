# from models.selector import *
# from utils.util import *
# from data_loader import get_test_loader, get_backdoor_loader
# from config import get_arguments


# def train_step(opt, train_loader, nets, optimizer, criterions, epoch):
#     cls_losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     snet = nets['snet']

#     criterionCls = criterions['criterionCls']
#     snet.train()

#     for idx, (img, target) in enumerate(train_loader, start=1):
#         if opt.cpu:
#             img = img.cpu()
#             target = target.cpu()

#         output_s = snet(img)

#         cls_loss = criterionCls(output_s, target)

#         prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
#         cls_losses.update(cls_loss.item(), img.size(0))
#         top1.update(prec1.item(), img.size(0))
#         top5.update(prec5.item(), img.size(0))

#         optimizer.zero_grad()
#         cls_loss.backward()
#         optimizer.step()

#         if idx % opt.print_freq == 0:
#             print('Epoch[{0}]:[{1:03}/{2:03}] '
#                   'cls_loss:{losses.val:.4f}({losses.avg:.4f})  '
#                   'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
#                   'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=cls_losses, top1=top1, top5=top5))


# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from sklearn.metrics import confusion_matrix

# def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
#     test_process = []

#     # Clean data evaluation
#     top1_clean = AverageMeter()
#     top5_clean = AverageMeter()

#     snet = nets['snet']
#     criterionCls = criterions['criterionCls']
#     snet.eval()

#     clean_predictions = []
#     clean_targets = []

#     for idx, (img, target) in enumerate(test_clean_loader, start=1):
#         img = img.cpu()
#         target = target.cpu()

#         with torch.no_grad():
#             output_s = snet(img)

#         prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
#         top1_clean.update(prec1.item(), img.size(0))
#         top5_clean.update(prec5.item(), img.size(0))

#         clean_predictions.extend(output_s.argmax(dim=1).cpu().numpy())
#         clean_targets.extend(target.cpu().numpy())

#     precision_clean = precision_score(clean_targets, clean_predictions, average='weighted')
#     recall_clean = recall_score(clean_targets, clean_predictions, average='weighted')
#     f1_clean = f1_score(clean_targets, clean_predictions, average='weighted')
#     ba_clean = top1_clean.avg / 100.0  # Convert percentage to decimal

#     print('[Clean Data] Prec@1: {:.2f}'.format(top1_clean.avg))
#     print('[Clean Data] Prec@5: {:.2f}'.format(top5_clean.avg))
#     print('[Clean Data] Precision: {:.4f}'.format(precision_clean))
#     print('[Clean Data] Recall: {:.4f}'.format(recall_clean))
#     print('[Clean Data] F1 Score: {:.4f}'.format(f1_clean))
#     print('[Clean Data] Benign Accuracy (BA): {:.4f}'.format(ba_clean))

#     # Backdoored data evaluation
#     cls_losses_bad = AverageMeter()
#     top1_bad = AverageMeter()
#     top5_bad = AverageMeter()

#     bad_predictions = []
#     bad_targets = []

#     for idx, (img, target) in enumerate(test_bad_loader, start=1):
#         img = img.cpu()
#         target = target.cpu()

#         with torch.no_grad():
#             output_s = snet(img)
#             cls_loss = criterionCls(output_s, target)

#         prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
#         top1_bad.update(prec1.item(), img.size(0))
#         top5_bad.update(prec5.item(), img.size(0))

#         cls_losses_bad.update(cls_loss.item(), img.size(0))

#         bad_predictions.extend(output_s.argmax(dim=1).cpu().numpy())
#         bad_targets.extend(target.cpu().numpy())

#     precision_bad = precision_score(bad_targets, bad_predictions, average='weighted')
#     recall_bad = recall_score(bad_targets, bad_predictions, average='weighted')
#     f1_bad = f1_score(bad_targets, bad_predictions, average='weighted')
#     asr = 1.0 - (top1_bad.avg / 100.0)  # Convert percentage to decimal
#     #confusion_bad = confusion_matrix(bad_targets, bad_predictions)

#     print('[Bad Data] Prec@1: {:.2f}'.format(top1_bad.avg))
#     print('[Bad Data] Prec@5: {:.2f}'.format(top5_bad.avg))
#     print('[Bad Data] Classification Loss: {:.4f}'.format(cls_losses_bad.avg))
#     print('[Bad Data] Attack Success Rate (ASR): {:.4f}'.format(asr))
#     print('[Bad Data] Precision: {:.4f}'.format(precision_bad))
#     print('[Bad Data] Recall: {:.4f}'.format(recall_bad))
#     print('[Bad Data] F1 Score: {:.4f}'.format(f1_bad))
#     #print('[Bad Data] Confusion Matrix:\n', confusion_bad)

#     # Save training progress
#     log_root = opt.log_root + '/run1.csv'
#     test_process.append((epoch, top1_clean.avg, top1_bad.avg, cls_losses_bad.avg))
#     df = pd.DataFrame(test_process, columns=("epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss"))
#     df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

#     return [top1_clean.avg, top5_clean.avg, precision_clean, recall_clean, f1_clean, ba_clean], [top1_bad.avg, top5_bad.avg, cls_losses_bad.avg, asr, precision_bad, recall_bad, f1_bad]

# # Rest of your code...

# # def train(opt):
# #     # Load models
# #     print('----------- Network Initialization --------------')
# #     student = select_model(dataset=opt.data_name,
# #                            model_name=opt.s_name,
# #                            pretrained=False,
# #                            pretrained_models_path=opt.model,
# #                            n_classes=opt.num_class).to(opt.device)
# #     print('finished student model init...')

# #     nets = {'snet': student}

# #     # initialize optimizer
# #     optimizer = torch.optim.SGD(student.parameters(),
# #                                 lr=opt.lr,
# #                                 momentum=opt.momentum,
# #                                 weight_decay=opt.weight_decay)

# #     # define loss functions
# #     if opt.cpu:
# #         criterionCls = nn.CrossEntropyLoss().cpu()
# #     else:
# #         criterionCls = nn.CrossEntropyLoss()

# #     print('----------- DATA Initialization --------------')
# #     train_loader = get_backdoor_loader(opt)
# #     test_clean_loader, test_bad_loader = get_test_loader(opt)

# #     print('----------- Train Initialization --------------')
# #     for epoch in range(1, opt.epochs):

# #         _adjust_learning_rate(optimizer, epoch, opt.lr)

# #         # train every epoch
# #         criterions = {'criterionCls': criterionCls}
# #         train_step(opt, train_loader, nets, optimizer, criterions, epoch)

# #         # evaluate on testing set
# #         print('testing the models......')
# #         acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch)

# #         # remember best precision and save checkpoint
# #         if opt.save:
# #             is_best = acc_bad[0] > opt.threshold_bad
# #             opt.threshold_bad = min(acc_bad[0], opt.threshold_bad)

# #             best_clean_acc = acc_clean[0]
# #             best_bad_acc = acc_bad[0]

# #             s_name = opt.s_name + '-' + opt.attack_method + '.pth'
# #             save_checkpoint({
# #                 'epoch': epoch,
# #                 'state_dict': student.state_dict(),
# #                 'best_clean_acc': best_clean_acc,
# #                 'best_bad_acc': best_bad_acc,
# #                 'optimizer': optimizer.state_dict(),
# #             }, is_best, os.path.join(opt.checkpoint_root, opt.dataset), s_name)


# # def _adjust_learning_rate(optimizer, epoch, lr):
# #     if epoch < 20:
# #         lr = 0.1
# #     elif epoch < 70:
# #         lr = 0.01
# #     else:
# #         lr = 0.001
# #     print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
# #     for param_group in optimizer.param_groups:
# #         param_group['lr'] = lr
# def train(opt):
#     # Load models
#     print('----------- Network Initialization --------------')
#     student = select_model(dataset=opt.data_name,
#                            model_name=opt.s_name,
#                            pretrained=False,
#                            pretrained_models_path=opt.model,
#                            n_classes=opt.num_class).to(opt.device)
#     print('Finished student model initialization...')

#     nets = {'snet': student}

#     # Initialize optimizer
#     optimizer = torch.optim.SGD(student.parameters(),
#                                 lr=opt.lr,
#                                 momentum=opt.momentum,
#                                 weight_decay=opt.weight_decay)

#     # Define loss functions
#     if opt.cpu:
#         criterionCls = nn.CrossEntropyLoss().cpu()
#     else:
#         criterionCls = nn.CrossEntropyLoss()

#     print('----------- DATA Initialization --------------')
#     train_loader = get_backdoor_loader(opt)
#     test_clean_loader, test_bad_loader = get_test_loader(opt)

#     print('----------- Train Initialization --------------')
#     for epoch in range(1, opt.epochs + 1):

#         _adjust_learning_rate(optimizer, epoch, opt.lr)

#         # Train every epoch
#         criterions = {'criterionCls': criterionCls}
#         train_step(opt, train_loader, nets, optimizer, criterions, epoch)

#         # Evaluate on testing set
#         print('Testing the models...')
#         acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch)

#         # Remember best precision and save checkpoint
#         if opt.save:
#             is_best = acc_bad[0] > opt.threshold_bad
#             opt.threshold_bad = min(acc_bad[0], opt.threshold_bad)

#             best_clean_acc = acc_clean[0]
#             best_bad_acc = acc_bad[0]

#             s_name = opt.s_name + '-' + opt.attack_method + '.pth'
#             save_checkpoint({
#                 'epoch': epoch,
#                 'state_dict': student.state_dict(),
#                 'best_clean_acc': best_clean_acc,
#                 'best_bad_acc': best_bad_acc,
#                 'optimizer': optimizer.state_dict(),
#             }, is_best, os.path.join(opt.checkpoint_root, opt.dataset), s_name)


# def _adjust_learning_rate(optimizer, epoch, lr):
#     if epoch < 20:
#         lr = 0.1
#     elif epoch < 70:
#         lr = 0.01
#     else:
#         lr = 0.001
#     print('Epoch: {}  Learning Rate: {:.4f}'.format(epoch, lr))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# # The rest of your code...

# def main():
#     # Prepare arguments
#     print("in main")
#     opt = get_arguments().parse_args()
#     print("opt attributes:", vars(opt))
#     train(opt)
#     print("end of main")
# import subprocess
# if (__name__ == '__main__'):
#     main()
from models.selector import *
from utils.util import *
from data_loader import get_test_loader, get_backdoor_loader
from config import get_arguments


def train_step(opt, train_loader, nets, optimizer, criterions, epoch):
    cls_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']

    criterionCls = criterions['criterionCls']
    snet.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cpu:
            img = img.cpu()
            target = target.cpu()

        output_s = snet(img)

        cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'cls_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=cls_losses, top1=top1, top5=top5))


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    test_process = []

    # Clean data evaluation
    top1_clean = AverageMeter()
    top5_clean = AverageMeter()

    snet = nets['snet']
    criterionCls = criterions['criterionCls']
    snet.eval()

    clean_predictions = []
    clean_targets = []

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cpu()
        target = target.cpu()

        with torch.no_grad():
            output_s = snet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1_clean.update(prec1.item(), img.size(0))
        top5_clean.update(prec5.item(), img.size(0))

        clean_predictions.extend(output_s.argmax(dim=1).cpu().numpy())
        clean_targets.extend(target.cpu().numpy())

    precision_clean = precision_score(clean_targets, clean_predictions, average='weighted')
    recall_clean = recall_score(clean_targets, clean_predictions, average='weighted')
    f1_clean = f1_score(clean_targets, clean_predictions, average='weighted')
    ba_clean = top1_clean.avg / 100.0  # Convert percentage to decimal

    print('[Clean Data] Prec@1: {:.2f}'.format(top1_clean.avg))
    print('[Clean Data] Prec@5: {:.2f}'.format(top5_clean.avg))
    print('[Clean Data] Precision: {:.4f}'.format(precision_clean))
    print('[Clean Data] Recall: {:.4f}'.format(recall_clean))
    print('[Clean Data] F1 Score: {:.4f}'.format(f1_clean))
    print('[Clean Data] Benign Accuracy (BA): {:.4f}'.format(ba_clean))

    # Backdoored data evaluation
    cls_losses_bad = AverageMeter()
    top1_bad = AverageMeter()
    top5_bad = AverageMeter()

    bad_predictions = []
    bad_targets = []

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cpu()
        target = target.cpu()

        with torch.no_grad():
            output_s = snet(img)
            cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1_bad.update(prec1.item(), img.size(0))
        top5_bad.update(prec5.item(), img.size(0))

        cls_losses_bad.update(cls_loss.item(), img.size(0))

        bad_predictions.extend(output_s.argmax(dim=1).cpu().numpy())
        bad_targets.extend(target.cpu().numpy())

    precision_bad = precision_score(bad_targets, bad_predictions, average='weighted')
    recall_bad = recall_score(bad_targets, bad_predictions, average='weighted')
    f1_bad = f1_score(bad_targets, bad_predictions, average='weighted')
    asr = 1.0 - (top1_bad.avg / 100.0)  # Convert percentage to decimal
    #confusion_bad = confusion_matrix(bad_targets, bad_predictions)

    print('[Bad Data] Prec@1: {:.2f}'.format(top1_bad.avg))
    print('[Bad Data] Prec@5: {:.2f}'.format(top5_bad.avg))
    print('[Bad Data] Classification Loss: {:.4f}'.format(cls_losses_bad.avg))
    print('[Bad Data] Attack Success Rate (ASR): {:.4f}'.format(asr))
    print('[Bad Data] Precision: {:.4f}'.format(precision_bad))
    print('[Bad Data] Recall: {:.4f}'.format(recall_bad))
    print('[Bad Data] F1 Score: {:.4f}'.format(f1_bad))
    #print('[Bad Data] Confusion Matrix:\n', confusion_bad)

    # Save training progress
    log_root = opt.log_root + '/run9.csv'
    test_process.append((epoch, top1_clean.avg, top1_bad.avg, cls_losses_bad.avg))
    df = pd.DataFrame(test_process, columns=("epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return [top1_clean.avg, top5_clean.avg, precision_clean, recall_clean, f1_clean, ba_clean], [top1_bad.avg, top5_bad.avg, cls_losses_bad.avg, asr, precision_bad, recall_bad, f1_bad]

# Rest of your code...

# def train(opt):
#     # Load models
#     print('----------- Network Initialization --------------')
#     student = select_model(dataset=opt.data_name,
#                            model_name=opt.s_name,
#                            pretrained=False,
#                            pretrained_models_path=opt.model,
#                            n_classes=opt.num_class).to(opt.device)
#     print('finished student model init...')

#     nets = {'snet': student}

#     # initialize optimizer
#     optimizer = torch.optim.SGD(student.parameters(),
#                                 lr=opt.lr,
#                                 momentum=opt.momentum,
#                                 weight_decay=opt.weight_decay)

#     # define loss functions
#     if opt.cpu:
#         criterionCls = nn.CrossEntropyLoss().cpu()
#     else:
#         criterionCls = nn.CrossEntropyLoss()

#     print('----------- DATA Initialization --------------')
#     train_loader = get_backdoor_loader(opt)
#     test_clean_loader, test_bad_loader = get_test_loader(opt)

#     print('----------- Train Initialization --------------')
#     for epoch in range(1, opt.epochs):

#         _adjust_learning_rate(optimizer, epoch, opt.lr)

#         # train every epoch
#         criterions = {'criterionCls': criterionCls}
#         train_step(opt, train_loader, nets, optimizer, criterions, epoch)

#         # evaluate on testing set
#         print('testing the models......')
#         acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch)

#         # remember best precision and save checkpoint
#         if opt.save:
#             is_best = acc_bad[0] > opt.threshold_bad
#             opt.threshold_bad = min(acc_bad[0], opt.threshold_bad)

#             best_clean_acc = acc_clean[0]
#             best_bad_acc = acc_bad[0]

#             s_name = opt.s_name + '-' + opt.attack_method + '.pth'
#             save_checkpoint({
#                 'epoch': epoch,
#                 'state_dict': student.state_dict(),
#                 'best_clean_acc': best_clean_acc,
#                 'best_bad_acc': best_bad_acc,
#                 'optimizer': optimizer.state_dict(),
#             }, is_best, os.path.join(opt.checkpoint_root, opt.dataset), s_name)


# def _adjust_learning_rate(optimizer, epoch, lr):
#     if epoch < 20:
#         lr = 0.1
#     elif epoch < 70:
#         lr = 0.01
#     else:
#         lr = 0.001
#     print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
def train(opt):
    # Load models
    print('----------- Network Initialization --------------')
    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=False,
                           pretrained_models_path=opt.model,
                           n_classes=opt.num_class).to(opt.device)
    print('Finished student model initialization...')

    nets = {'snet': student}

    # Initialize optimizer
    optimizer = torch.optim.SGD(student.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    # Define loss functions
    if opt.cpu:
        criterionCls = nn.CrossEntropyLoss().cpu()
    else:
        criterionCls = nn.CrossEntropyLoss()

    print('----------- DATA Initialization --------------')
    train_loader = get_backdoor_loader(opt)
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    print('----------- Train Initialization --------------')
    for epoch in range(1, opt.epochs + 1):

        _adjust_learning_rate(optimizer, epoch, opt.lr)

        # Train every epoch
        criterions = {'criterionCls': criterionCls}
        train_step(opt, train_loader, nets, optimizer, criterions, epoch)

        # Evaluate on testing set
        print('Testing the models...')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch)

        # Remember best precision and save checkpoint
        if opt.save:
            is_best = acc_bad[0] > opt.threshold_bad
            opt.threshold_bad = min(acc_bad[0], opt.threshold_bad)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]

            s_name = opt.s_name + '-' + opt.attack_method + '.pth'
            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, os.path.join(opt.checkpoint_root, opt.dataset), s_name)


def _adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 20:
        lr = 0.1
    elif epoch < 70:
        lr = 0.01
    else:
        lr = 0.001
    print('Epoch: {}  Learning Rate: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# The rest of your code...

def main():
    # Prepare arguments
    print("in main")
    opt = get_arguments().parse_args()
    print("opt attributes:", vars(opt))
    train(opt)
    print("end of main")
import subprocess
if (__name__ == '__main__'):
    main()
   