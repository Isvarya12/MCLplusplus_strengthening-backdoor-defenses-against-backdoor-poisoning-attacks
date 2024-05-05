import random
import copy

from data_loader import get_backdoor_loader
from data_loader import get_train_loader, get_test_loader
from inversion_torch import PixelBackdoor
from utils.util import *
from models.selector import *
from config import get_arguments
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
import matplotlib.pyplot as plt
from utils import Normalizer, Denormalizer

normalize = None
import os

# Suppress the UserWarning from joblib
os.environ["LOKY_MAX_CPU_COUNT"] = "12"  # Replace "8" with the number of physical cores on your system
class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def get_norm(args):
    global normalize
    normalize = Normalizer(args.dataset)
    print("Normalization function initialized.")

def inversion(args, model, target_label, train_loader):
    print('In inversion function')
    global normalize

    if args.dataset == 'imagenet':
        shape = (3, 224, 224)
    elif args.dataset == 'tinyImagenet':
        shape = (3, 64, 64)
    else:
        shape = (3, 32, 32)

    print("Processing label: {}".format(target_label))
    backdoor = PixelBackdoor(model,
                             shape=shape,
                             batch_size=args.batch_size,
                             normalize=normalize,
                             steps=100,
                             augment=False)

    pattern = backdoor.generate(train_loader, target_label, attack_size=args.attack_size)
    #print('@ inv fn, the pattern is '+pattern)
    print('\n')
    print('before attack w trigger fn')

    attack_with_trigger(args, model, train_loader, target_label, pattern)

    print('@after attack w trigger function')

    return pattern

import cv2
import os.path as osp
def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=1.0):
    
    return cv2.GaussianBlur(image, kernel_size, sigma)

def attack_with_trigger(args, model, train_loader, target_label, pattern):
    global normalize
    denormalize = Denormalizer(args.dataset)
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm.tqdm(train_loader):
            images = images.to(device)
            trojan_images = torch.clamp(images + pattern, 0, 1)
            trojan_images = normalize(trojan_images)
            
            # Apply adaptive filtering to remove perturbations
            for i in range(trojan_images.shape[0]):
                trojan_images[i] = apply_gaussian_filter(trojan_images[i].permute(1, 2, 0).cpu().numpy()).permute(2, 0, 1)
                
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)

            _, y_pred = y_pred.max(1)
            correct += y_pred.eq(y_target).sum().item()
            total += images.size(0)

        print("Accuracy on trojaned images after adaptive filtering:", correct / total)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

# def train_step(opt, train_loader, nets, optimizer, criterions, pattern, epoch, kmeans_clusters):
#     global normalize

#     model = nets['model']
#     backup = nets['victimized_model']

#     criterionCls = criterions['criterionCls']
#     cos = torch.nn.CosineSimilarity(dim=-1)
#     mse_loss = torch.nn.MSELoss()

#     model.train()
#     backup.eval()   

#     for idx, (data, label) in enumerate(train_loader, start=1):
#         data, label = data.clone().cpu(), label.clone().cpu()

#         negative_data = copy.deepcopy(data)
#         negative_data = torch.clamp(negative_data + pattern, 0, 1)

#         data = normalize(data)
#         negative_data = normalize(negative_data)

#         feature1 = model.get_final_fm(negative_data)

#         # Apply k-means clustering to features
#         cluster_assignments = kmeans_clustering(feature1, k=kmeans_clusters)

#         # Use cluster assignments as labels
#         labels = cluster_assignments

#         posi = cos(feature1, backup.get_final_fm(data).detach())
#         logits = posi.reshape(-1, 1)

#         feature3 = backup.get_final_fm(negative_data)
#         nega = cos(feature1, feature3.detach())
#         logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

#         logits /= opt.temperature
#         cmi_loss = criterionCls(logits, labels)

#         anchor = model.get_final_fm(data)
#         positive = backup.get_final_fm(data)
#         negative = backup.get_final_fm(negative_data)

#         triplet_loss = torch.triplet_margin_loss(anchor, positive, negative, margin=1.0)

#         alpha = 1.0  
#         beta = 1.0   

#         loss = alpha * cmi_loss + beta * triplet_loss
#         loss=loss.mean()

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        

#         if idx % opt.print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Loss {loss:.4f}'.format(epoch, idx, len(train_loader), loss=loss.item()))
    
#     print("Training step completed.")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from datetime import datetime

import os
from datetime import datetime

def visualize_tsne(features, labels, title):
    concatenated_features = np.concatenate(features, axis=0)  # Concatenate the list of features
    tsne = TSNE(n_components=2, random_state=42)
    projected_features = tsne.fit_transform(concatenated_features)

    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.scatter(projected_features[:, 0], projected_features[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    
 
    
    # Save the t-SNE visualization
  
    plt.show()

def train_step(opt, train_loader, nets, optimizer, criterions, pattern, epoch, kmeans_clusters, visualize=True):
    global normalize

    model = nets['model']
    backup = nets['victimized_model']

    criterionCls = criterions['criterionCls']
    cos = torch.nn.CosineSimilarity(dim=-1)
    mse_loss = torch.nn.MSELoss()

    model.train()
    backup.eval()

    features = []  # List to store features for visualization
    labels = []    # List to store labels for visualization

    for idx, (data, label) in enumerate(train_loader, start=1):
        data, label = data.clone().cpu(), label.clone().cpu()

        negative_data = copy.deepcopy(data)
        negative_data = torch.clamp(negative_data + pattern, 0, 1)

        data = normalize(data)
        negative_data = normalize(negative_data)

        feature1 = model.get_final_fm(negative_data)

        # Apply k-means clustering to features
        cluster_assignments = kmeans_clustering(feature1, k=kmeans_clusters)

        # Use cluster assignments as labels
        labels.extend(cluster_assignments.cpu().numpy())

        # Append features for visualization
        features.append(feature1.cpu().detach().numpy())

        posi = cos(feature1, backup.get_final_fm(data).detach())
        logits = posi.reshape(-1, 1)

        feature3 = backup.get_final_fm(negative_data)
        nega = cos(feature1, feature3.detach())
        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

        logits /= opt.temperature
        cmi_loss = criterionCls(logits, cluster_assignments)

        anchor = model.get_final_fm(data)
        positive = backup.get_final_fm(data)
        negative = backup.get_final_fm(negative_data)

        triplet_loss = torch.triplet_margin_loss(anchor, positive, negative, margin=1.0)

        alpha = 1.0  
        beta = 1.0   

        loss = alpha * cmi_loss + beta * triplet_loss
        loss=loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if visualize and idx % opt.print_freq == 0:
            # Visualize t-SNE clusters
            visualize_tsne(features, labels,title='tsne')

        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}'.format(epoch, idx, len(train_loader), loss=loss.item()))

    print("Training step completed.")



from sklearn.cluster import KMeans
import torch.nn.functional as F


def kmeans_clustering(data, k, num_classes=2):
    data_flattened = data.view(data.size(0), -1).detach().cpu().numpy()
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_assignments = kmeans.fit_predict(data_flattened)
    cluster_assignments = np.clip(cluster_assignments, 0, num_classes - 1)
    return torch.tensor(cluster_assignments, dtype=torch.long, device=data.device)


def fine_tuning(opt, train_loader, nets, optimizer, criterions, pattern, epoch):
    global normalize

    model = nets['model']
    backup = nets['victimized_model']

    criterionCls = criterions['criterionCls']

    cos = nn.CosineSimilarity(dim=1).cpu()

    model.train()
    backup.eval()

    for idx, (data, label) in enumerate(train_loader, start=1):
        data, label = data.clone().cpu(), label.clone().cpu()

        negative_data = copy.deepcopy(data)
        negative_data = torch.clamp(negative_data + pattern, 0, 1)

        data = normalize(data)
        negative_data = normalize(negative_data)

        feature1 = model.get_final_fm(negative_data)

        feature2 = backup.get_final_fm(data)

        loss = -cos(feature1, feature2.detach()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Fine-tuning completed.")

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    top1_clean = AverageMeter()
    top5_clean = AverageMeter()

    top1_bad = AverageMeter()
    top5_bad = AverageMeter()
    cls_losses_bad = AverageMeter()

    snet = nets['model']

    criterionCls = criterions['criterionCls']

    snet.eval()

    clean_predictions = []
    clean_targets = []

    bad_predictions = []
    bad_targets = []

    # Test on clean data
    for idx, (img_clean, target_clean) in enumerate(test_clean_loader, start=1):
        img_clean = img_clean.to(opt.device)
        target_clean = target_clean.to(opt.device)

        with torch.no_grad():
            output_clean = snet(img_clean)

        prec1_clean, prec5_clean = accuracy(output_clean, target_clean, topk=(1, 5))
        top1_clean.update(prec1_clean.item(), img_clean.size(0))
        top5_clean.update(prec5_clean.item(), img_clean.size(0))

      
        clean_predictions.extend(output_clean.argmax(dim=1).cpu().numpy())
        clean_targets.extend(target_clean.cpu().numpy())


    if test_bad_loader is not None:
        for idx, (img_bad, target_bad) in enumerate(test_bad_loader, start=1):
            img_bad = img_bad.to(opt.device)
            target_bad = target_bad.to(opt.device)
            with torch.no_grad():
                output_bad = snet(img_bad)

                cls_loss_bad = criterionCls(output_bad, target_bad)

            prec1_bad, prec5_bad = accuracy(output_bad, target_bad, topk=(1, 5))
            cls_losses_bad.update(cls_loss_bad.item(), img_bad.size(0))
            top1_bad.update(prec1_bad.item(), img_bad.size(0))
            top5_bad.update(prec5_bad.item(), img_bad.size(0))

            # Store predictions and targets for backdoored data
            bad_predictions.extend(output_bad.argmax(dim=1).cpu().numpy())
            bad_targets.extend(target_bad.cpu().numpy())

    # Combine predictions and targets for clean and backdoored data
    all_predictions = clean_predictions + bad_predictions
    all_targets = clean_targets + bad_targets

    # Calculate metrics for the entire dataset
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    benign_accuracy = accuracy_score(all_targets, all_predictions)
    attack_success_rate = 1 - benign_accuracy
    confusion = confusion_matrix(all_targets, all_predictions)

    # Print or log the results
    print('[Clean Data] Prec@1: {:.2f}'.format(top1_clean.avg))
    print('[Clean Data] Prec@5: {:.2f}'.format(top5_clean.avg))
    print('[Backdoored Data] Prec@1: {:.2f}'.format(top1_bad.avg))
    print('[Backdoored Data] Prec@5: {:.2f}'.format(top5_bad.avg))
    print('[Overall Data] Prec@1: {:.2f}'.format((top1_clean.sum + top1_bad.sum) / (top1_clean.count + top1_bad.count)))
    print('[Overall Data] Prec@5: {:.2f}'.format((top5_clean.sum + top5_bad.sum) / (top5_clean.count + top5_bad.count)))
    print('[Overall Data] Precision: {:.4f}'.format(precision))
    print('[Overall Data] Recall: {:.4f}'.format(recall))
    print('[Overall Data] F1 Score: {:.4f}'.format(f1))
    print('[Overall Data] Benign Accuracy (BA): {:.4f}'.format(benign_accuracy))
    print('[Overall Data] Attack Success Rate (ASR): {:.4f}'.format(attack_success_rate))
    print('[Overall Data] Confusion Matrix:\n', confusion)

    print("Testing completed.")

    return [top1_clean.avg, top5_clean.avg, precision, recall, f1, benign_accuracy, attack_success_rate], [top1_bad.avg, top5_bad.avg, cls_losses_bad.avg, attack_success_rate, confusion]

# Usage of the modified test function
# ...

def cl(model, opt, pattern, train_loader, kmeans_clusters):
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    nets = {'model': model, 'victimized_model': copy.deepcopy(model)}

    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    # define loss functions
    if opt.cpu:
        criterionCls = nn.CrossEntropyLoss().cpu()
    else:
        criterionCls = nn.CrossEntropyLoss()

    print('----------- Train Initialization --------------')
    for epoch in range(0, opt.epochs):

        # train every epoch
        criterions = {'criterionCls': criterionCls}

        if epoch == 0:
            # before training test firstly
            test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch)

        print("===Epoch: {}/{}===".format(epoch + 1, opt.epochs))

        fine_defense_adjust_learning_rate(optimizer, epoch, opt.lr, opt.dataset)

        train_step(opt, train_loader, nets, optimizer, criterions, pattern, epoch, kmeans_clusters, visualize=True)

        # evaluate on testing set
        print('testing the models......')
      
        test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch+1)

    print("Training completed.")

def reverse_engineer(opt):
    print('in rev engineering fn')

    # Create an instance of the PixelBackdoor class
    backdoor_attack = PixelBackdoor(
        model=select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=True,
                           pretrained_models_path=opt.model,
                           n_classes=opt.num_class).to(opt.device),
        shape=(3, 32, 32),
        num_classes=opt.num_class,
        steps=3,
        batch_size=32,
        asr_bound=0.9,
        init_cost=1e-3,
        lr=0.1,
        clip_max=1.0,
        normalize=None,
        augment=False
    )

    print('@in if else in rev eng')
    if opt.attack_method == 'wanet':
        if opt.dataset == 'tinyImagenet':
            opt.input_height = 64
            identity_grid = torch.load('./trigger/ResNet18_tinyImagenet_WaNet_identity_grid.pth').to(opt.device)
            noise_grid = torch.load('./trigger/ResNet18_tinyImagenet_WaNet_noise_grid.pth').to(opt.device)
        else:
            identity_grid = torch.load('./trigger/WRN-16-1_CIFAR-10_WaNet_identity_grid.pth').to(opt.device)
            noise_grid = torch.load('./trigger/WRN-16-1_CIFAR-10_WaNet_noise_grid.pth').to(opt.device)
        opt.identity_grid = identity_grid
        opt.noise_grid = noise_grid

    num_classes = opt.num_class
    kmeans_clusters = 2

    print("Getting normalization function based on the dataset...")
    get_norm(args=opt)

    print('----------- DATA Initialization --------------')
    train_loader = get_train_loader(opt)
    print('\n')

    # Use the PixelBackdoor class for inversion
    pattern = backdoor_attack.generate(data_loader=train_loader, target=opt.target_label, attack_size=100, trigger_type='constant')
    cl(backdoor_attack.model, opt, pattern, train_loader, kmeans_clusters)

if __name__ == '__main__':
    print("in main")
    device = torch.device("cpu:0" if torch.cuda.is_available() else "cpu")
    print('device used %s',device)
    opt = get_arguments().parse_args()
    print('arg used %s',opt)
    random.seed(opt.seed)  # torch transforms use this seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    print('reverse engineer trigger initiated')
    kmeans_clusters = 2
    reverse_engineer(opt)
    print("end of main")
