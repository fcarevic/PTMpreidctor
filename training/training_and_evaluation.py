import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import pandas as pd
import re

def evaluate_model(model, test_dataloader, device):
    '''
    Evaluates model on provided test data (test_dataloader).
    Prints AUC of Precision/Recall Curve, average value for precision
    and AUC of ROC
    
    Parameters
    ----------
    model:
        Model for evaluation
    test_dataloader:
        Test set dataloader
    device: torch.Device
        GPU or CPU
        
    Returns
    -------
    void
    '''
    # Model Evaluation
    model.eval()
    scores = []
    y = []
    for seq_batch, _, label_batch in test_dataloader:
        seqs = []
        labels = []
        # Obtaining model predictions
        for i, seq in enumerate(seq_batch):
            seq = torch.tensor(seq).to(device)
            seq = seq.unsqueeze(dim=0)
            labels.append(label_batch[i])
            seqs.append(model(seq).squeeze())
        seqs = torch.cat(seqs)
        scores.append(torch.sigmoid(seqs).to('cpu').detach().numpy())
        labels = torch.cat(labels)
        y.append(labels.to('cpu').detach().numpy()) 
    scores = np.concatenate(scores)
    y = np.concatenate(y)

    # Calculations of Precision/Recall curve
    precision, recall, thresh = precision_recall_curve(y, scores)
    rp_auc = auc(recall, precision)
    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall Precision Curve')
    plt.show()
    print('RecallPrecisionAUC: ' + str(rp_auc))
    
    # Calculations of average precision
    ap = average_precision_score(y, scores)
    print('AveragePrecision: ' + str(ap))

    # Calculations of ROC
    fpr, tpr, roc_thresh = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('ROC Curve')
    plt.show()
    print('ROC_AUC: ' + str(roc_auc))
    
    

def train_model(model, epoch_num, opt, sequence_dataset, loss_fn,  device, file_path, writer_comment, my_collate):
    '''
    Training function for LINEAR MODELS.
    Performs training and validation on the dataset provided as parameters of the function.
    Splitting of the dataset is performed inside the function.
    
    Parameters
    ----------
    model:
        Model to be trained
        
    epoch_num: int
        Number of epochs of the training process
        
    opt: 
        Optimizer used in training
        
    sequence_dataset:
        The training dataset
    
    loss_fn:
        Loss function used in training
        
    device:
        GPU or CPU
    
    file_path: string
        Path to a file where the model will be saved. If None, then saving will not be performed
    
    writer_comment: string
        Comment for the SummaryWriter
    
    my_collate: function
        Collate function to be used
        
    Returns
    -------
    void
    '''
    
    # Splitting on train and validation sets
    slen = len(sequence_dataset)
    val_set, train_set = random_split(sequence_dataset, [int(0.2 * slen), int(slen - int(0.2 * slen))],
                                      generator=torch.Generator().manual_seed(42))

    # Creating DataLoader for training
    train_dataloader = DataLoader(train_set, batch_size=300,
                            shuffle=True, num_workers=0, collate_fn=my_collate)

    # Creating DataLoader for validation
    val_dataloader = DataLoader(val_set, batch_size=10,
                            shuffle=True, num_workers=0, collate_fn=my_collate)
    # Instantiate SummaryWriter
    writer = SummaryWriter(comment=writer_comment)
    model.train()
    for epoch in range(epoch_num):
        print("Epoch: " + str(epoch))
        for batch_num, (seq_batch, _, label_batch) in enumerate(train_dataloader):
            batch_num = epoch*len(train_dataloader) + batch_num
            opt.zero_grad()
            seqs = []
            labels = []
            for i, seq in enumerate(seq_batch):
                seq = seq.to(device)
                seq = seq.unsqueeze(dim=0)
                labels.append(label_batch[i])
                seqs.append(model(seq.double()).squeeze())
            seqs = torch.cat(seqs)
            labels = torch.cat(labels).to(device)
            loss = loss_fn(seqs, labels)
            writer.add_scalar('Train/Loss', loss.item(), batch_num)


            loss.backward()
            opt.step()

            sum_=0
            for parm in model.parameters():
                sum_+= np.abs(parm.grad.data.cpu().numpy()).sum()
            writer.add_scalar("Train/Grad", sum_, batch_num)
            if (batch_num % 10 == 0 and batch_num != 0):
                # Model Evaluation
                model.eval()
                val_loss = 0
                val_batch_cnt = 0
                TP, TN, FP, FN = 0, 0, 0, 0
                scores = []
                y = []
                for seq_batch, _, label_batch in val_dataloader:
                    seqs = []
                    labels = []
                    for i, seq in enumerate(seq_batch):
                        seq = seq.to(device)
                        seq = seq.unsqueeze(dim=0)
                        labels.append(label_batch[i])
                        seqs.append(model(seq.double()).squeeze())
                    seqs = torch.cat(seqs)
                    scores.append(torch.sigmoid(seqs).to('cpu').detach().numpy())
                    labels = torch.cat(labels)
                    y.append(labels.to('cpu').detach().numpy())
                    labels = labels.to(device)
                    val_loss += loss_fn(seqs, labels).item()
                    seqs = seqs > 0
                    labels = labels > .5
                    TP += (seqs & labels).sum()
                    FP += (seqs & ~labels).sum()
                    TN += (~seqs & ~labels).sum()
                    FN += (~seqs & labels).sum()
                    val_batch_cnt += 1
                val_loss /= val_batch_cnt
                scores = np.concatenate(scores)
                y = np.concatenate(y)
                auc = roc_auc_score(y, scores)
                precision, recall, thresh = precision_recall_curve(y, scores)
                fpr, tpr, _ = roc_curve(y,scores)

                fig1 = plt.figure()
                plt.plot(fpr, tpr)
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.title('ROC')

                fig = plt.figure()
                plt.plot(recall, precision)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Recall Precision Curve')
                writer.add_figure('Validation/ROC', fig1, batch_num)
                writer.add_figure('Validation/RecallPrecisionCurve', fig, batch_num)
                writer.add_scalar('Validation/AUC', auc, batch_num)
                writer.add_scalar('Validation/Loss', val_loss, batch_num)
                writer.add_scalar('Validation/Precision', TP/(TP+FP), batch_num)
                writer.add_scalar('Validation/Recall', TP/(TP+FN), batch_num)
                writer.add_scalar('Validation/Accuracy', (TP+TN)/(TP+FP+TN+FN), batch_num)
                model.train()
        if file_path is None:
            torch.save(model, file_path)