import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryConfusionMatrix, MatthewsCorrCoef, BinaryF1Score
import sklearn
from sklearn import metrics 
from sklearn.metrics import roc_curve, auc


def test(model, test_loader, device, save_dir):
    # Test phase
    test_loss = 0.0
    batch_out_test = []
    batch_labels_test = []

    model.eval()    
    with torch.no_grad():
        for batch in test_loader:
            signals, labels = batch
            batch_labels_test.append(labels)              # append labels of all batches
            with torch.autocast(device_type="cuda"):
                out_T = model(signals)
                loss_T = F.cross_entropy(out_T, labels)
            batch_out_test.append(out_T)                       # append output of all batches

            # sum of all losses
            test_loss += loss_T.item()

            torch.cuda.empty_cache()

    # calculate average loss for each epoch
    epoch_loss_test = test_loss / len(test_loader)

    # concatenate output and labels of each epoch
    epoch_labels_test = torch.cat(batch_labels_test)
    epoch_out_test = torch.cat(batch_out_test) 

    # calculate evaluation metrics
    # prediction probability and labels
    _ , predicted_labels = torch.max(epoch_out_test, dim=1)      # use soft max for prediction labels
    probabilities = torch.softmax(epoch_out_test, dim=1)[:, 1]
   
    # Accuracy
    acc_metrics = BinaryAccuracy().to(device)
    acc = acc_metrics(probabilities, epoch_labels_test)
    # AUC
    auc_metrics = BinaryAUROC().to(device)
    auc = auc_metrics(probabilities, epoch_labels_test)
 
    confmat = BinaryConfusionMatrix().to(device)
    confmat(predicted_labels, epoch_labels_test)
    conf_matrix = confmat.compute()
    TN = conf_matrix[0, 0].item()
    FP = conf_matrix[0, 1].item()
    FN = conf_matrix[1, 0].item()
    TP = conf_matrix[1, 1].item()
    print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)

    f1score_metrics = BinaryF1Score().to(device)
    f1score = f1score_metrics(predicted_labels, epoch_labels_test)

    print('test_loss:', epoch_loss_test , 'val_acc:', round(acc.item()*100, 3) ,'val_AUC', round(auc.item()*100, 3), 'prediction', probabilities, 'target', epoch_labels_test, 'f1score', f1score.item())

    # Plot test AUC
    fpr, tpr, thresholds  = roc_curve(epoch_labels_test.cpu(), np.ravel(probabilities.cpu()))    #compare annotaion file labels with classifier prediction result 
    sensitivity = tpr
    specificity = 1-fpr

    plt.figure(figsize=(10,5))
    plt.plot(specificity, sensitivity, marker='.')
    plt.title(f'ROC Curve')
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    auc_value = metrics.auc(fpr, tpr)
    plt.savefig(f'{save_dir}/model_roc.png')


    # save prediction and target of test set in a dataframe
    test_dict = {'prediction': probabilities.cpu().numpy(), 'pred_labels':predicted_labels.cpu().numpy(), 'target': epoch_labels_test.cpu().numpy()}
    testdf = pd.DataFrame(test_dict)
    testdf.to_csv(f'{save_dir}/model_pred.csv', index=False)

