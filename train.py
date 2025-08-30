import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from torchmetrics.classification import  BinaryAccuracy, BinaryAUROC
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_warmup as warmup




# weight and biases configuration
def wandbinitialization(project_name, params):
    wandb.init(
        # Set the project where this run will be logged
        project=f"{project_name}",
        # Track hyperparameters and run metadata
        config=params
    )


def train(model, params, train_loader, validation_loader, device):

    start = time.time()

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999), weight_decay=params.weight_decay)

    scaler = torch.cuda.amp.GradScaler()

    if params.scheduler:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=params.epochs, eta_min=params.eta_min)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=params.warmup_period)
    
    best_auc = 0.0  
    best_model = None
    best_val_loss = float('inf')
    early_stop = 0
    
    for epoch in range(params.epochs):
        train_loss = 0.0
        val_loss = 0.0
        batch_out = []
        batch_labels = []

        # Training Phase 
        model.train()
        for batch in train_loader:
            signals, labels = batch
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                out = model(signals)
                loss = F.cross_entropy(out, labels) 
            scaler.scale(loss).backward()

            # Apply gradient clipping
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            torch.cuda.empty_cache()

        # Calculate the average loss for the epoch
        epoch_loss_train = train_loss / len(train_loader)

        # Validation phase
        model.eval()    
        with torch.no_grad():
            for batch in validation_loader:
                signals, labels = batch
                batch_labels.append(labels)              # append labels of all batches
                with torch.autocast(device_type="cuda"):
                    out_V = model(signals)
                    loss_V = F.cross_entropy(out_V, labels)
                batch_out.append(out_V)                   # append output of all batches

                # sum of all losses
                val_loss += loss_V.item()

                torch.cuda.empty_cache()
              
        # calculate average loss for each epoch
        epoch_loss_val = val_loss / len(validation_loader)

        # concatenate output and labels of each epoch
        epoch_labels = torch.cat(batch_labels)
        epoch_out = torch.cat(batch_out)    
        # calculate evaluation metrics
        # prediction probability and labels
        probabilities = torch.softmax(epoch_out, dim=1)[:,1]
  
        # Accuracy
        acc_metrics = BinaryAccuracy().to(device)
        acc = acc_metrics(probabilities, epoch_labels)
        # AUC
        auc_metrics = BinaryAUROC().to(device)
        auc = auc_metrics(probabilities, epoch_labels)
        
        # save the best model based on validation AUC
        if auc >= best_auc:
            best_auc = auc
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        # log wand b curve every 3 apochs
        if params.use_wandb and (epoch + 1) % 3 == 0:
            # weight and biases logs to 
            wandb.log({"epoch": epoch, "train loss": epoch_loss_train, "validation loss": epoch_loss_val, "validation acc": acc.item(), "AUC": auc.item(), "LR": optimizer.param_groups[0]["lr"]})

        print(epoch,'train_loss:', epoch_loss_train ,'val_loss',epoch_loss_val, 'val_acc:', round(acc.item()*100, 3) ,'val_AUC', round(auc.item()*100, 4), 'prediction', probabilities, 'target', epoch_labels )

        if params.scheduler:
            with warmup_scheduler.dampening():
                lr_scheduler.step()

        if params.apply_early_stop:
            if epoch_loss_val < best_val_loss:          #check if validation loss is decreasing
                best_val_loss = epoch_loss_val
                early_stop = 0
            else:
                early_stop += 1             # if the validation loss in inceasing w start cpunting till reaching the threshold
                
            if early_stop >= params.stop_threshold:             #stop the training based on threshold of epochs
                print('Early stopping tiggerd. Training stopped')
                break

    if params.use_wandb:   
        wandb.finish()

    end = time.time()
    total_time = end - start
    print(f'Total Execution Time: {total_time/3600}')
    print('best epoch number:', best_epoch)

    return model, best_epoch, best_model