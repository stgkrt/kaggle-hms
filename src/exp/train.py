import gc
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

import wandb
from src.data.dataloader import CustomDataset
from src.kaggle_metrics.kaggle_kl_div import competition_score
from src.log_utils import AverageMeter, timeSince
from src.model.get_model import CustomModel
from src.model.losses import KLDivBCEWithLogitsLoss, get_criterion, mixup_data
from src.model.optimizers import build_optimizer
from src.model.schedulers import get_scheduler


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, CFG):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = time.time()
    global_step = 0
    for step, batch in enumerate(train_loader):
        spectrogram = batch["spectrogram"].to(CFG.device)
        labels = batch["labels"].to(CFG.device)
        batch_size = labels.size(0)
        mixed_X, y_a, y_b, lam = mixup_data(spectrogram, labels, CFG, CFG.device)
        new_criterion = get_criterion(CFG, criterion)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(mixed_X)
            # loss = criterion(F.log_softmax(y_preds, dim=1), labels)
            loss = new_criterion(F.log_softmax(y_preds, dim=1), y_a, y_b, lam)
            # predにsigmoidをかける
            # loss = new_criterion(torch.sigmoid(y_preds), y_a, y_b, lam)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()

        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                )
            )
        if CFG.wandb:
            wandb.log(
                {
                    f"[fold{fold}] loss": losses.val,
                    f"[fold{fold}] lr": scheduler.get_lr()[0],
                }
            )
    return losses.avg


def valid_fn(valid_loader, model, criterion, CFG):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = time.time()
    for step, batch in enumerate(valid_loader):
        spectrogram = batch["spectrogram"].to(CFG.device)
        labels = batch["labels"].to(CFG.device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(spectrogram)
            loss = criterion(F.log_softmax(y_preds, dim=1), labels)
            # loss = criterion(torch.sigmoid(y_preds), labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(nn.Softmax(dim=1)(y_preds).to("cpu").numpy())
        # sigmoidをかけたものをpredsに追加(合計は1になるように正規化)
        # sigmoid_preds = torch.sigmoid(y_preds).to("cpu").numpy()
        # sigmoid_preds = sigmoid_preds / sigmoid_preds.sum(axis=1, keepdims=True)
        # preds.append(sigmoid_preds)
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def train_loop(folds, fold, directory, LOGGER, CFG):
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    if CFG.stage1_pop1:
        # train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
        valid_ids_path = f"/kaggle/input/valid_eeg_ids_fold{fold}.yaml"
        with open(valid_ids_path, "r") as file:
            valid_ids = yaml.load(file, Loader=yaml.FullLoader)
        train_folds = folds[~folds["eeg_id"].isin(valid_ids)].reset_index(drop=True)
    else:
        train_folds = folds[
            (folds["fold"] != fold) & (folds["total_evaluators"] >= 10)
        ].reset_index(drop=True)
    # valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    valid_folds = folds[folds["eeg_id"].isin(valid_ids)].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    LOGGER.info(f"train data num: {len(train_folds)}")
    LOGGER.info(f"valid data num: {len(valid_folds)}")

    train_dataset = CustomDataset(train_folds, CFG, augment=True, mode="train")
    valid_dataset = CustomDataset(valid_folds, CFG, augment=False, mode="train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG)
    if CFG.stage2_pop2:
        model_weight = (
            CFG.POP_1_DIR
            + f"{CFG.model_name}_fold{fold}_best_{CFG.exp_name}_stage1.pth"
        )
        checkpoint = torch.load(model_weight, map_location=CFG.device)
        model.load_state_dict(checkpoint["model"])
    # CPMP: wrap the model to use all GPUs
    model.to(CFG.device)
    if CFG.t4_gpu:
        model = nn.DataParallel(model)
    optimizer = build_optimizer(CFG, model)
    scheduler = get_scheduler(optimizer, CFG, train_loader)

    # ====================================================
    # loop
    # ====================================================
    if CFG.loss == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif CFG.loss == "KLDivLoss":
        criterion = nn.KLDivLoss(reduction="batchmean")
    elif CFG.loss == "KLDivBCEWithLogitsLoss":
        criterion = KLDivBCEWithLogitsLoss()
    else:
        print("loss function is not implemented. supported loss functions are:")
        print("BCEWithLogitsLoss or KLDivLoss or KLDivBCEWithLogitsLoss")
        raise NotImplementedError

    best_score = np.inf

    for epoch in range(CFG.epochs):
        start_time = time.time()
        # train
        avg_loss = train_fn(
            fold,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            CFG,
        )

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, CFG)
        pred_df = pd.DataFrame(predictions, columns=CFG.target_cols)
        target_df = pd.DataFrame(valid_labels, columns=CFG.target_cols)
        score = competition_score(target_df, pred_df)

        if not CFG.batch_scheduler:
            scheduler.step()
        elapsed = time.time() - start_time
        log_message = f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}"
        log_message += f", avg_val_loss: {avg_val_loss:.4f}"
        log_message += f", score : {score:.4f} time: {elapsed:.0f}s"
        LOGGER.info(log_message)

        if CFG.wandb:
            wandb.log(
                {
                    f"[fold{fold}] epoch": epoch + 1,
                    f"[fold{fold}] avg_train_loss": avg_loss,
                    f"[fold{fold}] avg_val_loss": avg_val_loss,
                    f"[fold{fold}] score": score,
                }
            )

        if best_score > score:
            best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best valid score: {score:.4f} Model")
            # CPMP: save the original model.
            # It is stored as the module attribute of the DP model.
            if CFG.stage1_pop1:
                state_dict = (
                    model.module.state_dict() if CFG.t4_gpu else model.state_dict()
                )
                model_name = CFG.model_name
                model_name += f"_fold{fold}_best_{CFG.exp_name}_stage1.pth"
                model_path = os.path.join(directory, model_name)
                torch.save(
                    {"model": state_dict, "predictions": predictions},
                    model_path,
                )
            else:
                state_dict = (
                    model.module.state_dict() if CFG.t4_gpu else model.state_dict()
                )
                model_name = CFG.model_name
                model_name += f"_fold{fold}_best_{CFG.exp_name}_stage2.pth"
                model_path = os.path.join(directory, model_name)
                torch.save(
                    {"model": state_dict, "predictions": predictions},
                    model_path,
                )

    if CFG.stage1_pop1:
        model_path = os.path.join(
            directory, f"{CFG.model_name}_fold{fold}_best_{CFG.exp_name}_stage1.pth"
        )
        predictions = torch.load(
            model_path,
            map_location=torch.device("cpu"),
        )["predictions"]
    else:
        model_path = os.path.join(
            directory, f"{CFG.model_name}_fold{fold}_best_{CFG.exp_name}_stage2.pth"
        )
        predictions = torch.load(
            model_path,
            map_location=torch.device("cpu"),
        )["predictions"]
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions
    valid_folds[CFG.target_cols] = valid_labels
    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds, best_score
