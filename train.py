from loguru import logger
import matplotlib.pyplot as plt
import os
import shutil
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pytorch_lightning.metrics.functional import accuracy, precision, recall, f1_score
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from data import MembranesDataset
from models.unet_orig import UNetOrig as net
from utils.json_parser import parse_json

if __name__ == "__main__":
    ####################################
    # PARSE CONFIG FILE
    ####################################

    config_params = parse_json('config.json')

    log_num = config_params["trainer"]["log_num"]
    input_channels = config_params["trainer"]["in_channels"]
    output_classes = config_params["trainer"]["out_channels"]
    device = config_params["trainer"]["device"]
    epochs = config_params["trainer"]["epochs"]
    lr = config_params["trainer"]["learning_rate"]
    batch_size = config_params["trainer"]["batch_size"]

    os.makedirs(os.path.join('experiments', f'{log_num}'), exist_ok=True)
    shutil.copy('config.json', os.path.join(f'./experiments/{log_num}', 'config.json'))

    ####################################
    # LOAD DATASET
    ####################################

    train_set = MembranesDataset(
        config_params['data_loader']['args']['train_image_paths'],
        config_params['data_loader']['args']['train_mask_paths'],
        mode='train'
    )
    val_set = MembranesDataset(
        config_params['data_loader']['args']['val_image_paths'],
        config_params['data_loader']['args']['val_mask_paths'],
        mode='val'
    )

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=31)
    # val_loader = DataLoader(val_set, batch_size=1, num_workers=31)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    ####################################
    # LOAD MODEL, SET LOSS, OPTIMIZER
    ####################################

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = net(n_channels=input_channels, n_classes=output_classes).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.add(f'info.log')

    history = {
        'train_loss': [],
        'train_acc': [],
        'train_recall': [],
        'train_f1': [],
        'train_cm': [],
        'val_loss': [],
        'val_acc': [],
        'val_recall': [],
        'val_f1': [],
        'val_cm': [],
    }

    # TRAIN LOOP
    for epoch in range(1, epochs + 1):

        model.train()

        train_loss = 0  # summation of loss for every batch
        train_acc = 0  # summation of accuracy for every batch
        train_recall = 0
        train_precision = 0
        train_f1 = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            train_loss += loss.item()
            train_acc += accuracy(y_batch, y_pred)  # calculate accuracy (on this single batch)
            train_recall += recall(y_batch, y_pred)
            train_precision += train_precision(y_batch, y_pred)
            train_f1 += f1_score(y_batch, y_pred)

            loss.backward()
            optimizer.step()

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc / len(train_loader))
        history['train_recall'].append(train_recall / len(train_loader))
        history['train_precision'].append(train_precision / len(train_loader))
        history['train_f1'].append(train_f1 / len(train_loader))

        logger.info(f"EPOCH: {e} (training)")
        logger.info(f"{'':<10}Loss{'':<5} ----> {train_loss / len(train_loader):.5f}")
        logger.info(f"{'':<10}Accuracy{'':<1} ----> {train_acc / len(train_loader):.3f}")
        logger.info(f"{'':<10}Recall{'':<1} ----> {train_recall / len(train_loader):.3f}")
        logger.info(f"{'':<10}Precision{'':<1} ----> {train_precision / len(train_loader):.3f}")
        logger.info(f"{'':<10}F1{'':<1} ----> {train_f1 / len(train_loader):.3f}")

        if epoch % 5 == 0:  # if the number of epoch is divided by 5 do the validation

            model.eval()

            val_loss = 0
            val_acc = 0
            val_recall = 0
            val_precision = 0
            val_f1 = 0

            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # send values to device (GPU)
                y_val_pred = model(X_batch)

                val_loss += criterion(y_val_pred, y_batch)
                val_acc += accuracy(y_val_pred, y_batch)
                val_recall += recall(y_batch, y_val_pred)
                val_precision += precision(y_batch, y_val_pred)
                val_f1 += f1_score(y_batch, y_val_pred)

            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc / len(val_loader))
            history['val_recall'].append(val_recall / len(val_loader))
            history['val_precision'].append(val_precision / len(val_loader))
            history['val_f1'].append(val_f1 / len(val_loader))

            logger.info(f"EPOCH: {e} (validation)")
            logger.info(f"{'':<10}Loss{'':<5} ----> {val_loss / len(val_loader):.5f}")
            logger.info(f"{'':<10}Accuracy{'':<1} ----> {val_acc / len(val_loader):.3f}")
            logger.info(f"{'':<10}Recall{'':<1} ----> {val_recall / len(val_loader):.3f}")
            logger.info(f"{'':<10}Precision{'':<1} ----> {val_precision / len(val_loader):.3f}")
            logger.info(f"{'':<10}F1{'':<1} ----> {val_f1 / len(val_loader):.3f}")

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

plt.plot(list(range(1, epochs + 1)), history['train_loss'], label='Training Loss')
plt.plot(list(range(5, epochs + 1, 5)), history['val_loss'], label='Validation Loss')
plt.legend(loc="upper right")
plt.show()
