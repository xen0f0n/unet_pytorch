from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Precision, ConfusionMatrix, Loss, Recall
from ignite.handlers import Checkpoint, DiskSaver, ModelCheckpoint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from loguru import logger
import os
import shutil
from models.unet_orig import UNetOrig as net
from data import MembranesDataset
from utils.losses.focal_loss import BinaryFocalLossWithLogits
from utils.json_parser import parse_json


def log_this(config_params, net, criterion, optimizer):
    logger.add(f'experiments/{config_params["trainer"]["log_num"]}/info.log')
    logger.info('\n\n\n\n\n\n-----------------------------------------------------\n'
                '-----------------------------------------------------\n\n\n\n\n\n')
    logger.info(f'Model -->  {config_params["model_name"]}')
    logger.info(f'Model short description --> {net.description}')
    logger.info(
        f'input_channels --> {config_params["trainer"]["in_channels"]} \t output_classes --> {config_params["trainer"]["out_channels"]}')
    logger.info(f'Loss --> {criterion}')
    logger.info(f'Epochs --> {config_params["trainer"]["epochs"]}')
    logger.info(f'Batch Size --> {config_params["trainer"]["batch_size"]}')
    logger.info(f'Optimizer --> {optimizer}')


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

    # criterion = nn.BCEWithLogitsLoss(reduction='mean')
    criterion = BinaryFocalLossWithLogits(alpha=.75, gamma=2, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log_this(config_params, model, criterion, optimizer)

    ####################################
    # TRAINER, EVALUATOR ENGINES, METRICS
    ####################################

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    def thresholded_output_transform(output):
        y_pred, y = output
        y_pred = torch.clamp(y_pred, 0., 1.)
        y_pred = torch.round(y_pred)
        return y_pred, y

    binary_accuracy = Accuracy(thresholded_output_transform)
    precision = Precision(average=True, output_transform=thresholded_output_transform)
    recall = Recall(average=True, output_transform=thresholded_output_transform)
    precision_f1 = Precision(average=False, output_transform=thresholded_output_transform)
    recall_f1 = Recall(average=False, output_transform=thresholded_output_transform)

    F1 = precision_f1 * recall_f1 * 2 / (precision_f1 + recall_f1 + 1e-20)

    metrics = {
        'Loss': Loss(criterion),
        'acc': binary_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': F1
        }

    train_evaluator = create_supervised_evaluator(model, device=device, metrics=metrics)
    val_evaluator = create_supervised_evaluator(model, device=device, metrics=metrics)

    def log_metrics(metrics, title):
        logger.info(f"EPOCH: {trainer.state.epoch} ({title})")
        logger.info(f"{'':<10}Loss{'':<5} ----> {metrics['Loss']}")
        logger.info(f"{'':<10}Acc{'':<5} ----> {metrics['acc']}")
        logger.info(f"{'':<10}Precision{'':<5} ----> {metrics['precision']}")
        logger.info(f"{'':<10}Recall{'':<5} ----> {metrics['recall']}")
        logger.info(f"{'':<10}F1{'':<5} ----> {metrics['f1']}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        log_metrics(metrics, 'train')


    @trainer.on(Events.EPOCH_COMPLETED(every=5))
    def log_validation_results(engine):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        log_metrics(metrics, 'validation')


    ####################################
    # CHECKPOINTING HANDLER
    ####################################

    to_save = {'model': model, 'optimizer': optimizer}
    ckpt_handler = ModelCheckpoint(f'./experiments/{log_num}/checkpoints', 'epoch',
                                   global_step_transform=lambda e, _: e.state.epoch, n_saved=None, create_dir=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), ckpt_handler, to_save)


    trainer.run(train_loader, max_epochs=epochs, epoch_length=4000//batch_size)