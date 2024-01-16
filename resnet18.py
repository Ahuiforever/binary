# !/usr/bin/env/binary python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 14:45
# @Author  : Ahuiforever
# @File    : resnet18.py
# @Software: PyCharm
"""
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)

"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import trange, tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from smoke import Smoke
from wjh.utils import PathChecker, LogWriter, ModelSaver


if __name__ == "__main__":

    band = 750

    # ? 实例化路径检查器
    pc = PathChecker()
    writer_path = f"{band}_tensorboard"
    pc(path=writer_path, del_=False)  # % del_=False if resume is True

    # ? 日志板
    writer = SummaryWriter(writer_path)
    # * tensorboard --logdir="resnet binary/450_tensorboard" --port=6006

    # ? 日志文件
    log = LogWriter("swin_v2_t.txt")  # % Change this

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # resnet18 = models.resnet18(
    #     weights=None
    #     # weights=models.ResNet18_Weights.IMAGENET1K_V1
    # )
    weights = None
    # weights = models.ResNet18_Weights.IMAGENET1K_V1
    # weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
    # resnet18 = models.resnet18(weights=weights)
    # resnet18 = models.resnext50_32x4d(weights=weights)
    # resnet18 = models.vgg16_bn(weights=weights)
    # resnet18 = models.inception_v3(weights=weights, init_weights=True)
    resnet18 = models.swin_v2_t(weights=weights)

    # __swin_v2_t modified =============================================================================================
    resnet18.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
    resnet18.head = nn.Linear(resnet18.head.in_features, out_features=1, bias=True)
    # __================================================================================================================

    # __inception_v3 modified ==========================================================================================
    # resnet18.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    # resnet18.fc = nn.Linear(resnet18.fc.in_features, 1)
    # __================================================================================================================

    # __vgg16_bn modified ==============================================================================================
    # resnet18.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    # num_features = resnet18.classifier[0].in_features
    # resnet18.classifier = nn.Sequential(
    #     nn.Linear(num_features, 4096),
    #     nn.ReLU(True),
    #     nn.Dropout(),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(True),
    #     nn.Dropout(),
    #     nn.Linear(4096, 1),  # Modify num_classes as needed
    # )
    # __================================================================================================================

    # __resnet18 modified =============================================================================================
    # resnet18.conv1 = nn.Conv2d(
    #     1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    # )
    # num_features = resnet18.fc.in_features
    # resnet18.fc = nn.Sequential(
    #     nn.Linear(num_features, 1),
    #     # nn.Sigmoid(),
    # )
    # __================================================================================================================
    resnet18.to(device)

    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(resnet18.parameters(), lr=0.001, weight_decay=1e-2)
    # optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.01, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if band == 450:
        mean = [3.2626e-05]
        std = [2.3301e-05]
    elif band == 540:
        mean = [0.0002]
        std = [0.0001]
    elif band == 750:
        mean = [0.0008]
        std = [0.0004]
    elif band == 900:
        mean = [0.0004]
        std = [0.0002]
    elif band == 950:
        mean = [0.0002]
        std = [7.4284e-05]
    else:
        raise ValueError("Invalid band.")

    transformation = transforms.Compose(
        [
            transforms.Resize((299, 299), antialias=True),
            # transforms.RandomResizedCrop(480),
            # * 232resnet18 | 232resnext50 | 232vgg16_bn | 299inception_v3
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                10
            ),  # You can adjust the angle of rotation as needed
            # transforms.ToTensor(),  # ! Normalize the image array number.
            # transforms.Normalize(mean=[0.9391], std=[0.2017]),
            # transforms.Normalize(mean=[0.2078], std=[0.2318])  # ! Standardize the distribution of image array.
            transforms.Normalize(mean=mean, std=std),
            # transforms.Normalize(mean=[3.2626e-05], std=[2.3301e-05]),  # ` 450
            # transforms.Normalize(mean=[0.0002], std=[0.0001]),  # ` 540
            # transforms.Normalize(mean=[0.0008], std=[0.0004]),  # ` 750
            # transforms.Normalize(mean=[0.0004], std=[0.0002]),  # ` 900
            # transforms.Normalize(mean=[0.0002], std=[7.4284e-05]),  # ` 950
            # % ! Calculate from my own dataset
        ]
    )

    # transformation = weights.transforms()
    # log(transformation)
    # % 450
    train_set = Smoke(
        root_dir=rf"X:\Work\try00\{band}\images\train", transform=transformation, show=False
    )
    dev_set = Smoke(
        root_dir=rf"X:\Work\try00\{band}\images\val", transform=transformation, show=False
    )

    train_loader = DataLoader(
        dataset=train_set, batch_size=32, shuffle=True, num_workers=3, drop_last=False
    )
    dev_loader = DataLoader(
        dataset=dev_set, batch_size=32, shuffle=True, num_workers=3, drop_last=False
    )

    # __Calculate the mean and standard deviation ======================================================================
    # mean = 0.0
    # std = 0.0
    # total_samples = 0
    #
    # for images, labels in tqdm(train_loader, ncols=100):
    #     batch_size = images.size(0)
    #     images = images.view(batch_size, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    #     std += images.std(2).sum(0)
    #     total_samples += batch_size
    #
    # mean /= total_samples
    # std /= total_samples
    #
    # log(f"950 -- Mean of the train set: {mean}, Standard deviation of the train set: {std}", True)
    #
    # mean = 0.0
    # std = 0.0
    # total_samples = 0
    # for images, labels in tqdm(dev_loader, ncols=100):
    #     batch_size = images.size(0)
    #     images = images.view(batch_size, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    #     std += images.std(2).sum(0)
    #     total_samples += batch_size
    #
    # mean /= total_samples
    # std /= total_samples
    #
    # log(f"Mean of the train set: {mean}, Standard deviation of the train set: {std}", True)
    # __================================================================================================================

    # __Count the Positive and Negative ================================================================================
    true = 0
    samples = 0
    for images, labels in tqdm(train_loader, ncols=100):
        samples += images.size(0)
        true += labels.sum(0)

    log(f'Train Positive {true}, Train Negative {samples-true}', printf=True)

    true = 0
    samples = 0
    for images, labels in tqdm(dev_loader, ncols=100):
        samples += images.size(0)
        true += labels.sum(0)

    log(f'Val Positive {true}, Val Negative {samples-true}', printf=True)
    # __================================================================================================================

    num_epochs = 101

    # ? 实例化模型保存
    ms = ModelSaver(
        model=resnet18,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_interval=1,
        max_checkpoints_to_keep=10,
        checkpoint_dir=f"./{band}_mlogs",  # % 450
    )

    resume = False  # % Resume from the last.pth

    if resume:
        checkpoint = torch.load("result02_inceptinov3/450_mlogs/VGG_92_0.1581_93.7057.pth")
        resnet18.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        break_epoch = checkpoint["epoch"]
        log(f'Resuming from epoch {break_epoch}th', printf=True)
    else:
        break_epoch = -1

    # i = 0  # % temp

    for epoch in trange(
        break_epoch + 1, num_epochs, desc="Epochs", leave=False, position=0, ncols=100
    ):
        resnet18.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.
        for images, labels in tqdm(
            train_loader, desc="Training", leave=False, position=1, ncols=100
        ):
            images, labels = images[:, 0:1, :, :].to(device), labels.to(device)
            outputs = resnet18(images).squeeze(axis=-1)
            # outputs = resnet18(images).logits.squeeze(axis=-1)  # inception v3
            # log(outputs, labels)
            # _, predicted = torch.max(outputs.data, dim=1)
            train_total += labels.size(0)
            binary_predictions = (outputs >= 0.5).float()
            train_correct += torch.eq(binary_predictions, labels).sum().item()
            # train_correct += torch.eq(predicted, labels).sum().item()
            # outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)
            loss = criterion(outputs, labels)
            # i += 1  # % temp
            # writer.add_scalar("training loss", loss.item(), i)  # % temp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_accuracy = 100 * train_correct / train_total
        train_loss /= train_loader.__len__()
        scheduler.step()

        val_correct = 0
        val_total = 0
        val_loss = 0.
        recall = 0.
        precision = 0.
        false_alarm = 0.
        _confusion_matrix = {'Band': band, 'Epoch': epoch, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}  # % 540
        misclassified = []
        resnet18.eval()
        with torch.no_grad():
            for images, labels in tqdm(
                dev_loader, desc="Validating", leave=False, position=1, ncols=100
            ):
                images, labels = images[:, 0:1, :, :].to(device), labels.to(device)
                val_total += labels.size(0)
                outputs = resnet18(images).squeeze(axis=-1)
                # ? max_values, max_indices = torch.max(tensor, dim=0)
                # _, predicted = torch.max(outputs.data, dim=1)
                binary_predictions = (outputs >= 0.5).float()
                comparison = torch.eq(binary_predictions, labels)
                val_correct += comparison.sum().item()

                # * Confusion Matrix
                indices_t = torch.where(comparison.int() == 1)[0]
                indices_f = torch.where(comparison.int() == 0)[0]
                tp = (binary_predictions[indices_t] == 1.).sum().item()
                tn = (binary_predictions[indices_t] == 0.).sum().item()
                fp = (binary_predictions[indices_f] == 1.).sum().item()
                fn = (binary_predictions[indices_f] == 0.).sum().item()
                _confusion_matrix['TP'] += tp
                _confusion_matrix['TN'] += tn
                _confusion_matrix['FP'] += fp
                _confusion_matrix['FN'] += fn

                recall += tp / (tp + fn + 1e-7)
                precision += tp / (tp + fp + 1e-7)
                false_alarm += fp / (fp + tn + 1e-7)

                # correct += torch.eq(prediction, labels).sum().item()
                val_loss += criterion(outputs, labels).item()

        val_accuracy = 100 * val_correct / val_total
        val_loss /= dev_loader.__len__()
        val_recall = 100 * recall / dev_loader.__len__()
        val_precision = 100 * precision / dev_loader.__len__()
        # print(val_precision, val_recall)
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-7)
        val_false_alarm = 100 * false_alarm / dev_loader.__len__()
        log(f"Confusion matrix : {_confusion_matrix}")

        # tqdm.write(
        #     "\n{}--Accuracy on the dev set: {:.2f}%".format(datetime.now(), val_accuracy)
        # )

        log(epoch, train_accuracy, train_loss, val_accuracy, val_loss)

        # writer.add_scalar('val_loss', val_loss, epoch)
        # writer.add_scalar('val_accuracy', val_accuracy, epoch)
        writer.add_scalars(
            "loss", {"train": train_loss, "val": val_loss}, epoch
        )
        writer.add_scalars(
            "accuracy", {"train": train_accuracy, "val": val_accuracy}, epoch
        )
        writer.add_scalars(
            "Val Index",
            {
                "Recall": val_recall,
                "Precision": val_precision,
                "F1": val_f1,
                "False Alarm": val_false_alarm}, epoch
        )

        torch.cuda.empty_cache()

        ms(epoch=epoch, val_loss=val_loss, val_accuracy=val_accuracy)

    print("Done")
