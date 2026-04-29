# TODO:Evaluate model accuracy
import csv
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from torch import nn

from ttach.base import Merger


class SegmentationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (segmentation model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): segmentation model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `mask`
    """

    def __init__(
        self,
        model: nn.Module,
        transforms,
        merge_mode: str = "mean",
        output_mask_key = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_mask_key

    def forward(
        self, image: torch.Tensor, *args
    ):
        merger = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_mask(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        """update confusion matrix"""
        for p, t in zip(preds, labels):
            self.matrix[int(p), int(t)] += 1

    def summary(self, classes):
        mIoU, mF1 = 0, 0
        table = PrettyTable()
        table.field_names = ["Per-class", "IoU (%)", "F1-score (%)"]
        for i in range(classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            IoU = TP / (TP + FP + FN) if TP + FN + FP != 0 else 1.
            P = TP / (TP + FP) if TP + FP != 0 else 1.
            R = TP / (TP + FN) if TP + FN != 0 else 1.
            F1 = 2 * P * R / (P + R) if TP + FN + FP != 0 else 1.
            table.add_row([self.labels[i], round(IoU * 100, 3), round(F1 * 100, 3)])
            mIoU += IoU
            mF1 += F1
        logging.info(f'Classes acc:\n{table}')
        mIoU = round(mIoU * 100 / classes, 3)
        mF1 = round(mF1 * 100 / classes, 3)
        OA = round(np.diag(self.matrix).sum() / self.matrix[:classes, :classes].sum() * 100, 3)
        logging.info(f'\tmIoU: {mIoU} %')
        logging.info(f'\tMacro F1-score: {mF1} %')
        logging.info(f'\tOA: {OA} %')
        return mIoU, mF1, OA

    def plot(self, save_path, n_classes):
        """plot confusion matrix"""
        matrix = self.matrix / (self.matrix.sum(0).reshape(1, n_classes) + 1e-8)  # normalize by classes
        # matrix = self.matrix / self.matrix.sum()
        plt.figure(figsize=(12, 9), tight_layout=True)
        sns.set(font_scale=1.0 if n_classes < 50 else 0.8)
        sns.heatmap(data=matrix,
                    annot=n_classes < 30,
                    annot_kws={"size": 12},
                    cmap='Blues',
                    fmt='.3f',
                    square=True,
                    linewidths=.16,
                    linecolor='white',
                    xticklabels=self.labels,
                    yticklabels=self.labels)

        plt.xlabel('True', size=16, color='purple')
        plt.ylabel('Predicted', size=16, color='purple')
        plt.title('Normalized confusion matrix', size=20, color='blue')
        plt.savefig(save_path + ".png", dpi=150)
        plt.close()

        # save as .csv file
        with open(save_path + ".csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + self.labels)
            for i, row in enumerate(self.matrix):
                writer.writerow([self.labels[i]] + list(row))


def eval_net(net, loader, device, json_label_path, n_img, bg):
    assert os.path.exists(json_label_path), 'Cannot find {} file'.format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for label, _ in class_indict.items()]
    n_classes = 2 if net.n_classes == 1 else net.n_classes
    confusion = ConfusionMatrix(num_classes=n_classes, labels=labels)
    evaluator = Evaluator(num_class=n_classes)
    evaluator.reset()
    net.eval()

    import ttach as tta
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            # tta.Rotate90(angles=[90]),
            tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
        ]
    )

    model = tta.SegmentationTTAWrapper(net, transforms)

    with torch.no_grad():
        for batch in loader:
            images = []
            for n in range(1, 1 + n_img):
                imgs = batch[f'image{n}']
                imgs = imgs.to(device=device, dtype=torch.float32)
                images.append(imgs)

            true_masks = batch['mask']
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            pred_masks = model(images[0])
            images.clear()

            for true_mask, pred in zip(true_masks, pred_masks):
                true_mask = true_mask.squeeze(dim=0).cpu().numpy()

                if net.n_classes > 1:
                    pred = F.softmax(pred, dim=0)
                    pred = torch.argmax(pred, dim=0)
                    pred = pred.cpu().numpy()
                else:
                    pred = (pred > 0.5).int().squeeze(dim=0).cpu().numpy()

                ##
                evaluator.add_batch(pre_image=pred, gt_image=true_mask)
                ##

                true_mask, pred = true_mask.flatten(), pred.flatten()
                # ignore background pixels
                if not bg:
                    mask = true_mask != (net.n_classes - 1)
                    pred, true_mask = pred[mask], true_mask[mask]
                    n_classes = net.n_classes - 1

                confusion.update(pred.astype(np.uint8), true_mask.astype(np.uint8))
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()
        for class_name, class_iou, class_f1 in zip(['ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter'],
                                                   iou_per_class, f1_per_class):
            logging.info('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
        logging.info('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA))
        mIoU, mF1, OA = confusion.summary(n_classes)
    return mIoU, mF1, OA, confusion, n_classes
