import utils.utils
import torch.nn.functional as F
import torch
import torch.nn as nn

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, obj='bbox'):
        super().__init__()
        self.obj = obj

    def forward(self, pred_bboxes, target_bboxes, target_scores):
        """IoU loss."""
        N_pos = torch.clamp(target_scores.sum(), min=1)
        
        if self.obj == 'bbox':
            iou = utils.utils.bbox_iou(pred_bboxes, target_bboxes, align=True, DIoU = True, CIoU = True)
        else:#cbbox
            iou = utils.utils.cbbox_iou(pred_bboxes, target_bboxes, align=True, DIoU = True)
            #iou, empty, missed = utils.utils.cbbox_iou(pred_bboxes, target_bboxes, align=True, DIoU = True)
        
        loss_iou = (1.0 - iou)
        loss_iou = weight_reduce_loss(loss_iou, avg_factor=N_pos)
        #loss_empty = (1.0 - empty)
        #loss_empty = weight_reduce_loss(loss_empty, avg_factor=N_pos)
        #loss_missed = (1.0 - missed)
        #loss_missed = weight_reduce_loss(loss_missed, avg_factor=N_pos)
        
        return loss_iou#, loss_empty, loss_missed

class HuberLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self):
        super().__init__()

    def forward(self, pred_bboxes, target_bboxes, delta=1):
        #N_pos = torch.clamp(target_scores.sum(), min=1)
        residual = torch.abs(target_bboxes - pred_bboxes)
        condition = residual < delta
        loss = torch.where(
            condition,
            0.5 * residual ** 2,  # MSE-like for small residuals
            delta * residual - 0.5 * delta ** 2  # MAE-like for large residuals
        )
        
        #loss = weight_reduce_loss(loss, avg_factor=N_pos)
        loss = torch.mean(loss)
        
        return loss
    
class smooth_l1_loss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    def forward(self, input_, target, label, beta=0.9):
        """Computes SmoothL1Loss"""
        if beta < 1e-5:
            loss = torch.abs(input_ - target)
        else:
            n = torch.abs(input_ - target)
            cond = n < beta
            loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
            #print(loss)
        
        N_pos = torch.clamp(label.sum(), min=1)
        #print(N_pos)
        loss = weight_reduce_loss(loss, avg_factor=N_pos)
        return loss#.mean(0).sum()

class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        print("dfl compiute")
        print(pred_dist.shape)
        print(target.shape)
        
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        
        loss = F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        
        N_pos = torch.clamp(target.sum(), min=1)
        loss = weight_reduce_loss(loss, avg_factor=N_pos)
        return loss#.mean(0).sum()
    
class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    def forward(self, pred, label, gamma=2.0, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        #print(loss.shape)
        
        N_pos = torch.clamp(label.sum(), min=1)
        loss = weight_reduce_loss(loss, avg_factor=N_pos)
        return loss#.mean(0).sum()

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    #@staticmethod
    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        #pred_score: refers to the predicted IACS
        #gt_score: refers to the IOU(gt box if exist, predicted box)
        #label: refers to 0/1 tensor to indicate object or not object 
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label

        loss = F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight
        
        N_pos = max(label.sum(), 1)
        
        loss = weight_reduce_loss(loss, avg_factor=N_pos)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward_one_class(self, inputs, target, weight=None, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)
        #target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] 
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        
        dice = self._dice_loss(inputs, target)
        class_wise_dice.append(1.0 - dice.item())
        loss += dice * weight[0]
        return loss
            
    def forward(self, inputs, target, weight=None, softmax=False):
        #if softmax:
        #    inputs = torch.softmax(inputs, dim=1)
        #target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes