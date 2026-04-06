import torch
import torch.nn as nn
import numpy as np

import utils.utils
import model.transformer_encoder as Transformer
import model.losses as losses
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader

# improved_pointrend.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import random

# optimized_light_pointrend.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from typing import Dict, Tuple, List

# optimized_light_pointrend.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from typing import Dict, Tuple, List
from model.layers_ours import *
from matplotlib import pyplot as plt 

class DetectionCircle(nn.Module):
    def __init__(self, anchors, image_size: int, obj: str, grid_size: int, device: str):
        super(DetectionCircle, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.image_size = image_size
        
        self.diou = losses.BboxLoss(obj)
        self.ldiou = 1 #lambda diou
        self.fl = losses.FocalLoss()
        self.lfl = 1 #lambda diou
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        
        self.device = device
        self.grid_size = grid_size
        self.metrics = {}
        
        #self.loss_shape = losses.smooth_l1_loss()
        #self.loss_texture = losses.smooth_l1_loss()
        #self.loss_color = losses.smooth_l1_loss()
        
        # Calculate offsets for each grid
        self.stride = self.image_size / self.grid_size #patch size
        self.cx = torch.arange(grid_size, dtype=torch.float, device=self.device).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size])
        self.cy = torch.arange(grid_size, dtype=torch.float, device=self.device).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])
        
        self.scaled_anchors = torch.as_tensor([(a_r / self.stride) for a_r in self.anchors],
                                         dtype=torch.float, device=self.device)# scale relative to patch size
        self.pr = self.scaled_anchors.view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets, gamma=2.0, alpha=0.25):
        """
            Compute the prediction and loss of the model output
             x shape : batch_size, anchors*(box_coordinates+score+n_class), grid_size, grid_size
             target shape: batch_size, box_coordinates+score+n_class, grid_size, grid_size
        """
        batch_size = x.size(0)

        prediction = (
            x.view(batch_size, self.num_anchors, 4, self.grid_size, self.grid_size)
             .permute(0, 1, 3, 4, 2)
             .contiguous()
        )
        #print(prediction)

        # raw logits (DO NOT sigmoid here for loss)
        tx = prediction[..., 0]
        ty = prediction[..., 1]
        tr = prediction[..., 2]
        tr = torch.clamp(tr, -3, 3)
        conf_logits = prediction[..., 3]

        # Decode boxes
        sigmoid_tx = torch.sigmoid(tx)
        sigmoid_ty = torch.sigmoid(ty)

        pred_boxes = torch.zeros_like(prediction[..., :3])

        pred_boxes[..., 0] = sigmoid_tx + self.cx
        pred_boxes[..., 1] = sigmoid_ty + self.cy
        pred_boxes[..., 2] = self.pr * torch.exp(tr)

        pred_boxes = pred_boxes / self.grid_size  # normalized 0–1

        pred_conf = torch.sigmoid(conf_logits)

        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 3) * self.image_size,
                pred_conf.view(batch_size, -1, 1),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        #aligned_targets, iou_scores, obj_mask, no_obj_mask = utils.utils.build_targets_circle(
        aligned_targets, pred_boxes, obj_mask, labels = utils.utils.build_targets_circle(
            pred_boxes=pred_boxes, #in range of 0 to 1
            target=targets, #range of 0-1
            anchors=self.scaled_anchors,
            device=self.device
        )
        #plt.figure(figsize=(12, 12))
        #for i in range(16):
        #    plt.subplot(4, 4, i+1)
        #    plt.imshow(obj_mask[i, 0, ...].detach().to('cpu'))
        #    plt.axis('off')
        if obj_mask.sum():
            #target ordered for comparison with corresponding pred_boxes
            loss_bbox = self.diou(pred_boxes, aligned_targets[...,:3], obj_mask)
        else:
            loss_bbox = torch.tensor(0.0, device=self.device)
        
        #loss_l1 = F.l1_loss(pred_boxes, aligned_targets)
        
        #N_pos = max(obj_mask.float().sum(), 1.0)
        
        loss_conf = self.fl(pred=conf_logits, label=obj_mask.float(), gamma=gamma, alpha=alpha)
        #loss_conf = self.bce(conf_logits, labels)
        #loss_conf = loss_conf.sum()/N_pos
        #print(obj_mask.sum(), pred_boxes.shape,aligned_targets.shape, targets.shape )
        
        #print("TEST")
        #print(tar2.shape)
        #print(shape_features.shape)
        
        #loss_shape = self.loss_shape(shape_features[obj_mask], tar2[obj_mask][...,3:8], label=obj_mask)
        #loss_texture = self.loss_texture(texture_features[obj_mask], tar2[obj_mask][...,8:13], label=obj_mask)
        #loss_color = self.loss_color(color_features[obj_mask], tar2[obj_mask][...,13:18], label=obj_mask)
        
        loss_layer = self.ldiou*loss_bbox + self.lfl*loss_conf #+ 0.5*(loss_shape+loss_texture+loss_color)#+ loss_empty + loss_missed#+ loss_cls

        # Write loss and metrics
        self.metrics = {
            "loss_bbox": loss_bbox.detach(),
            "loss_conf": loss_conf.detach(),
            "loss_layer": loss_layer.detach(),
        }
        
        return output, loss_layer
    
class DetectionRectangle(nn.Module):
    def __init__(self, anchors, image_size: int, obj: str, grid_size: int, device: str):
        super(DetectionRectangle, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.image_size = image_size
        
        self.diou = losses.BboxLoss(obj)
        self.ldiou = 1 #lambda diou
        self.fl = losses.FocalLoss()
        self.lfl = 1 #lambda diou

        self.device = device
        self.grid_size = grid_size
        self.metrics = {}
        
        # Calculate offsets for each grid
        self.stride = self.image_size / self.grid_size #patch size
        self.cx = torch.arange(grid_size, dtype=torch.float, device=self.device).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size])
        self.cy = torch.arange(grid_size, dtype=torch.float, device=self.device).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])
        self.scaled_anchors = torch.as_tensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors],
                                         dtype=torch.float, device=self.device)# scale relative to patch s
        self.pw = self.scaled_anchors[:, 0].view((1, self.num_anchors, 1, 1))
        self.ph = self.scaled_anchors[:, 1].view((1, self.num_anchors, 1, 1))
        
        print(self.scaled_anchors)
        #self.box = []

    def forward(self, x, targets, gamma=1.5, alpha=0.75):
        #Compute the prediction and loss of the model output
        # x shape : batch_size, anchors*(box_coordinates+score+n_class), grid_size, grid_size
        # target shape: batch_size, box_coordinates+score+n_class, grid_size, grid_size

        batch_size = x.size(0)

        prediction = (
            x.view(batch_size, self.num_anchors, 5, self.grid_size, self.grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )
        # batch_size, n anchors, grid size, grid size, classes + box
        
        # Get outputs
        sigmoid_tx = torch.sigmoid(prediction[..., 0])  # Center x
        sigmoid_ty = torch.sigmoid(prediction[..., 1])  # Center y
        tw = prediction[..., 2]  # Width
        th = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)
        #pred_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction
        
        # Add offset and scale with anchors
        pred_boxes = torch.zeros_like(prediction[..., :4], device=self.device)
        pred_boxes[..., 0] = sigmoid_tx + self.cx
        pred_boxes[..., 1] = sigmoid_ty + self.cy
        pred_boxes[..., 2] = self.pw * torch.exp(tw)
        pred_boxes[..., 3] = self.ph * torch.exp(th)
        
        pred_boxes = pred_boxes/self.grid_size #prediction in range of 0 to 1
        pred = (pred_boxes.view(batch_size, -1, 4) * self.image_size, #prediction in range of 0 to img_size
                pred_conf.view(batch_size, -1, 1))
        output = torch.cat(pred, -1)

        if targets is None:
            #output shape : batch_size, n boxes detections (grid_size**2), 4 (xc, yc, w, h, conf)
            #self.box=output[:,10,0:4]
            #output = output[:,10,-1].view(-1, 1)
            return output, 0
        #aligned_targets, iou_scores, obj_mask, no_obj_mask = utils.utils.build_targets_rec(
        aligned_targets, pred_boxes, obj_mask, _ = utils.utils.build_targets_rec(
            pred_boxes=pred_boxes, #in range of 0 to 1
            target=targets, #range of 0-1
            anchors=self.scaled_anchors,
            device=self.device,
            iou_type=True
        )
        #target ordered for comparison with corresponding pred_boxes
        loss_bbox = self.diou(pred_boxes, aligned_targets, obj_mask)
        loss_conf = self.fl(pred=prediction[..., 4], label=obj_mask.float(), gamma=gamma, alpha=alpha)

        loss_layer = self.ldiou*loss_bbox + self.lfl*loss_conf #+ loss_cls

        # Write loss and metrics
        self.metrics = {
            "loss_bbox": loss_bbox.detach(), #.cpu().item() reduce time not to get item 
            "loss_conf": loss_conf.detach(),
            "loss_layer": loss_layer.detach()
        }

        return output, loss_layer
    

class TransformerObjectDetection(nn.Module):
    def __init__(self, image_size, N_channels=3, n_model=512, num_blks=1, anchors=None, obj='cbbox', device='cpu', bitNet=False, gamma=1.5, alpha=0.75):
        super(TransformerObjectDetection, self).__init__()
        #anchors = {'scale1': [(10, 13), (16, 30), (33, 23)],
        #           'scale2': [(30, 61), (62, 45), (59, 119)],
        #           'scale3': [(116, 90), (58, 53), (373, 326)]}
        #anchors = {'scale1': [(12, 13), (15, 16), (17, 19)],
        #           'scale2': [(20, 23), (21, 19), (24, 23)],
        #           'scale3': [(24, 27), (28, 29), (33, 34)]}
        #anchors = {'scale2': [(13, 14), (16, 18), (19, 19)],
        #           'scale3': [(21, 23), (25, 26), (31, 32)]}
        assert obj in ['cbbox', 'bbox']
        
        self.n_model = n_model
        
        if anchors==None:
            if obj=='bbox':
                anchors = {'scale1': [(21, 21)], 'scale2': [(70, 69)]}
                #anchors = {'scale1': [(70, 69)], 'scale2': [(21, 21)]}
            else:#cbbox
                anchors = {'scale1': [(11)], 'scale2': [(35)]}
                
        #each anchor have 4 values for bboxes, 1 for object confidence, n num classes scores
        if obj == 'bbox':
            final_out_channel = len(anchors[list(anchors.keys())[0]]) * (4 + 1 ) 
        else: #cbbox
            final_out_channel = len(anchors[list(anchors.keys())[0]]) * (3 + 1 ) 
            
        
        #self.down_112 = self.make_conv(N_channels, n_model//4, kernel_size=3, stride=2, padding=1)# 224→112
        #self.down_56 = self.make_conv(n_model//4, n_model//2, kernel_size=3, stride=2, padding=1)# 112→56
        
        img_size=(image_size, image_size)

        patch_enc1 = 8
        patch_enc2 = 2
        
        img_size2=(image_size//patch_enc1, image_size//patch_enc1)

        self.encoder1 = Transformer.Encoder(img_size, N_channels, n_model//2, mult=2, 
                                           patch_size=patch_enc1, num_blks=num_blks, device=device, 
                                           bitNet=bitNet)#
        grid_size1 = img_size[0]//patch_enc1#only square images
        self.encoder2 = Transformer.Encoder(img_size2, n_model//2, n_model, mult=2, 
                                           patch_size=patch_enc2, num_blks=num_blks, device=device, bitNet=bitNet)
        
        grid_size2 = grid_size1//patch_enc2
        self.conv_final1 = self.make_conv_final(n_model, final_out_channel) #input is the output of enc2
        #self.conv_final1_box = self.make_conv_final_box(n_model, 4) #input is the output of enc2
        #self.conv_final1_cls = self.make_conv_final_cls(n_model, 1) #input is the output of enc2
        
        if obj=='bbox':
            self.detection1 = DetectionRectangle(anchors['scale1'], image_size, obj, grid_size2, device)
        elif obj=='cbbox':
            self.detection1 = DetectionCircle(anchors['scale1'], image_size, obj, grid_size2, device) # diou + fl
            
        self.upsample = self.make_upsample(n_model, n_model//2, scale_factor=patch_enc2)
        
        self.conv_final2 = self.make_conv_final(n_model, final_out_channel)
        #self.conv_final2_box = self.make_conv_final_box(n_model, 4) #input is the output of enc2
        #self.conv_final2_cls = self.make_conv_final_cls(n_model, 1) #input is the output of enc2

        if obj=='bbox':
            self.detection2 = DetectionRectangle(anchors['scale2'], image_size, obj, grid_size1, device)
        elif obj=='cbbox':
            self.detection2 = DetectionCircle(anchors['scale2'], image_size, obj, grid_size1, device)

        #parameters for Focal Loss
        self.gamma = gamma
        self.alpha = alpha

        self.layers = [self.detection1, self.detection2]
        
        self.upsample2 = self.make_upsample(n_model, n_model//2, scale_factor=patch_enc2)
        self.upsample3 = self.make_upsample(n_model//2+n_model//4, n_model//4, scale_factor=patch_enc2)
        self.upsample4 = self.make_upsample(n_model//4+n_model//8, n_model//8, scale_factor=patch_enc2)
        self.conv_final3 = self.make_conv2(n_model//8+n_model//16, 1, kernel_size=3)
        
        self.loss_mask = losses.DiceLoss(1)
        self.loss_coarses = losses.DiceLoss(1)
        
        self.clone1 = Clone()
        self.clone2 = Clone()
        self.clone3 = Clone()
        self.cat1 = Cat()
        
        self.cat2 = Cat()
        self.cat3 = Cat()
        self.cat4 = Cat()
        
        self.conv_coarse1 = Conv2d(n_model, 1, 3, padding=1)
        self.conv_coarse2 = Conv2d(n_model//2, 1, 3, padding=1)
        self.coarse_mask1 = None
        self.coarse_mask2 = None
        
    def forward(self, x, targets=None, y_true=None, explain=False):
        """
            x shape: batch_size, N channels, img_size, img_size
            targets shape: batch_size, N objects in the batch, coordinates box
        """
        loss = 0
        
        x, feats, x1_ln1 = self.encoder1(x) ##1/8 (28x28)
        feat_1_1 = feats[0]
        feat_1_2 = feats[1]
        feat_1_4 = feats[2]
        
        #residual_output = x
        residual_output, x1 = self.clone1(x, 2)
        
        x, x2_ln1 = self.encoder2(x)##1/16 (14x14)
        
        x1_ln1 = x1_ln1.permute(0, 2, 1).reshape(-1, self.n_model//2, 28, 28)
        x2_ln1 = x2_ln1.permute(0, 2, 1).reshape(-1, self.n_model, 14, 14)
        
        x2, x3, x4 = self.clone2(x, 3)
        scale1 = self.conv_final1(x2)
        x_coarse1 = self.conv_coarse1(x2_ln1)
        
        output1, layer_loss = self.detection1(scale1, targets, self.gamma, self.alpha)
        loss += layer_loss
        
        x = self.upsample(x3)
        #x = torch.cat((x, residual_output), dim=1)
        x = self.cat1((x, residual_output), dim=1)
        
        x5, x6 = self.clone3(x, 2)
        
        scale2 = self.conv_final2(x5)
        x_coarse2 = self.conv_coarse2(x1_ln1)
        self.coarse_mask1 = x_coarse1#14x14
        self.coarse_mask2 = x_coarse2#28x28
        
        output2, layer_loss = self.detection2(scale2, targets, self.gamma, self.alpha)
        loss += layer_loss
        
        #print("x",x.shape)#[32, 128, 28, 28]
        x = self.upsample2(x6) #1//4
        #print("x upsample2", x.shape)
        x = self.cat2((x, feat_1_4), dim=1)
        #print("x 1/4", x.shape)#x 1/4 torch.Size([64, 96, 56, 56])
        x = self.upsample3(x) #1//2
        #print("x upsample3", x.shape)
        x = self.cat3((x, feat_1_2), dim=1)
        #print("x 1/2", x.shape)
        x = self.upsample4(x) #1
        x = self.cat4((x, feat_1_1), dim=1)
        x = self.conv_final3(x)
        
        if targets is None:
            loss_seg = 0
            loss_additional=0
        else:
            
            #loss_seg = vi_regression_loss(x, y_true)
            loss_seg = self.loss_mask(x, y_true)
            
            downsampled1 = F.interpolate(y_true, size=(14, 14), mode='bicubic', align_corners = False)
            downsampled2 = F.interpolate(y_true, size=(28, 28), mode='bicubic', align_corners = False)
            #print(downsampled1.shape, downsampled2.shape)
            
            loss_additional = self.loss_coarses(torch.sigmoid(x_coarse1), downsampled1)+self.loss_coarses(torch.sigmoid(x_coarse2), downsampled2)
            #loss_seg = torch.tensor(0)
            #loss_additional = torch.tensor(0)
            self.metrics = {
                "loss_seg": loss_seg.detach()
            }
        loss += loss_seg+0.3*loss_additional
        #############################
        if explain:
            outputs = torch.cat([output1, output2], 1)
        else:
            outputs = torch.cat([output1.detach(), output2.detach()], 1)
        
        if targets is None:
            #print("none")
            return outputs, x
        else:
            #print("todo bien")
            return (loss, outputs, x)
        #return outputs, mask if targets is None else (loss, outputs, mask)
    
    def make_conv2(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1):
        module1 = Conv2d(in_channels, in_channels, kernel_size, stride, padding, bias=False)
        module2 = BatchNorm2d(in_channels, momentum=0.9, eps=1e-5)
        module3 = GELU()
        module4 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        modules = nn.Sequential(module1, module2, module3, module4)#nn.LeakyReLU(negative_slope=0.1))
        return modules
    def relprop_make_conv2(self, cam, **kwargs):
        #cam = cam.transpose(1,2)
        #cam = cam.reshape(cam.shape[0], cam.shape[1], 14, 14)
        #print("el batch", cam.shape)
        cam = self.conv_final3[-1].relprop(cam, **kwargs)
        cam = self.conv_final3[-2].relprop(cam, **kwargs)
        cam = self.conv_final3[2].relprop(cam, **kwargs)
        cam = self.conv_final3[1].relprop(cam, **kwargs)
        #print("convfinalseg", cam.shape) #[1, 128, 14, 14]
        
        return cam
    
    def make_conv(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1):
        module1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        module2 = BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)

        modules = nn.Sequential(module1, module2, GELU())#nn.LeakyReLU(negative_slope=0.1))
        return modules

    def make_conv_final(self, in_channels: int, out_channels: int):
        modules = nn.Sequential(
            #self.make_conv(in_channels, in_channels//4, kernel_size=3, padding=1),
            #self.make_conv(in_channels//4, in_channels//8, kernel_size=3, padding=1),
            self.make_conv(in_channels, in_channels//4, kernel_size=3),
            self.make_conv(in_channels//4, in_channels//8, kernel_size=3),
            
            #nn.Conv2d(in_channels//8, out_channels, kernel_size=3, padding=1, bias=True),
            Conv2d(in_channels//8, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        return modules

    def make_upsample(self, in_channels: int, out_channels: int, scale_factor: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return modules
    def make_upsample2(self, in_channels: int, out_channels: int, scale_factor: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=3),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return modules
    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad
    def relprop(self, cam=None,downsampled1=None, method='transformer_attribution', tokens=None, **kwargs):

        cam1 = cam[:,:196,:]#14x14 output1
        cam2 = cam[:,196:,:]#28x28 output2
        #cam1=cam1*self.detection1.grid_size
        #cam2=cam2*self.detection2.grid_size
        
        #cam = cam2
        cam2 = cam2.transpose(1,2)
        #cam = cam.reshape(cam.shape[0], cam.shape[1], 14, 14)
        cam2 = cam2.reshape(cam2.shape[0], cam2.shape[1], 28, 28)
        #print("aqui va ", cam.shape)#aqui va  torch.Size([1, 4, 28, 28])

        #plt.figure(figsize=(12, 12))
        #cam_test = cam2[0].detach().cpu()#permute(1, 0).reshape(-1, 28, 28)
        #for i in range(36):
        #    plt.subplot(6, 6, i+1)
        #    plt.imshow(cam_test[i,:,:])
        #    plt.axis('off')
        #    if (i+1)>=cam_test.shape[0]:
        #        break
                
        cam2 = self.relprop2(cam2, **kwargs)
        #self.relprop2.clean()
        #print("convs", cam2.shape) # torch.Size([1, 128, 14, 14]) # [1, 128, 28, 28]
        
        #############################################################
        #SEG BLOCK
        cam3 = self.relprop_make_conv2(downsampled1, **kwargs)
        
        cam3, cam_feats = self.cat4.relprop(cam3, **kwargs)
        cam3 = nn.MaxPool2d(2, stride=2)(cam3)
        cam3 = self.upsample4[0][2].relprop(cam3, **kwargs) #
        cam3 = self.upsample4[0][1].relprop(cam3, **kwargs) #
        cam3 = self.upsample4[0][0].relprop(cam3, **kwargs) #
        
        cam3, cam_feats = self.cat3.relprop(cam3, **kwargs)
        cam3 = nn.MaxPool2d(2, stride=2)(cam3)
        cam3 = self.upsample3[0][2].relprop(cam3, **kwargs) #
        cam3 = self.upsample3[0][1].relprop(cam3, **kwargs) #
        cam3 = self.upsample3[0][0].relprop(cam3, **kwargs) #
        
        cam3, cam_feats = self.cat2.relprop(cam3, **kwargs)
        cam3 = nn.MaxPool2d(2, stride=2)(cam3)
        cam3 = self.upsample2[0][2].relprop(cam3, **kwargs) #
        cam3 = self.upsample2[0][1].relprop(cam3, **kwargs) #
        cam3 = self.upsample2[0][0].relprop(cam3, **kwargs) #
        #print("SEG",cam3.shape)
        
        #cam3 = self.conv_coarse2.relprop(downsampled2, **kwargs)
        #print("cam coarse")
        #print(cam3.shape, cam2.shape, downsampled2.shape)
        #torch.Size([1, 64, 28, 28]) torch.Size([1, 128, 28, 28])
        #plt.figure(figsize=(12, 12))
        #cam_test = cam3[0].detach().cpu()#.permute(1, 0).reshape(-1, 28, 28)
        #for i in range(36):
        #    plt.subplot(6, 6, i+1)
        #    plt.imshow(cam_test[i,:,:])
        #    plt.axis('off')
        #    if (i+1)>=cam_test.shape[0]:
        #        break
        
        cam2 = self.clone3.relprop((cam2, cam3), **kwargs)
        
        #plt.figure(figsize=(12, 12))
        #cam_test = cam2[0].detach().cpu()#permute(1, 0).reshape(-1, 28, 28)
        #for i in range(36):
        #    plt.subplot(6, 6, i+1)
        #    plt.imshow(cam_test[i,:,:])
        #    plt.axis('off')
        #    if (i+1)>=cam_test.shape[0]:
        #        break
        
        #return cam2#, cam2
        #cam_up, cam2 = cam2[:,:64,...], cam2[:,64:,...]
        #return cam2[0].detach()
        cam_up, cam2 = self.cat1.relprop(cam2, **kwargs)
        cam_up = nn.MaxPool2d(2, stride=2)(cam_up)
        #print("after maxpool", cam_up.shape) #([1, 128, 14, 14])
        cam_up = self.upsample[0][2].relprop(cam_up, **kwargs) #
        #self.upsample[0][2].relprop.clean()
        cam_up = self.upsample[0][1].relprop(cam_up, **kwargs) #
        #self.upsample[0][1].relprop.clean()
        cam_up = self.upsample[0][0].relprop(cam_up, **kwargs) #
        #self.upsample[0][0].relprop.clean()
        #print("previous enc2",cam_up.shape) #[1, 128, 14, 14]
        
        ####################################################################################################
        #cam = cam[:,:64,...]
        #cam = self.encoder2.relprop(cam, **kwargs)

        cam1 = cam1.transpose(1,2)
        cam1 = cam1.reshape(cam1.shape[0], cam1.shape[1], 14, 14)
        cam1 = self.relprop1(cam1, **kwargs)
        #self.relprop1.clean()
        
        #plt.figure(figsize=(12, 12))
        #print("aqui vouy", cam1.shape)
        #cam_test = cam1[0].detach().cpu()#permute(1, 0).reshape(-1, 28, 28)
        #for i in range(36):
        #    plt.subplot(6, 6, i+1)
        #    plt.imshow(cam_test[i,:,:])
        #    plt.axis('off')
        #    if (i+1)>=cam_test.shape[0]:
        #        break
        #cam4 = self.conv_coarse1.relprop(downsampled1, **kwargs)
        #cam1 = self.clone2.relprop((cam1, cam_up, cam4), **kwargs)
        #self.clone2.clean()
        #print("antes enc2", cam1.shape) #[1, 128, 14, 14]
        
        #plt.figure(figsize=(12, 12))
        #cam_test = cam1[0].detach().cpu()#.permute(1, 0).reshape(-1, 28, 28)
        #for i in range(36):
        #    plt.subplot(6, 6, i+1)
        #    plt.imshow(cam_test[i,:,:])
        #    plt.axis('off')
        #    if (i+1)>=cam_test.shape[0]:
        #        break
        cam1 = self.encoder2.relprop(cam1, **kwargs)
        #cam1 = cam1.clip(0, 1)
        #self.encoder2.relprop.clean()
        #print("tentative feature map", cam1.shape)#([1, 196, 128])
        #plt.figure(figsize=(12, 12))
        #cam_test = cam1[0].detach().cpu().permute(1, 0).reshape(-1, 14, 14)
        #for i in range(36):
        #    plt.subplot(6, 6, i+1)
            #plt.imshow((cam_test[i,:,:]/cam_test[i,:,:].max()).clip(0, 1))
        #    plt.imshow(cam_test[i,:,:])
        #    plt.axis('off')
        #    if (i+1)>=cam_test.shape[0]:
        #        break
        #return cam1, cam1

        
        #patch embeding
        (cam1, cam_pos) = self.encoder2.add1.relprop(cam1, **kwargs)
        #self.encoder2.add1.relprop.clean()
        #print("los cams", cam1.shape, cam2.shape) #[1, 196, 128]) torch.Size([1, 196, 128]
        cam1 = self.encoder2.dropout.relprop(cam1, **kwargs)
        #self.encoder2.dropout.relprop.clean()
        #print(cam1.shape)#torch.Size([1, 196, 128])
        
        cam1 = self.encoder2.patch_embedding.relprop(cam1, **kwargs) 
        #cam1 = cam1.clip(0, 1)
        #self.encoder2.patch_embedding.relprop.clean()
        #print("after patch emb", cam1.shape, cam2.shape) #torch.Size([1, 64, 28, 28]) torch.Size([1, 64, 28, 28])
        #plt.figure(figsize=(12, 12))
        #print(cam1.shape)
        #cam_test = cam1[0].detach().cpu()#.permute(1, 0).reshape(-1, 14, 14)
        #for i in range(36):
        #    plt.subplot(6, 6, i+1)
        #    plt.imshow(cam_test[i,:,:].clip(0, 1))
        #    plt.axis('off')
        #    if (i+1)>=cam_test.shape[0]:
        #        break
                
        cam = self.clone1.relprop((cam2, cam1), **kwargs)
        #self.clone1.relprop.clean()
        #cam.clip(0, 1)
        
        cam = self.encoder1.relprop(cam, **kwargs)
        
        #self.encoder1.relprop.clean()
        #print("tentative feature map", cam.shape)#[1, 784, 64]
        #plt.figure(figsize=(12, 12))
        #cam_test = cam[0].detach().cpu().permute(1, 0).reshape(-1, 28, 28)
        #for i in range(36):
        #    plt.subplot(6, 6, i+1)
        #    plt.imshow(cam_test[i,:,:])
        #    plt.axis('off')
        #    if (i+1)>=cam_test.shape[0]:
        #        break
        #return cam, cam
        #return cam[0].detach().cpu().permute(1, 0).reshape(-1, 28, 28)
        #return cam[0].permute(1, 0).reshape(-1, 28, 28)
        #print(cam)
        
        #cam_f = cam
        
        #for blk in self.encoder2.blks:
        #    grad = blk.attention.get_attn_gradients()
            #print(grad.shape)
            #plt.figure(figsize=(12, 12))
            #cam_test = grad[0].detach().cpu().sum(0).permute(1, 0).reshape(-1, 14, 14)
            #for i in range(36):
            #    plt.subplot(6, 6, i+1)
            #    plt.imshow(cam_test[i+100,:,:])
            #    if (i+1)>=cam_test.shape[0]:
            #        break
            
        #    return grad[0].detach().cpu().sum(0).permute(1, 0).reshape(-1, 14, 14)
        
        if method == "rollout":
        ########################################################################
        ############################## ROLLOUT #################################
        ########################################################################
            attn_cams = []
            for blk in self.encoder1.blks:
                attn_heads = blk.attention.get_attn_cam()#.clamp(min=0)
                avg_heads = attn_heads.mean(dim=1).detach()#(attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            rollout = compute_rollout_attention(attn_cams, start_layer=0)
            
        ########################################################################
        ########################################################################
        elif method == "transformer_attribution":
            cams = []
            grads = []
            cams_att = []
            for blk in self.encoder1.blks:
                grad = blk.attention.get_attn_gradients()        
                cam_attn = blk.attention.get_attn_cam()#.clamp(min=0)
                cam_attn = cam_attn[0].reshape(-1, cam_attn.shape[-1], cam_attn.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                
                cam = grad * cam_attn
                cams_att.append(cam_attn)
                grads.append(grad)
               

                #plt.figure(figsize=(12, 12))
                #cam_test = cam_attn[0].detach().cpu().permute(1, 0).reshape(-1, 28, 28)
                #for i in range(36):
                #    plt.subplot(6, 6, i+1)
                #    plt.imshow(cam_test[i+100,:,:])
                #    if (i+1)>=cam_test.shape[0]:
                #        break

                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
                blk.attention.clean_hooks()
            rollout = compute_rollout_attention(cams, start_layer=0)
        #print(cams[0].shape, rollout.shape, rollout[:, 0, :].shape)
        #print(rollout.shape)#torch.Size([1, 784, 784])
        #"""
        #add the relevant tokens of the detection
        sal = torch.zeros_like(rollout[:,0,:])
        count = 0
        if len(tokens)>0:
            for token in tokens:
                if token<196:
                    t_i = (token//14)*2
                    t_j = (token%14)*2
                    token = t_i*28+t_j
                else:
                    token = token-196
                    t_i = (token//28)
                    t_j = (token%28)
                
                
                sal+=rollout[:,token,:]
            
            cam = sal/len(tokens)
        else:
            cam=sal
        #"""    
        #cam = rollout.mean(1)
        
        for blk in self.encoder1.blks:
            blk.attention.clean_hooks()
        
        #rollout = compute_rollout_attention(cams, start_layer=0)
        #print("rollout ", rollout.shape)
        #if self.token!=None:
        #    cam = rollout[:, self.token, :]
        #else:
        #    cam = rollout[:, 0, :]
        #self.clear_model_gradients(self)
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
            
        #del cams, cam, grad, cam_attn, cam1,cam2, cam_up, cam_pos
        #import gc
        #gc.collect()
    
        # Clear GPU cache
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
        #    torch.cuda.synchronize()
        #return rollout[0].detach().reshape(-1, 28, 28).cpu()#, cams, cam_f, cam_attn, grad
        return cam.detach()#, grads, cams_att#, rollout.detach()# cam_f.detach()#, cams, cam_f, cam_attn, grad
    def relprop1(self, cam, **kwargs):
        #cam = cam.transpose(1,2)
        #cam = cam.reshape(cam.shape[0], cam.shape[1], 14, 14)
        #print("el batch", cam.shape)
        cam = self.conv_final1[-1].relprop(cam, **kwargs)
        #print("primera conv", cam.shape, self.conv_final1[-1]) #[1, 16, 14, 14]
        cam = self.conv_final1[1][2].relprop(cam, **kwargs) # batch torch.Size([1, 16, 14, 14])
        cam = self.conv_final1[1][1].relprop(cam, **kwargs) # batch torch.Size([1, 16, 14, 14])
        cam = self.conv_final1[1][0].relprop(cam, **kwargs) # conv torch.Size([1, 16, 14, 14])
        cam = self.conv_final1[0][2].relprop(cam, **kwargs) # batch torch.Size([1, 16, 14, 14])
        cam = self.conv_final1[0][1].relprop(cam, **kwargs) # batch torch.Size([1, 16, 14, 14])
        cam = self.conv_final1[0][0].relprop(cam, **kwargs) # conv torch.Size([1, 16, 14, 14])
        #print("primer makeconvfinal", cam.shape) #[1, 128, 14, 14]
        
        return cam

    def relprop2(self, cam, **kwargs):
        #cam = cam.transpose(1,2)
        #cam = cam.reshape(cam.shape[0], cam.shape[1], 28, 28)
        #print("el batch", cam.shape, self.conv_final2[-1])
        
        cam = self.conv_final2[-1].relprop(cam, **kwargs)
        #print("primera conv", cam.shape, self.conv_final1[-1]) #[1, 16, 14, 14]
        cam = self.conv_final2[1][2].relprop(cam, **kwargs) # batch torch.Size([1, 16, 14, 14])
        cam = self.conv_final2[1][1].relprop(cam, **kwargs) # batch torch.Size([1, 16, 14, 14])
        cam = self.conv_final2[1][0].relprop(cam, **kwargs) # conv torch.Size([1, 16, 14, 14])
        cam = self.conv_final2[0][2].relprop(cam, **kwargs) # batch torch.Size([1, 16, 14, 14])
        cam = self.conv_final2[0][1].relprop(cam, **kwargs) # batch torch.Size([1, 16, 14, 14])
        cam = self.conv_final2[0][0].relprop(cam, **kwargs) # conv torch.Size([1, 16, 14, 14])
        #print("primer makeconvfinal", cam.shape) #[1, 128, 14, 14]
        
        return cam
    
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1] #1, 196, 196
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                           for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


    
if __name__ == '__main__':
    model = TransformerObjectDetection(image_size=224, N_channels=3, n_model=128, num_blks=1, device='cpu')
    print(model)

    test = torch.rand([1, 3, 224, 224])
    y = model(test)
    print(y.shape)
#modelo 6, conv predicen todo