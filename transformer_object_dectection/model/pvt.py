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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNet
#from transformers import PyViTModel, PyViTConfig
from transformers import AutoImageProcessor, PvtForImageClassification
from transformers import AutoModelForImageClassification, AutoImageProcessor
from typing import Dict, List, Optional, Tuple

class PVTFeatureExtractor(nn.Module):
    """
    Feature extractor using Pyramid Vision Transformer (PVT) from HuggingFace
    """
    def __init__(self, model_name: str = "OpenGVLab/pvt_v2_b0", pretrained: bool = True):
        super().__init__()
        # Load PVT model from HuggingFace
        self.pvt = AutoModelForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0")
        
        # Get feature dimensions from the model
        self.feature_dim = self.pvt.config.hidden_sizes
        self.num_stages = 4  # PVT typically has 4 feature map stages
        
        # These are the output channels for each feature level in RetinaNet
        self.out_channels = 128
        
        # Lateral and output convolutions to match RetinaNet requirements
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        # Create conv layers for each feature level
        for i in range(self.num_stages):
            self.lateral_convs.append(
                nn.Conv2d(self.feature_dim[i], self.out_channels, kernel_size=1)
            )
            self.output_convs.append(
                nn.Conv2d(self.out_channels, self.out_channels, 
                         kernel_size=3, padding=1)
            )
            #print("feature dim  out", self.out_channels)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # PVT expects channels-last format (batch, height, width, channels)
        #x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        
        # Forward pass through PVT
        #with torch.no_grad():
        outputs = self.pvt(**x, output_hidden_states=True)
        #print('np ', outputs.logits.shape)
        
        # Get hidden states (skip the first which is embeddings)
        hidden_states = outputs.hidden_states  # Typically 4 feature maps
        
        # We'll use the last hidden state from each stage
        features = []
        for i, h_state in enumerate(hidden_states):
            #print(h_state.shape)
            # Apply lateral and output convs
            h_state = self.lateral_convs[i](h_state)
            h_state = self.output_convs[i](h_state)
            
            
            features.append(h_state)
        
        # Return as a dictionary to match FPN interface
        return {f"feat_{i}": feat for i, feat in enumerate(features)}
    
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
        
        self.device = device
        self.grid_size = grid_size
        self.metrics = {}
        
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
        #grid_size = x.size(2)

        prediction = (
            x.view(batch_size, self.num_anchors, 4, self.grid_size, self.grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )
        # batch_size, n anchors, grid size, grid size, box

        # Get outputs
        sigmoid_tx = torch.sigmoid(prediction[..., 0])  # Center x
        sigmoid_ty = torch.sigmoid(prediction[..., 1])  # Center y
        tr = prediction[..., 2]  # Width
        pred_conf = torch.sigmoid(prediction[..., 3])  # Object confidence (objectness)
        #pred_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction

        # Add offset and scale with anchors
        pred_boxes = torch.zeros_like(prediction[..., :3], device=self.device)
        pred_boxes[..., 0] = sigmoid_tx + self.cx
        pred_boxes[..., 1] = sigmoid_ty + self.cy
        pred_boxes[..., 2] = self.pr * torch.exp(tr)
        
        pred_boxes = pred_boxes/self.grid_size #prediction in range of 0 to 1
        pred = (pred_boxes.view(batch_size, -1, 3) * self.image_size, #prediction in range of 0 to img_size
                pred_conf.view(batch_size, -1, 1))
        output = torch.cat(pred, -1)
        
        if targets is None:
            #output shape : batch_size, n boxes detections (grid_size**2), 4 (xc, yc, r, conf)
            return output, 0
        #aligned_targets, iou_scores, obj_mask, no_obj_mask = utils.utils.build_targets_circle(
        aligned_targets, pred_boxes, obj_mask, no_obj_mask = utils.utils.build_targets_circle(
            pred_boxes=pred_boxes, #in range of 0 to 1
            target=targets, #range of 0-1
            anchors=self.scaled_anchors,
            device=self.device,
            iou_type=True # if true the aligned targets are the same as input but aligned
        )
        
        #target ordered for comparison with corresponding pred_boxes
        loss_bbox = self.diou(pred_boxes, aligned_targets, obj_mask)
       
        loss_conf = self.fl(pred=prediction[..., 3], label=obj_mask.float(), gamma=gamma, alpha=alpha)
        
        loss_layer = self.ldiou*loss_bbox + self.lfl*loss_conf #+ loss_cls

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

class PVT(nn.Module):
    def __init__(self, image_size, N_channels=3, n_model=512, num_blks=1, anchors=None, obj='cbbox', loss_type='diou', device='cpu', bitNet=False, gamma=2.0, alpha=0.75):
        super(PVT, self).__init__()
        #anchors = {'scale1': [(10, 13), (16, 30), (33, 23)],
        #           'scale2': [(30, 61), (62, 45), (59, 119)],
        #           'scale3': [(116, 90), (58, 53), (373, 326)]}
        #anchors = {'scale1': [(12, 13), (15, 16), (17, 19)],
        #           'scale2': [(20, 23), (21, 19), (24, 23)],
        #           'scale3': [(24, 27), (28, 29), (33, 34)]}
        #anchors = {'scale2': [(13, 14), (16, 18), (19, 19)],
        #           'scale3': [(21, 23), (25, 26), (31, 32)]}
        assert obj in ['cbbox', 'bbox']
        assert loss_type in ['mse', 'diou', 'diou2', 'mse2']
        
        if anchors==None:
            if obj=='bbox':
                #anchors = {'scale1': [(21, 21)], 'scale2': [(70, 69)]}
                anchors = {'scale1': [(15, 16)], 'scale2': [(26, 26)], 'scale3': [(57, 57)], 'scale4': [(85, 82)]}
                #anchors = {'scale1': [(70, 69)], 'scale2': [(21, 21)]}
            else:#cbbox
                #anchors = {'scale1': [(11)], 'scale2': [(35)]}
                anchors = {'scale1': [(8)], 'scale2': [(13)], 'scale3': [(27)], 'scale4': [(41)]}
                
        #each anchor have 4 values for bboxes, 1 for object confidence, n num classes scores
        if obj == 'bbox':
            final_out_channel = len(anchors[list(anchors.keys())[0]]) * (4 + 1 ) 
        else: #cbbox
            final_out_channel = len(anchors[list(anchors.keys())[0]]) * (3 + 1 ) 
        
        img_size=(image_size, image_size)
        
        self.pvt_backbone = PVTFeatureExtractor("OpenGVLab/pvt_v2_b0", pretrained=True).to(device)
        
        grid_size1 = 56
        grid_size2 = 28
        grid_size3 = 14
        grid_size4 = 7
        
        self.conv_final1 = self.make_conv_final(128, final_out_channel) #input is the output of enc2
        #self.conv_final1_box = self.make_conv_final_box(n_model, 4) #input is the output of enc2
        #self.conv_final1_cls = self.make_conv_final_cls(n_model, 1) #input is the output of enc2
        if loss_type=='diou' and obj=='bbox':
            self.detection1 = DetectionRectangle(anchors['scale1'], image_size, obj, grid_size4, device)
        elif loss_type=='diou' and obj=='cbbox':
            self.detection1 = DetectionCircle(anchors['scale1'], image_size, obj, grid_size4, device) # diou + fl
            
            
        #self.conv_residual = self.make_conv(n_model//2, n_model//4, kernel_size=1, stride=1, padding=0)
        #self.upsample1 = self.make_upsample(256, 128, scale_factor=2)
        #self.upsample2 = self.make_upsample(288, 288//2, scale_factor=2)
        #self.upsample3 = self.make_upsample(208, 208//2, scale_factor=2)
        
        self.upsample1 = self.make_upsample(128, 64, scale_factor=2)
        self.upsample2 = self.make_upsample(192, 192//2, scale_factor=2)
        self.upsample3 = self.make_upsample(224, 224//2, scale_factor=2)
        
        #self.conv_final2 = self.make_conv_final(160+128, final_out_channel)
        self.conv_final2 = self.make_conv_final(64+128, final_out_channel)
        #self.conv_final2_box = self.make_conv_final_box(n_model, 4) #input is the output of enc2
        #self.conv_final2_cls = self.make_conv_final_cls(n_model, 1) #input is the output of enc2
        if loss_type=='diou' and obj=='bbox':
            self.detection2 = DetectionRectangle(anchors['scale2'], image_size, obj, grid_size3, device)
        elif loss_type=='diou' and obj=='cbbox':
            self.detection2 = DetectionCircle(anchors['scale2'], image_size, obj, grid_size3, device)
            
        #self.conv_final3 = self.make_conv_final(208, final_out_channel)
        self.conv_final3 = self.make_conv_final(96+128, final_out_channel)
        if loss_type=='diou' and obj=='bbox':
            self.detection3 = DetectionRectangle(anchors['scale3'], image_size, obj, grid_size2, device)
        elif loss_type=='diou' and obj=='cbbox':
            self.detection3 = DetectionCircle(anchors['scale3'], image_size, obj, grid_size2, device)
            
        #self.conv_final4 = self.make_conv_final(136, final_out_channel)    
        self.conv_final4 = self.make_conv_final(112+128, final_out_channel)    
        if loss_type=='diou' and obj=='bbox':
            self.detection4 = DetectionRectangle(anchors['scale4'], image_size, obj, grid_size1, device)
        elif loss_type=='diou' and obj=='cbbox':
            self.detection4 = DetectionCircle(anchors['scale4'], image_size, obj, grid_size1, device)
        
        #parameters for Focal Loss
        self.gamma = gamma
        self.alpha = alpha

        self.layers = [self.detection1, self.detection2]
        
    def forward(self, x, targets=None):
        """
            x shape: batch_size, N channels, img_size, img_size
            targets shape: batch_size, N objects in the batch, coordinates box
        """
        loss = 0
        
        feats = self.pvt_backbone(x)
        
        residual_output1 = feats['feat_0'] # [1, 128, 56, 56]
        residual_output2 = feats['feat_1'] # [1, 128, 28, 28]
        residual_output3 = feats['feat_2'] # [1, 128, 14, 14]
        residual_output4 = feats['feat_3'] # [1, 128, 7, 7]
        
        #torch.Size([64, 32, 56, 56])
        #torch.Size([64, 64, 28, 28])
        #torch.Size([64, 160, 14, 14])
        #torch.Size([64, 256, 7, 7])
        
        scale1 = self.conv_final1(residual_output4)
        #print(scale1.shape)
        output1, layer_loss = self.detection1(scale1, targets, self.gamma, self.alpha)
        loss += layer_loss
        
        #print(residual_output4.shape)
        x = self.upsample1(residual_output4)
        #print(x.shape)
        x = torch.cat((x, residual_output3), dim=1)
        #print(x.shape)
        scale2 = self.conv_final2(x)
        output2, layer_loss = self.detection2(scale2, targets, self.gamma, self.alpha)
        loss += layer_loss
        
        x = self.upsample2(x)
        #print(x.shape)
        x = torch.cat((x, residual_output2), dim=1)
        
        scale3 = self.conv_final3(x)
        output3, layer_loss = self.detection3(scale3, targets, self.gamma, self.alpha)
        loss += layer_loss
        
        x = self.upsample3(x)
        x = torch.cat((x, residual_output1), dim=1)
        
        scale4 = self.conv_final4(x)
        output4, layer_loss = self.detection4(scale4, targets, self.gamma, self.alpha)
        loss += layer_loss
        

        outputs = torch.cat([output1.detach(), output2.detach(), output3.detach(), output4.detach()], 1)
        return outputs if targets is None else (loss, outputs)

    def make_conv(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1):
        module1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        module2 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)

        modules = nn.Sequential(module1, module2, nn.GELU())#nn.LeakyReLU(negative_slope=0.1))
        return modules

    def make_conv_final(self, in_channels: int, out_channels: int):
        modules = nn.Sequential(
            #self.make_conv(in_channels, in_channels//4, kernel_size=3, padding=1),
            #self.make_conv(in_channels//4, in_channels//8, kernel_size=3, padding=1),
            self.make_conv(in_channels, in_channels//4, kernel_size=3),
            self.make_conv(in_channels//4, in_channels//8, kernel_size=3),
            
            #nn.Conv2d(in_channels//8, out_channels, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(in_channels//8, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        return modules
    def make_conv_final_box(self, in_channels: int, out_channels: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, in_channels//4, kernel_size=3),
            
            self.make_conv(in_channels//4, in_channels//8, kernel_size=1, stride=1, padding=0),
            
            nn.Conv2d(in_channels//8, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        return modules
    
    def make_conv_final_cls(self, in_channels: int, out_channels: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, in_channels//4, kernel_size=3),
            
            self.make_conv(in_channels//4, in_channels//8, kernel_size=1, stride=1, padding=0),
            
            nn.Conv2d(in_channels//8, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            
        )
        return modules

    def make_upsample(self, in_channels: int, out_channels: int, scale_factor: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return modules


    
if __name__ == '__main__':
    model = TransformerObjectDetection(image_size=224, N_channels=3, n_model=512, num_blks=1, device='cpu')
    print(model)

    test = torch.rand([1, 3, 224, 224])
    y = model(test)
    print(y.shape)
