import argparse
import torch
import torch.nn.functional as F
import numpy as np
from numpy import *
import utils.utils

class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, tokens, method='rollout'):
        output, mask_seg = self.model(input, explain=True)
        #output = self.model(input, explain=True)
        kwargs = {"alpha": 1.0}
        
        #cam1 = utils.utils.build_target_circle_explain(preds, scores, self.model.detection1.scaled_anchors, input.device, nG = 224//16)#14x14
        #cam1 = np.array(cam1.cpu())
        #cam2 = utils.utils.build_target_circle_explain(preds, scores, self.model.detection2.scaled_anchors, input.device, nG = 224//8)#28x28
        #cam2 = np.array(cam2.cpu())
        #cam = np.concatenate((cam1, cam2), axis=1)

        cam1 = torch.zeros(1, 196, 4)
        cam2 = torch.zeros(1, 784, 4)
        #cam = np.concatenate((cam1, cam2), axis=1)
        #cam[0, self.model.token, :] = 1
        
        cam3 = (mask_seg>0.5).float()
        #out_mask1 = self.model.coarse_mask1 
        #out_mask2 = self.model.coarse_mask2 
        #downsampled1 = F.interpolate(mask_seg, size=(14, 14), mode='bilinear', align_corners=False)
        #downsampled2 = F.interpolate(mask_seg, size=(28, 28), mode='bilinear', align_corners=False)
        
        #cam3 = (out_mask1>0.5).float()
        #cam4 = (out_mask2>0.5).float()
        #token = self.model.token
        for token in tokens:
            if token<196:
                cam1[0, token, :] = 1
                t_i = (token//14)*2
                t_j = (token%14)*2
                new_token = t_i*28+t_j
                cam2[0, new_token, :] = 1
            else:
                token = token-196
                cam2[0, token, :] = 1
                ############ transform token enc1 to token enc2############
                t_i = (token//28)//2
                t_j = (token%28)//2
                new_token = t_i*14+t_j
                cam1[0, new_token, :] = 1
            
        cam = np.concatenate((cam1, cam2), axis=1)        
        
        one_hot = cam#np.zeros((1, output.size()[-1]), dtype=np.float32)
        #one_hot[0, :] = cam[0,103,:]
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        #output[...,:3]=output[...,:3]/224
        #output = (output != 0).int()
        #print("out", one_hot.shape, output.shape)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        one_hot_m1 = cam3
        #one_hot_m2 = cam4
        
        one_hot_vector_m1 = one_hot_m1
        #one_hot_vector_m2 = one_hot_m2
        one_hot_m1 = one_hot_m1.requires_grad_(True)
        #one_hot_m1 = one_hot_m1.requires_grad_(True)
        
        one_hot_m1 = torch.sum(one_hot_m1.cuda()*mask_seg)
        #one_hot_m1 = torch.sum(one_hot_m1.cuda()*out_mask1)
        #one_hot_m2 = torch.sum(one_hot_m2.cuda()*out_mask2)
        #downsampled1 = torch.sum(downsampled1.cuda()*)
        

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #print(torch.tensor(one_hot_vector))
        #return 0
        return self.model.relprop(torch.tensor(one_hot_vector).to(input.device),one_hot_vector_m1, method=method,tokens=tokens,**kwargs)

    def generate_LRP2(self, input, preds, scores, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output, _ = self.model(input)
        kwargs = {"alpha": 1}
        #if index == None:
        #    index = 0
        cam = utils.utils.build_target_circle_explain(preds, scores, self.model.detection1.scaled_anchors, input.device)
        cam = np.array(cam.cpu())
        #one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        #one_hot[0, :] = 1
        one_hot_vector = cam
        one_hot = torch.from_numpy(cam).requires_grad_(True)
        one_hot = one_hot.cuda()

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #print(torch.tensor(one_hot_vector))
        #return 0
        return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)



    def generate_LRP_original(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)