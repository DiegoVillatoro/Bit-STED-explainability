import torch
import torchvision
#import ttach as tta
import torch.nn as nn
from torchvision import transforms
#from torchvision.transforms import Compose, Normalize, ToTensor

import os
import sys

import copy
from skimage.transform import resize
from sklearn.cluster import AgglomerativeClustering
#from scipy.special import softmax

import numpy as np
import cv2
import gc

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

class causal_score(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(causal_score, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch


    def forward(self, x, masks_input, class_p):
        x=x[0]
        self.masks =  masks_input.reshape(-1, 1, *self.input_size)
        self.N = self.masks.shape[0]
        N = self.N 
        H=self.input_size[0]
        W=self.input_size[1]
        masks=self.masks
        # Generate the inverse of masks, i.e., 1-M_i
        masks_inverse=torch.from_numpy(np.repeat((1-self.masks.cpu().numpy())[:, :, np.newaxis,:], 3, axis=2)).cuda()
        
        # Generate the random Gaussian noise
        random_whole=torch.randn([N]+list((3,H,W))).cuda()* 0.1
        
        # Define tensors holding the mask images with noise and original images with noise 
        mask_image_with_noise=torch.empty((N,3,H,W)).cuda()
        original_image_with_noise=torch.empty((N,3,H,W)).cuda()

        for i in range(N):
            # noise to add: Gaussian noise*(1-M_i)
            noise_to_add=random_whole[i]*masks_inverse[i]
            temp_mask=masks[i]
            #thres=np.percentile(temp_mask.cpu().numpy(),50)
            #temp_mask_thres=temp_mask>thres
            mask_image_with_noise[i]=x*temp_mask+noise_to_add
            original_image_with_noise[i]=x+noise_to_add
        
        # Get prediction score for mask images with noise and original images with noise 
        stack_whole=torch.cat((mask_image_with_noise, original_image_with_noise), 0).cuda()
        p_whole = []
        for i in range(0, 2*N, self.gpu_batch):
            p_whole.append(self.model(stack_whole[i:min(i + self.gpu_batch, 2*N)]))
        p_whole = torch.cat(p_whole)
        p_mask_image_with_noise=p_whole[:N]
        p_original_image_with_noise=p_whole[N:]

        # Compute the final causal impact score
        CL = p_mask_image_with_noise.size(1)
        masks_divide=masks/torch.sum(masks,axis=0)
        p_final=p_mask_image_with_noise.data.transpose(0, 1)-p_original_image_with_noise.data.transpose(0, 1)+class_p
        sal = torch.matmul(p_final,masks_divide.view(N,H*W))
        sal = sal.view((CL, H, W))
        sal = sal / N 
        sal = sal.cpu()
        return sal

def get_cos_similar_matrix(v1, v2):
    num = torch.mm(v1,torch.transpose(v2, 0, 1)) 
    denom = torch.linalg.norm(v1,  dim=1).reshape(-1, 1) * torch.linalg.norm(v2,  dim=1)
    res = num / denom
    res[torch.isnan(res)] = 0
    return res

def norm_matrix(act):
    row_mins=torch.min(act,1).values[:, None]
    row_maxs=torch.max(act,1).values[:, None] 
    act=(act-row_mins)/(row_maxs-row_mins)
    return act

#-----------------------Function to Reshape the Extracted Feature Maps----------------------------------
def reshape_function_vit(tensor):#, height=14, width=14):
    _, p, _ = tensor.shape # [b, patches, n_model]
    height = int(np.sqrt(p))
    width = height
    result = tensor.reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    #print(result.shape)
    return result
def get_cam_weights(input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))

def get_loss(output, target_category):
    loss = 0
    for i in range(len(target_category)):
        loss = loss + output[i, target_category[i]]
    return loss

def get_cam_image(input_tensor,
                  target_layer,
                  target_category,
                  activations,
                  grads,
                  eigen_smooth=False):
    weights = get_cam_weights(input_tensor, target_layer,
                                   target_category, activations, grads)
    weighted_activations = weights[:, :, None, None] * activations
    if eigen_smooth:
        cam = get_2d_projection(weighted_activations)
    else:
        cam = weighted_activations.sum(axis=1)
    return cam

def get_target_width_height(input_tensor):
    width, height = input_tensor.size(-1), input_tensor.size(-2)
    return width, height

def compute_cam_per_layer(
        input_tensor,
        target_category, target_layers,
        eigen_smooth):
    activations_list = [a.cpu().data.numpy()
                        for a in featuremaps]
    grads_list = [g.cpu().data.numpy()
                  for g in gradients]
    target_size = get_target_width_height(input_tensor)

    cam_per_target_layer = []
    # Loop over the saliency image from every layer

    for target_layer, layer_activations, layer_grads in \
            zip(target_layers, activations_list, grads_list):
        cam = get_cam_image(input_tensor,
                                 target_layer,
                                 target_category,
                                 layer_activations,
                                 layer_grads,
                                 eigen_smooth)
        cam[cam<0]=0 # works like mute the min-max scale in the function of scale_cam_image
        scaled = scale_cam_image(cam, target_size)
        cam_per_target_layer.append(scaled[:, None, :])

    return cam_per_target_layer

def aggregate_multi_layers(cam_per_target_layer):
    cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
    cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
    result = np.mean(cam_per_target_layer, axis=1)
    return scale_cam_image(result)

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result
def save_activation(module, input, output):
    activation = output
    if reshape_function_vit is not None:
        activation = reshape_function_vit(activation)
    featuremaps.append(activation.cpu().detach())

def save_gradient(module, grad_input, grad_output):
    # Gradients are computed in reverse order
    grad = grad_output[0]
    if reshape_function_vit is not None:
        grad = reshape_function_vit(grad)
    #gradients = [grad.cpu().detach()] + gradients
    #gradients.append(grad.cpu().detach())
    gradients.insert(0, grad.cpu().detach().clone())