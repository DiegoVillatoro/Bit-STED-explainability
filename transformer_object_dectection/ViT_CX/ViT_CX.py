import os
import sys
# Get the current working directory
cwd = os.getcwd()
#sys.path.insert(1, str(cwd)+'\\ViT_CX')#add path for windows
#sys.path.insert(1, str(cwd)+'\\ViT_CX\\py_cam')
sys.path.insert(1, os.path.join(cwd, "ViT_CX"))#add path for linux
sys.path.insert(1, os.path.join(cwd, "ViT_CX", "py_cam"))
from cam import get_feature_map
from causal_score import causal_score
import numpy as np
import cv2
import copy
from skimage.transform import resize
from sklearn.cluster import AgglomerativeClustering
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
#cudnn.benchmark = True
import numpy as np
from matplotlib import pyplot as plt
########################################################################################################
########################################################################################################
import cv2
import ttach as tta
#from activations_and_gradients import ActivationsAndGradients
#from cam.utils.svd_on_activations import get_2d_projection

import torch
import torch.nn as nn
import numpy as np
from skimage.transform import resize
import random
from explainability_generator import LRP

def save_activation(module, input, output):
    activation = output
    if reshape_function_vit is not None:
        activation = reshape_function_vit(activation)
    #featuremaps.append(activation.cpu().detach())
    featuremaps.insert(0, activation.cpu().detach().clone())

def save_gradient(module, grad_input, grad_output):
    # Gradients are computed in reverse order
    grad = grad_output[0]
    if reshape_function_vit is not None:
        grad = reshape_function_vit(grad)
    #gradients = [grad.cpu().detach()] + gradients
    gradients.append(grad.cpu().detach())
    #gradients.insert(0, grad.cpu().detach().clone())
def get_loss(output, target_category):
    loss = torch.tensor(0)
    for i in range(len(target_category)):
        loss = loss + output[0, target_category[i],-1]
        #loss = loss + output[i, target_category[i]]
    return loss
class causal_score(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(causal_score, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch


    def forward(self, x, masks_input, target_category, class_p):
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
            #p_whole.append(self.model(stack_whole[i:min(i + self.gpu_batch, 2*N)]))
            #pred = self.model(stack_whole[i:min(i + self.gpu_batch, 2*N)])
            pred,_ = self.model(stack_whole[i:min(i + self.gpu_batch, 2*N)])
            pred = pred[...,3].view(-1, 980)#3rd for cbbox
            p_whole.append(pred)            
        p_whole = torch.cat(p_whole)
        #dets = torch.sum(p_whole>0.5, -1)
        p_mask_image_with_noise=p_whole[:N]
        p_original_image_with_noise=p_whole[N:]
        
        """
        print("stack",stack_whole.shape)
        test = stack_whole.detach().to('cpu').reshape(-1,3,224,224).permute(0, 2, 3, 1)
        print(stack_whole.shape)
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(test[i][...,[2, 1, 0]])
            plt.title(int(dets[i].item()))
            print(test[i].min(), test[i].max())
            plt.axis('off')
            if i>len(test)-2:
                break
        """
        
        #print("p whole", p_whole.shape, p_mask_image_with_noise.shape)#torch.Size([10, 980]) torch.Size([5, 980])

        # Compute the final causal impact score
        CL = p_mask_image_with_noise.size(1)
        masks_divide=masks/(torch.sum(masks,axis=0)+ 1e-6)
        
        p_mask_image_with_noise = p_mask_image_with_noise[:,target_category]
        p_original_image_with_noise = p_original_image_with_noise[:,target_category]
        
        #print("antes p final", p_mask_image_with_noise.data.transpose(0, 1).shape, torch.tensor(class_p).to('cuda').view(-1, 1).shape)
        p_final=p_mask_image_with_noise.detach().transpose(0, 1)-p_original_image_with_noise.detach().transpose(0, 1)+torch.tensor(class_p).view(-1, 1).to('cuda')
    
        sal = torch.matmul(p_final, masks_divide.view(N, H*W))
        class_p = torch.as_tensor(class_p, device=sal.device).view(1, -1)
        sal2 = (torch.matmul(class_p, sal)/ N).cpu()
        sal2 = sal2.view(H, W)
                
        sal = sal.view((len(target_category), H, W))
        sal = sal / N 
        sal = sal.cpu()
        
        class_p
        return sal, sal2
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.featuremaps = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compitability with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def __call__(self, x):
        self.gradients = []
        self.featuremaps = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
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
def get_cam_image(
                  input_tensor,
                  target_layer,
                  target_category,
                  activations,
                  grads,
                  eigen_smooth=False):
    weights = get_cam_weights(input_tensor, target_layer,
                                   target_category, activations, grads)
    #print("cams",weights.shape)
    #print(activations.shape)
    #print(weights[:, :, None, None].shape)
    weighted_activations = grads*activations#weights[:, :, None, None] * activations

    if eigen_smooth:
        cam = get_2d_projection(weighted_activations)
    else:
        cam = weighted_activations.sum(axis=1)
    #print(cam.shape)
    #print(cam)
    return cam
def get_cam_weights(
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        #grads shape : b, n features, h, w (1, 64, 28, 28)
        return np.mean(grads, axis=(2, 3))# (b, n_features) (1, 64)
########################################################################################################
########################################################################################################
########################################################################################################

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
# Users may need to adjust the reshape_transform function for different ViT Models
# For instance, in DEiT, the first two tokens are [CLS] and [Dist], should the patch tokens start from the third element, thus we shall have:
# result = tensor[:, 2:, :].reshape(tensor.size(0),height, width, tensor.size(2))
#def reshape_function_vit(tensor, height=14, width=14):
#    result = tensor[:, 1:, :].reshape(tensor.size(0),
#                                      height, width, tensor.size(2))
#    result = result.transpose(2, 3).transpose(1, 2)
#    return result
def reshape_function_vit(tensor):#, height=14, width=14):
    _, p, _ = tensor.shape # [b, patches, n_model]
    height = int(np.sqrt(p))
    width = height
    result = tensor.reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


#--------------------------------Function to make the ViT-CX explanation-----------------------------
'''
1. model: ViT model to be explained;
2. image: input image in the tensor form (shape: [1,#channels,width,height]);
3. target_layer: the layer to extract feature maps  (e.g. model.blocks[-1].norm1);
4. target_category: int (class to be explained), in default - the top_1 prediction class;
5. distance_threshold: float between [0,1], distance threshold to make the clustering where  
   feature maps with similarity<distance_threshold will be merged together, in default - 0.1; 
6. reshape_function: function to reshape the extracted feature maps, in default - a reshape function for vanilla vit;
7. gpu_batch: batch size the run the prediction for the masked images, in default - 50.
'''
gradients = []
featuremaps = []
def ViT_CX(model,image,target_layer,target_category=None,distance_threshold=0.1,reshape_function=reshape_function_vit,gpu_batch=50):
    
    image=image.cuda()
    model_softmax=copy.deepcopy(model)
    model=model.eval()
    model=model.cuda()
    #model_softmax = nn.Sequential(model_softmax, nn.Softmax(dim=1))
    model_softmax = model_softmax.eval()
    model_softmax = model_softmax.cuda()
    for p in model_softmax.parameters():
        p.requires_grad = False
    #y_hat = model_softmax(image)
    y_hat,_ = model_softmax(image)
    y_hat_1=y_hat.detach().cpu().numpy()[0,:,3]
    if target_category==None:
        top_1=np.argsort(y_hat_1[...,3])[::-1][0]# selection of 3rd position for score in cbbox
        #top_1=np.argsort(y_hat_1)[::-1][0]
        target_category = np.array([top_1])
        
    class_p=y_hat_1[target_category]

    input_size=(image.shape[2],image.shape[3])
    transform_fp = transforms.Compose([transforms.Resize(input_size)])
    
    gradients.clear()
    featuremaps.clear()
    # Extract the ViT feature maps 
    #GetFeatureMap = get_feature_map(model=model,target_layers=[target_layer],use_cuda=True,reshape_transform=reshape_function)
    #_ = GetFeatureMap(input_tensor=image,target_category=target_category)
    #feature_map=GetFeatureMap.featuremap_and_grads.featuremaps[0][0].cuda()
    target_layers=[target_layer]   
    handles = []
    for target_layer in target_layers:
        handles.append(
                target_layer.register_forward_hook(
                    save_activation))
        # Backward compitability with older pytorch versions:
        if hasattr(target_layer, 'register_full_backward_hook'):
            handles.append(
                target_layer.register_full_backward_hook(
                    save_gradient))
        else:
            handles.append(
                target_layer.register_backward_hook(
                    save_gradient))
    #output = model(image, explain=True)
    output,_ = model(image, explain=True)
        
    model.zero_grad()
    #print("voy aqui ", target_category)
    loss = get_loss(output, target_category)
    loss.backward(retain_graph=True)
    
    """
    test = gradients[0].to('cpu')[0]
    #test = featuremaps[0].to('cpu')[0]
    print(test.shape)
    plt.figure(figsize=(8, 8))
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(test[0])
        plt.axis('off')
       """ 
    cam_per_layer = compute_cam_per_layer(image, target_category, target_layers,
                                                       eigen_smooth=False)
    _ = aggregate_multi_layers(cam_per_layer)
    
    feature_map=featuremaps[0][0].cuda()
       
    for handle in handles:
        handle.remove()
    handles.clear()
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    # Reshape and normalize the ViT feature maps to get ViT masks
    feature_map=transform_fp(feature_map)
    #print(feature_map.shape)
    mask=norm_matrix(torch.reshape(feature_map, (feature_map.shape[0],input_size[0]*input_size[1])))
    
    # Compute the pairwise cosine similarity and distance of the ViT masks
    similarity = get_cos_similar_matrix(mask,mask)
    distance = 1 - similarity

    # Apply the  AgglomerativeClustering with a given distance_threshold
    cluster = AgglomerativeClustering(n_clusters = None, distance_threshold=distance_threshold,metric='precomputed', linkage='complete') 
    cluster.fit(distance.cpu())
    cluster_num=len(set(cluster.labels_))
    #print('number of masks after the clustering:'+str(cluster_num))
      
    # Use the sum of a clustering as a representation of the cluster
    cluster_labels=cluster.labels_
    cluster_labels_set=set(cluster_labels)
    mask_clustering=torch.zeros((len(cluster_labels_set),input_size[0]*input_size[1])).cuda()
    for i in range(len(mask)):
        mask_clustering[cluster_labels[i]]+=mask[i]

    # normalize the masks
    mask_clustering_norm=norm_matrix(mask_clustering).reshape((len(cluster_labels_set),input_size[0],input_size[1]))
    #"""
    test = mask_clustering_norm.to('cpu').reshape(-1,224,224)
    #print(test.shape)
    #plt.figure(figsize=(8, 8))
    #for i in range(64):
    #    plt.subplot(8, 8, i+1)
    #    plt.imshow(test[i])
    #    print(test[i].min(), test[i].max())
    #    plt.axis('off')
    #    if i>len(test)-2:
    #        break
    #"""    
    # compute the causal impact score
    compute_causal_score = causal_score(model_softmax, (input_size[0], input_size[1]),gpu_batch=gpu_batch)
    #sal = compute_causal_score(image,mask_clustering_norm, target_category, class_p)[target_category].cpu().numpy()
    sal, sal2 = compute_causal_score(image,mask_clustering_norm, target_category, class_p)
    sal = sal.cpu().numpy()
    sal2 = sal2.cpu().numpy()
    
    import gc
    # Break all references
    model.cpu()
    model_softmax.cpu()
    del model
    del model_softmax
    del y_hat
    del transform_fp
    
    #del output
    #del handles
    #del loss
    #del cam_per_layer
    del feature_map
    
    del image            # Remove reference

    # Force cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    #GetFeatureMap.featuremap_and_grads.release()
    return torch.tensor(sal),torch.tensor(sal2)
