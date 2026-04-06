import torch
import utils.counting as counting
import utils.utils
from ViT_CX.ViT_CX import ViT_CX
from explainability_generator import LRP
import cv2
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def anomaly_detection(boxes_explains, bins=256, box_size=48):
    min_, max_ = 0, 1
    bin_width = 1#/bins
    
    grays = np.zeros((len(boxes_explains), box_size, box_size))
    for i, box_im in enumerate(boxes_explains):
        grays[i,:,:] = cv2.resize(box_im, (box_size, box_size), interpolation = cv2.INTER_LINEAR)
    print("Boxes resized...")
    histograms = np.zeros((box_size, box_size, bins,1))
    for i in range(box_size):
        for j in range(box_size):
            counts, bins_ = np.histogram(grays[:, i, j], bins=bins, range=(min_, max_))
            counts = counts.reshape(bins, 1)/(len(boxes_explains)*bin_width)
            histograms[i,j, ...] = counts
    fingerprint = np.argmax(histograms, axis=2)[...,0]
    print("Histograms computed...")

    adnormal_i = []
    x_coords, y_coords = np.meshgrid(np.arange(box_size), np.arange(box_size))
    
    for k, box_im in enumerate(boxes_explains):
        print('\r', "%4d "%(k), end='')
        #box = np.array(box.cpu()).astype('int')
        #box_im = imgnp[box[1]:box[3],box[0]:box[2]]#, :]
        gray2 = grays[k, ...]#cv2.resize(box_im, (100, 100), interpolation = cv2.INTER_LINEAR)
        im = ((bins-1)*gray2).astype('uint8')
        #gray2 = box_im
        #gray2 = cv2.cvtColor(box_im, cv2.COLOR_BGR2GRAY)
        
        result = histograms[y_coords, x_coords, im, 0]
        # log-likelihood (stable)
        #result = np.log(result + 1e-12)
    
        S = np.sum(result) / (box_size*box_size)
    
        adnormal_i.append(S)

    scores = np.array(adnormal_i)
    
    # Step 2: Define bounds
    #lower_bound = 0.35
    #lower_bound = np.percentile(scores, 5)  # anomalies = lowest likelihood
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    
    # Step 3: Identify anomalies
    anomalies = scores[(scores < lower_bound)]
    print()
    print("Anomalies: %d, with thr: %.4f"%( len(anomalies), lower_bound ) )

    return scores, lower_bound, fingerprint#, histograms
    
    
def inference(model, im, obj, conf_thr, diou_thr, adjust_ij=(0, 0), xyxy=True, device='cuda'):
    model.eval()
    image_preds = model(im)
    #image_preds = image_preds.cpu()
    
    boxes_n = [torch.zeros([0, 4]) for _ in range(len(image_preds))]
    scores_n = [torch.zeros([0]) for _ in range(len(image_preds))]
    tokens_n = [torch.zeros([0]) for _ in range(len(image_preds))]
    adjust_n = [torch.zeros([0]) for _ in range(len(image_preds))]
    masks_n = [torch.zeros([224, 224]) for _ in range(len(image_preds))]
    
    #for image_i, prediction in enumerate(image_preds): #each image of the batch
    for image_i, (prediction, mask) in enumerate(zip(image_preds[0], image_preds[1])): #each image of the batch
        #filter occording to min confidence threshold
        #print("pred", prediction.shape)
        if obj == 'bbox':
            b = prediction[..., 4] >= conf_thr
        else:#'cbbox'
            b = prediction[..., 3] >= conf_thr
        tokens = b.nonzero()
        prediction = prediction[b]
        
        #apply non_max_suppression
        index_b = utils.utils.non_max_suppression(prediction, diou_thr, obj, DIoU = True, CIoU=True, device=device)
        prediction = prediction[index_b]
        tokens = tokens[index_b]
        #print(tokens)
        
        if obj=='bbox': 
            if xyxy: #else return the boxes xywh
                #convert from xywh to x1y1x2y2
                xc, yc, w, h = prediction[:,0].clone(), prediction[:,1].clone(), prediction[:,2].clone(), prediction[:,3].clone()
                prediction[:,0] = xc-w/2
                prediction[:,1] = yc-h/2
                prediction[:,2] = xc+w/2
                prediction[:,3] = yc+h/2
        
            boxes = prediction[:,0:4]
            scores = prediction[:,4].to(device) #scores of boxes
            
            #update boxes to orthomosaic map
            boxes[:,0] = boxes[:,0] + adjust_ij[1] #x1
            boxes[:,1] = boxes[:,1] + adjust_ij[0] #y1
            boxes[:,2] = boxes[:,2] + adjust_ij[1] #x2
            boxes[:,3] = boxes[:,3] + adjust_ij[0] #y2
            boxes = boxes.to(device)
        
        else:
            boxes = prediction[:,0:3]
            scores = prediction[:,3].to(device) #scores of boxes
            
            #update boxes to orthomosaic map
            boxes[:,0] = boxes[:,0] + adjust_ij[1] #xc
            boxes[:,1] = boxes[:,1] + adjust_ij[0] #yc
            boxes = boxes.to(device) 
            
        boxes_n[image_i] = boxes
        scores_n[image_i] = scores
        masks_n[image_i] = mask
        tokens_n[image_i] = tokens
        adjust_n[image_i] = ((adjust_ij,)*len(boxes))
        
    return boxes_n, scores_n, tokens_n, adjust_n, masks_n

########## METRICS ################
def deletion_auc(model, image, saliency, tokens, boxes, expanded_box=1.4, steps=100, plot=False):
    """
    image: [1,C,H,W]
    saliency: [H,W]
    """
    if len(boxes)==0:
        return 0

    B, C, H, W = image.shape
    #sal = saliency.view(-1)
    #sorted_idx = torch.argsort(sal, descending=True)
    sorted_idxs = []
    max_areas = []
    for i in range(len(boxes)):
        sal_b = np.zeros_like(saliency)
        #print(boxes[i][0])
        #cv2.circle(sal_b, (int(boxes[i][0]), int(boxes[i][1])), int(boxes[i][2]), color=(1), thickness=cv2.FILLED)
        
        xc, yc, r = boxes[i]
        r = r*expanded_box#expanded region
        x1 = xc-r if xc-r>0 else 0
        x2 = xc+r if xc+r<saliency.shape[0] else saliency.shape[0]
        y1 = yc-r if yc-r>0 else 0
        y2 = yc+r if yc+r<saliency.shape[1] else saliency.shape[1]

        box_x1y1 = (int(x1), int(y1))
        box_x2y2 = (int(x2), int(y2))

        cv2.rectangle(sal_b, box_x1y1, box_x2y2, color=(1), thickness=cv2.FILLED)
    
        sal_b = torch.where(torch.tensor(sal_b)==1, saliency, 0)
        
        #plt.imshow(sal_b)
        sal_b = sal_b.view(-1)
        sorted_idx = torch.argsort(sal_b, descending=True)
        sorted_idxs.append(sorted_idx)
        
        #print(boxes[i][2])
        #average_area += boxes[i][2]**2
        max_areas.append((x2-x1)*(y2-y1))
        
    image_flat = image.view(C, -1)
    scores_n = []
    model.to(device)
    model.eval()

    for i in range(steps):
        frac = i / steps
        #k = int(frac * average_area)

        perturbed = image_flat.clone()

        #if k > 0:
        for j in range(len(sorted_idxs)):
            #area_box = torch.pi*(boxes[j][2]**2)
            k = int(frac * max_areas[j])
            #print(k)

            perturbed[:, sorted_idxs[j][:k]] = 0  # remove pixels
            #break

        perturbed = perturbed.view(1, C, H, W).to(device)
        #print(perturbed.shape)
        #if i==63:
        #plt.imshow(perturbed[0,0,...].detach().cpu())

        pred, masks = model(perturbed)
        #print(pred.shape)
        scores = pred[0,tokens,-1].to('cpu')
        #print(scores.shape)

        scores_n.append(scores)
    scores_n = np.array(scores_n)
    auc=[]
    for i in range(scores_n.shape[1]):
        ###################
        curve = scores_n[:, i, 0]

        baseline = curve[0]
        final = curve[-1]

        # normalize between 0 and 1
        curve = (curve - final) / (baseline - final + 1e-8)
        curve = np.clip(curve, 0, 1)

        auc.append(np.trapz(curve, dx=1/steps))
        ##################¿
        if plot:
            plt.plot(scores_n[:,i,0])
    return np.array(auc).mean()#, scores

def insertion_auc(model, image, saliency, tokens, boxes, expanded_box=1.4, steps=100, plot=False):
    B, C, H, W = image.shape
    
    if len(boxes)==0:
        return 1
    
    blurred = F.avg_pool2d(image, kernel_size=51, stride=1, padding=25)
    #plt.imshow(blurred[0].permute(1, 2, 0).detach().cpu())

    #sal = saliency.view(-1)
    #sorted_idx = torch.argsort(sal, descending=True)
    sorted_idxs = []
    max_areas = []
    for i in range(len(boxes)):
        sal_b = np.zeros_like(saliency)

        xc, yc, r = boxes[i]
        r = r*expanded_box#expanded region
        x1 = xc-r if xc-r>0 else 0
        x2 = xc+r if xc+r<saliency.shape[0] else saliency.shape[0]
        y1 = yc-r if yc-r>0 else 0
        y2 = yc+r if yc+r<saliency.shape[1] else saliency.shape[1]

        box_x1y1 = (int(x1), int(y1))
        box_x2y2 = (int(x2), int(y2))

        cv2.rectangle(sal_b, box_x1y1, box_x2y2, color=(1), thickness=cv2.FILLED)
    
        sal_b = torch.where(torch.tensor(sal_b)==1, saliency, torch.tensor(-float("inf")))

        sal_b = sal_b.view(-1)
        sorted_idx = torch.argsort(sal_b, descending=True)
        sorted_idxs.append(sorted_idx)

        max_areas.append((x2-x1)*(y2-y1))
                            

    image_flat = image.view(C, -1)
    blurred_flat = blurred.view(C, -1)

    scores_n = []
    model.to(device)
    model.eval()

    for i in range(steps):
        frac = i / steps
        #k = int(frac * H * W)

        inserted = blurred_flat.clone()

        #if k > 0:
        for j in range(len(sorted_idxs)):
            k = int(frac * max_areas[j])

            inserted[:, sorted_idxs[j][:k]] = image_flat[:, sorted_idxs[j][:k]]

        inserted = inserted.view(1, C, H, W)

        pred, masks = model(inserted)
        scores = pred[0,tokens,-1].to('cpu')

        scores_n.append(scores)
    scores_n = np.array(scores_n)
    auc=[]
    for i in range(scores_n.shape[1]):
        curve = scores_n[:, i, 0]

        baseline = curve[0]
        max_val = curve.max()

        curve = (curve - baseline) / (max_val - baseline + 1e-8)
        curve = np.clip(curve, 0, 1)

        auc.append(np.trapz(curve, dx=1/steps))
        if plot:
            plt.plot(scores_n[:,i,0])
    return np.array(auc).mean()

def pointing_game_circle(saliency, boxes_pr, gt_mask, expanded_box=1.4):
    #B, C, H, W = image.shape
    if len(boxes_pr)==0:
        return 0
    
    gt_mask = gt_mask.flatten()
    hits = []
    for i in range(len(boxes_pr)):
        sal_b = np.zeros_like(saliency)

        xc, yc, r = boxes_pr[i]
        r = r*expanded_box#expanded region
        x1 = xc-r if xc-r>0 else 0
        x2 = xc+r if xc+r<saliency.shape[0] else saliency.shape[0]
        y1 = yc-r if yc-r>0 else 0
        y2 = yc+r if yc+r<saliency.shape[1] else saliency.shape[1]

        box_x1y1 = (int(x1), int(y1))
        box_x2y2 = (int(x2), int(y2))

        cv2.rectangle(sal_b, box_x1y1, box_x2y2, color=(1), thickness=cv2.FILLED)
    
        sal_b = torch.where(torch.tensor(sal_b)==1, saliency, torch.tensor(-float("inf")))

        sal_b = sal_b.view(-1)
        #sorted_idx = torch.argsort(sal_b, descending=True)
        idx = sal_b.argmax()
        
        hit = int(gt_mask[idx]==1)
        
        hits.append(hit)

    return np.mean(hits)
def energy_pointing(saliency, gt_mask):

    saliency = saliency / (saliency.sum() + 1e-8)

    energy = saliency[gt_mask==1].sum()

    return energy
def topk_localization(saliency, boxes_pr, gt_mask, k=75, expanded_box=1.4):
    #B, C, H, W = image.shape
    if len(boxes_pr)==0:
        return 0
    
    #gt_mask = gt_mask.flatten()
    hits = []
    for i in range(len(boxes_pr)):
        sal_b = np.zeros_like(saliency)

        xc, yc, r = boxes_pr[i]
        r = r*expanded_box#expanded region
        x1 = xc-r if xc-r>0 else 0
        x2 = xc+r if xc+r<saliency.shape[0] else saliency.shape[0]
        y1 = yc-r if yc-r>0 else 0
        y2 = yc+r if yc+r<saliency.shape[1] else saliency.shape[1]

        box_x1y1 = (int(x1), int(y1))
        box_x2y2 = (int(x2), int(y2))

        cv2.rectangle(sal_b, box_x1y1, box_x2y2, color=(1), thickness=cv2.FILLED)
    
        sal_b = torch.where(torch.tensor(sal_b)==1, saliency, torch.tensor(-float("inf")))

        sal_b = sal_b.view(-1)
        #sorted_idx = torch.argsort(sal_b, descending=True)
        #idx = sal_b.argmax()
        flat_idx = torch.argsort(sal_b.flatten(), descending=True)[:k]
        #print(flat_idx, sal_b.shape)
        ys, xs = np.unravel_index(flat_idx, saliency.shape)
        hit = gt_mask[ys, xs].sum() / k
        
        hits.append(hit)

    return np.mean(hits)

def explain_VIT_CX(model_dir, image, tokens, img_size=224, device='cuda'):
   
    #extract model features from model_dir
    n_model, num_blks, obj, N_channels, channels, loss_type, bitNet = counting.get_model_pars_from_dir(model_dir)
    
    import model.transformer
    model = model.transformer.TransformerObjectDetection(img_size, N_channels, n_model, num_blks, 
                                                         obj = obj, device=device, bitNet=bitNet).to(device)
    checkpoint = torch.load(model_dir, map_location=torch.device(device))
    if type(checkpoint) == dict:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict( checkpoint )
                
    image_cpu = image.cpu()
    target_layer=model.encoder1.blks[-1].ln1
    
    if len(tokens)>0:
        _,result = ViT_CX(model,image_cpu,target_layer,target_category=tokens,distance_threshold=0.1,gpu_batch=8)
        #result = result.mean(0)
    else:
        result = torch.zeros(224, 224)
        
    model.cpu()
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result

#REQUIRE SAVE GRADIENTS
def explain_chefer(model_dir, image, tokens, img_size=224, method='transformer_attribution', device='cuda'):
    #extract model features from model_dir
    n_model, num_blks, obj, N_channels, channels, loss_type, bitNet = counting.get_model_pars_from_dir(model_dir)
    
    import model.transformer
    model = model.transformer.TransformerObjectDetection(img_size, N_channels, n_model, num_blks, 
                                                         obj = obj, device=device, bitNet=bitNet).to(device)
    checkpoint = torch.load(model_dir, map_location=torch.device(device))
    if type(checkpoint) == dict:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict( checkpoint )
                
    attribution_generator = LRP(model)
    
    if len(tokens)>0:
        cam  = attribution_generator.generate_LRP(image, tokens, method=method)
        #cam  = attribution_generator.generate_LRP(image, toks, method='rollout')
        transformer_attribution = cam.detach().cpu().permute(1, 0)
        #transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = transformer_attribution.reshape(1, 1, 28, 28)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=8, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
        result = torch.tensor((transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min()))

    else:
        result = torch.zeros(224, 224)
        
    model.cpu()
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result