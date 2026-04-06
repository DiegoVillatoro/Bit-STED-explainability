import pandas as pd
import utils.utils
import cv2
from matplotlib import pyplot as plt
import torch
import numpy as np
import rasterio as rio
from utils.utils import VI
from model.trainer import count_parameters
from model.trainer import model_size
from model.transformer_encoder import BitLinear
import time

def read_orthomap_full(map_dir):
    with rio.open(map_dir) as img :
        imgnp = img.read() 
    #B = imgnp[0,:,:]; 
    #G = imgnp[1,:,:]
    #R = imgnp[2,:,:]
    #Re = imgnp[3,:,:]
    #NIR = imgnp[4,:,:]
    print("Orthomap read at "+map_dir)
    return imgnp

def get_model_pars_from_dir(model_dir):
    #split dir of form .../dtenc_subName_modelRun_obj_channels_loss_BitNet/weights.pt
    model_pars = model_dir.split('/')[-2].split('_')
    if model_pars[0] == "explain":
        model_pars.pop(0)
    if len(model_pars)>7:
        print("Error in model dir")
        return 128, 1, 'cbbox', 3, 'bgr', 'diou', True#default
    sub_name = model_pars[2]
    if sub_name=='nano':
        n_model=128
        num_blks=1
    elif sub_name=='small':
        n_model=256
        num_blks=1
    elif sub_name=='medium':
        n_model=512
        num_blks=2
    elif sub_name=='large':
        n_model=1024
        num_blks=2
    else:
        print("Model size error")
        n_model=128
        num_blks=1

    obj = model_pars[3]
    channels = model_pars[4]
    if channels=='bgr':
        N_channels = 3
    elif channels=='multispectral':
        N_channels = 5
    elif channels=='viSH':
        N_channels = 6
    elif channels=='vi':
        N_channels = 6
    else:
        print("Channel selection error")
        N_channels = 3
    loss_type = model_pars[5]
    if model_pars[6]=='BitNet':
        bitNet=True
    else:
        bitNet=False
        
    return n_model, num_blks, obj, N_channels, channels, loss_type, bitNet

#im shape : batch size, n channels, weight image, height image
#return list of boxes and list of scores for each image
def inference(model, im, obj, conf_thr, diou_thr, adjust_ij=(0, 0), xyxy=True, device='cuda'):
    model.eval()
    #flag1 = time.time()
    image_preds = model(im)
    #flag2 = time.time()
    #image_preds = image_preds.cpu()
    
    boxes_n = [torch.zeros([0, 4]) for _ in range(len(image_preds))]
    scores_n = [torch.zeros([0]) for _ in range(len(image_preds))]
    classes_n = [torch.zeros([0]) for _ in range(len(image_preds))]
    masks_n = [torch.zeros([224, 224]) for _ in range(len(image_preds))]
    
    for image_i, (prediction, mask) in enumerate(zip(image_preds[0], image_preds[1])): #each image of the batch
    #for image_i, prediction in enumerate(image_preds): #each image of the batch
        #filter occording to min confidence threshold
        if obj == 'bbox':
            prediction = prediction[prediction[..., 4] >= conf_thr]
        else:#'cbbox'
            prediction = prediction[prediction[..., 3] >= conf_thr]
        
        #apply non_max_suppression
        index_b = utils.utils.non_max_suppression(prediction, diou_thr, obj, DIoU = True, CIoU=True, device=device)
        prediction = prediction[index_b]
        
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
            #if prediction.shape[1]>5:
            #    cls = prediction[:,5].to(device)
            #else:
            cls = torch.zeros(len(boxes)).to(device)
            
            #update boxes to orthomosaic map
            boxes[:,0] = boxes[:,0] + adjust_ij[1] #x1
            boxes[:,1] = boxes[:,1] + adjust_ij[0] #y1
            boxes[:,2] = boxes[:,2] + adjust_ij[1] #x2
            boxes[:,3] = boxes[:,3] + adjust_ij[0] #y2
            boxes = boxes.to(device)
        
        else:
            boxes = prediction[:,0:3]
            scores = prediction[:,3].to(device) #scores of boxes
            #if prediction.shape[1]>4:
            #    cls = prediction[:,4].to(device)
            #else:
            cls = torch.zeros(len(boxes)).to(device)
            
            #update boxes to orthomosaic map
            boxes[:,0] = boxes[:,0] + adjust_ij[1] #xc
            boxes[:,1] = boxes[:,1] + adjust_ij[0] #yc
            boxes = boxes.to(device) 
            
        boxes_n[image_i] = boxes
        scores_n[image_i] = scores
        classes_n[image_i] = cls
        masks_n[image_i] = mask
        
    return boxes_n, scores_n, classes_n, masks_n

def get_boxes_fraction_map(model, imgnp, ch, obj, conf_thr, iou_thr, diou_thr, img_size=224, device='cuda', bitNet=False, n_classes=2, frac_size=336, nms=False):
    
    #load fraction   
    center = np.array(imgnp[0,:,:].shape)//2
    x1 = center[0]-frac_size//2 if center[0]-frac_size//2 > 0 else 0
    y1 = center[1]-frac_size//2 if center[1]-frac_size//2 > 0 else 0
    x2 = center[0]+frac_size//2 if center[0]+frac_size//2 < imgnp[0,:,:].shape[0] else imgnp[0,:,:].shape[0]
    y2 = center[1]+frac_size//2 if center[1]+frac_size//2 < imgnp[0,:,:].shape[1] else imgnp[0,:,:].shape[1]
    coords = np.array([[x1, y1], [x2, y2]]) #x1y1x2y2
    
    B = imgnp[0,:,:][coords[0, 0]:coords[1, 0], coords[0, 1]:coords[1, 1]]
    G = imgnp[1,:,:][coords[0, 0]:coords[1, 0], coords[0, 1]:coords[1, 1]]
    R = imgnp[2,:,:][coords[0, 0]:coords[1, 0], coords[0, 1]:coords[1, 1]]
    Re = imgnp[3,:,:][coords[0, 0]:coords[1, 0], coords[0, 1]:coords[1, 1]]
    NIR = imgnp[4,:,:][coords[0, 0]:coords[1, 0], coords[0, 1]:coords[1, 1]]
    
    if ch == 'bgr':
        img = cv2.merge((B, G, R))
    elif ch == 'multispectral':
        img = cv2.merge((B, G, R, Re, NIR))
    elif ch == 'vi':
        VI_N = ['NDCI', 'TSAVI2', 'ARVI']
        channels = (B, G, R, Re, NIR) 
        VI_1 = VI(channels, VI_N[0], ranges=None)
        VI_2 = VI(channels, VI_N[1], ranges=None)
        VI_3 = VI(channels, VI_N[2], ranges=None)
        img = cv2.merge((B, G, R, VI_1, VI_2, VI_3))       
    elif ch == 'viShadowAttenuation': #incomplete
        VI_1 = imgnp[5,:,:][coords[0, 0]:coords[1, 0], coords[0, 1]:coords[1, 1]]
        VI_2 = imgnp[6,:,:][coords[0, 0]:coords[1, 0], coords[0, 1]:coords[1, 1]]
        VI_3 = imgnp[7,:,:][coords[0, 0]:coords[1, 0], coords[0, 1]:coords[1, 1]]
        img = cv2.merge((B, G, R, VI_1, VI_2, VI_3))
    else:
        print("¡¡¡Error loading fraction map!!!")
    print("Fraction map loaded: "+str(imgnp.shape) )   
    
    all_boxes = torch.tensor([]).to(device)
    all_scores = torch.tensor([]).to(device)
    all_categories = torch.tensor([]).to(device)
    
    preds_boxes_test = []
    preds_scores_test = []
    imgs_test = []
    
    for i in range(0, img.shape[0], img_size//2):
        for j in range(0, img.shape[1], img_size//2):
            i2 = i+img_size if i+img_size <= img.shape[0] else img.shape[0]
            j2 = j+img_size if j+img_size <= img.shape[1] else img.shape[1]

            im = np.zeros((img_size, img_size, img.shape[-1]))
            im[0:i2-i, 0:j2-j, :] = img[i:i2,j:j2, :]
            im = (torch.Tensor(im).permute(2, 0, 1).unsqueeze(0)).to(device) #1 x n channels x W x H
            
            boxes, scores, _, _  = inference(model, im, obj, conf_thr, diou_thr, adjust_ij=(i, j), device=device)
            boxes, scores = boxes[0], scores[0]
            print('\r', "Tile %4d:%4d, %4d:%4d, Objects Detected: %3d"%(i, i2, j, j2, len(boxes)), end='')
            
            if boxes!=None:
                all_boxes = torch.concat((all_boxes, boxes))
                all_scores = torch.concat((all_scores, scores))
            
            #for plot testing take in account only completed patches
            if len(preds_boxes_test)<4 and i+img_size<=img.shape[0] and j+img_size<=img.shape[1]: 
                boxes, scores, _, _ = inference(model, im, obj, conf_thr, diou_thr, adjust_ij=(0, 0), device=device)
                boxes, scores = boxes[0], scores[0]
                preds_boxes_test.append(boxes)
                preds_scores_test.append(scores)
                imgs_test.append(im)
    
    print()
    print("All boxes detected: %d"%(len(all_boxes)))
    #bboxes x1y1x2y2 and circles xc yc r
    index_filter_boxes = utils.utils.non_max_suppression_area(all_boxes, all_scores, iou_threshold=iou_thr, obj=obj, device=device)
    if nms:
        predictions = torch.cat((all_boxes, all_scores), dim=1)
        index_filter_boxes = utils.utils.non_max_suppression(predictions, iou_thr, obj, DIoU = False, CIoU=False, device=device)
    N_boxes = len(index_filter_boxes)
    print("Boxes filtered detected: %d"%(N_boxes))
    
    boxes_filtered = all_boxes[[index_filter_boxes]]
    scores_filtered = all_scores[[index_filter_boxes]]
    
    return img, all_boxes, all_scores, boxes_filtered, scores_filtered, preds_boxes_test, preds_scores_test, imgs_test, obj

def load_model(model_dir, img_size, N_channels, n_model, num_blks, obj, device, bitNet):
    import model.transformer
    model = model.transformer.TransformerObjectDetection(img_size, N_channels, n_model, num_blks, 
                                                         obj = obj, device=device, 
                                                         bitNet=bitNet).to(device)
    checkpoint = torch.load(model_dir, map_location=torch.device(device))
    if type(checkpoint) == dict:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict( checkpoint )
    print("Model loaded")
    
    return model
    
def get_boxes_map(model, imgnp, ch, obj, conf_thr, iou_thr, diou_thr, img_size=224, device='cuda', nms=False):
    
    n_pars = count_parameters(model)
    n_size_mb = model_size(model)
    
    B = imgnp[0,:,:]
    G = imgnp[1,:,:]
    R = imgnp[2,:,:]
    Re = imgnp[3,:,:]
    NIR = imgnp[4,:,:]
    
    #imgnp = imgnp.transpose((1, 2, 0)) # W x H x N channels
    #imgnp = imgnp[:,:,channels]
    
    if ch == 'bgr':
        img = cv2.merge((B, G, R))
    elif ch == 'multispectral':
        img = cv2.merge((B, G, R, Re, NIR))
    elif ch == 'vi':
        VI_N = ['NDCI', 'TSAVI2', 'ARVI']
        channels = (B, G, R, Re, NIR) 
        VI_1 = VI(channels, VI_N[0], ranges=None)
        VI_2 = VI(channels, VI_N[1], ranges=None)
        VI_3 = VI(channels, VI_N[2], ranges=None)
        img = cv2.merge((B, G, R, VI_1, VI_2, VI_3))       
    elif ch == 'viSH': #incomplete
        VI_1 = imgnp[5,:,:]
        VI_2 = imgnp[6,:,:]
        VI_3 = imgnp[7,:,:]
        img = cv2.merge((B, G, R, VI_1, VI_2, VI_3))
    else:
        print("Error loading map for "+ch)  
    print("Map loaded: "+str(imgnp.shape) )
    
    all_boxes = torch.tensor([]).to(device)
    all_scores = torch.tensor([]).to(device)
    all_categories = torch.tensor([]).to(device)
    all_masks = []
    times_predict = []
    
    for i in range(0, img.shape[0], img_size//2):
        row_mask = []
        for j in range(0, img.shape[1], img_size//2):
            i2 = i+img_size if i+img_size < img.shape[0] else img.shape[0]-1
            j2 = j+img_size if j+img_size < img.shape[1] else img.shape[1]-1

            im = np.zeros((img_size, img_size, img.shape[-1]))
            im[0:i2-i, 0:j2-j, :] = img[i:i2,j:j2, :]
            im = (torch.Tensor(im).permute(2, 0, 1).unsqueeze(0)).to(device) #1 x n channels x W x H
            
            start_b = time.time()        
            boxes, scores, classes,masks = inference(model, im, obj, conf_thr, diou_thr, adjust_ij=(i, j), device=device)            
            end_b = time.time()
            times_predict.append(end_b-start_b)
            #times_predict.append(ts)
            boxes, scores, classes = boxes[0], scores[0], classes[0]
            print('\r', "Tile %4d:%4d, %4d:%4d, Objects Detected: %5d, im_shape %3d %3d %3d %3d, t:%2.2f"%(i, i2, j, j2, len(boxes), im.shape[0], im.shape[1], im.shape[2], im.shape[3], (end_b-start_b)*1000), end='')
            
            if boxes!=None:
                all_boxes = torch.concat((all_boxes, boxes))
                all_scores = torch.concat((all_scores, scores))
                all_categories = torch.concat((all_categories, classes))
                
            masks = masks[0][0].detach().cpu()
            row_mask.append(masks)
        all_masks.append(row_mask)
                
    print()
    print("All boxes detected: %d"%(len(all_boxes)))
    #bboxes x1y1x2y2 and circles xc yc r
    if nms:
        predictions = torch.cat((all_boxes, all_scores.unsqueeze(1)), dim=1)
        index_filter_boxes = utils.utils.non_max_suppression(predictions, iou_thr, obj, DIoU = False, CIoU=False, x1y1x2y2=True, device=device)
    else:
        index_filter_boxes = utils.utils.non_max_suppression_area(all_boxes, all_scores, iou_threshold=iou_thr, obj=obj, device=device)
        
    N_boxes = len(index_filter_boxes)
    print("Boxes filtered detected: %d"%(N_boxes))
    
    boxes_filtered = all_boxes[[index_filter_boxes]]
    scores_filtered = all_scores[[index_filter_boxes]]
    classes_filtered = all_categories[[index_filter_boxes]]
    
    #masks_filtered = [all_masks[i] for i in index_filter_boxes] 
    
    return imgnp, all_boxes, all_scores, all_categories, boxes_filtered, scores_filtered, classes_filtered, all_masks, n_pars, n_size_mb, np.array(times_predict).mean(), obj

#def get_counting_orthomap(folder_maps, zone_name, model, channels, bitNet, obj, imgnp, conf_thr, iou_thr, diou_thr, img_size, device, nms, n_classes=1):
    
    #n_model, num_blks, obj, N_channels, channels, loss_type, bitNet = get_model_pars_from_dir(model_dir)
    
#    result = get_boxes_map(model, imgnp, channels, obj, conf_thr, iou_thr, diou_thr, img_size, device, bitNet, n_classes, nms)
#    imgnp, all_boxes, all_scores, all_categories, boxes_filtered, scores_filtered, classes_filtered, n_pars, n_size_mb, time_avg_pred, obj = result

def get_metrics_orthomap(folder_maps, zone_name, device, all_boxes, all_scores, all_categories, boxes_filtered, scores_filtered, classes_filtered, n_pars, n_size_mb, time_avg_pred, obj, n_classes=1):  
    ############ Object Detection Metrics ##############################
    file_name = folder_maps + zone_name + "_labels.txt"
    boxes_gt = torch.tensor([]).to(device)
    classes_gt = torch.tensor([]).to(device)
    with open(file_name, "r") as f:
        lines = f.readlines()
    for line in lines:
        data = np.fromstring(line, sep=' ')
        
        cnt = data[5:]
        cnt = cnt.reshape(-1, 2)
        if obj == 'bbox':
            box_gt = torch.tensor(np.array([data[1:5]])).to(device)#x1y1x2y2
        else:
            (xc,yc),radius = cv2.minEnclosingCircle(cnt.astype('int'))
            box_gt = torch.tensor(np.array([[xc, yc, radius]])).to(device)
        #class_gt = torch.tensor(np.array([data[0]])).to(device)
        class_gt = torch.tensor(np.array([0])).to(device)#all 0 when onnly one class

        boxes_gt = torch.concat((boxes_gt, box_gt))
        classes_gt = torch.concat((classes_gt, class_gt))
    print("Ground Truth read for %s with %d boxes"%(zone_name, boxes_gt.shape[0]))
    
    #classes_gt = torch.zeros_like(classes_gt)#to check only one class
    #classes_filtered=1
    if boxes_filtered.shape[0]>0:
        mAP25, N_agaves_mAP25, aps25 = utils.utils.mAP(boxes_gt, boxes_filtered, classes_gt, classes_filtered, num_classes=n_classes, scores=scores_filtered, obj=obj, iou_thr=0.25, DIoU=True, CIoU=True, device=device)
        mAP40, N_agaves_mAP40, aps40 = utils.utils.mAP(boxes_gt, boxes_filtered, classes_gt, classes_filtered, num_classes=n_classes, scores=scores_filtered, obj=obj, iou_thr=0.4, DIoU=True, CIoU=True, device=device)
        mAP50, N_agaves_mAP50, aps50 = utils.utils.mAP(boxes_gt, boxes_filtered, classes_gt, classes_filtered, num_classes=n_classes, scores=scores_filtered, obj=obj, iou_thr=0.5, DIoU=True, CIoU=True, device=device)
        mAP75, N_agaves_mAP75, aps75 = utils.utils.mAP(boxes_gt, boxes_filtered, classes_gt, classes_filtered, num_classes=n_classes, scores=scores_filtered, obj=obj, iou_thr=0.75, DIoU=True, CIoU=True, device=device)
        mAP95, N_agaves_mAP95, aps95 = utils.utils.mAP(boxes_gt, boxes_filtered, classes_gt, classes_filtered, num_classes=n_classes, scores=scores_filtered, obj=obj, iou_thr=0.95, DIoU=True, CIoU=True, device=device)
    else:#if no detected boxes
        mAP25, N_agaves_mAP25 = 0.0, 0
        mAP40, N_agaves_mAP40 = 0.0, 0
        mAP50, N_agaves_mAP50 = 0.0, 0
        mAP75, N_agaves_mAP75 = 0.0, 0
        mAP95, N_agaves_mAP95 = 0.0, 0

    #error_mAP50 = '%1.2f%%' %(((N_agaves_mAP50-len(boxes_gt))/len(boxes_gt))*100)
    N_agaves_real = boxes_gt.shape[0]
    #error_mAP50 = ((N_agaves_mAP50-N_agaves_real)/N_agaves_real)*100
    #error_mAP75 = ((N_agaves_mAP75-N_agaves_real)/N_agaves_real)*100
    #error_mAP95 = ((N_agaves_mAP95-N_agaves_real)/N_agaves_real)*100

    #detection metrics
    #sort_index = scores_filtered.argsort(descending=True) #sort of the largest score to the smallest
    boxes_pred = boxes_filtered#[sort_index]
    classes_pred = classes_filtered#[sort_index]
    #classes_pred = torch.zeros_like(scores_filtered)
    average_precisions = []
    epsilon = 1e-6
    corrects=0
    
    TP_index_25 = []
    FP_index_25 = []
    FN_index_25 = []
    
    TP_index_40 = []
    FP_index_40 = []
    FN_index_40 = []
    
    TP_index_50 = []
    FP_index_50 = []
    FN_index_50 = []
    
    #Get ground truth data considering only one class
    for c in range(n_classes):
        boxes_gt_c = torch.tensor([]).to(device)
        boxes_pred_c = torch.tensor([]).to(device)
        selected_ind_class_gt = []
        selected_ind_class_pred = []
        for i, (box_gt, class_gt) in enumerate(zip(boxes_gt, classes_gt)):
            if class_gt == c:
                boxes_gt_c = torch.concat((boxes_gt_c, box_gt.unsqueeze(0) ))
                selected_ind_class_gt.append(i)
        for i, (box_pred, class_pred) in enumerate(zip(boxes_pred, classes_pred)):
            if class_pred == c:
                boxes_pred_c = torch.concat((boxes_pred_c, box_pred.unsqueeze(0) ))
                selected_ind_class_pred.append(i)
        
        if boxes_pred_c.shape[0]>0:
            if obj == 'bbox':
                ious = utils.utils.bbox_iou(boxes_gt_c, boxes_pred_c, x1y1x2y2=True, align=False, DIoU = True, CIoU = True)
            else:
                ious = utils.utils.cbbox_iou(boxes_gt_c, boxes_pred_c, align=False, DIoU = True)
        else:
            ious = torch.zeros(len(boxes_gt_c), 0).to(device)
        print("IoU with GT computed")
        
        ###################### Detection according diou_thr 0.25 ###########################
        ious_filter = ious.where(ious>0.25,torch.tensor(0.)) #match with some overlap
        mappings = utils.utils.get_mappings(ious_filter) if boxes_pred_c.shape[0]>0 else torch.zeros(len(boxes_gt_c), 0).to(device)

        TP_index1 = mappings.sum(0).eq(1).long() #sum cols and select only the results eq to 1 and conver boolean to long : array of len of predictions
        FP_index1 = mappings.sum(0).eq(0).long() #sum cols and select only the results eq to 0 and conver boolean to long : array of len of predictions
        FN_index1 = mappings.sum(1).eq(0).long() #sum rows and select only the results eq to 0 and conver boolean to long : array of len of ground truths

        #TP_index2 = torch.where(TP_index2[sort_index.argsort()])[0]
        #FP_index2 = torch.where(FP_index2[sort_index.argsort()])[0]
        #FN_index2 = torch.where(FN_index2)[0]
        TP_index1 = torch.where(TP_index1)[0]; TP_index1 = np.array(selected_ind_class_pred)[TP_index1.cpu()]
        FP_index1 = torch.where(FP_index1)[0]; FP_index1 = np.array(selected_ind_class_pred)[FP_index1.cpu()]
        FN_index1 = torch.where(FN_index1)[0]; FN_index1 = np.array(selected_ind_class_gt)[FN_index1.cpu()]
        
        TP_index_25.append(TP_index1)
        FP_index_25.append(FP_index1)
        FN_index_25.append(FN_index1)
        
        ###################### Detection according diou_thr 0.4 ###########################
        ious_filter = ious.where(ious>0.4,torch.tensor(0.)) #match with some overlap
        mappings = utils.utils.get_mappings(ious_filter) if boxes_pred_c.shape[0]>0 else torch.zeros(len(boxes_gt_c), 0).to(device)

        TP_index2 = mappings.sum(0).eq(1).long() #sum cols and select only the results eq to 1 and conver boolean to long : array of len of predictions
        FP_index2 = mappings.sum(0).eq(0).long() #sum cols and select only the results eq to 0 and conver boolean to long : array of len of predictions
        FN_index2 = mappings.sum(1).eq(0).long() #sum rows and select only the results eq to 0 and conver boolean to long : array of len of ground truths

        TP_index2 = torch.where(TP_index2)[0]; TP_index2 = np.array(selected_ind_class_pred)[TP_index2.cpu()]
        FP_index2 = torch.where(FP_index2)[0]; FP_index2 = np.array(selected_ind_class_pred)[FP_index2.cpu()]
        FN_index2 = torch.where(FN_index2)[0]; FN_index2 = np.array(selected_ind_class_gt)[FN_index2.cpu()]
        
        TP_index_40.append(TP_index2)
        FP_index_40.append(FP_index2)
        FN_index_40.append(FN_index2)
        
        ###################### Detection according diou_thr 0.5 ###########################
        ious_filter = ious.where(ious>0.5,torch.tensor(0.)) #match with some overlap
        mappings = utils.utils.get_mappings(ious_filter) if boxes_pred_c.shape[0]>0 else torch.zeros(len(boxes_gt_c), 0).to(device)

        TP_index3 = mappings.sum(0).eq(1).long() #sum cols and select only the results eq to 1 and conver boolean to long : array of len of predictions
        FP_index3 = mappings.sum(0).eq(0).long() #sum cols and select only the results eq to 0 and conver boolean to long : array of len of predictions
        FN_index3 = mappings.sum(1).eq(0).long() #sum rows and select only the results eq to 0 and conver boolean to long : array of len of ground truths

        TP_index3 = torch.where(TP_index3)[0]; TP_index3 = np.array(selected_ind_class_pred)[TP_index3.cpu()]
        FP_index3 = torch.where(FP_index3)[0]; FP_index3 = np.array(selected_ind_class_pred)[FP_index3.cpu()]
        FN_index3 = torch.where(FN_index3)[0]; FN_index3 = np.array(selected_ind_class_gt)[FN_index3.cpu()]
        
        TP_index_50.append(TP_index3)
        FP_index_50.append(FP_index3)
        FN_index_50.append(FN_index3)
        
    eps = 1e-6
    ###################### Detection for min overlap ###########################
    #d_acc = len(TP_index)/(boxes_gt.shape[0]+eps)
    d_recall = len(TP_index1)/(len(TP_index1)+len(FN_index1)) if (len(TP_index1)+len(FN_index1))!=0 else 1
    d_precision = len(TP_index1)/(len(TP_index1)+len(FP_index1)) if (len(TP_index1)+len(FP_index1))!=0 else 1
    d_f1 = (2*d_precision*d_recall)/(d_precision+d_recall+eps)
    d_recall*=100; d_precision*=100; d_f1*=100
    
    ###################### Detection according diou_thr 0.25 ###########################
    #d_acc2 = len(TP_index2)/(boxes_gt.shape[0]+eps)
    d_recall2 = len(TP_index2)/(len(TP_index2)+len(FN_index2)) if (len(TP_index2)+len(FN_index2))!=0 else 1
    d_precision2 = len(TP_index2)/(len(TP_index2)+len(FP_index2)) if (len(TP_index2)+len(FP_index2))!=0 else 1
    d_f12 = (2*d_precision2*d_recall2)/(d_precision2+d_recall2+eps)
    d_recall2*=100; d_precision2*=100; d_f12*=100
    
    ###################### Detection according diou_thr 0.4 ###########################
    #d_acc3 = len(TP_index3)/(boxes_gt.shape[0]+eps)
    d_recall3 = len(TP_index3)/(len(TP_index3)+len(FN_index3)) if (len(TP_index3)+len(FN_index3))!=0 else 1
    d_precision3 = len(TP_index3)/(len(TP_index3)+len(FP_index3)) if (len(TP_index3)+len(FP_index3))!=0 else 1
    d_f13 = (2*d_precision3*d_recall3)/(d_precision3+d_recall3+eps)
    d_recall3*=100; d_precision3*=100; d_f13*=100
    
    ###################### Save results ###########################
    #final_result = (zone_name, boxes_filtered, scores_filtered, classes_filtered, boxes_gt, TP_index, FP_index, FN_index, TP_index2, FP_index2, FN_index2, TP_index3, FP_index3, FN_index3, TP_index4, FP_index4, FN_index4, TP_index5, FP_index5, FN_index5, obj)
    
    if n_classes==2:
        final_result = (zone_name, boxes_filtered, scores_filtered, classes_filtered, boxes_gt, TP_index_25[0], TP_index_25[1], FP_index_25[0], FP_index_25[1], FN_index_25[0], FN_index_25[1], TP_index_40[0], TP_index_40[1], FP_index_40[0], FP_index_40[1], FN_index_40[0], FN_index_40[1], TP_index_50[0], TP_index_50[1], FP_index_50[0], FP_index_50[1], FN_index_50[0], FN_index_50[1], obj)
        
        data = np.array([int(len(scores_filtered)), N_agaves_real, n_pars, np.round(n_size_mb, 3), 
                         np.round(time_avg_pred*1000, 2), 
                         np.round(mAP25, 2), int(N_agaves_mAP25), np.round(aps25[0], 2), np.round(aps25[1], 2),
                         np.round(mAP40, 2), int(N_agaves_mAP40), np.round(aps40[0], 2), np.round(aps40[1], 2),

                         np.round(mAP50, 2), int(N_agaves_mAP50), np.round(aps50[0], 2), np.round(aps50[1], 2),
                         np.round(mAP75, 2), int(N_agaves_mAP75), np.round(aps75[0], 2), np.round(aps75[1], 2),
                         np.round(mAP95, 2), int(N_agaves_mAP95), np.round(aps95[0], 2), np.round(aps95[1], 2),
                         np.round(d_recall, 2), np.round(d_precision, 2), np.round(d_f1, 2),
                         np.round(d_recall2, 2), np.round(d_precision2, 2), np.round(d_f12, 2),
                         np.round(d_recall3, 2), np.round(d_precision3, 2), np.round(d_f13, 2),
                         len(TP_index1), len(FP_index1), len(FN_index1),
                        len(TP_index2), len(FP_index2), len(FN_index2),
                        len(TP_index3), len(FP_index3), len(FN_index3)
                        ])
    else:
        final_result = (zone_name, boxes_filtered, scores_filtered, classes_filtered, boxes_gt, TP_index_25[0], 
                        FP_index_25[0], FN_index_25[0], TP_index_40[0], FP_index_40[0], FN_index_40[0], TP_index_50[0], FP_index_50[0], FN_index_50[0], obj)
        
        data = np.array([int(len(scores_filtered)), N_agaves_real, n_pars, np.round(n_size_mb, 3), 
                     np.round(time_avg_pred*1000, 2), 
                     np.round(mAP25, 2), int(N_agaves_mAP25), np.round(aps25[0], 2),
                     np.round(mAP40, 2), int(N_agaves_mAP40), np.round(aps40[0], 2),
                     
                     np.round(mAP50, 2), int(N_agaves_mAP50), np.round(aps50[0], 2),
                     np.round(mAP75, 2), int(N_agaves_mAP75), np.round(aps75[0], 2),
                     np.round(mAP95, 2), int(N_agaves_mAP95), np.round(aps95[0], 2),
                     np.round(d_recall, 2), np.round(d_precision, 2), np.round(d_f1, 2),
                     np.round(d_recall2, 2), np.round(d_precision2, 2), np.round(d_f12, 2),
                     np.round(d_recall3, 2), np.round(d_precision3, 2), np.round(d_f13, 2),
                     len(TP_index1), len(FP_index1), len(FN_index1),
                    len(TP_index2), len(FP_index2), len(FN_index2),
                    len(TP_index3), len(FP_index3), len(FN_index3)
                    ])

    return data, final_result