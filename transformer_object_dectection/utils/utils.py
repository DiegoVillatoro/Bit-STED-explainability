import torch
import numpy as np
import math

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0.0)
        torch.nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, torch.nn.BatchNorm2d):    
        torch.nn.init.constant_(m.bias, 0.0)
        torch.nn.init.constant_(m.weight, 1.0)

#iou based on only weight and heigh to get the best anchor for each target #each wh1 with each wh2
def bbox_wh_iou(wh1, wh2): #one anchor to multiple predictions
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area
#iou based on only radius to get the best anchor for each target #each r1 with each r2
def cbbox_r_iou(r1, r2): #one anchor to multiple predictions
    r2 = r2.t()
    r2 = r2[0]
    inter_proportion = torch.min(r1, r2)**2#r
    union_proportion = torch.max(r1, r2)**2#R
    return inter_proportion / union_proportion

def bbox_iou(box1, box2, x1y1x2y2=False, align=True, DIoU=False, CIoU=False, eps = 1e-8):
    """
        Returns the IoU of each pair of bounding boxes one to one if align is true else iou of each box1 to each box2
    
    """  
    if not x1y1x2y2:#boxes xc yc w h (default)
        # Transform from center and width to corner coordinates
        if align:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b1_xc, b1_yc = box1[:, 0], box1[:, 1] 
        else:
            b1_x1, b1_x2 = box1[:, None, 0] - box1[:, None, 2] / 2, box1[:, None, 0] + box1[:, None, 2] / 2
            b1_y1, b1_y2 = box1[:, None, 1] - box1[:, None, 3] / 2, box1[:, None, 1] + box1[:, None, 3] / 2
            b1_xc, b1_yc = box1[:, None, 0], box1[:, None, 1] 
            
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        b2_xc, b2_yc = box2[:, 0], box2[:, 1] 
    else:
        if align:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b1_xc, b1_yc = (box1[:, 2]+box1[:, 0]) / 2, (box1[:, 3]+box1[:, 1]) / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, None, 0], box1[:, None, 1], box1[:, None, 2], box1[:, None, 3]
            b1_xc, b1_yc = (box1[:, None, 2]+box1[:, None, 0]) / 2, (box1[:, None, 3]+box1[:, None, 1]) / 2
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        b2_xc, b2_yc = (box2[:, 2]+box2[:, 0]) / 2, (box2[:, 3]+box2[:, 1]) / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + eps)

    if DIoU:
        
        #find minenclousing rec for each pair
        ext_rect_x1 = torch.min(b1_x1, b2_x1)
        ext_rect_y1 = torch.min(b1_y1, b2_y1)
        ext_rect_x2 = torch.max(b1_x2, b2_x2)
        ext_rect_y2 = torch.max(b1_y2, b2_y2)
        
        # w_ext_rect**2 + h_ext_rect**2
        C2 = (ext_rect_x2-ext_rect_x1)**2 + (ext_rect_y2-ext_rect_y1)**2
        rho2 = (b2_xc-b1_xc)**2 + (b2_yc-b1_yc)**2
        
        if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
            v = (4 / torch.pi**2) * (((b2_x2 - b2_x1 + eps) / (b2_y2 - b2_y1 + eps)).atan() - ((b1_x2 - b1_x1 + eps) / (b1_y2 - b1_y1 + eps)).atan()).pow(2)
            alpha = v / (v - iou + (1 + eps))
            
            
            return iou - (rho2 / C2 + v * alpha)  # CIoU

        return iou - rho2 / C2  # DIoU
    
    return iou

#box1: preds
#box2: targets
def cbbox_iou(box1, box2, align=True, DIoU = False, eps = 1e-8):  #one to one cbbox
    """
    Returns the IoU of circular bounding boxes.
    
    If align=True:
        one-to-one IoU
        
    If align=False:
        pairwise IoU (box1 vs box2)

    box format: (xc, yc, r)
    """
    if align:
        b1_x, b1_y, b1_r = box1.chunk(3, -1)#box1[:, 0], box1[:, 1], box1[:, 2]
        b2_x, b2_y, b2_r = box2.chunk(3, -1)
    else:
        b1_x, b1_y, b1_r = box1[:, None, 0], box1[:, None, 1], box1[:, None, 2]
        b2_x, b2_y, b2_r = box2[:, 0], box2[:, 1], box2[:, 2]#box2.chunk(3, -1) #box2[:, 0], box2[:, 1], box2[:, 2]
    
    rho2 = (b1_x-b2_x)**2+(b1_y-b2_y)**2
    d_centers = torch.sqrt(rho2) #distance of each circle center to each other
    zeros = torch.zeros_like(d_centers)
    
    R = torch.maximum(b1_r, b2_r)
    r = torch.minimum(b1_r, b2_r)

    checks_overlap = torch.logical_and( (R-r)<d_centers, d_centers<=(R+r) ) # condition for circles intersection
    checks_inside = d_centers<=(R-r) #condition for circle inside the other or be the same
    
    R2 = torch.pow(R, 2)
    r2 = torch.pow(r, 2)
    
    num1 = torch.where(checks_overlap, R2+rho2-r2, 1)
    den1 = torch.where(checks_overlap, torch.add(eps, torch.mul(torch.mul(2, R), d_centers) ), 1)
    num2 = torch.where(checks_overlap, r2+rho2-R2, 1)
    den2 = torch.where(checks_overlap, torch.add(eps, torch.mul(torch.mul(2, r), d_centers) ), 1)
    
    theta = torch.where(checks_overlap, torch.arccos( torch.divide( num1, den1 ) ), 0)
    phi = torch.where(checks_overlap, torch.arccos( torch.divide( num2, den2 ) ), 0)

    # get intersection area of two circles
    inter_area = torch.sub( torch.add(torch.mul(theta, R2), torch.mul(phi, r2) ), 
                              torch.add(torch.mul(R2, torch.sin(torch.mul(2,theta))), 
                                        torch.mul(r2, torch.sin(torch.mul(2,phi))) ), 
                               alpha=0.5 )

    iou = torch.divide( inter_area, (torch.pi*(R2+r2) - inter_area + eps) )
    iou = torch.where(checks_inside, torch.divide(r2, R2 + eps ), iou)

    if DIoU:
        #find minenclousing diameter of the circle for each pair
        C2 = torch.where(checks_inside, 4*R2, (d_centers+R+r)**2)
        
        return iou - rho2 / C2 #- (torch.pi*R2-inter_area)/r - (torch.pi*r2-inter_area)/R # DIoU
    
    return iou+eps

def non_max_suppression(predictions, iou_threshold = 0.5, obj= 'bbox', DIoU = False, CIoU = False, x1y1x2y2=False, device='cpu'):
#CIoU and xyxy only applies for BBOX
    rows, columns = predictions.shape

    if obj == 'bbox':
        sort_index = torch.flip(predictions[:, 4].argsort(), (0, ))
        predictions = predictions[sort_index]
        recs = predictions[:, :4]
        ious = bbox_iou(recs, recs, x1y1x2y2=x1y1x2y2, align = False, DIoU = DIoU, CIoU = CIoU)#x1y1wh
    else: #ccbox
        sort_index = torch.flip(predictions[:, 3].argsort(), (0, ))
        predictions = predictions[sort_index]
        circles = predictions[:, :3]
        ious = cbbox_iou(circles, circles, align = False, DIoU = DIoU)
    
    ious = ious - torch.eye(rows).to(device)
    keep = torch.ones(rows, dtype=bool).to(device)

    for index, iou in enumerate(ious):#for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        condition = (iou > iou_threshold) #& (categories == category)
        keep = keep & ~condition
    
    keep = keep[sort_index.argsort()]
    return torch.where(keep)[0]
def build_targets_circle(pred_boxes, target, anchors, device):

    nB, nA, nG = pred_boxes.shape[:3]

    obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool, device=device)
    tar = torch.zeros(nB, nA, nG, nG, 3, dtype=torch.float, device=device)
    labels = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)

    # Scale GT to grid
    target_boxes = target[:, 2:5] * nG
    gxy = target_boxes[:, :2]
    gt = target[:, 2:5]
    b = target[:, 0].long()

    gi = gxy[:, 0]
    gj = gxy[:, 1]

    offsets = torch.tensor(
        [[0,0]],
         #,[0,-1],[0,1],[-1,0],[1,0]],
        device=device
    )

    num_gt = len(target)
    num_offsets = offsets.size(0)

    # Expand GT with offsets
    gx_exp = gi[:, None] + offsets[:, 0]
    gy_exp = gj[:, None] + offsets[:, 1]

    gx_exp = gx_exp.reshape(-1)
    gy_exp = gy_exp.reshape(-1)
    b_exp = b[:, None].repeat(1, num_offsets).reshape(-1)
    gt_exp = gt[:, None].repeat(1, num_offsets, 1).reshape(-1, 3)
    gxy_exp = gxy[:, None].repeat(1, num_offsets, 1).reshape(-1, 2)

    # Keep valid grid cells
    valid = (
        (gx_exp >= 0) & (gx_exp < nG) &
        (gy_exp >= 0) & (gy_exp < nG)
    )

    gx_exp = gx_exp[valid]
    gy_exp = gy_exp[valid]
    b_exp = b_exp[valid]
    gt_exp = gt_exp[valid]
    gxy_exp = gxy_exp[valid]

    # Compute squared distance to grid center
    grid_cx = gx_exp + 0.5
    grid_cy = gy_exp + 0.5

    dist = (grid_cx - gxy_exp[:,0])**2 + \
           (grid_cy - gxy_exp[:,1])**2

    # Create unique flat index per (batch, grid)
    flat_idx = b_exp * (nG*nG) + gy_exp * nG + gx_exp

    # Sort by (flat_idx, distance)
    sort_idx = torch.argsort(flat_idx * 1e6 + dist)
    flat_idx = flat_idx[sort_idx]
    dist = dist[sort_idx]

    gx_exp = gx_exp[sort_idx]
    gy_exp = gy_exp[sort_idx]
    b_exp = b_exp[sort_idx]
    gt_exp = gt_exp[sort_idx]

    # Keep first occurrence per flat_idx (minimum distance wins)
    unique_mask = torch.ones_like(flat_idx, dtype=torch.bool)
    unique_mask[1:] = flat_idx[1:] != flat_idx[:-1]

    gx_exp = gx_exp[unique_mask].long()
    gy_exp = gy_exp[unique_mask].long()
    b_exp = b_exp[unique_mask].long()
    gt_exp = gt_exp[unique_mask]

    # Assign to all anchors
    obj_mask[b_exp, :, gy_exp, gx_exp] = True
    tar[b_exp, :, gy_exp, gx_exp] = gt_exp.unsqueeze(1)
    labels[b_exp, :, gy_exp, gx_exp] = 1.0

    # Compute IoU for assigned cells
    #pred_sel = pred_boxes[b_exp, :, gy_exp, gx_exp]  # (num_cells, nA, 3)
    #gt_sel = gt_exp.unsqueeze(1).expand_as(pred_sel)

    #ious = cbbox_iou(
    #    pred_sel.reshape(-1, 3),
    #    gt_sel.reshape(-1, 3),
    #    align=True,
    #    DIoU=False
    #)

    #ious = ious.clamp(0, 1).detach()
    #ious = ious.view(pred_sel.shape[0], nA)

    #labels[b_exp, :, gy_exp, gx_exp] = ious

    return (
        tar[obj_mask],
        pred_boxes[obj_mask],
        obj_mask,
        labels
    )

def build_targets_rec(pred_boxes, target, anchors, device, ignore_thres=0.5, iou_type = False):

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    #nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)
    # pred_cls shape: batch_size, n anchors, grid size, grid size, classes
    # pred_boxes shape: batch_size, n anchors, grid size, grid size, box

    # Output tensors
    obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool, device=device)
    iou_scores = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tar = torch.zeros(nB, nA, nG, nG, 4, dtype=torch.float, device=device)
    labels = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    #th = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    #tcls = torch.zeros(nB, nA, nG, nG, nC, dtype=torch.float, device=device)
    pos = torch.ones(nB, nA, nG, nG, dtype=torch.bool, device=device)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    _, best_ious_idx = ious.max(0)

    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gi, gj = gxy.long().t() #get int coordinates of corner left of the grid cell
    # Set masks
    #obj_mask[b, best_ious_idx, gj, gi] = 1 # (nB, nA, nG, nG) grid cell with center of the box is assigned to predict box with best anchor
    #for experimentation it was faster to assign matrix of 1s than directly 1*
    obj_mask[b, best_ious_idx, gj, gi] = pos[b, best_ious_idx, gj, gi]
    
    # One-hot encoding of label
    #tcls[b, best_ious_idx, gj, gi, target_labels] = 1
    
    # target arrange
    if iou_type:
        #noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool, device=device)
        #noobj_mask[b, best_ious_idx, gj, gi] = 0 #cell with objects are power off in non object mask
        # Set noobj mask to zero where iou exceeds ignore threshold, ignore boxes in the non object mask with high iou 
        #for i, anchor_ious in enumerate(ious.t()):
        #    noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
        #######################
        
        #iou_scores[b, best_ious_idx, gj, gi] = bbox_iou(pred_boxes[b, best_ious_idx, gj, gi][:,0:4], target[:,2:6], 
        #                                                x1y1x2y2=False, align = True, DIoU=False).view(-1,) 
        #iou_scores[b, best_ious_idx, gj, gi] = (1-f)+f*iou_scores[b, best_ious_idx, gj, gi] #for low iou start in 1-f value as score
        labels[b, best_ious_idx, gj, gi] = target[:, 1] #gx - gx.floor()
        
        tar[b, best_ious_idx, gj, gi, 0] = target[:, 2] #gx - gx.floor()
        tar[b, best_ious_idx, gj, gi, 1] = target[:, 3] #gy# - gy.floor()
        tar[b, best_ious_idx, gj, gi, 2] = target[:, 4] #torch.log(gr / anchors[best_ious_idx] + 1e-16)
        tar[b, best_ious_idx, gj, gi, 3] = target[:, 5] #torch.log(gr / anchors[best_ious_idx] + 1e-16)
        #return tar, iou_scores, obj_mask, noobj_mask
        return tar[b, best_ious_idx, gj, gi, :], pred_boxes[b, best_ious_idx, gj, gi, :], obj_mask, labels
    else:
        noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool, device=device)
        noobj_mask[b, best_ious_idx, gj, gi] = 0 #cell with objects are power off in non object mask
        # Set noobj mask to zero where iou exceeds ignore threshold, ignore boxes in the non object mask with high iou 
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
        
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        tar[b, best_ious_idx, gj, gi, 0] = gx - gx.floor()
        tar[b, best_ious_idx, gj, gi, 1] = gy - gy.floor()
        tar[b, best_ious_idx, gj, gi, 2] = torch.log(gw / anchors[best_ious_idx][:, 0] + 1e-16)
        tar[b, best_ious_idx, gj, gi, 3] = torch.log(gh / anchors[best_ious_idx][:, 1] + 1e-16)
        return tar[b, best_ious_idx, gj, gi, :], pred_boxes[b, best_ious_idx, gj, gi, :], obj_mask, noobj_mask

#################### OBJECT DETECTION #####################################
def bbox_area(box):#receive x1, y1, x2, y2
    return (box[2] - box[0]) * (box[3] - box[1])
#box proportional area
def bbox_pa_batch(box1, box2, area_b): #receive x1, y1, x2, y2

    #area_a = box_area(boxes_a.transpose(0, 1))
    #area_b = box_area(boxes_b.transpose(0, 1))
    
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, None, 0], box1[:, None, 1], box1[:, None, 2], box1[:, None, 3]
    b1_xc, b1_yc = (box1[:, None, 2]+box1[:, None, 0]) / 2, (box1[:, None, 3]+box1[:, None, 1]) / 2
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    b2_xc, b2_yc = (box2[:, 2]+box2[:, 0]) / 2, (box2[:, 3]+box2[:, 1]) / 2
        
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)    

    iou = inter_area / (area_b)
    
    """
    ext_rect_x1 = torch.min(b1_x1, b2_x1)
    ext_rect_y1 = torch.min(b1_y1, b2_y1)
    ext_rect_x2 = torch.max(b1_x2, b2_x2)
    ext_rect_y2 = torch.max(b1_y2, b2_y2)

    # w_ext_rect**2 + h_ext_rect**2
    C2 = (ext_rect_x2-ext_rect_x1)**2 + (ext_rect_y2-ext_rect_y1)**2
    rho2 = (b2_xc-b1_xc)**2 + (b2_yc-b1_yc)**2

    return iou - rho2 / C2  # DIoU  
    """
    
    return iou

def cbbox_area(box):
    return (np.pi*box[2]**2)
#box proportional area
def cbbox_pa_batch(box1, box2, area_b2, eps= 1e-8):
    b1_x, b1_y, b1_r = box1[:, None, 0], box1[:, None, 1], box1[:, None, 2]
    b2_x, b2_y, b2_r = box2[:, 0], box2[:, 1], box2[:, 2]
    
    rho2 = (b1_x-b2_x)**2+(b1_y-b2_y)**2
    d_centers = torch.sqrt(rho2) #distance of each circle center to each other
    
    R = torch.maximum(b1_r, b2_r)
    r = torch.minimum(b1_r, b2_r)
    iou = torch.zeros_like(d_centers)

    checks_overlap = torch.logical_and( (R-r)<d_centers, d_centers<=(R+r) ) # condition for circles intersection
    checks_inside = d_centers<=(R-r) #condition for circle inside the other or be the same
    
    R2 = torch.pow(R, 2)
    r2 = torch.pow(r, 2)
    
    num1 = torch.where(checks_overlap, R2+rho2-r2, 1)
    den1 = torch.where(checks_overlap, torch.add(eps, torch.mul(torch.mul(2, R), d_centers) ), 1)
    num2 = torch.where(checks_overlap, r2+rho2-R2, 1)
    den2 = torch.where(checks_overlap, torch.add(eps, torch.mul(torch.mul(2, r), d_centers) ), 1)
    
    theta = torch.where(checks_overlap, torch.arccos( torch.divide( num1, den1 ) ), 0)
    phi = torch.where(checks_overlap, torch.arccos( torch.divide( num2, den2 ) ), 0)
    
    # get the coordinates of the intersection rectangle
    #area_b2 = area_b2[:,None].expand(-1, len(R))
    #print(area_b2.shape)

    inter_area = torch.sub( torch.add(torch.mul(theta, R2), torch.mul(phi, r2) ), 
                              torch.add(torch.mul(R2, torch.sin(torch.mul(2,theta))), 
                                        torch.mul(r2, torch.sin(torch.mul(2,phi))) ), 
                               alpha=0.5 )
    
    iou = torch.divide( inter_area, (area_b2 + eps) )
    iou = torch.where(checks_inside, torch.divide(torch.pi*r2, area_b2 + eps ), iou)
    
    """
    #find minenclousing circle for each pair
    angle = torch.arctan2(torch.sub(b1_y,b2_y), torch.sub(b1_x,b2_x))
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    lineExtrema_x1 = torch.add(b1_x, torch.mul(b1_r, cos) )
    lineExtrema_y1 = torch.add(b1_y, torch.mul(b1_r, sin) )
    lineExtrema_x2 = torch.sub(b2_x, torch.mul(b2_r, cos) )
    lineExtrema_y2 = torch.sub(b2_y, torch.mul(b2_r, sin) )

    C2 = torch.pow(torch.sub(lineExtrema_x1,lineExtrema_x2),2)+torch.pow(torch.sub(lineExtrema_y1,lineExtrema_y2),2)
    C2 = torch.where(checks_inside, 4*R2, C2)

    return iou - rho2 / C2  # DIoU
    """

    return iou

class LenOfBatchesError(Exception):
    pass
#max_batch: limit the amount of boxes to evaluate to avoid use all memory which can produce error out of memory
#           it need to match with the maximun number of agaves in the image
#indices_to_keep_r: useful to avoid infinity loop by count the number of valid boxes in recursion
def non_max_suppression_area( boxes, scores, iou_threshold: float = 0.5, min_size: float = 15.0, obj='bbox', indices_to_keep_r=[], max_batch = 10000, device = 'cpu') :
    keep = torch.ones(boxes.shape[0], dtype=bool).to(device)
    flag_one_batch = True if len(boxes)<=max_batch else False
    
    for r in range(0, len(boxes), max_batch):
        boxes_temp = boxes[r:r+max_batch] if r+max_batch<len(boxes) else boxes[r:]
        scores_temp = scores[r:r+max_batch] if r+max_batch<len(scores) else scores[r:] 
    
        rows, columns = boxes_temp.shape
        
        if obj=='bbox':
            boxes_area = bbox_area(boxes_temp.transpose(0, 1))
        else:
            boxes_area = cbbox_area(boxes_temp.transpose(0, 1))
        sort_index = boxes_area.argsort(descending=True) #sort of the largest box to the smallest
        #sort_index = scores_temp.argsort(descending=True) #sort of the largest box to the smallest
        boxes_temp = boxes_temp[sort_index]
        boxes_area = boxes_area[sort_index]

        if obj=='bbox':
            ious = bbox_pa_batch(boxes_temp, boxes_temp, boxes_area)
        else:
            ious = cbbox_pa_batch(boxes_temp, boxes_temp, boxes_area)
        ious = ious - torch.eye(rows).to(device)
        keep_temp = torch.ones(rows, dtype=bool).to(device)

        for index, iou, in enumerate(ious):
            if not keep_temp[index]: # if keep[index]==False
                continue
            condition = iou > iou_threshold
            keep_temp = keep_temp & ~condition #shut down the index of overlapped rectangles with the rectangle of index position
        keep_temp = keep_temp & (boxes_area>min_size) #filter the smaller areas than min_size
        
        if r+max_batch<len(boxes):
            keep[r:r+max_batch] = keep[r:r+max_batch] & keep_temp[sort_index.argsort()]
        else:
            keep[r:] = keep[r:r+max_batch] & keep_temp[sort_index.argsort()]
    
    indices_to_keep = torch.where(keep)[0]
    if len(indices_to_keep_r) == len(indices_to_keep) and len(indices_to_keep)>max_batch:
        print("Error trying to reduce boxes, max batches need to be greater")
        #raise LenOfBatchesError("Error trying to reduce boxes, max batches need to be greater")
        return indices_to_keep
            
    if not flag_one_batch:
        boxes_temp = boxes[[indices_to_keep]]
        scores_temp = scores[[indices_to_keep]]
        indices_temp = non_max_suppression_area( boxes_temp, scores_temp, iou_threshold, min_size, obj, indices_to_keep, max_batch, device)
        return indices_to_keep[indices_temp]
    else:                         
        #return only the index of elements to keep
        return indices_to_keep

def get_mappings(iou_mat): # match each predicted box with the ground truth with the larges iou value if exist
    mappings = torch.zeros_like(iou_mat) #ground truth x predicted
    gt_count, pr_count = iou_mat.shape
    
    #first mapping (max iou for first pred_box (first column))
    if not iou_mat[:,0].eq(0.).all():
        # if not a zero column
        mappings[iou_mat[:,0].argsort()[-1],0] = 1

    for pr_idx in range(1,pr_count):
        # sum rows from column 0 to pr_idx to know which gt-boxes are already assigned
        not_assigned = torch.logical_not(mappings[:,:pr_idx].sum(1)).long()

        # Considering unassigned gt-boxes for further evaluation 
        targets = not_assigned * iou_mat[:,pr_idx]

        # If no gt-box satisfy the previous conditions
        # for the current pred-box, ignore it (False Positive)
        if targets.eq(0).all():
            continue

        # max-iou from current column after all the filtering
        # will be the column element for mapping
        mappings[targets.argsort()[-1], pr_idx] = 1
    return mappings

def mAP(boxes_gt, boxes_pred, classes_gt, classes_pred, num_classes, scores, obj, iou_thr=0.5, DIoU=False, CIoU=False, device = 'cpu'):
    #boxes_gt : total boxes of ground truth (Nx4)
    #boxes_pred : total boxes pred boxes (Mx4)
    #classes_gt : class of each real boxes (N)
    #classes_pred : class of each pred boxes (M)
    #num_classes : int
    #scores : score of each pred box (M)
    #iou_thr : overlapping threshold to consider valid a box
    #CIoU only for BBOX
    
    sort_index = scores.argsort(descending=True) #sort of the largest score to the smallest
    boxes_pred = boxes_pred[sort_index]
    classes_pred = classes_pred[sort_index]
    #classes_pred = torch.zeros_like(scores)
    
    average_precisions = []
    epsilon = 1e-6
    corrects=0
    
    for c in range(num_classes):
        boxes_gt_c = torch.tensor([]).to(device)
        boxes_pred_c = torch.tensor([]).to(device)
        for box_gt, class_gt in zip(boxes_gt, classes_gt):
            if class_gt == c:
                boxes_gt_c = torch.concat((boxes_gt_c, box_gt.unsqueeze(0) ))
        for box_pred, class_pred in zip(boxes_pred, classes_pred):
            if class_pred == c:
                boxes_pred_c = torch.concat((boxes_pred_c, box_pred.unsqueeze(0) ))
                
        if obj=='bbox':
            ious = bbox_iou(boxes_gt_c, boxes_pred_c, x1y1x2y2=True, align=False, DIoU = DIoU, CIoU = CIoU)
        else:
            ious = cbbox_iou(boxes_gt_c, boxes_pred_c, align=False, DIoU = DIoU)
        ious = ious.where(ious>iou_thr,torch.tensor(0.))
        mappings = get_mappings(ious)

        TP_cumsum = torch.cumsum(mappings.sum(0).eq(1).long(), dim=0)
        FP_cumsum = torch.cumsum(mappings.sum(0).eq(0).long(), dim=0)
        FN_cumsum = torch.cumsum(mappings.sum(1).eq(0).long(), dim=0)
        total_true_boxes, _ = boxes_gt_c.shape 

        recalls = TP_cumsum/(total_true_boxes+epsilon)
        precisions = torch.divide(TP_cumsum, TP_cumsum+FP_cumsum+epsilon)

        precisions = torch.cat((torch.tensor([1]).to(device), precisions))
        recalls = torch.cat((torch.tensor([0]).to(device), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
        
        corrects += mappings.sum()
        
    average_precisions = torch.tensor(average_precisions)
    return np.array((sum(average_precisions)/len(average_precisions)).cpu())*100, np.array(corrects.cpu()), average_precisions*100

################################ VI #########################################
#bands are between 0 and 1, the returned VI is also between 0 and 1
#ranges determine the bounds when bounded is true,
# also ranges[0] can move the 0 values or indefined (background of orthomosaic) to other position by initialize the output array of the VI
def VI(bands, vi_name, ranges=None, bounded = True):
    B = bands[0]
    G = bands[1]
    R = bands[2]
    Re = bands[3]
    NIR = bands[4]
    V_I = np.zeros_like(R)

    ranges_dic = {'RVI':(0, 1),
         'VIN':(1, 20),#hight sensibility to data
         'NDVI':(0,1),                                                  #####
         'PVI1':(0,1),                                                  #####
         'PVI2':(0,0.4), #medium sensibility to data                    #####
         'SAVI':(0,1),                                                  #####
         'TSAVI1':(-1,1),#'TSAVI1':(-3,3), #hight sensibility to data
         'TSAVI2':(-0.9,0.9), #hight sensibility to data                #####
         'NDGI':(-0.5,0.6),#medium sensibility to data                  #####
         'RI':(-0.5,0.5),
         'RI_NDVI':(0,1.1),#medium sensibility to data
         'RI_SAVI':(0,1),
         'ARVI':(0,1),                                                  #####
         'NDCI':(0,1),#'NDCI':(0,0.7),                                  #####

         'MSAVI':(0,1),#'MSAVI':(0,0.8),                                #####
         'NLI':(-1,1),                                                  #####
         'RDVI':(0,1),                                                  #####
         'ExG':(-0.25,0.25), #medium sensibility to data 
         'SIPI':(0.5, 2.7), #medium sensibility to data
         'GNDVI':(0,1),                                                 #####
         'OSAVI':(0,1),                                                 #####
         'MSR':(0,3.5), #medium sensibility to data                    
         'NDWI':(-1, 0),                                                #####
         'ExR':(-0.1,0.25), #medium sensibility to data
         'MCARI':(0,4.5),#hight sensibility to data

         'NDRE':(0, 0.5),
         'TVI':(0,34), #hight sensibility to data
         'ARI':(-2.4, 36.4), #hight sensibility to data
         'EVI':(0,3),                                                  #####
         'CIVE':(18.7, 18.9),
         'NRI':(0.15,0.5),
         'NGI':(0.15, 0.6),
         'NBI':(0.2, 0.6),
         'CCCI':(0.3, 3.5), #medium sensibility to data
         'EVI2':(0, 1.1)}#medium sensibility to data                    #####

    #select ranges from input or dictionary if exist key
    if ranges!=None:
        ranges = ranges
    elif vi_name in ranges_dic:
        ranges = ranges_dic[vi_name]
    else:
        ranges = (0, 1)

    if vi_name == 'RVI':#rango abierto (0-INF) rango particular (0 - 2.7) rango acotado(0, 1)
        V_I = np.divide(R, NIR, out=np.zeros_like(R)+ranges[0], where=NIR!=0)
    elif vi_name == 'VIN':#rango abierto (0-INF) rango particular (0 - 32), rango acotado (0,15)
        V_I = np.divide(NIR, R, out=np.zeros_like(R)+ranges[0], where=R!=0)
    elif vi_name == 'NDVI': #rango (-1-1), rango particular (-0.5 - 1.0)
        V_I = np.divide(NIR-R, NIR+R, out=np.zeros_like(R)+ranges[0], where=(NIR+R)!=0)
    elif vi_name == 'PVI1': #rango (-0.385-0.923)   rango particular (-0.1 - 0.8)
        #V_I = (a*NIR-R-b)/np.sqrt(a**2+1) with a=2.4 and b=0.01
        V_I = np.add(0.923*NIR, -0.385*R, out=np.zeros_like(R)+ranges[0], where=((NIR+R)!=0))
    elif vi_name == 'PVI2': #rango (-0.676-0.737) rango particula (-0.1 - 0.4)
        #V_I = (a*Re-R-b)/np.sqrt(a**2+1) with a=1.091 and b=5.49
        V_I = np.add(0.737*Re, -0.676*R, out=np.zeros_like(R)+ranges[0], where=((Re+R)!=0))
    elif vi_name == 'SAVI': #rango (-1-1) rango particular (-0.3 - 0.8)
        V_I = np.divide(1.5*(NIR-R), NIR+R+0.5, out=np.zeros_like(R)+ranges[0], where=(NIR+R)!=0)
    elif vi_name == 'TSAVI1':#rango abierto (-INF-INF) rango acotado(-5, 5)
        a = 1.901#0.999
        b = 0.111#0.136
        V_I = np.divide(a*(NIR-a*R-b), a*NIR+R-a*b, out=np.zeros_like(R)+ranges[0], where=np.logical_and((NIR+R)!=0, (a*NIR+R-a*b)!=0))
    elif vi_name == 'TSAVI2':#rango abierto (-INF-INF)  rango particular (-0.71 - 0.7) rango acotado (-0.4-0.5)
        a = 1.901#0.999
        b = 0.111#0.136
        V_I = np.divide(a*(NIR-a*R-b), a*NIR+R-a*b+0.08*(1+a**2), out=np.zeros_like(R)+ranges[0], where=(NIR+R)!=0)
    elif vi_name == 'NDGI': #rango (-1-1) rango particular (-0.9 - 1)
        V_I = np.divide(G-R, G+R, out=np.zeros_like(R)+ranges[0], where=(G+R)!=0)
    elif vi_name == 'RI':#rango (-1-1) rango particular (-1.0 - 0.9)
        V_I = np.divide(R-G, R+G, out=np.zeros_like(R)+ranges[0], where=(R+G)!=0)
    elif vi_name == 'RI_NDVI':  #rango (-1.45-1.45) rango particular (-1 - 1.45)
        k = 0.45
        NDVI = np.divide(NIR-R, NIR+R, out=np.zeros_like(R), where=(NIR+R)!=0)
        RI = np.divide(R-G, R+G, out=np.zeros_like(R), where=(R+G)!=0)
        V_I = np.add(NDVI, -k*RI, out=np.zeros_like(R)+ranges[0], where=((NDVI)!=0))
    elif vi_name == 'RI_SAVI': #rango (-1.26-1.26) rango particular (-1.24 - 0.8)
        k = 0.26
        SAVI = np.divide(1.5*(NIR-R), NIR+R+0.5, out=np.zeros_like(R), where=(NIR+R)!=0)
        RI = np.divide(R-G, R+G, out=np.zeros_like(R), where=(R+G)!=0)
        V_I = np.add(SAVI, -k*RI, out=np.zeros_like(R)+ranges[0], where=((SAVI)!=0))
    elif vi_name == 'ARVI': #rango (-INF, INF) rango particular (-0.6-11) rango acotado (0-1)
        y=0.5
        RB = R-y*(B-R)
        V_I = np.divide(NIR-RB, NIR+RB, out=np.zeros_like(R)+ranges[0], where=np.logical_and((NIR+R)!=0, (NIR+RB)!=0)) #out is shiftted to initialize the 0's with the corresponding min values

    elif vi_name == 'NDCI':#rango (-1-1) rango particular (-0.8 - 0.9)
        V_I = np.divide(Re-R, Re+R, out=np.zeros_like(R)+ranges[0], where=(Re+R)!=0)
    elif vi_name == 'MSAVI':#rango (-1-1 o 1.5)? rango particular (-0.2 - 0.8)
        V_I = np.divide(2*NIR+1-np.sqrt((2*NIR+1)**2-8*(NIR-R)), 2, out=np.zeros_like(R)+ranges[0], where=(NIR+R)!=0)
    elif vi_name == 'NLI':#rango (-1-1) rango particular (-1 - 0.9)
        V_I = np.divide(NIR**2-R, NIR**2+R, out=np.zeros_like(R)+ranges[0], where=(NIR+R)!=0)
    elif vi_name == 'RDVI':#rango (-1-1) rango particular (-0.3, 0.7)
        V_I = np.divide(NIR-R, np.sqrt(NIR+R), out=np.zeros_like(R)+ranges[0], where=(np.sqrt(NIR+R))!=0)
    elif vi_name == 'ExG': #rango (-2, 2) rango particular (-0.7, 0.9)
        V_I = np.add(2*G, -R-B, out=np.zeros_like(R)+ranges[0], where=((R+G+B)!=0))
    elif vi_name == 'NDWI':
        V_I = np.divide(G-NIR, G+NIR, out=np.zeros_like(R)+ranges[0], where=(G+NIR)!=0)
    elif vi_name == 'GNDVI': #rango (-1-1) rango particular (-0.7 - 1)
        V_I = np.divide(NIR-G, NIR+G, out=np.zeros_like(R)+ranges[0], where=(NIR+G)!=0)
    elif vi_name == 'OSAVI': #rango (-1, 1) rango particular (-0.3 - 0.7)
        V_I = np.divide(NIR-R, NIR+R+0.16, out=np.zeros_like(R)+ranges[0], where=(NIR+R)!=0)

    elif vi_name == 'MSR': #rango (-1, INF) rango particular (-0.4 - 4.7) rango acotado (0 - 2.8)
        V_I = np.divide(NIR-R, R+np.sqrt(NIR*R), out=np.zeros_like(R)+ranges[0], where=R+np.sqrt(NIR*R)!=0)
    elif vi_name == 'SIPI': #rango (-INF, INF)
        V_I = np.divide(NIR-B, NIR-R, out=np.zeros_like(R)+ranges[0], where=(NIR-R)!=0)
    elif vi_name == 'ExR': #rango (-1-1.3) rango particular (-0.4 - 0.4)
        V_I = np.add(1.3*R, -G, out=np.zeros_like(R)+ranges[0], where=((R+G)!=0))
    elif vi_name == 'MCARI': #rango(-INF-INF) rango particular (-0.2 - 2.7) rango acotado (-0.1-0.9)
        V_I = np.divide((Re-R-0.2*(Re-G))*Re, R, out=np.zeros_like(R)+ranges[0], where=(R)!=0)
    elif vi_name == 'NDRE':  #rango (-1-1) rango particular (-0.6 - 0.9)
        V_I = np.divide(NIR-Re, NIR+Re, out=np.zeros_like(R)+ranges[0], where=(NIR+Re)!=0)
    elif vi_name == 'TVI': #rango (-100, 100) rango particular (-14 - 39)
        V_I = np.add(60*NIR+40*G, -100*R, out=np.zeros_like(R)+ranges[0], where=((NIR+R+G)!=0))
    elif vi_name == 'ARI': #rango (-INF, INF) rango particular (-121 - 359) rango acotado (-10, 86)
        V_I = np.divide(Re-G, G*Re, out=np.zeros_like(R)+ranges[0], where=(G*Re)!=0)
    elif vi_name == 'EVI': #rango (-INF, INF) rango particular (-372637 - 1416), rango acotado (0, 1.25)
        L, c1, c2, g = 1, 6, 7.5, 2.5
        V_I = np.divide(g*(NIR-R), NIR+c1*R-c2*B+L, out=np.zeros_like(R)+ranges[0], where=(NIR+R+B)!=0)
    elif vi_name == 'CIVE': #rango(18 - 20) rango particular(18.4 - 19.1)
        V_I = np.add(0.441*R, -0.811*G+0.385*B+18.78745, out=np.zeros_like(R)+ranges[0], where=((R+G+B)!=0))
    elif vi_name == 'NRI': #rango(0-1), rango particular (0.0-0.8)
        V_I = np.divide(R, R+G+B, out=np.zeros_like(R)+ranges[0], where=(R+G+B)!=0)
    elif vi_name == 'NGI': #rango(0-1), rango particular (0.0-0.92)
        V_I = np.divide(G, R+G+B, out=np.zeros_like(R)+ranges[0], where=(R+G+B)!=0)
    elif vi_name == 'NBI': #rango(0-1), rango particular (0.0-0.85)
        V_I = np.divide(B, R+G+B, out=np.zeros_like(R)+ranges[0], where=(R+G+B)!=0)
    elif vi_name == 'CCCI': #rango (-INF, INF) rango particular (-73005 - 263474) rango de datos (0, 1.7)
        NDRE = VI(bands, 'NDRE', ranges=(0,0.4), bounded=True)
        NDVI = VI(bands, 'NDVI', ranges=(0,1), bounded=True)
        V_I = np.divide(NDRE, NDVI, out=np.zeros_like(R)+ranges[0], where=(NDVI)!=0)
    elif vi_name == 'CCCI2': #rango (-INF, INF) rango particular (-60594 - 218683) rango de datos (-0.375, 1.02)
        NDRE = VI(bands, 'NDRE', ranges=(0,0.4), bounded=True)
        NDVI = VI(bands, 'NDVI', ranges=(0,1), bounded=True)
        V_I = np.divide(NDRE, NDVI, out=np.zeros_like(R)+ranges[0], where=(NDVI)!=0)
        V_I = 0.92*V_I-0.03
    elif vi_name == 'EVI2': #rango (-0.74 - 1.25), rango particular (-0.2 - 0.8)
        g, c1, L = 2.5, 2.4, 1
        V_I = np.divide(g*(NIR-R), NIR+c1*R+L, out=np.zeros_like(R)+ranges[0], where=(NIR+R)!=0)

    if bounded: #bound the range of the VI and scale to be between 0 and 1
        V_I = np.clip(V_I, ranges[0], ranges[1]-0.001) #set little small to upper bound to avoid overflow in uin8
        V_I = (V_I-ranges[0])/(ranges[1]-ranges[0])

    return V_I.astype('float32')