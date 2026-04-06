from collections import OrderedDict
import pandas as pd
import utils.utils
import utils.counting as counting
import cv2
from matplotlib import pyplot as plt
import torch
import numpy as np
import rasterio as rio
import time

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import torch.nn.functional as F
import random

def plot_train_losses(dir_losses, title, start_epoch=0, y_max=5, folder_to_save=None, dpi=400):
    linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
    
    df = pd.read_csv(dir_losses)
    losses_train = df['Losses train'][start_epoch:]
    times_train = df['Times train'][start_epoch:]*1000 #ms
    losses_eval = df['Losses eval'][start_epoch:]
    times_eval = df['Times eval'][start_epoch:]*1000 #ms
    l1_losses_box = df['L1 Losses cbbox'][start_epoch:]
    l1_losses_conf = df['L1 Losses conf'][start_epoch:]
    l2_losses_box = df['L2 Losses cbbox'][start_epoch:]
    l2_losses_conf = df['L2 Losses conf'][start_epoch:]

    epochs=len(losses_train)
    x = range(1,epochs+1)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(17, 9))
    plt.rcParams.update({'font.size': 10})
    #times = np.array(losses)/2

    ax1 = axs[0][0]
    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.plot(x, losses_train, color=color, linestyle=linestyles['densely dashed'], label="Loss train")
    color = 'tab:blue'
    ax1.plot(x, losses_eval, color=color, linestyle=linestyles['densely dashdotdotted'], label="Loss eval")
    ax1.tick_params(axis='y')
    ax1.legend(loc="upper left")
    ax1.axis(ymin=0, ymax=y_max)
    
    y2 = min(losses_eval)
    x2 = x[list(losses_eval).index(y2)]
    ax1.annotate('Best Ep:{0:d}\n Loss: {1:0.2f}'.format(x2, y2),xy=(x2,y2),xytext=(x2-epochs/4,y2+y_max/7), weight="bold", arrowprops={'width':0.05})
    
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel('time (ms)')  # we already handled the x-label with ax1
    ax2.plot(x, times_train, color=color, linestyle=linestyles['densely dashed'], label='Avg time for train epoch')
    color = 'tab:pink'
    ax2.plot(x, times_eval, color=color, linestyle=linestyles['densely dashdotdotted'], label='Avg time for eval epoch')
    ax2.tick_params(axis='y')
    ax2.legend(loc="upper right")
    ax2.set_title('Lossess for all epochs')
    
    ax2 = axs[1][0]
    ax2.set_ylabel('loss')
    color = 'tab:orange'
    ax2.plot(x, l1_losses_box, color=color, linestyle=linestyles['loosely dashdotdotted'], label="L1 Loss Box")
    color = 'tab:purple'
    ax2.plot(x, l1_losses_conf, color=color, linestyle=linestyles['densely dashed'], label="L1 Loss Conf")
    color = 'tab:orange'
    ax2.plot(x, l2_losses_box, color=color, linestyle=linestyles['loosely dashed'], label="L2 Loss Box")
    color = 'tab:purple'
    ax2.plot(x, l2_losses_conf, color=color, linestyle=linestyles['densely dashdotdotted'], label="L2 Loss Conf")
    ax2.legend(loc="upper left")
    ax2.set_title('Components of training losses for all epochs')
    ax2.axis(ymin=0, ymax=y_max)

    start_plot = 2*epochs//3
    ax1 = axs[0][1]
    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.plot(x[start_plot:], losses_train[start_plot:], color=color, linestyle=linestyles['densely dashed'], label="Loss train")
    color = 'tab:blue'
    ax1.plot(x[start_plot:], losses_eval[start_plot:], color=color, linestyle=linestyles['densely dashdotdotted'], label="Loss eval")
    ax1.tick_params(axis='y')
    ax1.legend(loc="upper left")
    
    y_min2 = 0.9*min(losses_eval[start_plot:].min(), losses_train[start_plot:].min())
    y_max2 = 1.1*max(losses_eval[start_plot:].max(), losses_train[start_plot:].min())
    ax1.axis(ymin=y_min2, ymax=y_max2)
    ax1.annotate('Best Ep:{0:d}\n Loss: {1:0.2f}'.format(x2, y2),xy=(x2,y2),xytext=(x2-(epochs-start_plot)/4,y2+(y_max2-y_min2)/4), weight="bold", arrowprops={'width':0.05})
    
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel('time (ms)')  # we already handled the x-label with ax1
    ax2.plot(x[start_plot:], times_train[start_plot:], color=color, linestyle=linestyles['densely dashed'], label='Avg train epoch')
    color = 'tab:pink'
    ax2.plot(x[start_plot:], times_eval[start_plot:], color=color, linestyle=linestyles['densely dashdotdotted'], label='Avg eval epoch')
    ax2.tick_params(axis='y')
    ax2.legend(loc="upper right")
    ax2.set_title('Losses for last epochs')
    
    ax2 = axs[1][1]
    ax2.set_ylabel('loss')
    color = 'tab:orange'
    ax2.plot(x[start_plot:], l1_losses_box[start_plot:], color=color, linestyle=linestyles['loosely dashdotdotted'], marker='o', label="L1 Loss Box")
    color = 'tab:purple'
    ax2.plot(x[start_plot:], l1_losses_conf[start_plot:], color=color, linestyle=linestyles['densely dashed'], marker='o', label="L1 Loss Conf")
    color = 'tab:orange'
    ax2.plot(x[start_plot:], l2_losses_box[start_plot:], color=color, linestyle=linestyles['loosely dashed'], marker='+', label="L2 Loss Box")
    color = 'tab:purple'
    ax2.plot(x[start_plot:], l2_losses_conf[start_plot:], color=color, linestyle=linestyles['densely dashdotdotted'], marker='+', label="L2 Loss Conf")
    ax2.legend(loc="upper left")
    ax2.set_title('Components of training losses for last epochs')
    
    fig.suptitle(title, fontsize = 20)
    if folder_to_save!=None:
        plt.savefig(folder_to_save, format="png", dpi=dpi)
    plt.close(fig)
    return fig


def plot_model_test(preds_boxes_test, preds_scores_test, imgs_test, imgnp, obj, all_boxes, boxes_filtered, scores_filtered):
    plt.figure(figsize=(4*6, 4*2))

    ######### PLOT SAMPLE IMAGE FOR TEST
    cmap = np.array(plt.cm.get_cmap('Paired').colors)
    cmap_rgb: list = np.multiply(cmap, 255).astype(np.int32).tolist()
    color = tuple(cmap_rgb[int(0) % len(cmap_rgb)])
    index_p = [1, 2, 7, 8]
    for i_p, pred_boxes_test, pred_scores_test, img_test in zip(index_p, preds_boxes_test, preds_scores_test, imgs_test):
        #print(img_test.shape)
        img = img_test[0, ...].permute(1, 2, 0).cpu()
        #img = img[:,:,channels]
        img = img[:,:,-3:]#select last 3 channels to plot
        img = np.array(img[:,:,[2, 1, 0]])
        img = (255*img/img.max()).astype('uint8')
        #print(img.shape)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        for i in range(len(pred_boxes_test)):
            if obj=='bbox':
                x1, y1, x2, y2 = pred_boxes_test[i][0], pred_boxes_test[i][1], pred_boxes_test[i][2], pred_boxes_test[i][3]
                xc, yc = (x1+x2)/2, (y1+y2)/2
                #cv2.rectangle(img, (int(pred_boxes_test[i][0]), int(pred_boxes_test[i][1])), (int(pred_boxes_test[i][2]), int(pred_boxes_test[i][3])), (255, 0, 0), 2)
                draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)
            else:
                xc, yc, r = pred_boxes_test[i][0], pred_boxes_test[i][1], pred_boxes_test[i][2]
                #cv2.circle(img, (int(pred_boxes_test[i][0]), int(pred_boxes_test[i][1])), int(pred_boxes_test[i][2]), (255, 0, 0), 2)
                draw.ellipse((xc-r, yc-r, xc+r, yc+r), outline=color, width=2)

            # label
            text = '{} {:.1f}'.format(" ", pred_scores_test[i] * 100)
            font = ImageFont.truetype('calibri.ttf', size=12)
            text_width, text_height = font.getbbox(text)[-2:]
            draw.rectangle(((xc, yc), (xc + text_width, yc + text_height)), fill=color)
            draw.text((xc, yc), text, fill=(0, 0, 0), font=font)

        plt.subplot(2, 6, i_p)
        plt.imshow(img); plt.axis('off')

    ###### PLOT FRACTION OF ORTHOPHOTO FOR SEE ALL DETECTED BOXES 
    print(imgnp.shape)
    #img = imgnp[:,:,channels]
    #imgnp = imgnp.transpose(1, 2, 0)
    img = imgnp[:,:,-3:]#select last 3 channels to plot
    img = img[:,:,[2, 1, 0]].copy()
    img = (255*img/img.max()).astype('uint8')
    print(img.shape)
    for i in range(len(all_boxes)):
        if obj=='bbox':
            cv2.rectangle(img, (int(all_boxes[i][0]), int(all_boxes[i][1])), (int(all_boxes[i][2]), int(all_boxes[i][3])), (0, 255, 0), 2)
        else:
            cv2.circle(img, (int(all_boxes[i][0]), int(all_boxes[i][1])), int(all_boxes[i][2]), (0, 255, 0), 2)

    plt.subplot(1, 3, 2)
    plt.imshow(img); plt.axis('off')
    for i in range(0, img.shape[1], 112):
        plt.axvline(i, 0, 1)
    for i in range(0, img.shape[0], 112):
        plt.axhline(i, 0, 1)
    plt.title('All detected boxes')

    ###### PLOT FRACTION OF ORTHOPHOTO FOR SEE FILTERED DETECTED BOXES 
    img = imgnp[:,:,[2, 1, 0]].copy()
    img = (255*img/img.max()).astype('uint8')
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    color = (32, 255, 32) #green
    for i in range(len(boxes_filtered)):
        if obj=='bbox':
            #cv2.rectangle(img, (int(boxes_filtered[i][0]), int(boxes_filtered[i][1])), (int(boxes_filtered[i][2]), int(boxes_filtered[i][3])), (0, 255, 0), 2)
            x1, y1, x2, y2 = boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3]
            xc, yc = (x1+x2)/2, (y1+y2)/2
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)
        else:
            #cv2.circle(img, (int(boxes_filtered[i][0]), int(boxes_filtered[i][1])), int(boxes_filtered[i][2]), (0, 255, 0), 2)
            xc, yc, r = boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2]
            draw.ellipse((xc-r, yc-r, xc+r, yc+r), outline=color, width=2)
        # label
        text = '{} {:.1f}'.format(" ", scores_filtered[i] * 100)
        font = ImageFont.truetype('calibri.ttf', size=12)
        text_width, text_height = font.getbbox(text)[-2:]
        draw.rectangle(((xc, yc), (xc + text_width, yc + text_height)), fill=color)
        draw.text((xc, yc), text, fill=(0, 0, 0), font=font)
        
    plt.subplot(1, 3, 3)
    plt.imshow(img); plt.axis('off')
    plt.title('Filtered boxes')

def draw_boxes(imgnp, obj, correctionFactor, boxes_filtered, scores_filtered, classes_filtered, boxes_gt, TP_index, FP_index, FN_index):
    img = imgnp[:,:,-3:] #select last 3 channels
    img = img[:,:,[2, 1, 0]].copy()
    img = (255*img/correctionFactor).astype('uint8')
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    color = (32, 255, 32) #green
    for i in TP_index:
        if classes_filtered[i]==0:
                color = (32, 255, 25) #green
        else:
            color = (32, 255, 25)
                
        if obj=='bbox':
            #cv2.rectangle(img, (int(boxes_filtered[i][0]), int(boxes_filtered[i][1])), (int(boxes_filtered[i][2]), int(boxes_filtered[i][3])), (0, 255, 0), 2)
            x1, y1, x2, y2 = boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3]
            xc, yc = (x1+x2)/2, (y1+y2)/2
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)
        else:
            #cv2.circle(img, (int(boxes_filtered[i][0]), int(boxes_filtered[i][1])), int(boxes_filtered[i][2]), (0, 255, 0), 2)
            xc, yc, r = boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2]
            draw.ellipse((xc-r, yc-r, xc+r, yc+r), outline=color, width=2)
        # label
        text = '{} {:.1f}'.format(" ", scores_filtered[i] * 100)
        font = ImageFont.truetype('calibri.ttf', size=12)
        text_width, text_height = font.getbbox(text)[-2:]
        draw.rectangle(((xc, yc), (xc + text_width, yc + text_height)), fill=color)
        draw.text((xc, yc), text, fill=(0, 0, 0), font=font)
    color = (255, 32, 32) #red
    for i in FP_index:
        if obj=='bbox':
            #cv2.rectangle(img, (int(boxes_filtered[i][0]), int(boxes_filtered[i][1])), (int(boxes_filtered[i][2]), int(boxes_filtered[i][3])), (255, 0, 0), 2)
            x1, y1, x2, y2 = boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3]
            xc, yc = (x1+x2)/2, (y1+y2)/2
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)
        else:
            #cv2.circle(img, (int(boxes_filtered[i][0]), int(boxes_filtered[i][1])), int(boxes_filtered[i][2]), (255, 0, 0), 2)
            xc, yc, r = boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2]
            draw.ellipse((xc-r, yc-r, xc+r, yc+r), outline=color, width=2)
        # label
        text = '{} {:.1f}'.format(" ", scores_filtered[i] * 100)
        font = ImageFont.truetype('calibri.ttf', size=12)
        text_width, text_height = font.getbbox(text)[-2:]
        draw.rectangle(((xc, yc), (xc + text_width, yc + text_height)), fill=color)
        draw.text((xc, yc), text, fill=(0, 0, 0), font=font)
    color = (32, 32, 255) #blue
    for i in FN_index:
        if obj=='bbox':
            #cv2.rectangle(img, (int(boxes_gt[i][0]), int(boxes_gt[i][1])), (int(boxes_gt[i][2]), int(boxes_gt[i][3])), (0, 0, 255), 2)
            x1, y1, x2, y2 = boxes_gt[i][0], boxes_gt[i][1], boxes_gt[i][2], boxes_gt[i][3]
            xc, yc = (x1+x2)/2, (y1+y2)/2
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)
        else:
            #cv2.circle(img, (int(boxes_gt[i][0]), int(boxes_gt[i][1])), int(boxes_gt[i][2]), (0, 0, 255), 2)
            xc, yc, r = boxes_gt[i][0], boxes_gt[i][1], boxes_gt[i][2]
            draw.ellipse((xc-r, yc-r, xc+r, yc+r), outline=color, width=2)
    return img
            
def plot_model_eval(imgnp, correctionFactor, obj, boxes_filtered, scores_filtered, classes_filtered, boxes_gt, TP_index, FP_index, FN_index, TP_index2, FP_index2, FN_index2, TP_index3, FP_index3, FN_index3, frac_size=300):
    
    plt.figure(figsize=(9*2, 9*2))

    ######### PLOT SAMPLE IMAGE FOR TEST
    #cmap = np.array(plt.cm.get_cmap('Paired').colors)
    #cmap_rgb: list = np.multiply(cmap, 255).astype(np.int32).tolist()
    #color = tuple(cmap_rgb[int(0) % len(cmap_rgb)])

    ###### PLOT FRACTION OF ORTHOPHOTO FOR SEE ALL DETECTED BOXES
    img0 = draw_boxes(imgnp, obj, correctionFactor, boxes_filtered, scores_filtered, classes_filtered, boxes_gt, TP_index=np.concatenate((TP_index, FP_index)), FP_index=[], FN_index=[])
    
    img1 = draw_boxes(imgnp, obj, correctionFactor, boxes_filtered, scores_filtered, classes_filtered, boxes_gt, TP_index, FP_index, FN_index)
    img2 = draw_boxes(imgnp, obj, correctionFactor, boxes_filtered, scores_filtered, classes_filtered, boxes_gt, TP_index2, FP_index2, FN_index2)
    img3 = draw_boxes(imgnp, obj, correctionFactor, boxes_filtered, scores_filtered, classes_filtered, boxes_gt, TP_index3, FP_index3, FN_index3)
    
    #crop fraction
    center = np.array(imgnp[...,0].shape)//2
    (xmin, ymin), (xmax, ymax) = np.array([center-frac_size//2, center+frac_size//2]) #x1y1x2y2
    print(xmin, xmax, ymin, ymax)
    crop0 = img0.crop((xmin, ymin, xmax, ymax))
    crop1 = img1.crop((xmin, ymin, xmax, ymax))
    crop2 = img2.crop((xmin, ymin, xmax, ymax))
    crop3 = img3.crop((xmin, ymin, xmax, ymax))
    
    plt.subplot(2, 2, 1)
    plt.imshow(crop0); plt.axis('off')
    plt.title('Boxes filtered')
    
    plt.subplot(2, 2, 2)
    plt.imshow(crop1); plt.axis('off')
    plt.title('Detection with min overlaping')
    
    plt.subplot(2, 2, 3)
    plt.imshow(crop2); plt.axis('off')
    plt.title('Detection with iou threshold 0.5')
    
    plt.subplot(2, 2, 4)
    plt.imshow(crop3); plt.axis('off')
    plt.title('Detection with iou threshold 0.75')

def reduction_att(att_model, reduction='mean'):
    if reduction == 'mean':
        att = att_model.mean(axis=1).cpu() #[batch_size, n patches, n patches]
    elif reduction == 'max':
        att = att_model.max(axis=1).values.cpu() #[batch_size, n patches, n patches]
    elif reduction == 'min':
        att = att_model.min(axis=1).values.cpu() #[batch_size, n patches, n patches]
    else:
        print("Error, reduction could be: ['mean', 'max', 'min']")
        return None
    return att
    
def attention_rollout(encoders, reduction='mean'):
    """Computes attention rollout from the given list of attention matrices.
    https://arxiv.org/abs/2005.00928
    """
    #print( encoders.blks[0].att.shape )
    rollout = reduction_att( encoders.blks[0].att.detach().clone(), reduction )
    #print(rollout.shape)
    for i in range(1,len(encoders.blks)):
        att = reduction_att( encoders.blks[i].att.detach().clone(), reduction )
        rollout = torch.matmul(
            0.5*att + 0.5*torch.eye(att.shape[1], device=att.device),
            rollout
        ) # the computation takes care of skip connections
    #batch_size x nq x nk
    return rollout

def get_map_att(encoders, imgs, boxes, reduction='mean'):
    # att in encoder -> [batch_size, n patches, n patches, n heads]
    # imgs: [batch size, num channels, Height, Weight] (float 0-1)
    # boxes: list of len batch size and each element shape: [n detections, xywh or xyr]
    
    batch_size = imgs.shape[0]
    img_size = imgs.shape[-2:]
    
    cls_rollout = attention_rollout(encoders, reduction) #batch_size, n patches, n patches
    #print(cls_rollout.shape)
    n_patches = cls_rollout.shape[1]
    grid_size = int(np.sqrt(n_patches))
    
    #get the available tokens of each image in batch
    tokens = [ [] for _ in range(batch_size)] 
    for i in range(batch_size):
        x, y = (boxes[i][:, 0:2]/img_size[0] * grid_size).long().t() #get int coordinates of corner left of the grid cell
        tokens_temp = grid_size*y+x
    
        for tt in tokens_temp:
            if i>=batch_size:#if there are more targets than imgs available
                continue
            tokens[i].append(int(tt))

    #select the first token of each image
    selected_tokens = []
    for t in tokens:
        if len(t)>0:
            selected_tokens.append(t[random.randint(0, len(t)-1)]) #random valid token
            #selected_tokens.append(t[len(t)//2]) #token in middle
        else:
            selected_tokens.append(random.randint(0, n_patches))
            
    #attention of selected token of the image
    temp = torch.zeros((batch_size, n_patches)) #[batch_size, n patches]
    #print(temp.shape)
    for i in range(batch_size):
        minimo, maximo = cls_rollout[i,selected_tokens[i],:].min(), cls_rollout[i,selected_tokens[i],:].max()
        #print(minimo, maximo)
        #print(cls_rollout[i,selected_tokens[i],:].shape)
        temp[i, :] = (cls_rollout[i,selected_tokens[i],:]-minimo)/(maximo-minimo)#scale 0 to 1 each image of the batch
    cls_rollout = temp 
    
    cls_rollout = F.interpolate (cls_rollout.view(-1, 1, grid_size, grid_size), (img_size[0], img_size[1]), mode='bicubic') # upsample to original image size
    cls_rollout = (cls_rollout-cls_rollout.min())/(cls_rollout.max()-cls_rollout.min())#scale 0 to 1
    
    grays = np.zeros((batch_size, img_size[0], img_size[1], 3)) #3 channels
    heatmap = np.zeros((batch_size, img_size[0], img_size[1], 3)) #3 channels
    masks = np.zeros((batch_size, img_size[0], img_size[1], 3)) #3 channels
    
    alpha = 0.5
    sc = img_size[0]/grid_size
    
    for i in range(batch_size):
        heatmap[i,...] = cv2.applyColorMap(np.uint8(255 * cls_rollout[i,0,...]), cv2.COLORMAP_JET) #0 to 255
        heatmap[i,...] = np.float32(heatmap[i,...])/255 #0 to 1
        
        gray = cls_rollout[i,0,...] # 0 to 1 #np.uint8(255 * cls_rollout[i,0,...]) #0 to 255
        grays[i,:,:,0] = gray; grays[i,:,:,1] = gray; grays[i,:,:,2] = gray
        
        #img = imgs[i,...].permute(1, 2, 0) # W x H x n channels
        img = imgs[0,0:3,...].permute(1, 2, 0) # W x H x n channels, select first 3 channels
        img = np.float32(img[:,:,[2, 1, 0]]/img.max()) # 0 to 1 
        
        mask = alpha*heatmap[i,...] + img
        masks[i,...] = mask / np.max(mask) # 0 to 1
        
        x = int(sc*(selected_tokens[i]%grid_size)) #it goes left to right
        y = int(sc*(selected_tokens[i]//grid_size)) #it goes up to down
        masks[i,...] = cv2.circle(masks[i,...], (x,y), radius=0, color=(0.1, 1, 0.1), thickness=16)
        heatmap[i,...]  = cv2.circle(heatmap[i,...], (x,y), radius=0, color=(0.1, 1, 0.1), thickness=8)
        grays[i,...]  = cv2.circle(grays[i,...], (x,y), radius=0, color=(0.1, 1, 0.1), thickness=8)
    
    return (255*heatmap).astype('uint8'), (255*grays).astype('uint8'), (255*masks).astype('uint8')

def get_data_atts(imgs, enc, model_dir, N_channels, n_model, num_blks, loss_type, obj, conf_thr, iou_thr, diou_thr, img_size=224, reduction='mean', device='cuda'):
    import model.transformer
    model = model.transformer.TransformerObjectDetection(img_size, N_channels, n_model, num_blks, 
                                                         obj = obj, loss_type= loss_type, device=device, bitNet=True).to(device)
    checkpoint = torch.load(model_dir, map_location=torch.device(device))
    if type(checkpoint) == dict:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict( checkpoint )
    print("Model loaded")
    
    imgs = imgs.to(device)
    boxes, scores  = counting.inference(model, imgs, obj, conf_thr, diou_thr, adjust_ij=(0, 0), xyxy=False, device=device)
    
    if enc=='encoder1':
        heatmap, grays, masks = get_map_att(model.encoder1, imgs.to('cpu'), boxes, reduction)
    elif enc=='encoder2':
        heatmap, grays, masks = get_map_att(model.encoder2, imgs.to('cpu'), boxes, reduction)
    
    return heatmap, grays, masks