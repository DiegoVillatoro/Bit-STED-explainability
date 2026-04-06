import utils.datasets
import utils.utils
import time
import numpy as np
import pandas as pd
import os
import torch
import torch.utils.data
import torch.nn.functional as F
#from ViT_CX.ViT_CX import ViT_CX
#from matplotlib import pyplot as plt

def count_parameters(model, grad=True):
    return sum(p.numel() for p in model.parameters() if not grad or p.requires_grad)
def model_size(model):#return the size of the model in mb
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 10**6
    return size_all_mb


def train(model, dataloader, optimizer, clip, device):

    model.train()
    epoch_loss = 0
    times = 0
    
    l1_lcbbox, l1_lconf, l2_lcbbox, l2_lconf = 0, 0, 0, 0
    l1_shape, l1_texture, l1_color = 0,0,0
    l2_shape, l2_texture, l2_color, l_seg = 0,0,0, 0

    for batch_idx, (outs) in enumerate(dataloader):
        if len(outs)==4:
            (_, images, targets, masks) = outs
            masks = masks.to(device)
        else:
            (_, images, targets) = outs
            masks = None
            
        images = images.to(device)
        targets = targets.to(device)  
        
        start_b = time.time()
        #if masks != None:
        loss, outputs, y = model(images, targets, masks)
        #else:
        #loss, outputs = model(images, targets)
        
        loss.backward()
        
        if batch_idx % 10 == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            optimizer.step()
            optimizer.zero_grad()
            
        end_b = time.time()
        times+=(end_b-start_b)
        
        epoch_loss += loss.item()
        l1_lcbbox += model.layers[0].metrics["loss_bbox"].cpu().item() 
        l1_lconf += model.layers[0].metrics["loss_conf"].cpu().item() 
        l2_lcbbox += model.layers[1].metrics["loss_bbox"].cpu().item() 
        l2_lconf += model.layers[1].metrics["loss_conf"].cpu().item() 
        
        #l1_shape += model.layers[0].metrics["loss_shape"].cpu().item() 
        #l1_texture += model.layers[0].metrics["loss_texture"].cpu().item() 
        #l1_color += model.layers[0].metrics["loss_color"].cpu().item() 
        #l2_shape += model.layers[1].metrics["loss_shape"].cpu().item() 
        #l2_texture += model.layers[1].metrics["loss_texture"].cpu().item() 
        #l2_color += model.layers[1].metrics["loss_color"].cpu().item() 
        #l_seg += model.metrics["loss_seg"].cpu().item()
        
        print('\r', '          Batch: %4d,      Loss Tr: %6.3f,         Time: %6.3fs'%(batch_idx, loss.item(), end_b - start_b), end=' ')
    #print() 
    #return mean epoch loss and mean time of all the batches
    N = batch_idx+1
    return epoch_loss/N, times/N, (l1_lcbbox/N, l1_lconf/N, l2_lcbbox/N, l2_lconf/N), N

def evaluate(model, dataloader, device):
    model.eval()
            
    epoch_loss = 0
    times = 0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():

        for batch_idx, (outs) in enumerate(dataloader):
            
            if len(outs)==4:
                (_, images, targets, masks) = outs
                masks = masks.to(device)
            else:
                (_, images, targets) = outs
                masks = None
            
            images = images.to(device)
            targets = targets.to(device)
            
            start_b = time.time()
            #loss, outputs = model(images, targets)
            loss, outputs, y = model(images, targets, masks)
            end_b = time.time()
            
            epoch_loss += loss.cpu().item()
            times+=end_b-start_b
            print('\r', '          Batch: %4d,      Loss Ev: %6.3f,         Time: %6.3fs'%(batch_idx, loss.item(), end_b - start_b), end=' ')
        #print()
    #return epoch loss mean and accuracy mean of all the batches
    N = batch_idx+1
    return epoch_loss/N, times/N, N

def get_previous_time_from_txt(root_save_results):
    #load previous time of training
    f = open(root_save_results+'total_training_time.txt', "r") 
    previous_time_string = f.read().splitlines()[-1]
    if previous_time_string[38:45]=='minutes':
        previous_time = float(previous_time_string[28:37])
    elif previous_time_string[38:43]=='hours':
        previous_time = 60*float(previous_time_string[28:37])
    else:
        print("Fail loading previous time from txt")
        previous_time = 0
    f.close()
    
    return previous_time

def get_pretrained_data(folder_save_results, model, optimizer, scheduler, use_optimizer_dict, use_scheduler_dict, device):
    
    if os.path.isfile(folder_save_results+'last.pt'):
        checkpoint = torch.load(folder_save_results+'last.pt', map_location=torch.device(device))
        if type(checkpoint) != dict:
            print("Loaded only weights from the last training")
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint['model']) 
            print("Loaded weights and parameters of the optimizer from the last training")
            if use_scheduler_dict: #else use the scheduler without change
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("Loaded parameters of the scheduler from the last training")
            if use_optimizer_dict: #else use the scheduler without change
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Loaded parameters of the optimizer from the last training")
        
        df = pd.read_csv(folder_save_results+'Losses.csv')
        losses_train = list(df['Losses train'])
        times_train = list(df['Times train'])
        losses_eval = list(df['Losses eval'])
        times_eval = list(df['Times eval'])
        l1_losses_cbbox = list(df['L1 Losses cbbox'])
        l1_losses_conf = list(df['L1 Losses conf'])
        l2_losses_cbbox = list(df['L2 Losses cbbox'])
        l2_losses_conf = list(df['L2 Losses conf'])
        
        
        best_valid_loss = min(losses_eval)
        x = range(1,len(losses_train)+1)
        print("pre-trained weights loaded")
        
        previous_time = get_previous_time_from_txt(folder_save_results)
    else:
        model.apply(utils.utils.init_weights)
        losses_train = []
        times_train = []
        losses_eval = []
        times_eval = []
        l1_losses_cbbox, l1_losses_conf, l2_losses_cbbox, l2_losses_conf = [], [], [], []
       
        best_valid_loss = float('inf')
        #create folder if available name and doest not exist
        if folder_save_results!='':
            if not os.path.exists(folder_save_results):
                os.makedirs(folder_save_results)
        #x = range(1,epochs+1)#numbered epochs
        print("Weights normal init applied to model")
        previous_time=0
    return model, optimizer, scheduler, losses_train, times_train, losses_eval, times_eval, l1_losses_cbbox, l1_losses_conf, l2_losses_cbbox, l2_losses_conf, best_valid_loss, previous_time

def saveBestTraining(root_save_results, epoch, optimizer, model, scheduler):
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, root_save_results+'best.pt')
    print("{:67}".format(" ")+'Best saved at Epoch: {0:4d}'.format(epoch))
    
def saveLastTraining(root_save_results, optimizer, model, scheduler, 
                                    losses_train, times_train, losses_eval, times_eval, l1_losses_cbbox, 
                                    l1_losses_conf, l2_losses_cbbox, l2_losses_conf,
                                    previous_time, start, epochs):
    end = time.time()
    ################## SAVE MODEL #######################
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, root_save_results+'last.pt')
    ################## SAVE LOSSES #######################
    x = range(1,len(losses_train)+1)
    df = pd.DataFrame(np.array([x, losses_train, times_train, losses_eval, times_eval, 
                                l1_losses_cbbox, l1_losses_conf, l2_losses_cbbox, l2_losses_conf]).transpose(),
                       columns=["Epoch", 'Losses train', 'Times train', "Losses eval", "Times eval", 
                                "L1 Losses cbbox", "L1 Losses conf", "L2 Losses cbbox", "L2 Losses conf",
                                ])
    df.to_csv(root_save_results+'Losses.csv')
    
    ################## SAVE TIME #######################
    total_time = previous_time + (end - start)/60 #minutes
    total_time_string = "Epoch: {0:4d}, Training time: {1:9.4f} minutes, last lf: {2:0.6f}, saved at {3:s}\n".format(epochs, total_time, scheduler.get_last_lr()[0], root_save_results)
    if total_time>59:
        total_time = total_time/60 #hours
        total_time_string = "Epoch: {0:4d}, Training time: {1:9.4f} hours, last lf: {2:0.6f}, saved at {3:s}\n".format(epochs, total_time, scheduler.get_last_lr()[0], root_save_results)
        print("{:68}".format(" ")+'Epoch: {0:4d} finished in {1:6.3f} hours'.format(epochs, total_time ) )
    else:
        print("{:68}".format(" ")+'Epoch: {0:4d} finished in {1:6.3f} minutes'.format(epochs, total_time) )
    
    f = open(root_save_results+'total_training_time.txt', "a")   # 'r' for reading, 'w' for writing, 'a' to append text
    f.write(total_time_string)                                   # Write inside file 
    f.close()                                                    # Close file 
    #print("Last lr: %0.6f and total training time: %s saved in %s"%(scheduler.get_last_lr()[0], total_time_string, root_save_results))
    

def get_dataloaders(folder, obj, augment, prob, batch_size, extract_masks, n_channels):
    if obj == 'bbox':
        #train_dataset = utils.datasets.Load_data_agave('/home/a01328525/Datasets_YOLO/Zones_bboxes_dataset_01/', 'train')
        train_dataset = utils.datasets.Load_data_agave_multispectral(folder, 'train', augment=augment, prob=prob, extract_masks=extract_masks, n_channels=n_channels)
        #test_dataset = utils.datasets.Load_data_agave('/home/a01328525/Datasets_YOLO/Zones_bboxes_dataset_01/', 'test')
        test_dataset = utils.datasets.Load_data_agave_multispectral(folder, 'val', augment=False, prob=0, extract_masks=extract_masks, n_channels=n_channels)
    else:#cbbox
        #train_dataset = utils.datasets.Load_data_agave_circles('/home/a01328525/Datasets_YOLO/Zones_cbboxes_dataset_01/', 'train')
        train_dataset = utils.datasets.Load_data_agave_circles_multispectral(folder, 'train', augment=augment, prob=prob, extract_masks=extract_masks, n_channels=n_channels)
        #test_dataset = utils.datasets.Load_data_agave_circles('/home/a01328525/Datasets_YOLO/Zones_cbboxes_dataset_01/', 'test')
        test_dataset = utils.datasets.Load_data_agave_circles_multispectral(folder, 'val', augment=False, prob=0, extract_masks=extract_masks, n_channels=n_channels)
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   collate_fn=train_dataset.collate_fn,
                                                   pin_memory=True)  # note that we're passing the collate function here

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                                   collate_fn=test_dataset.collate_fn,
                                                   pin_memory=True)  # note that we're passing the collate function here
    return train_dataloader, test_dataloader

def trainer(model, folder_data, obj, optimizer, scheduler, 
            epochs, folder_save_results='', device='gpu', 
            augment=True, batch_size=32, use_optimizer_dict=True, 
            use_scheduler_dict=True, ReduceLROnPlateau=True, 
            probAugment=0.5, saveEach=10, extract_masks=False, n_channels=3):
    
    ############################ READ PREVIOUS WEIGHTS IF EXIST ###################################
    model, optimizer, scheduler, losses_train, times_train, losses_eval, times_eval, l1_losses_cbbox, l1_losses_conf, l2_losses_cbbox, l2_losses_conf, best_valid_loss, previous_time = get_pretrained_data(folder_save_results, model, optimizer, scheduler, use_optimizer_dict, use_scheduler_dict, device)
    
    #l1_losses_shape, l1_losses_texture, l1_losses_color, l2_losses_shape, l2_losses_texture, l2_losses_color, l_losses_seg = [],[],[],[],[],[],[]
    #folder_bboxes = '/home/a01328525/Datasets_YOLO/Zones_bbox_dataset_10/'
    #folder_cbboxes = '/home/a01328525/Datasets_YOLO/Zones_cbboxes_dataset_03/' 
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    print('The model size: {:.3f}MB'.format(model_size(model)))
    #model = torch.compile(model)
    #print("Model compiled!")
    ############################ LOAD DATASET ###################################
    train_dataloader, test_dataloader = get_dataloaders(folder_data, obj, augment, probAugment, batch_size, extract_masks, n_channels)
    ############################ DETERMINE EPOCHS ###################################
    flag_training=False
    previous_epochs = len(losses_train)
    epochs = epochs-previous_epochs
    if epochs>0:
        print("Training for %d epochs"%(epochs) )
    else:
        print("No training developed")
    ############################ EPOCH TRAINING ###################################
    start = time.time()
    for epoch in range(epochs):
        
        start_ep = time.time()
        loss_train, time_train, (l1_lcbbox, l1_lconf, l2_lcbbox, l2_lconf), Nb = train(model, train_dataloader, optimizer, clip=0.5, device=device)
        print('\r', 'Ep: %4d, Batches: %2d,  Avg Loss Tr: %6.3f,     Avg Time: %6.3fs\n'%(epoch+previous_epochs, Nb, loss_train, time_train), end=' ')
        loss_eval, time_eval, Nb = evaluate(model, test_dataloader, device=device)
        print('\r', '          Batches: %2d,  Avg Loss Ev: %6.3f,     Avg Time: %6.3fs\n'%(Nb, loss_eval, time_eval), end=' ')
        
        losses_train.append(loss_train)
        times_train.append(time_train)
        losses_eval.append(loss_eval)
        times_eval.append(time_eval)
        l1_losses_cbbox.append( l1_lcbbox )
        l1_losses_conf.append( l1_lconf )
        l2_losses_cbbox.append( l2_lcbbox )
        l2_losses_conf.append( l2_lconf )
    
        ############################ SAVE BEST ###################################
        if loss_eval < best_valid_loss:
            best_valid_loss = loss_eval
            if folder_save_results!='':
                #save model, optimizer, scheduler params of best
                saveBestTraining(folder_save_results, epoch+previous_epochs, optimizer, model, scheduler)
                
        ############################ SAVE LAST EACH 10 EPOCHS ###################################
        if (epoch+previous_epochs)%saveEach==0 and folder_save_results!='':
            #save model params, losses and total time 
            saveLastTraining(folder_save_results, optimizer, model, scheduler, 
                                    losses_train, times_train, losses_eval, times_eval, l1_losses_cbbox, 
                                    l1_losses_conf, l2_losses_cbbox, l2_losses_conf,
                                    previous_time, start, epoch+previous_epochs)
        if ReduceLROnPlateau:
            scheduler.step(loss_eval)#when using ReduceLROnPlateau require loss_eval
        else:
            scheduler.step()
        flag_training = True
        end_ep = time.time()
        
    ############################ SAVE LAST RESULTS ###################################
    if folder_save_results!='' and flag_training:
        #save model params, losses and total time 
        saveLastTraining(folder_save_results, optimizer, model, scheduler, 
                                losses_train, times_train, losses_eval, times_eval, l1_losses_cbbox, 
                                l1_losses_conf, l2_losses_cbbox, l2_losses_conf,
                                previous_time, start, epoch+previous_epochs)
    else:
        end = time.time()
        print("{:68}".format(" ")+'Finished in {1:6.3f} minutes'.format(epochs, (end-start)/60) )
    
    return model # MODEL