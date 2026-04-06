""" A program which performs k-means clustering to get the the dimentions of anchor boxes """
import numpy as np
import torch
import os
import warnings

def avg_iou(boxes, clusters):
    """ Calculates average IoU between the GT boxes and clusters 
        Input:
            boxes: array, having width and height of all the GT boxes
        Output:
            Returns numpy array of average IoU with all the clusters
    """
    return np.mean([np.max(iou_distance_Kmeans(boxes, clusters), axis=1)])
"""
def iou_distance_Kmeans(boxes, clusters):
    Calculates Intersection over Union between the provided boxes and cluster centroids
        Input:
            boxes: Bounding boxes -> N boxes x 2
            clusters: cluster centroids -> k clusters x 2
        Output:
            IoU between boxes and cluster centroids
    
    n = boxes.shape[0]
    k = clusters.shape[0]

    box_area = boxes[:, 0] * boxes[:, 1] # Area = width * height -> N boxes x 1
    # Repeating the area for every cluster as we need to calculate IoU with every cluster
    box_area = box_area.repeat(k).reshape(n, k) # N boxes x k clusters

    cluster_area = clusters[:, 0] * clusters[:, 1] # k clusters x 1
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))

    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    inter_area = np.multiply(min_w_matrix, min_h_matrix)

    result = inter_area / (box_area + cluster_area - inter_area) # N boxes x k clusters
    return result
"""
def iou_distance_Kmeans(boxes, clusters):
    """ Calculates Intersection over Union between the provided boxes and cluster centroids
        Input:
            boxes: Bounding boxes -> N boxes x 2
            clusters: cluster centroids -> k clusters x 2
        Output:
            IoU between boxes and cluster centroids
    """
    n = boxes.shape[0]
    k = clusters.shape[0]

    box_area = np.pi*boxes[:, 0]**2
    # Repeating the area for every cluster as we need to calculate IoU with every cluster
    box_area = box_area.repeat(k).reshape(n, k) # N boxes x k clusters

    cluster_area = np.pi*clusters[:, 0]**2 # k clusters x 1
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))

    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    #box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    #cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
    #min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    inter_area = np.pi*min_w_matrix**2

    result = inter_area / (box_area + cluster_area - inter_area) # N boxes x k clusters
    return result

def kmeans(boxes, k, max_iters=10000):
    """ Executes k-means clustering on the provided dimentions of boxes with IoU as
        distance metric.
        Input:
            boxes: numpy array containing dimentions of all the boxes
            k: num of clusters
            max_iters: maximum iterations to try to define clusters before resetting cluster centroids and trying again
        Output:
            clusters after convergence
    """
    num_boxes = boxes.shape[0]
    distances = np.empty((num_boxes, k))
    last_cluster = np.zeros((num_boxes, ))

    # Initializing the clusters
    np.random.seed()
    clusters = boxes[np.random.choice(num_boxes, k, replace=False)]
    count = 0
    # Optimizarion loop
    while True:
        if count == max_iters:
            clusters = boxes[np.random.choice(num_boxes, k, replace=False)]
            count=0
            print('Clusters reset for {0:d} clusters'.format(k))
        distances = 1 - iou_distance_Kmeans(boxes, clusters)
        mean_distance = np.mean(distances)
        
        current_nearest = np.argmin(distances, axis=1)
        if(last_cluster == current_nearest).all():
            break # The model is converged
        for cluster in range(k):
            clusters[cluster] = np.mean(boxes[current_nearest == cluster], axis=0)

        last_cluster = current_nearest
        count+=1
    return clusters

def get_boxes(file_path, img_size=224, bt='bbox'):
    """ Extracts the bounding boxes from the coco train.txt file 
        Input:
            file_path: path of annotations of the images
            img_size: original size of the images
            bt: box type, i.e. (targetClass, xc, yc, w, h) or (targetClass, xc, yc, r)
        Output:
            numpy array containing all the bouding boxes (considering values in the dataset files are in the range 0-1)
    """
    assert bt in ['bbox', 'cbbox']
    
    p = 5 if bt=='bbox' else 4 #number of values for each box
    file_names = os.listdir(file_path)
    for i in range(len(file_names)):
        label_path = file_path+file_names[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boxes = np.loadtxt(label_path).reshape(-1, p)
            
        dims = boxes[:,3:]
        dims = dims*img_size  
        dataSet = dims if i == 0 else np.concatenate((dataSet, dims))
        
    return dataSet

def get_clusters(num_clusters, file_path, img_size=224, dt = 'bbox'):
    """ Calls all the required functions to run k-means and get good anchor boxes 
    Input:
        num_clusters: number of clusters
        file_path: path of annotations of the images
        img_size: original size of the images
        bt: box type, i.e. (targetClass, xc, yc, w, h) or (targetClass, xc, yc, r) 
    Output:
        Returns avg_accuracy of computer anchor box over the whole dataset and the anchors
    """
    all_boxes = get_boxes(file_path, img_size, dt)
    
    result = kmeans(all_boxes, num_clusters)
    result = result[np.lexsort(result.T[0, None])]
    
    avg_acc = avg_iou(all_boxes, result)*100

    return avg_acc, result