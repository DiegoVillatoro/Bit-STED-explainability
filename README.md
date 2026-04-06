# Explainable Bit-STED
Explaining an Object Detection Model

Related paper: <b>[Explainable Transformer-Based Anomaly Detection in Agave Crops](https://authors.elsevier.com/a/1ltmucFCSf6XG) </b>

This paper deep in <b>[Bit-STED](https://authors.elsevier.com/a/1ltmucFCSf6XG) </b> model, a novel and simplified transformer encoder architecture for efficient agave plant detection and accurate counting using unmanned aerial vehicle (UAV) imagery.

Monitoring agave crops is essential for the tequila industry; however, automated anomaly detection is hindered by a scarcity of labeled datasets. Although vision transformers (ViTs) offer superior global feature representations, their "black-box" nature and lack of ground-truth labels limit their practical deployment in complex agricultural terrains. We propose a novel, unsupervised anomaly detection pipeline utilizing a lightweight Bit-STED model. By implementing strategic data augmentation (TrivialAugment), we improved the average precision from 86.33\% to 92.97\%. To bridge the explainability gap, we adapted ViT-CX to generate causal saliency maps, which were fused into field-scale explainability orthomaps to preserve spatial context. These maps provide the input for a robust statistical framework that identifies anomalies by obtaining an abnormal index based on the probabilities of each pixel location in the heatmap. Subsequently, it is thresholded using a 1.5 IQR threshold. Validation through Moran’s I, PERMANOVA, and Ripley's H function confirmed that the detected anomalies represented biologically meaningful spatial aggregations rather than stochastic noise. This framework provides a transparent "visual attention" tool that enables farmers to trust AI-assisted decisions in agricultural environments.


<p>Data can be available on request</p>
The data of the images are tif files with the multispectral data of the image
The image size is 224x224

DSTAdam optimizer used in training model was obtained from 
https://github.com/kunzeng/DSTAdam

## General overview of followed methodology.
<img width="2648" height="778" alt="general" src="https://github.com/user-attachments/assets/8054bee6-4529-4ce0-a14e-7267a0f1ae76" />

The specific contributions of this study can be described as follows:

**1.** We demonstrate that strategic data augmentation enhances the attention-based saliency maps by focusing on pixels that belong to the plants.

**2.** We propose a method for fusing explainability tiles into a comprehensive orthomap for field-scale anomaly detection.

**3.** We present an automated anomaly detection pipeline that identifies anomalous plants without prior ground-truth knowledge.

**4.** We propose a novel statistical evaluation framework for cluster validation, utilizing a set of metrics to ensure the reliability and significance of detected anomalous patterns.

The trained models are provided in Datasets_STED folder for 1001 the model with Data Augmentation and 1002 the model withoud Data Augmentation

A simple test of the process to merge saliency maps from an orthophoto is develop on jupyter notebook "Explainability-VIT-CX-orthomap-test"

