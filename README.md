# Explainable Bit-STED

Explaining an Object Detection Model for Screening Anomalies

Related paper: <b>[Explainable Transformer-Based Anomaly Screening in Agave Crops](https://authors.elsevier.com/a/1ltmucFCSf6XG) </b>

A useful paper on the used model for detection and then for explanation can be consulted in <b>[Bit-STED](https://www.sciencedirect.com/science/article/abs/pii/S0168169925011536) </b>.

Monitoring agave crops is essential for the tequila industry; however, the scarcity of labeled datasets constrains automated anomaly screening. Vision transformers (ViTs) provide strong global feature representations; however, their “black-box” nature and the lack of ground-truth labels hinder their practical deployment in complex agricultural environments. Herein, a novel, explainability-guided anomaly-screening framework based on a lightweight Bit-STED model is proposed. The explainability gap was addressed by adapting ViT-CX to generate saliency tiles. Strategic data augmentation using TrivialAugment alone improved pointing-game accuracy from 0.56 to 0.86. Saliency tiles were fused into field-scale explainability orthomaps to preserve spatial context. These orthomaps were used as the input in a robust statistical framework that quantified anomaly scores based on pixel-wise probabilities derived from heatmaps and applied a 1.5 interquartile range (IQR) threshold for anomaly screening. A comprehensive statistical validation, including Moran’s I, PERMANOVA, and Ripley’s H function, showed that the screened candidate plants exhibited feature differences and non-random spatial organization relative to the remaining detected population. These findings support the internal statistical consistency of the screening output. The proposed framework provides a transparent “visual attention” tool that enables farmers to support transparent prioritization of plants for subsequent inspection.

<p>Data can be available on request</p>

The data of the images are TIF files with the multispectral data of the image.

The image size is 224x224.

## General overview of the followed methodology.

<img width="3470" height="783" alt="general" src="https://github.com/user-attachments/assets/849103fe-a779-4fd0-843f-232b5260fe27" />

The specific contributions of this study can be described as follows:

1. We demonstrate that strategic data augmentation (DA) improves attention-based saliency maps by focusing on plant-relevant pixels.

2. We introduce a method for merging tile-level saliency maps into spatially coherent explainability orthomaps for field-scale analysis.

3. We develop an automated, field-relative pipeline for screening statistically atypical plants without requiring disease or stress labels.

4. We propose a statistical evaluation framework to characterize screened candidates through shape, texture, and color variables.

The trained models are provided in the Datasets_STED folder for run 1001, corresponding to the model with data augmentation, and 1002, corresponding to the model without data augmentation. Both with an auxiliary segmentation head.

Three explainability methods were tested in this study. The comparison according to the Bit-STED model is shown below.

<img width="724" height="850" alt="explainability_tiles_examples" src="https://github.com/user-attachments/assets/f037ff82-1d33-4888-8427-7120b7d4f7e4" />

A simple test of the process to merge saliency maps from an orthophoto is developed on a Jupyter notebook "Explainability-VIT-CX-orthomap-test"

<img width="2475" height="4768" alt="mergingProcess" src="https://github.com/user-attachments/assets/ed062b17-2c90-443e-9040-2cae1fb8862f" />

In addition, the computed features per agave are provided in csv files for each orthomap to run statistical framework on Jupyter notebok "Explainability-VIT-CX-orthomap-statistic-test"

