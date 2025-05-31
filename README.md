# Brain Tumor Classification Using Images with Convolutional Neural Networks (CNN)

## Autor
Andres Martin Bañuelos Elias
Ingeniería Biomédica, Departamento de Ingeniería Eléctrica y Electrónica, Tecnológico Nacional de México/IT Tijuana. Blvd. Alberto Limón Padilla s/n, Tijuana, C.P. 22454, B.C., México. Email: l21212142@tectijuana.edu.mx

# Abstract

A brain tumor is a collection or mass of abnormal cells in the brain. Because the skull is a rigid and enclosed space, any growth inside it can lead to increased intracranial pressure, potentially causing brain damage or even becoming life-threatening. Brain tumors can be classified as benign (noncancerous) or malignant (cancerous). Early detection and classification of brain tumors is a critical area of research in medical imaging, as it helps determine the most appropriate treatment and can ultimately save lives.

In recent years, deep learning techniques have provided impactful solutions in medical diagnosis. According to the World Health Organization (WHO), proper brain tumor diagnosis includes detection, tumor localization, and classification based on malignancy, grade, and type. This study focuses on using Convolutional Neural Networks (CNNs) for automated brain tumor detection and classification from Magnetic Resonance Imaging (MRI) scans.

Unlike traditional approaches that use separate models for each classification task, this work proposes a multi-task CNN-based model capable of performing multiple tasks simultaneously. The model not only classifies the tumor (based on grade, type, and malignancy) but also identifies its location by segmenting the tumor region within the brain. This unified approach improves efficiency and has the potential to support clinical decision-making.

The dataset used in this project is publicly available and contains MRI images labeled by tumor types (glioma, meningioma, pituitary tumor, and no tumor). It can be accessed through the following link:

Brain Tumor MRI Dataset on Kaggle:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download
