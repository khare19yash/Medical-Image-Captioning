# Multilabel Classification and Caption Generation for Medical Images using Graph Neural Networks

## Abstract

   Medical image captioning is the task of automatically generating sentences that describe input medical images in the best way possible in the form of natural language. It requires an effective way to deal with understanding and evaluating the similarity among visual and text-based components and generating a sequence of output words. Automatic medical image caption generation has been an appealing research problem for computer-aided diagnosis to lighten the responsibility of doctors in recent years. The deep learning methodology for natural image captioning is effectively adapted to generating the respective captions. However, medical image captioning is different from the natural image captioning task as clinical and diagnostic keywords referenced are significant in medical image captioning in contrast with the equal importance of every word in a natural image caption. 
   Hence, we propose a novel heterogeneous graph and transformer decoder based image captioning technique which first predicts keywords for each image using link prediction between image and word nodes. Then combine the image and word embeddings to generate the final caption for each image.

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Method

We propose a graph-based approach to create a mapping between image and text (keywords extracted from captions). Then use the combined image+text embedding to generate the final caption. Overall the proposed approach is divided into two tasks namely, Link Prediction and Caption Generation.

![pipeline](https://user-images.githubusercontent.com/17990196/208685370-4244e4d5-e6fb-4153-a86d-edeceea8c098.jpg)

### Multilabel Classification using Link Prediction
We build a heterogeneous graph with two node types: image and words (keywords extracted from training corpus). The image node embedding is extracted from a pretrained ResNet152 model, and the word node embedding is extracted from BERT. The graph contains three types of links between these nodes: image-image links, word-word links, and image-word links.
We do the link prediction task on the heterogeneous graph for the image-word type links to get the keywords for each image. For this, we apply the GraphSAGE layer with a mean aggregator for each type of node in the link (image and text).

![graph](https://user-images.githubusercontent.com/17990196/208685288-dad4c1e5-582f-4954-bade-b7d8c0e95cb1.jpg)
   
### Caption Generation
In this task, we pass the combined image+text embedding to a seq2seq transformer-based encoder-decoder model to generate the final caption.

<!-- ## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
``` -->

## Multilabel Classification Results

Our model achieves the following performance on :
### Open-I Dataset
![openi_result](https://user-images.githubusercontent.com/17990196/208685422-7fbbc476-f375-46c8-84fe-4d6b1e641930.jpeg)

| Dataset         | Method  | B1 | B2 | B3| B4 | Rogue-L | Meteor |  
| ------------------ |---------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| Open-I  | Our  | 0.374 | 0.242 | 0.162 | 0.106 | 0.340 | 0.180 |
| ------------------ |---------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| Open-I  | SOTA  | 0.473 | 0.305 | 0.217 | 0.162 | 0.378 | 0.186 |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

### Qualitative Results
Results
