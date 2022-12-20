>📋  A template README.md for code accompanying a Machine Learning paper

# Multilabel Classification and Caption Generation for Medical Images

## Abstract

   Medical image captioning is the task of automatically generating sentences that describe input medical images in the best way possible in the form of natural language. It requires an effective way to deal with understanding and evaluating the similarity among visual and text-based components and generating a sequence of output words. Automatic medical image caption generation has been an appealing research problem for computer-aided diagnosis to lighten the responsibility of doctors in recent years. The deep learning methodology for natural image captioning is effectively adapted to generating the respective captions. However, medical image captioning is different from the natural image captioning task as clinical and diagnostic keywords referenced are significant in medical image captioning in contrast with the equal importance of every word in a natural image caption. 
   Hence, we propose a novel heterogeneous graph and transformer decoder based image captioning technique which predicts links between images and words to obtain the keywords for each image. The combined image and word embeddings are then used to generate coherent medical captions.

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Method

![Model Pipeline](/images/pipeline.jpg?raw=true "Pipeline")

![Heterogeneous Graph](/images/graph.jpg?raw=true "Graph")

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

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
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Multilabel Classification Results

Our model achieves the following performance on :
\
### Open-I Dataset

![Results](/images/openi_result.jpeg?raw=true "Pipeline")

| Dataset         | Method  | B1 | B2 | B3| B4 | Rogue-L | Meteor |  
| ------------------ |---------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| Open-I  | Our  | B1 | B2 | B3| B4 | Rogue-L | Meteor |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

### Qualitative Results

Results
