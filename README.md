# Sarcasm Detection in Social Network

The provided repository serves as a comprehensive resource for the code files developed during the course of this thesis.

In addition to the code files, a presentation of the thesis is attached as a PDF file to provide a deeper understanding of the concepts in my thesis.

[Click here to see the presentation](Thesis_defense.pdf) 



To facilitate a better understanding of the repository's structure, it is organized as follows:

### The Data
In Datasets Folder you can find the datasets utilized in the study for experimentation and evaluation.
every dataset contain at least two fields:sentence and label.
most of the datasets contain extactly two labels: positive or negative.

The Table below show the sources for the datasets and additional intel:

<table border="1">
  <thead>
    <tr>
      <th>Dataset Name</th>
      <th>Source</th>
      <th>Link</th>
      <th>Original Size</th>
      <th>Current Sample Size</th>
      <th>Positive Percentage</th>
      <th>Comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>abuseDataset</td>
      <td>the text is same as olid, but the labels given by Tommaso Caselli</td>
      <td><a href="https://github.com/tommasoc80/AbuseEval">https://github.com/tommasoc80/AbuseEval</a></td>
      <td>13241</td>
      <td>5000</td>
      <td>0.2</td>
      <td>explanation on the labels can be found here</td>
    </tr>
    <tr>
      <td>empathyDataset</td>
      <td></td>
      <td><a href="https://github.com/behavioral-data/Empathy-Mental-Health/blob/master/dataset/emotional-reactions-reddit.csv">https://github.com/behavioral-data/Empathy-Mental-Health/blob/master/dataset/emotional-reactions-reddit.csv</a></td>
      <td>3085</td>
      <td>3085</td>
      <td>0.33</td>
      <td>The original table contains 3 kinds of labels. I changed it to 2 labels for binary classification.</td>
    </tr>
    <tr>
      <td>hateDataset</td>
      <td>Valerio Basile</td>
      <td><a href="https://github.com/msang/hateval">https://github.com/msang/hateval</a></td>
      <td>100</td>
      <td>100</td>
      <td>0.5</td>
      <td></td>
    </tr>
    <tr>
      <td>hateDataset2</td>
      <td>data.world</td>
      <td><a href="https://data.world/thomasrdavidson/hate-speech-and-offensive-language">https://data.world/thomasrdavidson/hate-speech-and-offensive-language</a></td>
      <td></td>
      <td>1400</td>
      <td>0.5</td>
      <td>The positive class is hate speech</td>
    </tr>
    <tr>
      <td>hopeDataset</td>
      <td></td>
      <td><a href="https://codalab.lisn.upsaclay.fr/competitions/10215#participate-get_starting_kit">https://codalab.lisn.upsaclay.fr/competitions/10215#participate-get_starting_kit</a></td>
      <td>22652</td>
      <td>5000</td>
      <td>0.2</td>
      <td></td>
    </tr>
    <tr>
      <td>humorDataset</td>
      <td>ColBERT: Using BERT Sentence Embedding in Parallel Neural Networks for Computational Humor</td>
      <td><a href="https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection">https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection</a></td>
      <td>200k</td>
      <td>5000</td>
      <td>0.5</td>
      <td></td>
    </tr>
    <tr>
      <td>ironyDataset</td>
      <td>kaggle</td>
      <td><a href="https://www.kaggle.com/datasets/rtatman/ironic-corpus">https://www.kaggle.com/datasets/rtatman/ironic-corpus</a></td>
      <td>1950</td>
      <td>1950</td>
      <td>0.27</td>
      <td></td>
    </tr>
    <tr>
      <td>OffensiveDataset</td>
      <td>olid</td>
      <td><a href="https://sites.google.com/site/offensevalsharedtask/olid">https://sites.google.com/site/offensevalsharedtask/olid</a></td>
      <td>13241</td>
      <td>5000</td>
      <td>0.2</td>
      <td>The link for downloading the zip file is at the bottom of the page</td>
    </tr>
    <tr>
      <td>offensiveDataset2</td>
      <td>data.world</td>
      <td><a href="https://data.world/thomasrdavidson/hate-speech-and-offensive-language">https://data.world/thomasrdavidson/hate-speech-and-offensive-language</a></td>
      <td></td>
      <td>1400</td>
      <td>0.5</td>
      <td>The positive class is offense</td>
    </tr>
    <tr>
      <td>toxicityDataset</td>
      <td>Vladyslav Kozhukhov</td>
      <td><a href="https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset?select=toxicity_parsed_dataset.csv">https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset?select=toxicity_parsed_dataset.csv</a></td>
      <td>160k</td>
      <td>5000</td>
      <td>0.5</td>
      <td>The related dataset is "toxicity parsed dataset"</td>
    </tr>
    <!-- Continue adding rows as needed -->
  </tbody>
</table>


### The model
We use `bert-base-uncased` for the binary calssifiction task. 
BERTmodels Folder hosts essential Python scripts related to BERT models' development and evaluation,
Here a brief explanation on the files: 

    1) bertModels.py - A Python file encompassing the class definitions of BERT models, thoughtfully tailored to the specific objectives of the study.
    
    2) runBERTModels.py - A script file offering a command-line interface (CLI) to evaluate the BERT models. By appending specific parameters, users can effortlessly assess various aspects 
      of the models' performance. (example attached below).
      
    3) open_terminal.ipynb - This notebook serves as an illustrative guide, demonstrating how to employ command-line instructions to create specialized BERT models. By following the      
       instructions, users can generate customized models to cater to their particular requirements.
       
    4) load_evaluated_model.ipynb - This notebook is a versatile tool to load and manipulate evaluated BERT models. Users can utilize this notebook to load the models they've generated, gain          insights into their performance metrics (such as F-scores), predict sentences, visualize histograms, and more. This interactive notebook streamlines the exploration of model    
        capabilities.
       	

### how to initiate this model
after download the this project, go the directory where
runBERTmodels.py is located. then execute in shell this command:

```
python  runBERTmodels.py
[--absolute_path ABSOLUTE_PATH]
[--datasets DATASETS]
[--output_model_dir OUTPUT_MODEL_DIR]
```


 	
-     ABSOLUTE_PATH is the path to the datasets folder
-     DATASETS  is short path to datasets for cross domain evaluation:
      The first dataset used for training the model and other for testing it.
-     OUTPUT_MODEL_DIR is the path where to save the model

Example for the command above:

```
python  runBERTmodels.py  --absolute_path "C:/Thesis/datasets/"--datasets "SARC_dataset_train.csv sarc_test.csv SemEval2022_test.csv" --output_model_dir "Thesis/savedModels"
```
