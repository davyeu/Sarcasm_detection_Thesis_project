# Sarcasm Detection in Social Network

The provided repository serves as a comprehensive resource for the code files developed during the course of this thesis.

In addition to the code files, a presentation of the thesis is attached as a PDF file to provide a deeper understanding of the concepts in my thesis.

[View the PDF](Thesis defense.pdf) 



To facilitate a better understanding of the repository's structure, it is organized as follows:

### The Data
In Datasets Folder you can find the datasets utilized in the study for experimentation and evaluation.
every dataset contain at least two fields:sentence and label.
most of the datasets contain extactly two labels: positive or negative.

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
