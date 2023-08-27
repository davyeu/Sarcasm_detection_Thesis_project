# Sarcasm_detection_Thesis_project

The provided repository serves as a comprehensive resource for the code files developed during the course of this thesis. To facilitate a better understanding of the repository's structure, it is organized as follows:
1.	Datasets Folder: Contains the datasets utilized in the study for experimentation and evaluation.
2.	BERTmodels Folder: Hosts essential Python scripts related to BERT models' development and evaluation:
    - 	bertModels.py: A Python file encompassing the class definitions of BERT models, thoughtfully tailored to the specific objectives of the study.
    - 	runBERTModels.py: A script file offering a command-line interface (CLI) to evaluate the BERT models. By appending specific parameters, users can effortlessly assess various aspects of         the models' performance.
3.	Jupyter Notebooks: These notebooks provide hands-on demonstrations of key procedures and functionalities:
     - 	open_terminal.ipynb: This notebook serves as an illustrative guide, demonstrating how to employ command-line instructions to create specialized BERT models. By following the      
       instructions, users can generate customized models to cater to their particular requirements.
     - 	load_evaluated_model.ipynb: This notebook is a versatile tool to load and manipulate evaluated BERT models. Users can utilize this notebook to load the models they've generated, gain          insights into their performance metrics (such as F-scores), predict sentences, visualize histograms, and more. This interactive notebook streamlines the exploration of model    
        capabilities.



To effectively utilize the repository, follow these steps:
1.	Utilize the provided datasets in the "datasets" folder to conduct experiments.
2.	Leverage the "bertModels.py" file to define custom BERT models that align with the objectives of your research.
3.	Execute the "runBERTModels.py" script with appropriate command-line parameters to evaluate your BERT models and assess their performance.
4.	Refer to the "open_terminal.ipynb" notebook to gain insights into creating specialized BERT models via command-line instructions.
5.	Employ the "load_evaluated_model.ipynb" notebook to load, analyze, and manipulate your evaluated BERT models. This notebook streamlines various aspects of model assessment.
In summary, this repository encapsulates the meticulous development and evaluation of BERT models tailored to the nuances of sarcasm detection. By judiciously organizing code files, datasets, and informative notebooks, the repository fosters an efficient workflow for understanding, implementing, and refining the techniques and insights gleaned from this study.



**insturction for run BERT model with command line**

the name of the script: runBERTmodels.py

first enter the files directory.

execute in shell this command:

python runBERTmodels.py
[--absolute_path ABSOLUTE_PATH]
[--datasets DATASETS]
[--output_model_dir OUTPUT_MODEL_DIR]
[-- model_mode  MODEL_MODE]


 	

- 	where  ABSOLUTE_PATH is the path to the datasets folder
- where DATASETS  is short path to datasets for cross domain evaluation
- 	where OUTPUT_MODEL_DIR is the path where to save the model
- 	where MODEL_MODE is “training” or “testing”

for example, the command:



python  runBERTmodels.py --datasets "SARC_dataset2_train.csv sarc_pos_class.csv sarc_neg_class.csv SemEval2022_pos_class.csv SemEval2022_neg_class.csv sarc_test.csv SemEval2022_test.csv SARC_dataset2_test.csv" --output_model_dir "/content/drive/MyDrive/Thesis project/Thesis/BERTmodels/savedModels"
