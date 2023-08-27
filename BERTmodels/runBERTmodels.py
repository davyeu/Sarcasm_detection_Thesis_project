



import argparse
from  bertModel import *



def parse_args():
    """
    Parse command line arguments.
    :return: the parsered arguments
    """
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument(
        "--absolute_path",
        default="/content/drive/MyDrive/Thesis project/Thesis/datasets/",
        type=str,
        required=False,
        help="the absolute path for datasets",
    )
    
    parser.add_argument(
        "--datasets",
        default="dataset/",
        type=str,
        required=False,
        help="the paths to datasets where the first is using for training and the else for tests",
    )
    
   
    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        type=str,
        required=False,
        help="for the completed list of models names, please look in https://huggingface.co/models",
    )
    
    parser.add_argument(
        "--model_mode",
        default="training",
        type=str,
        required=False,
        help="set the mode of the model: training or testing. If 'training' is chossen so the" 
                +"first dataset use for train and the rest for test. If 'test' is chossen so" 
                +" the model is pre-trained and all the datasets will be use for testing"
    )
    
    parser.add_argument(
        "--output_model_dir",
        #default="/content/drive/MyDrive/Thesis project/Thesis/BERTmodels/savedModels",
        default=None,
        type=str,
        required=False,
        help="the a path to the folder where we save the model, if not mention the model will not be saved ",
    )
    
    parser.add_argument(
        "--topic",
        default=0,
        type=int,
        required=False,
        help="whether to append the text column with topic column in the dataset. the defualt is not ",
    )

    args = parser.parse_args()
    return args
    


def main():
    args = parse_args()
    print("my args",args,sep="\n")
    myBERTmodel=BERTModel(args.model_name,
                                args.absolute_path,
                                args.datasets,
                                args.output_model_dir,
                                args.model_mode,
                                int(args.topic))
    myBERTmodel.load_data()
    myBERTmodel.train_model()
    myBERTmodel.evaluate_model()
    myBERTmodel.saveModel()

if __name__ == "__main__":
    main()