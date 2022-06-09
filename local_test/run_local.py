import os, shutil
import sys
import pandas as pd
import pprint
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sys.path.insert(0, './../app')
import algorithm.utils as utils 
import algorithm.model_trainer as model_trainer
import algorithm.model_server as model_server
import algorithm.model_tuner as model_tuner
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.elasticnet as elasticnet



hyper_param_path = "./ml_vol/model/model_config/"
hpt_results_path = "./ml_vol/hpt/results/"
data_schema_path = "./ml_vol/inputs/data_config/"
train_data_path = "./ml_vol/inputs/data/training/regressionBaseMainInput/"
test_data_path = "./ml_vol/inputs/data/testing/regressionBaseMainInput/"
model_path = "./ml_vol/model/artifacts/"
output_path = "./ml_vol/outputs"

'''
this script is useful for doing the algorithm testing locally without needing 
to build the docker image and run the container.
make sure you create your virtual environment, install the dependencies
from requirements.txt file, and then use that virtual env to do your testing. 
This isnt foolproof. You can still have host os-related issues, so beware. 
'''



def create_ml_vol():    
    dir_tree = {
        "ml_vol": {
            "inputs": {
                "data_config": None,
                "data": {
                    "training": {
                        "regressionBaseMainInput": None
                    },
                    "testing": {
                        "regressionBaseMainInput": None
                    }
                }
            },
            "model": {
                "model_config": None,
                "artifacts": None,
            }, 
            "hpt": {
                "results": None
            }, 
            "outputs": None
        }
    }    
    def create_dir(curr_path, dir_dict): 
        for k in dir_dict: 
            dir_path = os.path.join(curr_path, k)
            if os.path.exists(dir_path): shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            if dir_dict[k] != None: 
                create_dir(dir_path, dir_dict[k])

    create_dir("", dir_tree)



def copy_example_files():     
    # data schema
    shutil.copyfile("./examples/abalone_schema.json", data_schema_path + "abalone_schema.json")    
    # train data    
    shutil.copyfile("./examples/abalone_train.csv", train_data_path + "abalone_train.csv")    
    # test data     
    shutil.copyfile("./examples/abalone_test.csv", test_data_path + "abalone_test.csv")    
    # hyperparameters
    shutil.copyfile("./examples/hyperparameters.json", hyper_param_path  + "hyperparameters.json")


def run_HPT(): 
    # Read data
    train_data = utils.get_data(train_data_path)    
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)  
    # run hyper-parameter tuning. This saves results in each trial, so nothing is returned
    num_trials = 6
    model_tuner.tune_hyperparameters(train_data, data_schema, num_trials, hyper_param_path, hpt_results_path)


def train_and_save_algo():        
    # Read hyperparameters 
    hyper_parameters = utils.get_hyperparameters(hyper_param_path)    
    # Read data
    train_data = utils.get_data(train_data_path)    
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)  
    # get trained preprocessor, model, training history 
    preprocessor, model, history = model_trainer.get_trained_model(train_data, data_schema, hyper_parameters)            
    # Save the processing pipeline   
    pipeline.save_preprocessor(preprocessor, model_path)
    # Save the model 
    elasticnet.save_model(model, model_path)
    # Save training history
    elasticnet.save_training_history(history, model_path)    
    print("done with training")


def load_and_test_algo(): 
    # Read data
    test_data = utils.get_data(test_data_path)   
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)    
    # instantiate the trained model (does lazy loading)
    predictor = model_server.ModelServer(model_path)
    # make predictions
    predictions = predictor.predict(test_data, data_schema)
    # score the results
    score(test_data, predictions)  
    print("done with predictions")


def score(test_data, predictions): 
    predictions = predictions.merge(test_data[["Id", "Rings"]], on="Id")
    rmse = mean_squared_error(predictions["Rings"], predictions['pred_Rings'], squared=False)
    r2 = r2_score(predictions["Rings"], predictions['pred_Rings'])
    print(f"rmse: {rmse},  r2: {r2}")



if __name__ == "__main__": 
    create_ml_vol()   # create the directory which imitates the bind mount on container
    copy_example_files()   # copy the required files for model training    
    # run_HPT()    
    train_and_save_algo()        # train the model and save
    load_and_test_algo()        # load the trained model and get predictions on test data
    