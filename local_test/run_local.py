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


inputs_path = "./ml_vol/inputs/"

data_schema_path = os.path.join(inputs_path, "data_config")

data_path = os.path.join(inputs_path, "data")
train_data_path = os.path.join(data_path, "training", "regressionBaseMainInput")
test_data_path = os.path.join(data_path, "testing", "regressionBaseMainInput")

model_path = "./ml_vol/model/"
hyper_param_path = os.path.join(model_path, "model_config")
model_artifacts_path = os.path.join(model_path, "artifacts")

output_path = "./ml_vol/outputs"
hpt_results_path = os.path.join(output_path, "hpt_results")
testing_outputs_path = os.path.join(output_path, "testing_outputs")
errors_path = os.path.join(output_path, "errors")

'''
this script is useful for doing the algorithm testing locally without needing 
to build the docker image and run the container.
make sure you create your virtual environment, install the dependencies
from requirements.txt file, and then use that virtual env to do your testing. 
This isnt foolproof. You can still have host os or python-version related issues, so beware. 
'''

dataset_name = "abalone"; id_col = "Id"; target_col = "Rings";
# dataset_name = "auto_prices"; id_col = "id"; target_col = "price";
# dataset_name = "computer_activity"; id_col = "id"; target_col = "usr";
# dataset_name = "heart_disease"; id_col = "Id"; target_col = "num";
# dataset_name = "white_wine"; id_col = "id"; target_col = "quality";


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
            
            "outputs": {
                "hpt_outputs": None,
                "testing_outputs": None,
                "errors": None,                
            }
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
    shutil.copyfile(f"./examples/{dataset_name}_schema.json", os.path.join(data_schema_path, f"{dataset_name}_schema.json"))
    # train data    
    shutil.copyfile(f"./examples/{dataset_name}_train.csv", os.path.join(train_data_path, f"{dataset_name}_train.csv"))    
    # test data     
    shutil.copyfile(f"./examples/{dataset_name}_test.csv", os.path.join(test_data_path, f"{dataset_name}_test.csv"))    
    # hyperparameters
    shutil.copyfile("./examples/hyperparameters.json", os.path.join(hyper_param_path, "hyperparameters.json"))


def run_HPT(): 
    # Read data
    train_data = utils.get_data(train_data_path)    
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)  
    # run hyper-parameter tuning. This saves results in each trial, so nothing is returned
    num_trials = 20
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
    pipeline.save_preprocessor(preprocessor, model_artifacts_path)
    # Save the model 
    elasticnet.save_model(model, model_artifacts_path)
    # Save training history
    elasticnet.save_training_history(history, model_artifacts_path)    
    print("done with training")


def load_and_test_algo(): 
    # Read data
    test_data = utils.get_data(test_data_path)   
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)    
    # instantiate the trained model 
    predictor = model_server.ModelServer(model_artifacts_path)
    # make predictions
    predictions = predictor.predict(test_data, data_schema)
    # save predictions
    predictions.to_csv(os.path.join(testing_outputs_path, "test_predictions.csv"), index=False)
    # score the results
    score(test_data, predictions)  
    print("done with predictions")


def score(test_data, predictions): 
    predictions = predictions.merge(test_data[[id_col, target_col]], on=id_col)
    rmse = mean_squared_error(predictions[target_col], predictions['prediction'], squared=False)
    r2 = r2_score(predictions[target_col], predictions['prediction'])
    print(f"rmse: {rmse},  r2: {r2}") 



if __name__ == "__main__": 
    # create_ml_vol()   # create the directory which imitates the bind mount on container
    # copy_example_files()   # copy the required files for model training    
    # run_HPT()               # run HPT and save tuned hyperparameters
    # train_and_save_algo()        # train the model and save
    load_and_test_algo()        # load the trained model and get predictions on test data
     