################################################################################################################
################################################################################################################
# import overall usefull libraries
import os
import sys
import time
from datetime import datetime
import platform
import joblib
from multiprocessing import cpu_count

# import specific libraries for this notebook
import pandas as pd

# import specific functions from torch and sklearn
from functools import reduce
from sklearn.model_selection import ParameterGrid
import torch
from torch import optim, nn

### import from parent directory with a little help from sys.path.insert()
sys.path.insert(0, '..')

# from util.py (file which once contained all classes and functions):
from util import train_model_double, dueling_net

### Configuration file to determine root directory and subdirectories
import conf

# from configuration file set working directory
os.chdir(os.path.join(conf.ROOT_DIR, 'SEPSIS'))

################################################################################################################
print('\n\n-----------------\nDQN MODEL TRAINING\n-----------------')

### Print system configuration
conf.print_python_environment()

### pretty print
import pprint
pp = pprint.PrettyPrinter(indent=4)


use_gpu = torch.cuda.is_available()

################################################################################################################
################################################################################################################
### Experiment name
exp_name = 'FINAL'

### OPTIONAL: Continue training
continue_training = False
cont_exp_grid_run = 0
original_exp_name = exp_name

#### super OPTIONAL: Continue training a previous interim model
continue_interim = False
continue_interim_i = 0

############################################
### LOAD DATA AND SET PARAMETERS
# This is horrible practice: https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
if not os.path.exists(os.path.join(conf.EXP_DIR, exp_name)):
    raise Exception('Cannot find experiment directory, run create_exp_dataset.ipnyb to create directories.')
else:
    exp_dir = os.path.join(conf.EXP_DIR, exp_name)
    ############################################
    # load data files    
    try:
        data_dict = joblib.load(os.path.join(exp_dir, 'data/FINAL_data_dict.pkl'))
    except:
        raise Exception('Cannot load dataset, run create_exp_dataset.ipynb to create new data pickle files.')
    try:
        # Action probabilities of physician's action used for intermediate evaluateion
        train_pi_behavior = pd.read_pickle(os.path.join(exp_dir, 'KNN/KNN_pi_behavior_' + 'train' + 'data.pkl')) # pi_evaluation
        val_pi_behavior = pd.read_pickle(os.path.join(exp_dir, 'KNN/KNN_pi_behavior_' + 'val' + 'data.pkl')) # pi_evaluation
        test_pi_behavior = pd.read_pickle(os.path.join(exp_dir, 'KNN/KNN_pi_behavior_' + 'test' + 'data.pkl')) # pi_evaluation
    except: 
        raise Exception('Cannot load KNN files, run Physician_KNN.py to create new KNN pickle files.')
    try:
        # dataset MDP Q function (FQI-SARSA)
        train_MDP_Q = pd.read_pickle(os.path.join(exp_dir, 'FQI/FQI_QValues_continuous_' + 'train' + 'data.pkl'))
        val_MDP_Q = pd.read_pickle(os.path.join(exp_dir, 'FQI/FQI_QValues_continuous_' + 'val' + 'data.pkl'))
        test_MDP_Q = pd.read_pickle(os.path.join(exp_dir, 'FQI/FQI_QValues_continuous_' + 'test' + 'data.pkl'))
    except: 
        raise Exception('Cannot load FQI files, run Physician_FQI.py to create new FQI pickle files,')
    
    
    ############################################
    data_dict_MDP = {'train_pi_behavior': train_pi_behavior,
                 'val_pi_behavior': val_pi_behavior,
                 'test_pi_behavior': test_pi_behavior,
                 'train_MDP_Q': train_MDP_Q,
                 'val_MDP_Q': val_MDP_Q,
                 'test_MDP_Q': test_MDP_Q
                }

###################
# define hyperparameter tuning grid
param_options = {   'state_dim' :         [data_dict['train']['X'].shape[1]],  
                    'action_dim' :        [21],                                 
                    'gamma' :             [0.9],                              
                    'batch_size' :        [32,128],                        
                    'lr' :                [1e-4],                               
                    'num_epochs' :        [250000],                             
                    'hidden_dim' :        [128],                           
                    'num_hidden' :        [2],                       
                    'drop_prob' :         [0.0],                         
                    'option' :            ['linear'],                  
                    'use_scheduler':      ['StepLR'],                       
                    'scheduler_gamma':    [0.9,0.99],                       
                    'sched_step_size':    [25000,50000],                  
                    'tau':                [1e-4],                       
                    'reg_lambda' :        [5],                            
                    'REWARD_THRESHOLD' :  [15],                               
                    'PER_beta_start' :    [0.9],                             
                    'PER_alpha' :         [0.6],                        
                    'PER_epsilon' :       [0.01],                      
                    'PER_sample_alpha' :  [0.0],                            
                    'PER_start_prob' :    [0.001],                          
                    'use_GPU':            [use_gpu],                            
                    'tracking_step_eval':           [5000],                     
                    'tracking_step_interim_model':  [5000],                     
                    'tracking_console_print':       [1000]                      
         }

###################
# Create config grid:
config_grid = ParameterGrid(param_options)

################################################################################################################
################################################################################################################
### TRAINING LOOP
print('\n-----------------\nSTART EXPERIMENT')
print("experiment: " + exp_name)
print("Started at: " + str(datetime.now()) + "\n-----------------")
total_since = time.time()
exp_grid_run = 0
# For each configuration LOOP
for config in config_grid:
    exp_grid_run +=1
    if not os.path.exists(os.path.join(exp_dir, 'models/' + exp_name + "_" + str(exp_grid_run) )):
        os.makedirs(os.path.join(exp_dir, 'models/' + exp_name + "_" + str(exp_grid_run) ))

    ###################
    ### Models
    model = dueling_net(D_in=config['state_dim'],
                        H=config['hidden_dim'],
                        D_out=config['action_dim'],
                        drop_prob=config['drop_prob'],
                        num_hidden=config['num_hidden'],
                        option=config['option']
                        )

    target_model = dueling_net(D_in=config['state_dim'],
                               H=config['hidden_dim'],
                               D_out=config['action_dim'],
                               drop_prob=config['drop_prob'],
                               num_hidden=config['num_hidden'],
                               option=config['option']
                               )
        
    ###################
    ### Optimizer
    optimizer = optim.Adam([{'params': model.parameters()}],
                           lr=config['lr'])

    ###################
    ### Scheduler
    if config['use_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
    elif config['use_scheduler'] == 'StepLR':      
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['scheduler_gamma']) # step size applied with config['sched_step_size'] in util.py train function
    else:
        scheduler = None

    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data).float()   
            
    ###################
    ### Create CONFIG dataframe for saving to disk
    config_df = pd.DataFrame(config, index=[0])
            
    ###################
    # OPTIONAL: Load previous model    
    if continue_training: 
        
        ###################
        # This is a mess
        exp_name = original_exp_name + '_' + str(cont_exp_grid_run) + '_continued'
        new_exp_dir = os.path.join(exp_dir, 'models/' + exp_name + "_" + str(exp_grid_run) )
        
        # create model_continued model directory
        if not os.path.exists(new_exp_dir):
            os.makedirs(new_exp_dir)
        
        # load moel
        if continue_interim: 
            exp_model = original_exp_name + '/models/' + original_exp_name + '_' + str(cont_exp_grid_run) + '/' + original_exp_name + '_' + str(cont_exp_grid_run) + '_interim_' + str(continue_interim_i) + '_iteration_model.chk'
        else:
            exp_model = original_exp_name + '/models/' + original_exp_name + '_' + str(cont_exp_grid_run) + '_model.chk'
            
        selected_model = os.path.join(conf.EXP_DIR, exp_model) 

        ###################    
        # CPU or GPU, let the budget decide!  
        if use_gpu:
            model = model.cuda()
            target_model = target_model.cuda()
            model.load_state_dict(torch.load(selected_model))
            target_model.load_state_dict(torch.load(selected_model))
        else:
            model.load_state_dict(torch.load(selected_model, map_location=lambda storage, loc: storage))
            target_model.load_state_dict(torch.load(selected_model, map_location=lambda storage, loc: storage))
        
        ###################
        # experiment name and location for interim model saving
        experiment = { 'exp_name': exp_name, 'exp_dir': exp_dir, 'exp_grid_run': exp_grid_run, 'exp_grid_total_runs': len(config_grid) }
            
        # Save CONFIG dataframe to disk and set location for FINAL model saving 
        checkpoints_path = os.path.join(exp_dir, 'models/' + exp_name + "_" + str(exp_grid_run) + '_model.chk')
        config_df.to_csv(os.path.join(exp_dir, 'models/' + exp_name + "_" + str(exp_grid_run) + '_config.csv'), index=False)
        
    ###################
    # IF NOT LOADING PREVIOUS MODEL, START FRESH
    else:
        ###################
        # initialize model weights
        model.apply(weights_init)
        target_model.apply(weights_init)
        if use_gpu:
            model = model.cuda()
            target_model = target_model.cuda()
            
        # experiment name and location for interim model saving
        experiment = { 'exp_name': exp_name, 'exp_dir': exp_dir, 'exp_grid_run': exp_grid_run, 'exp_grid_total_runs': len(config_grid) }
    
        ###################
        # Save CONFIG dataframe to disk and set location for FINAL model saving
        checkpoints_path = os.path.join(exp_dir, 'models/' + exp_name + "_" + str(exp_grid_run) + '_model.chk')
        config_df.to_csv(os.path.join(exp_dir, 'models/' + exp_name + "_" + str(exp_grid_run) + '_config.csv'), index=False)

        
    ############################################################################  
    if continue_training:
        print('\n-----------------\nSTART CONTINUED MODEL TRAINING')
    else:        
        print('\n-----------------\nSTART MODEL TRAINING')
    print("Experiment name: " + exp_name)
    print("Experiment " + str(exp_grid_run) + ' out of ' + str(len(config_grid)))
    print("configuration of model: ")
    pp.pprint(config)
    print('-----------------')
   
    ###################
    # This is probably what your looking for, the actual model training... it's at the end of util.py, you're welcome!
    tracking_performance_dict, performance_dict, best_model = train_model_double(model=model,
                                                                                 target_model=target_model,
                                                                                 data_dict=data_dict,
                                                                                 data_dict_MDP=data_dict_MDP,
                                                                                 config=config,
                                                                                 optimizer=optimizer,
                                                                                 scheduler=scheduler,
                                                                                 experiment=experiment,
                                                                                 use_gpu=use_gpu)

    ###################
    # save best model of this config run
    torch.save(best_model.state_dict(), checkpoints_path)
    
    # create dictionaries
    performance_df = pd.DataFrame.from_dict(performance_dict)
    tracking_performance_df = pd.DataFrame.from_dict(tracking_performance_dict)

    # write to csv
    tracking_performance_df.to_csv(os.path.join(exp_dir, 'performance/' + exp_name + "_" + str(exp_grid_run) + '_tracking_performance.csv'), index=False)
    performance_df.to_csv(os.path.join(exp_dir, 'performance/' + exp_name + "_" + str(exp_grid_run) + '_performance.csv'), index=False)
    
################################################################################################################
################################################################################################################
time_elapsed = time.time() - total_since
hours = time_elapsed//3600
temp = time_elapsed - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('\n-----------------\nFINISHED TRAINING ALL EXPERIMENTS\n-----------------')
print(exp_name)
print("Finished at: " + str(datetime.now()))
print('Experiment complete in %d hours, %d minutes and %d seconds' %(hours,minutes,seconds))

