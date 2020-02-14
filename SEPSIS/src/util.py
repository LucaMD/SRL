# import overall usefull libraries
import os
import platform
import sys
import time
import gc

# import project specific external libraries
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# project specific imports
import conf
from per import PrioritizedExperienceReplay

################################################################################################################
# Model architecture - A dueling feedforward net with an arbitrary number of hidden layers
################################################################################################################
class dueling_net(torch.nn.Module):

    def __init__(self, D_in, H, D_out, drop_prob, num_hidden, option='linear'):
        # Initialize the network
        super(dueling_net, self).__init__()

        if option == 'linear':
            self.input_layer = hidden_linear_layer(D_in, H, drop_prob)
            self.value_layer = nn.Linear(H, 1)
            self.advantage_layer = nn.Linear(H, D_out)
            self.layers = nn.ModuleList([self.input_layer])

        if num_hidden > 1:
            self.layers.extend([hidden_linear_layer(H, H, drop_prob) for i in range(num_hidden - 1)])

    def forward(self, x):
        pre_output = nn.Sequential(*self.layers).forward(x)
        value = self.value_layer(pre_output)
        advantage = self.advantage_layer(pre_output)
        advantage_diff = advantage - advantage.mean(1, keepdim=True)
        y_pred = value + advantage_diff

        return y_pred

# A hidden linear layer with dropout, activation, and batch norm
class hidden_linear_layer(torch.nn.Module):

    def __init__(self, D_in, D_out, drop_prob):
        super(hidden_linear_layer, self).__init__()

        self.linear = nn.Linear(D_in, D_out).float()
        self.dropout = nn.Dropout(p=drop_prob)
        self.activation = nn.ELU()
        self.batch_norm = nn.BatchNorm1d(num_features=D_out)

    def forward(self, x):
        result = self.dropout(self.activation(self.batch_norm(self.linear(x))))
        return result


################################################################################################################
# Data loader
################################################################################################################    
def load_data(data, batch_ids, next_state_batch_ids, use_gpu):
    # load the data into a tensor
    if use_gpu:  # TODO make this work properly on the GPU
        batch_ids = batch_ids.cpu()
    states_np = data['X'][(batch_ids)]
    states_np[np.where(np.isinf(states_np))] = 0 # impute inf by mean, something with feature 42, ironic isn't it?
    states = torch.from_numpy(states_np).float()
    
    actions = torch.LongTensor(data['action'][(batch_ids)])
    rewards = torch.from_numpy(data['reward'][(batch_ids)]).float()
    next_states = torch.from_numpy(data['X'][(next_state_batch_ids)]).float()
    
    # async=true to non_blocking-true (SOURCE: https://forums.fast.ai/t/cuda-syntax-error/20177/4)
    if use_gpu:
        states, actions, rewards, next_states = Variable(states.cuda(non_blocking=True)), \
                                                Variable(actions.cuda(non_blocking=True)), \
                                                Variable(rewards.cuda(non_blocking=True)), \
                                                Variable(next_states.cuda(non_blocking=True))
    else:
        states, actions, rewards, next_states = Variable(states), \
                                                Variable(actions), \
                                                Variable(rewards), \
                                                Variable(next_states)
    return states, actions, rewards, next_states




################################################################################################################
# Performance tracker
################################################################################################################
class PerformanceTracker:
    def __init__(self):
        self.performance_dict = {
            'loss': [],
            'avg_best_Q': [],
            'avg_current_Q': [],
            'epoch_action_prob': [],
            'epoch_error': [],
            'epoch_td_error': [],
            'epoch_per_error': [],
            'epoch_reg_term': []
        }
        
        self.tracking_performance_dict = {
            'iteration': [],
            'eval_type': [],
            'model_WDR': [],
            'model_wis': [] ,
            'iteration_action_prob': []
        }
    # function for tracking Q-learning performance on the training datset
    def append(self, loss, avg_best_Q, avg_current_Q, epoch_action_prob,epoch_error, epoch_td_error, epoch_per_error, epoch_reg_term):
        self.performance_dict['loss'].append(loss)
        self.performance_dict['avg_best_Q'].append(avg_best_Q)
        self.performance_dict['avg_current_Q'].append(avg_current_Q)
        self.performance_dict['epoch_action_prob'].append(epoch_action_prob)
        self.performance_dict['epoch_error'].append(epoch_error)
        self.performance_dict['epoch_td_error'].append(epoch_td_error)
        self.performance_dict['epoch_per_error'].append(epoch_per_error)
        self.performance_dict['epoch_reg_term'].append(epoch_reg_term)
        
    # function for tracking Q-learning performance on the validation datset
    def track_append(self, iteration, eval_type, model_WDR, model_wis, best_action_prob):
        self.tracking_performance_dict['iteration'].append(iteration)
        self.tracking_performance_dict['eval_type'].append(eval_type)
        self.tracking_performance_dict['model_WDR'].append(model_WDR)
        self.tracking_performance_dict['model_wis'].append(model_wis)
        self.tracking_performance_dict['iteration_action_prob'].append(best_action_prob)

def track_performance(data_dict, i, gamma, model, pt, pi_behavior, MDP_Q, eval_type, use_gpu):        
        ############################################
        # Model evaluation function
        outputs, best_actions, best_action_probabilities, outputs_prob, state_Q_values, best_policy_values = evaluate_model(model, data_dict, eval_type, use_gpu)
        best_action_prob = best_action_probabilities.mean()
        
        # create an output dataframe with for the Q values and action probabilit
        pi_evaluation = np.around(pd.DataFrame.from_records(outputs_prob),3)

        #############################################
        # Perform WDR analysis
        model_WDR, model_wis = eval_WDR(data_dict, eval_type, gamma, pi_evaluation, pi_behavior, MDP_Q)

        #############################################
        # Track results
        print( str(eval_type) + "\nDQN WDR: " + str(round(model_WDR, 3)) 
                              + "\nDQN WIS: " + str(round(model_wis, 3)))
        
        # track 
        pt.track_append(i, eval_type, model_WDR, model_wis, best_action_prob)
        print('-' * 10)

        
################################################################################################################
# Evaluation
################################################################################################################
def evaluate_model(model, data_dict, eval_type, use_gpu = False):
    # Key = starting state, value = next state
    if (eval_type == 'val'):
        transition_dict = dict(zip(data_dict['val']['state_id'], data_dict['val']['next_state_id']))
    elif (eval_type == 'test'):
        transition_dict = dict(zip(data_dict['test']['state_id'], data_dict['test']['next_state_id']))
    elif (eval_type == 'train'):
        transition_dict = dict(zip(data_dict['train']['state_id'], data_dict['train']['next_state_id']))
    else:
        print("error using evaluate_model: Incorrect eval type")
        return (False)

    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Set model training mode to false
    model.train(False)

    # get next state id's # keep this " for X in batch loop" because we can use it in later in case we need to evaluate in batch modus...
    batch_ids = data_dict[eval_type]['state_id']
    next_state_batch_ids = [transition_dict[x] for x in batch_ids]

    # get data
    states, actions, rewards, next_states = load_data(data_dict[eval_type], batch_ids, next_state_batch_ids, use_gpu)

    # define how many rows this batch of data has
    batch_size = states.shape[0]
    batch_idx = torch.arange(batch_size, device=device)

    # Run model
    outputs = model(states)

    # choose best actions based on highest Q values
    best_actions = torch.max(outputs, dim=1)[1].data

    # Action probability based on a softmax of the Qvalue distribution for each state
    output_prob = torch.nn.functional.softmax(outputs.data, dim = 1)

    # get the probability of the DQN choosen best action
    best_action_probabilities = output_prob[torch.LongTensor(np.arange(batch_size).tolist()),best_actions.data]

    # Get Q(s, a) and Q(s, a*)
    state_Q_values = outputs[batch_idx, actions.data]
    best_policy_values = outputs[batch_idx, best_actions.data]

    # reset model training setting
    model.train(True)
    if use_gpu:
        return outputs.cpu().data.numpy(), best_actions.cpu().numpy(), best_action_probabilities.cpu().numpy(), output_prob.cpu().data.numpy(), state_Q_values.cpu().data.numpy(), best_policy_values.cpu().data.numpy()
    else:
        return outputs.data.numpy(), best_actions.numpy(), best_action_probabilities.numpy(), output_prob.data.numpy(), state_Q_values.data.numpy(), best_policy_values.data.numpy()

    
############################################
# WDR evaluation function
def eval_WDR(data_dict,eval_type, gamma, pi_evaluation, pi_behavior, Q, V='Qmax'):
    # Get actions, rewards and trajectory ids from data dictionary
    batch_ids           = data_dict[eval_type]['state_id']
    actions_sequence    = data_dict[eval_type]['action'][(batch_ids)] # actual actions from data dictionary
    rewards_sequence    = data_dict[eval_type]['reward'][(batch_ids)] # actual rewards from data dictionary
    fence_posts         = get_fence_post(data_dict, eval_type) # list of indexes from data_dictionary to mark the beginning of a trajectory by it' index (state ID)

    # construct value estimates by taking the maximum Q value
    if V == 'Qmax':
        V = Q.max(axis=1)

    # execute WDR evaluation
    eval_wdr, eval_wis  = WDR(actions_sequence, rewards_sequence, fence_posts, gamma, pi_evaluation, pi_behavior, V, Q)
    return eval_wdr, eval_wis

############################################
# get ICU-stay trajectories for each patient, say, row 1-4 (index 0:3) are the records for patient 1, row 5-10 (index (4:9) are for patient 2, etc...
# then, fence_post returns [0, 4, 10 ...] so on and so forth, this is used for WDR estimate.
def get_fence_post(data_dict, eval_type):
    data_dict[eval_type]['state_id']
    data_dict[eval_type]['next_state_id']
    fence_post_array = np.equal(np.array(data_dict[eval_type]['state_id']),np.array(data_dict[eval_type]['next_state_id']))
    index_of_end_of_trajectory = np.append([-1], np.argwhere(fence_post_array==True).flatten(order='C')) # add index 0, the start of the first trajectory
    index_of_beginning_of_new_trajectory = index_of_end_of_trajectory
    index_of_beginning_of_new_trajectory[:] = [x + 1 for x in index_of_end_of_trajectory]
    return index_of_beginning_of_new_trajectory[:-1] # remove the last value

############################################
# WDR in numpy version
def WDR(actions_sequence,
           rewards_sequence,
           fence_posts,
           gamma,
           pi_evaluation,
           pi_behavior,
           V = None,
           Q = None
          ):

    # number of trajectories
    num_of_trials = len( fence_posts )

    # initialize weight table... --> matrix of trajectories by 21 (maximum length of a trajectory?)
    whole_rho = np.zeros((num_of_trials, 21))

    ########################################################
    # for each trajectory calculate the weight for each step
    for trial_i in range( num_of_trials ):

        # initialize the vector (trail_rho) of weights for this trajectory.
        # Predefine the weight of the first state as rho = 1 (equally likely under both pi_eval / pi_behavior)
        rho = 1
        trial_rho = np.zeros(21)
        trial_rho[0] = rho

        # Determine the amount of states in this trajectory. The else statement handles the last trajectory in the dataset.
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( actions_sequence) - fence_posts[-1]

        # for each step in the trajectory
        for t in range(fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            previous_rho = rho
            rho *= pi_evaluation.iloc[ t, actions_sequence[t]] / pi_behavior.iloc[ t, actions_sequence[ t]]
            trial_aux = np.zeros(21)
            trial_aux[t - fence_posts[ trial_i]+1] = 1
            trial_rho = trial_rho + trial_aux*rho

        # for trajectory of shorter then the maximum lenght add [1] to the weights table for those steps in the trial that did not happen for this patient
        if steps_in_trial < 20:

            # for each step after the last true actual step in this trajectory.
            # The Range function turns dataset index into a trial_aux index. this allows to dynamically adept to different trajectory lengths.
            for t in range(fence_posts[ trial_i] + steps_in_trial, fence_posts[trial_i] + 20):
                trial_aux = np.zeros(21)

                # the steps in the trial (index of trail_aux) that correspond to a step after the last step but before the end of the maximum length of a trajectory...
                # set the rho at 1. ~ equally likely/unlikely to happen under both pi_eval and pi_behavior
                trial_aux[t - fence_posts[ trial_i]+1] = 1

                # Trial_rho is already of the true length of the trajectory and gets now added a "fake rho" for each step after the last real step.
                trial_rho = trial_rho + trial_aux*rho

        # the last part of the trajectory weight calculation is to normalize is over the entire dataset
        whole_aux = np.zeros((num_of_trials, 21))
        whole_aux[trial_i, :] = 1
        whole_rho += whole_aux*trial_rho

    weight_table = whole_rho/np.sum(whole_rho, axis = 0)

    ######################################################
    # calculate the doubly robust estimator of the policy
    wdr_estimator = 0
    wis_estimator = 0
    for trial_i in range(num_of_trials):

        # Initialise a new discount factor at the beginning of the trajectory
        discount = 1/gamma

        # Get the index (steps_in_trail) from data datafile that correspond to the states states in this trajectory
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len(actions_sequence) - fence_posts[-1]

        # for each step in this trajectory
        for t in range(fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):

            previous_weight = weight_table[trial_i, t - fence_posts[ trial_i]]

            weight = weight_table[trial_i, t - fence_posts[ trial_i]+1]

            discount *= gamma

            r =  rewards_sequence[ t ]

            # Get the Qvalue for this state (t) and action (action_sequency[t] corresponds to the column index where the Qvalue for the action of interest is located)
            Q_value=  Q.iloc[ t, actions_sequence[ t ] ]
            V_value =  V[t]

            # recursively calculate the estimator value. estimator = sum_of_all_WDR_step_values + next_step_WDR_value
            wdr_estimator =  wdr_estimator + weight * discount * r - discount * ( weight * Q_value - previous_weight * V_value )
            wis_estimator =  wis_estimator + weight * discount * r

    return wdr_estimator, wis_estimator

       
#################################################################################################################################################################
#################################################################################################################################################################
# Model Training function
#################################################################################################################################################################
#################################################################################################################################################################
# Train model with Double Dueling Q-Learning with Prioritised Experience Replay on CPU or GPU if cuda if available
def train_model_double(model,
                       target_model,
                       data_dict,
                       data_dict_MDP,
                       config,
                       optimizer,
                       scheduler,
                       experiment,
                       use_gpu=False):

    # Key = starting state ID, value = next state ID
    transition_dict_train = dict(zip(data_dict['train']['state_id'], data_dict['train']['next_state_id']))

    ############################################
    ### Initialize learning settings
    
    # LEARNING hyperparameters 
    reg_lambda = config['reg_lambda']
    REWARD_THRESHOLD = config['REWARD_THRESHOLD']

    # don't learn to much to fast
    scheduler_step_size = config['sched_step_size']
    

    ###########################################
    ### Prioritised Experience Replay

    # Initialize Prioritized Experience Replay with *fixed* parameters 
    per = PrioritizedExperienceReplay(beta_start=config['PER_beta_start'], 
                                      alpha=config['PER_alpha'], 
                                      sample_alpha = config['PER_sample_alpha'], 
                                      epsilon=config['PER_epsilon'], 
                                      use_gpu=use_gpu)
    # create weights
    per.create_weights(rewards=data_dict['train']['reward'],start_prob=config['PER_start_prob'])

    ############################################
    ### Physician behavior policy and dataset MDP
    
    # Action probabilities of physician's action used for intermediate evaluateion
    train_pi_behavior = data_dict_MDP['train_pi_behavior']
    val_pi_behavior = data_dict_MDP['val_pi_behavior']
    test_pi_behavior = data_dict_MDP['test_pi_behavior']
       
    # dataset MDP Q function (FQI-SARSA)
    train_MDP_Q = data_dict_MDP['train_MDP_Q']
    val_MDP_Q = data_dict_MDP['val_MDP_Q']
    test_MDP_Q = data_dict_MDP['test_MDP_Q']
    
    ############################################
    ### Initialize TRACKING
    tracking_step_eval = int(config['tracking_step_eval'])
    tracking_step_interim_model = int(config['tracking_step_interim_model'])
    tracking_console_print = int(config['tracking_console_print'])
    pt = PerformanceTracker()
    since = time.time()
    running_loss = 0.0

    ############################################
    # Set models training mode
    target_model.train(False)
    model.train(True)
    
    ### CPU or GPU
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    
    ### beginning of number of training loop (EPOCHS)
    for epoch in range(config['num_epochs']):
        model.train(True)

        batch_ids = per.sample(config['batch_size'])
        next_state_batch_ids = [transition_dict_train[x.item()] for x in batch_ids]

        states, actions, rewards, next_states = load_data(data_dict['train'], batch_ids, next_state_batch_ids, use_gpu)        

        # define how many rows this batch of data has
        batch_size = states.shape[0]
        batch_idx = torch.arange(batch_size, device=device)

        ################################################################################################################
        ### BEGINNING OF ACTUAL REINFORCEMENT LEARNING ALGORITHM

        # firstly get the chosen actions at the next timestep using main Q network
        action_prime = torch.max(model(next_states), 1)[1].data

        # Q values for the next timestep from target network, as part of the Double DQN update
        target_output = target_model(next_states)

        # Get the value of the next states at the action_prime from the target model
        # ~== predicted Q valuesb (batch size X 1 vector)
        next_state_values = target_output[batch_idx, action_prime]

        # empirical hack to make the Q values never exceed the threshold - helps learning
        next_state_values = torch.clamp(next_state_values, min=-REWARD_THRESHOLD, max=REWARD_THRESHOLD)

        # the expected_state_values = "next_state_value"  * "config_gamma" * "reward mask = 0" + reward = 0 )
        reward_mask = rewards == 0

        # the target Q value
        with torch.no_grad():
            expected_state_values = (next_state_values * config['gamma']) * reward_mask.float() + rewards.float()

        # calculate new importance sampling weights
        imp_sampling_weights = per.calculate_importance_sampling_weights(batch_ids)

         # Zero the parameter gradients before outputting the model. Documentation: https://pytorch.org/docs/stable/optim.html
        optimizer.zero_grad()

        # Compute the Q-values for the model at the current state values at the actions actually taken
        outputs = model(states)
        
        # choose best actions based on highest Q values
        best_actions = torch.max(outputs, dim=1)[1].data

        # Action probability based on a softmax of the Qvalue distribution for each state-action pair  (state in batch_size * Unique amount of actions = 20)
        output_prob = torch.nn.functional.softmax(outputs.data, dim = 1)

        # get the probability of the DQN choosen best action
        best_action_probabilities = output_prob[torch.LongTensor(np.arange(batch_size).tolist()),best_actions.data]

        # Get Q(s, a)
        current_state_values = outputs[batch_idx, actions.data]

        # Get Q(s, a*)
        best_policy_values = outputs[batch_idx, best_actions.data]

        ################################################################################################################
        ### LOSS & ERROR
        def custom_loss(current_state_values, expected_state_values):                         # input of loss function: input, target
            reg_vector = torch.clamp(torch.abs(current_state_values)-REWARD_THRESHOLD, min=0) # vector of Q-values of "input part of loss function"
            reg_term = reg_vector.sum()                                                       # sum(vector)
            abs_error = torch.abs(expected_state_values - current_state_values)               # (A-B)=vector
            td_error = abs_error.double()**2                                                  # vector^2 (each value gets multiplied by itself)
            per_error = td_error * imp_sampling_weights.double()                              # vector*imp_vector (torch * np.array)
            loss = torch.mean(per_error) + reg_lambda*reg_term.double()                       # mean(vector) + float*float
            return(reg_term,td_error.mean(),per_error.mean(),loss)
        
        # Execute loss functions
        reg_term, td_error, per_error, loss = custom_loss(current_state_values,expected_state_values)
        
        # Backpropagate loss
        loss.backward()
        optimizer.step()
        
        # calculate the 'absolute error'
        error = torch.abs(expected_state_values - current_state_values)
        
        ################################################################################################################
        ### UPDATE the PER weights and update the target model
        
        # update PER weights
        per.update_weights(batch_ids, error)

        # Updating Target Model
        tau = config['tau']
        for main_param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data = (main_param * tau) + (target_param * (1-tau))

        ################################################################################################################
        ### TRACKING METRICS
        num_samples = batch_size

        # keeping track of performance
        cpu = torch.device('cpu')
        epoch_loss          = loss.to(cpu).item()
        epoch_avg_best_Q    = best_policy_values.to(cpu).data.sum().numpy()/num_samples
        epoch_avg_current_Q = current_state_values.to(cpu).data.sum().numpy()/num_samples
        epoch_action_prob   = best_action_probabilities.to(cpu).data.sum().numpy()/num_samples
        epoch_error         = error.to(cpu).data.sum().numpy()/num_samples
        epoch_td_error      = td_error.to(cpu).item()
        epoch_per_error     = per_error.to(cpu).item()
        epoch_reg_term      = reg_term.to(cpu).item()

        # update performance dictionary with new results for this epoch on the training dataset
        pt.append(epoch_loss,epoch_avg_best_Q,epoch_avg_current_Q, epoch_action_prob, epoch_error, epoch_td_error, epoch_per_error, epoch_reg_term)
        
        ############################################
        ### UPDATE SCHEDULER

        # keep a running_value of the loss
        running_loss += epoch_error
        running_loss_normalised = running_loss / (epoch+1)
              
        # Adjust learning rate if the use of a scheduler is set to TRUE in the configuration EVERY 'scheduler_step_size' STEPS
        if (epoch+1) % scheduler_step_size == 0 and epoch > 0:
            if scheduler:
                if config['use_scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(running_loss_normalised)
                elif config['use_scheduler'] == 'StepLR':      
                    scheduler.step()

                
        ############################################
        ### CONSOLE TRACKING (stepsizes are set in the configuration dictionary)  
        if (epoch+1) % tracking_console_print == 0 and epoch > 0:
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch+1, config['num_epochs']),
                   '\n batch loss: ', np.around(epoch_loss,3) ,
                   '\n running loss: ', np.around(running_loss_normalised,3), 
                   '\n batch error: ', np.around(epoch_error,3) ,
                   '\n batch avg Phy Q value: ', np.around(epoch_avg_current_Q,3) ,
                   '\n batch avg DQN Q value: ', np.around(epoch_avg_best_Q,3) ,
                   '\n batch mean DQN best action probability: ', np.around(epoch_action_prob*100,3), '%')

        ############################################
        ### WDR/WIS performance analysis on train and validation dataset
        if (epoch+1) % tracking_step_eval == 0 and epoch > 0:
            print('-' * 10)  
            print('intermediate evaluation:')
            track_performance(data_dict, (epoch+1), config['gamma'], model, pt, train_pi_behavior, train_MDP_Q, 'train', use_gpu)
            track_performance(data_dict, (epoch+1), config['gamma'], model, pt, val_pi_behavior, val_MDP_Q, 'val', use_gpu)
            track_performance(data_dict, (epoch+1), config['gamma'], model, pt, test_pi_behavior, test_MDP_Q, 'test', use_gpu)
            print('-' * 10)
            
            ############################################
            ### INTERIM results

            ### create DataFrames from dictionaries
            performance_df = pd.DataFrame.from_dict(pt.performance_dict)
            tracking_performance_df = pd.DataFrame.from_dict(pt.tracking_performance_dict)

            ### write to csv
            tracking_performance_df.to_csv(os.path.join(str(experiment['exp_dir']), 'performance/' + str(experiment['exp_name']) + "_" + str(experiment['exp_grid_run']) + '_tracking_performance.csv'), index=False)
            performance_df.to_csv(os.path.join(str(experiment['exp_dir']), 'performance/' + str(experiment['exp_name']) + "_" + str(experiment['exp_grid_run']) + '_performance.csv'), index=False)
            
        ############################################
        ### SAVE INTERIM MODEL
        if (epoch+1) % tracking_step_interim_model == 0 and epoch > 0:
            print('Saving interim Model ' + str(epoch+1))            
            interim_path = os.path.join(str(experiment['exp_dir']), 'models/' + str(experiment['exp_name']) + "_" + str(experiment['exp_grid_run']) + '/' + str(experiment['exp_name']) + "_" + str(experiment['exp_grid_run']) + '_interim_' + str(epoch+1) + '_iteration_model.chk')  
            torch.save(model.state_dict(), interim_path)
            print('-' * 10)      

    ############################################
    ### End of total training
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    ### best_model_wts is only available after the phase=val. it's normal for this
    return pt.tracking_performance_dict, pt.performance_dict, model
