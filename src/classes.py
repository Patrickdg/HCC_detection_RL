#LIBRARIES
import tensorflow as tf
#Set gpu device for multi-gpu setups (can be customized)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

from src.generate_data import PARAMS
import pandas as pd 
import numpy as np
import math

import random
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#CLASSES
class Subject():
    def __init__(self, subject_params):
        #Subject info
        self.id = subject_params['id']
        self.cl = subject_params['class']
        #Static risk factors
        self.age = subject_params['age']
        self.black = subject_params['black']
        self.platelets = subject_params['platelets']
        self.ever_smoked = subject_params['ever_smoked']
        self.alkaline_phosphatase = subject_params['alkaline_phosphatase']
        self.esophageal_varices = subject_params['esophageal_varices']
        #initialize AFP factors -> 0 to restrict DM knowledge to 'B' vector when subject not yet screened
        self.afp_std = 0 
        self.afp_std_var = 0 
        self.afp_rise = 0
        self.afp_rise_var = 0

        #Variale risk factors
        self.cancer_stage = None #determined when screened
        self.afp_history = pd.DataFrame(columns=['afp_std', 'afp_rise']) #index increments each screening update
    
    def sample_cancer_stage(self):
        """Samples tumor size to determine cancer stage."""
        s = np.random.normal(3.50, 2.00)
        if 1 <= s <= 5: 
            stage = 'early'
        elif 5 < s: 
            stage = 'late'
        else: 
            stage = 'cancer-free'
        return stage

    def update_AFP(self): 
        """AFP Reading Module:: Update patient AFP readings
        @init: True when instantiating current Subject() --> do not update AFP readings, only _var readings"""
        #Sample AFP std. and rise from distribution parameters
        cl = 'hcc' if self.cl == 1 else 'no_hcc'
        for f in ['afp_std', 'afp_rise']:
            val = PARAMS[cl][f]['val']
            std = PARAMS[cl][f]['std']
            type_ = PARAMS[cl][f]['type']

            f_val = np.random.normal(val, std, 1)[0]
            setattr(self, f, f_val)

        #Update patient history
        update_ind = len(self.afp_history)
        update_df = pd.DataFrame({
                            'afp_std': self.afp_std, 
                            'afp_rise': self.afp_rise}, 
                            index=[update_ind])
        self.afp_history = pd.concat([self.afp_history, update_df])

        for f in ['afp_std', 'afp_rise']:
            f_var = self.afp_history[f].var()
            if math.isnan(f_var): 
                f_var = 0
            setattr(self, f+'_var', f_var)

class Panel():
    def __init__(self): 
        self.subjects = {} #dict of {id: Subject(i)} 
        self.n = 0 #total num. subjects in current panel
        self.C = 0 #num. cancer subjects in current panel
        self.NC = 0 #num. non-cancer subjects in current panel

    def add_subject(self, i):
        """Add Subject() to panel"""
        self.subjects[i.id] = i
        self.n += 1 

        self.C += i.cl #cl == 1 if cancer subject, else 0
        self.NC = self.n - self.C
    
    def remove_subject(self, i):
        """Remove Subject() from panel"""
        del self.subjects[i.id]
        self.n -= 1

        self.C -= i.cl
        self.NC = self.n - self.C
    
    def update_subject(self, i): 
        """Update patient 'i' after screening modifications"""
        self.remove_subject(i)
        self.add_subject(i)
    
    def replace_subject(self, i, pool, replace):
        """New Patient Module:: Replace subject 'i' with new patient from pool."""
        pool_remaining = pool.copy()
        #remove old Subject()
        self.remove_subject(i)
        #add new Subject() to Panel(), then remove from pool if replace=False
        i_new = pool_remaining.sample(n=1, replace=replace)
        i_new = i_new.to_dict('records')[0]
        i_new = Subject(i_new)
        self.add_subject(i_new)
        if replace == False:  
            pool_remaining.query(f'id != {i_new.id}', inplace=True)
        return pool_remaining

class Agent():
    def __init__(self, policy, policy_params, k, n, T, log_model): 
        self.policy = policy
        self.policy_params = policy_params
        self.k = k
        self.n = n 
        self.T = T
        self.log_model = log_model
        self.log_coefs = self.log_model.coef_[0]

        self.panel = None

        #Metric storage
        self.E = 0
        self.L = 0
        self.X = 0
        self.C_count = 0 #cumulative count of cancer patients in panel during each iteration 't'

    def init_panel(self, pool, replace):
        """Decision maker randomly selects an initial panel of size 'n'
        @pool: dataframe containing all remaining subjects from subjects.csv
        """
        #Random sampling
        pool_remaining = pool.copy()
        samples = pool_remaining.sample(n = self.n, replace=replace)
        samples = samples.to_dict('records')

        #add Subject()s to Panel(), then remove from pool if replace=False
        self.panel = Panel()
        for i in samples:
            subject = Subject(i)
            self.panel.add_subject(subject)
            if replace == False: 
                pool_remaining.query(f'id != {subject.id}', inplace=True)
        self.C_count += self.panel.C
        return pool_remaining
    
    def get_k_subjects(self): 
        """Determine 'k' subjects to screen based on agent's current policy function"""
        k_subjects = self.policy(self.panel, self.policy_params, self.k, self.log_coefs)
        #Increment X for each screening spent on cancer subjects ('cl' attribute)
        for id, i in k_subjects.items():
            self.X += i.cl
        return k_subjects
    
    def screen(self, k_subjects): 
        """
        3a) if cancer-free, output cancer-free state. 
        3b) if cancer, determine current cancer stage (early/late/cancer-free) based on random sampling from tumor size (s) distribution: 
            1 <= s <= 5 --> early-stage
            5 <  s      --> late-stage
            Otherwise   --> cancer-free
        """
        for id, i in k_subjects.items():
            if i.cl == 0: 
                i.cancer_stage = 'cancer-free'
            else: #Tumor size sampling 
                i.cancer_stage = i.sample_cancer_stage()
        return k_subjects

    def update_screened(self, k_subjects, pool, replace):
        """
        3c) Updates: 
        - if cancer-free, AFP Reading Module:: assign new AFP reading for this patient, add to DM's knowledge of panel
        - if early- or late-stage, increment E (or L) += 1, then replace patient in New Patient Module:: draw new random patient. 

        - update relevent metric trackers
        """
        pool_remaining = pool.copy()
        for id, i in k_subjects.items():
            if i.cancer_stage == 'cancer-free': 
                i.update_AFP()
                self.panel.update_subject(i)
            else: 
                if i.cancer_stage == 'early': 
                    self.E += 1
                elif i.cancer_stage == 'late': 
                    self.L += 1
                pool_remaining = self.panel.replace_subject(i, pool_remaining, replace)
        self.C_count += self.panel.C
        return k_subjects, pool_remaining

    def patient_exit_module(self, pool, d):
        """Patient Exit Module:: randomly sample 'D' subjects in panel to exit.
        Increment L metric with any selected subjects with cancer."""
        pool_remaining = pool.copy()

        num_d = math.floor(d*self.n)
        D = np.random.choice(
            list(self.panel.subjects.keys()), 
            size=num_d, 
            replace=False)
        
        for d in D:
            d = self.panel.subjects[d]
            self.L += d.cl #cancer subjects in early-stage will eventually develop late-stage outside of simulation
            pool_remaining = self.panel.replace_subject(d, pool_remaining, replace=False)
        return pool_remaining

    def report_metrics(self, n_iter):
        metrics = {
            'E': self.E,  
            'L': self.L,  
            'X': self.X,
            'detection_rate': (self.E+self.L)/self.C_count,
            'early_detection_rate': self.E/(self.E + self.L),
            'resource_efficiency': self.X/(self.k*self.n*self.T)  
        }
        metrics = pd.DataFrame(metrics, index=[n_iter])
        return metrics

class Agent_DL(Agent): 
    def __init__(self, policy, policy_params, k, n, T, log_model): 
        super().__init__(policy, policy_params, k, n, T, log_model)

        self.policy.build_model()

    def get_k_subjects(self): 
        """Determine 'k' subjects to screen based on agent's current policy function"""
        k_subjects = self.policy.get_k(self.panel, self.policy_params, self.k, self.log_coefs)
        #Training on memory, experience replay: 
        if len(self.policy.memory) > self.policy.batch: 
            self.policy.replay(self.policy.batch)
        #Increment X for each screening spent on cancer subjects ('cl' attribute)
        for id, i in k_subjects.items():
            self.X += i.cl
        return k_subjects

class PolicyDL():
    def __init__(self, policy_params):
        self.model = None
        self.batch = policy_params['batch']
        self.memory = deque(maxlen=policy_params['memory_size'])

        self.state_size = 9 #6 static risk factors, 2 variable AFP factors, 1 remaining 'k' counter
        self.action_size = 2 #{screen, don't screen}
        #Parameters
        self.gamma = 0.95 
        self.epsilon = 0.05
        self.learning_rate = 0.001

    def remember(self, state, action, reward, terminal): 
        self.memory.append((state, action, reward, terminal))

    def take_action(self, state): 
        #Exploration
        if np.random.random() <= self.epsilon: 
            return random.randrange(self.action_size)
        action = self.model.predict(state)
        action = 1 if action > 0 else 0
        return action

    def replay(self, batch_size): 
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, terminal in batch: 
            target = reward
            curr_pred = self.model.predict(state)
            target_ = curr_pred + self.gamma*(target - curr_pred)

            self.model.fit(state, target_, epochs=1, verbose=0)

    def build_model(self): 
        model = Sequential()
        model.add(Dense(64, input_dim = self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(
                loss='mse', 
                optimizer=Adam(learning_rate=self.learning_rate))
        self.model = model

    def get_k(self, panel, policy_params, k, coefs):
        num_k = math.floor(k*panel.n)
        rem_k = math.floor(k*panel.n) #counter

        #Randomly shuffle keys since training is dependent on order
        subject_ids = list(panel.subjects.keys())
        random.shuffle(subject_ids)
        
        #Trackers for training loop
        subject_states = []
        subject_classes = []
        k_inds = []
        k_classes = []
        not_k_classes = []
        num_not_screened = 0

        TP_rate = 0
        TN_rate = 0
        #Training loop
        for id in subject_ids:
            i = panel.subjects[id] 
            state = [
                i.age,
                i.black,
                i.platelets,
                i.ever_smoked,
                i.alkaline_phosphatase,
                i.esophageal_varices,
                i.afp_std, 
                i.afp_rise, 
                rem_k]
            state = [float(s_i) for s_i in state]  
            state = np.reshape(np.array(state), [1, len(state)])

            subject_states.append(state)
            subject_classes.append(i.cl)

            action = self.take_action(state)
            if action == 1: #screen
                k_inds.append(id)
                k_classes.append(i.cl)
                rem_k -= 1
                terminal = 1 if rem_k == 0 else 0
                #Reward calculation
                allocated_k = num_k - rem_k 
                new_TP_rate = sum(k_classes) / allocated_k 
                reward_r = new_TP_rate - TP_rate
                TP_rate = new_TP_rate
                self.remember(state, action, reward_r*100, terminal)
            
                if terminal: #no screenings left to be allocated
                    break
            elif action == 0: #don't screen
                num_not_screened += 1 
                not_k_classes.append(i.cl)
                #Reward calculation
                true_negatives = len(not_k_classes) - sum(not_k_classes) 
                new_TN_rate = true_negatives/num_not_screened
                reward_r = new_TN_rate - TN_rate
                TN_rate = new_TN_rate
                self.remember(state, action, reward_r*100, 0)

        return {k: panel.subjects[k] for k in k_inds}