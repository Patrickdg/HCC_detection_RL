#LIBRARIES
import numpy as np
import random
import math

"""METHODOLOGY of policy functions:
Inputs: Panel() object of Subject()s of size 'n' 
Outputs: Subset of panel to screen with proportion 'k' based on myopic selection

Methods included: 
- random selection 
- myopic (pure-greedy)
- e-greedy
- Interval estimation
- Boltzmann exploration
"""

#FUNCTIONS
def policy_random(panel, policy_params, k, coefs):
    """Random sampling of subjects for screening."""
    i_inds = random.sample(list(panel.subjects.keys()), math.floor(k*len(panel.subjects)))
    return {i: panel.subjects[i] for i in i_inds}

def policy_myopic(panel, policy_params, k, coefs):
    """Myopic selection: select top k subjects based on ascending x_it values"""
    num_k = math.floor(k*panel.n)

    subject_ids = []
    subject_states = []
    for id, i in panel.subjects.items():
        state = [
            i.age,
            i.black,
            i.platelets,
            i.ever_smoked,
            i.alkaline_phosphatase,
            i.esophageal_varices,
            i.afp_std, 
            i.afp_rise
        ]
        x = np.dot(state, np.transpose(coefs))
        subject_ids.append(id)
        subject_states.append(x)
    k_inds = np.argsort(subject_states)[::-1][:num_k]
    k_inds = np.array(subject_ids)[k_inds]
    return {k: panel.subjects[k] for k in k_inds}

def policy_egreedy(panel, policy_params, k, coefs):
    """e-greedy selection: select top (1-e)*k subjects based on ascending x_it values, 
        randomly select remaining e*k subjects"""
    e = policy_params['e']
    num_k = math.floor(k*panel.n)

    subject_ids = []
    subject_states = []

    num_greedy = math.floor((1-e)*num_k)
    num_explore = num_k - num_greedy
    #GREEDY
    for id, i in panel.subjects.items():
        state = [
            i.age,
            i.black,
            i.platelets,
            i.ever_smoked,
            i.alkaline_phosphatase,
            i.esophageal_varices,
            i.afp_std, 
            i.afp_rise
        ]
        x = np.dot(state, np.transpose(coefs))
        subject_ids.append(id)
        subject_states.append(x)
    k_inds = np.argsort(subject_states)[::-1][:num_greedy]
    k_inds = list(np.array(subject_ids)[k_inds])
    
    #EXPLORATION
    while num_explore > 0:
        #Randomly select subject id, then keep if not already in selected greedy subjects in k_inds
        sample_id = np.random.choice(subject_ids, size=1)
        if sample_id not in k_inds:
            k_inds.append(sample_id)
            num_explore -= 1
    k_inds = [int(k) for k in k_inds]
    return {int(k): panel.subjects[k] for k in k_inds}

def policy_interval_estimation(panel, policy_params, k, coefs):
    """Interval estimation: select  top k subjects based on distorted risk score."""
    z = policy_params['z']
    num_k = math.floor(k*panel.n)

    subject_ids = []
    subject_states = []
    #Split coefficients for risk factors
    c1 = coefs[:-2]
    c2 = coefs[-2]
    c3 = coefs[-1]
    for id, i in panel.subjects.items():
        s1 = [
            i.age,
            i.black,
            i.platelets,
            i.ever_smoked,
            i.alkaline_phosphatase,
            i.esophageal_varices
        ]

        s1 = np.dot(c1, np.transpose(s1))
        s2 = c2 * (i.afp_std+z*math.sqrt(i.afp_std_var))
        s3 = c3 * (i.afp_rise+z*math.sqrt(i.afp_rise_var))
        x = s1 + s2 + s3

        subject_ids.append(id)
        subject_states.append(x)

    k_inds = np.argsort(subject_states)[::-1][:num_k]
    k_inds = list(np.array(subject_ids)[k_inds])

    return {k: panel.subjects[k] for k in k_inds}

def policy_boltzmann(panel, policy_params, k, coefs):
    """Boltzmann exploration selection: select k patients with probabilities weighted by risk score"""
    tau = policy_params['tau']
    num_k = math.floor(k*panel.n)

    subject_ids = []
    subject_probs = []
    for id, i in panel.subjects.items():
        state = [
            i.age,
            i.black,
            i.platelets,
            i.ever_smoked,
            i.alkaline_phosphatase,
            i.esophageal_varices,
            i.afp_std, 
            i.afp_rise
        ]
        e_x = math.exp(np.dot(state, np.transpose(coefs))/tau)

        subject_ids.append(id)
        subject_probs.append(e_x)
    
    subject_probs = [p/sum(subject_probs) for p in subject_probs]
    k_inds = np.random.choice(subject_ids, size=num_k, replace=False, p=subject_probs)
    return {k: panel.subjects[k] for k in k_inds}