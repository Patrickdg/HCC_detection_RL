#LIBRARIES
import os
import numpy as np
import pandas as pd

"""
Module to generate synthetic data via population parameters. 

*Static risk factors are determined for period 0 across N subjects
*Variable risk factors are calculated on run-time by subject after screening of decision maker, per reference paper. 

Procedure: 
Step 1) Sample N total at-risk subjects and sample HCC vs. no HCC population based on HCC rate 82/885.
        *HCC rate selected from reference paper [Lee et al. (2014)] 
Step 2) For each subject, sample each risk factors based on HCC/no HCC params: 
        Numerical variables: Normal distribution sampling
        Boolean/rate variables: Binomial distribution sampling
"""

PARAMS = {
    'hcc': {
        'class_rate': 
            {'val': 82/967, 'type': 'rate'},
        'age': 
            {'val': 53, 'std': 7, 'type': 'numerical'},
        'black': 
            {'val': 0.24, 'std': None, 'type': 'rate'},
        'platelets': 
            {'val': 126, 'std': 51,'type': 'numerical'},
        'ever_smoked': 
            {'val': 0.41, 'std': None, 'type': 'rate'},
        'alkaline_phosphatase': 
            {'val': 117, 'std': 59,'type': 'numerical'},
        'esophageal_varices': 
            {'val': 0.04, 'std': None, 'type': 'rate'},
        'afp_std': 
            {'val': 51, 'std': 86, 'type': 'numerical'},
        'afp_rise': 
            {'val': 5, 'std': 11, 'type': 'numerical'},
    },
    'no_hcc': {
        'class_rate': 
            {'val': 885/967, 'type': 'rate'},
        'age': 
            {'val': 50, 'std': 7, 'type': 'numerical'},
        'black': 
            {'val': 0.18, 'std': None, 'type': 'rate'},
        'platelets': 
            {'val': 169, 'std': 65,'type': 'numerical'},
        'ever_smoked': 
            {'val': 0.24, 'std': None, 'type': 'rate'},
        'alkaline_phosphatase': 
            {'val': 97, 'std': 43,'type': 'numerical'},
        'esophageal_varices': 
            {'val': 0.34, 'std': None, 'type': 'rate'},
        'afp_std': 
            {'val': 9, 'std': 19, 'type': 'numerical'},
        'afp_rise': 
            {'val': 0.11, 'std': 2.1, 'type': 'numerical'}
    }
}

def main(num_subjects, output_dir):
    class_rates = [
        PARAMS['no_hcc']['class_rate']['val'], 
        PARAMS['hcc']['class_rate']['val']
    ]
    risk_factors = list(PARAMS['hcc'].keys())[1:]

    #Step 1) Get subjects
    N = np.random.choice([0, 1], size=num_subjects, p=class_rates)
    N_df = pd.DataFrame(columns=['id', 'class'] + risk_factors)
    #Step 2) Sample population params per subject
    for i, c in enumerate(N):
        cl = 'hcc' if c == 1 else 'no_hcc'
        i_dict = {'id': i, 'class': c}
        for risk_factor in risk_factors:
            val = PARAMS[cl][risk_factor]['val']
            std = PARAMS[cl][risk_factor]['std']
            type_ = PARAMS[cl][risk_factor]['type']

            if type_ == 'rate': 
                f = np.random.binomial(1, val)
            elif type_ == 'numerical':
                f = np.random.normal(val, std, 1)[0]
            i_dict[risk_factor] = f
        N_df = pd.concat([N_df, pd.DataFrame(i_dict, index=[0])])
    
    save_path = os.path.join(output_dir, 'subjects.csv')
    N_df.to_csv(save_path, index=False)

    #Initialize results.csv to store simulation run metrics
    if not os.path.exists('results/results.csv'): 
        results_df = pd.DataFrame(columns=[
                                    'T', 
                                    'D',
                                    'num_iters', 
                                    'n', 
                                    'k', 
                                    'policy',
                                    'policy_params',
                                    'replace',
                                    'd',
                                    'E', 
                                    'L', 
                                    'X', 
                                    'detection_rate', 
                                    'early_detection_rate', 
                                    'resource_efficiency'])
    return N_df

#RUN
if __name__ == "__main__":
    main(10000, 'data')
