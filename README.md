# HCC Detection, Reinforcement Learning Implementation 
Replication study of 'Applying reinforcement learning techniques to detect hepatocellular carcinoma under limited screening capacity' with extension of MDP and DQN learning application. 

**Final report can be found [here](https://github.com/Patrickdg/HCC_detection_RL/blob/main/Final%20Report.pdf).**

## Reference Paper: 
Lee E, Lavieri MS, Volk ML, and Xu Y. Applying reinforcement learning techniques to detect 
hepatocellular carcinoma under limited screening capacity. Health Care Manag Sci. 2015 Sep,
18(3):363-75. doi: 10.1007/s10729-014-9304-0. Epub 2014 Oct 12. PMID: 25308168.

## Usage  
Below is an example function call for the main python module. The DQN method currently uses Tensorflow to detect GPU #0 for initialization (adjust this as needed in the src.classes module, line 5!)

Example 1: Single model simulation: 
```python
df = main(
        'data/subjects.csv', #points to the data file available
        T=3650, #planning horizon (days)
        D=90,   #decision epoch spacing (days)
        num_iters=20, #iterations to run per model simulation
        n=500, #panel size
        k=0.4, #constraint level
        policy='deep', #RL method name
        policy_params={}, #policy params if necessary (select policies only)
        replace=True, #sample with replacement to update panel
        d=0.05, #dropout rate of participants in study pool
        store_results=True #flag to store metrics in results/results.csv log
    )
```
Example 2: Run simulation for all models, model parameters, and constraint level permutations (this takes a long time!): 
```python
run_full_sim() #runs 
```
