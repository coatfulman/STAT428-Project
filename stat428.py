import pandas as pd
import numpy as np
import pymc3 as pm
import os
import theano.tensor as tt
import sys

# Example to load data as numpy array
user = sys.argv[1]
data = pd.read_csv('stat428/' + user + '.csv').values
data = data[data[:,-1].argsort()]

for idx, _ in enumerate(data):
    if idx != len(data) - 1:
        data[idx][-1] = data[idx+1][-1] - data[idx][-1]

data = data[:-1]

# First scale scores
data[:, 2] = np.log(np.abs(data[:, 2]).astype(float) + 1)
data[:, 3] = np.log(data[:, 3].astype(float) + 1)

if np.max(data[:, 2]) - np.min(data[:, 2]) != 0:
    data[:, 2] = (data[:, 2] - np.min(data[:, 2])) / (np.max(data[:, 2]) - np.min(data[:, 2]))

if np.max(data[:, 3]) - np.min(data[:, 3]) != 0:
    data[:, 3] = (data[:, 3] - np.min(data[:, 3])) / (np.max(data[:, 3]) - np.min(data[:, 3]))

def community_mapping(communities):
    str_int, int_str = {}, {}
    cnt = 0

    for community in communities:
        if community not in str_int:
            str_int[community] = cnt
            int_str[cnt] = community
            cnt += 1

    return (str_int, int_str)

# Make mapping between community/action string to integer
action_str_int = {'question':0, 'answer':1, 'others':2}
action_int_str = {0:'question', 1:'answer', 2:'others'}
(community_str_int, community_int_str) = community_mapping(data[:, 0])

# Make community and action as matrix (n, m), n is number of rows in data, m is 93/3
community_matrix, action_matrix = np.zeros((len(data), 93)), np.zeros((len(data), 3))
community_idx = [community_str_int[community] for community in data[:, 0]]
community_matrix[np.arange(len(data)), community_idx] = 1

action_idx = [action_str_int[community] for community in data[:, 1]]
action_matrix[np.arange(len(data)), action_idx] = 1

# 93 communities in total.
# 3 actions in total.

with pm.Model() as model:

    # Community's prior.
    community_prior = pm.HalfCauchy('community_diric', beta=1, shape=93)
    # Community distribution for this user
    community_weight = pm.Dirichlet('community_weight', a=community_prior, shape=93)

    # Action's prior.
    action_prior = pm.HalfCauchy('action_diric', beta=1, shape=3)
    # Action distribution for this user
    action_weight = pm.Dirichlet('action_weight', a=action_prior, shape=3)

    # Score Prior
    score_sd = pm.Exponential('score_sd', lam=1)
    # Score for this action
    score_numeral = pm.Lognormal('score_numeral', mu=data[:,2].astype(float), sd=score_sd, shape=len(data))

    # Numerize community and action
    community_numeral = tt.dot(community_matrix, community_weight)
    action_numeral = tt.dot(action_matrix, action_weight)

    # Draw coefficient of community, action, score and intercept
    community_coef = pm.Normal('community_coef', mu=0, sd=1)
    action_coef = pm.Normal('action_coef', mu=0, sd=1)
    score_coef = pm.Normal('score_coef', mu=0, sd=1)
    intercept = pm.Normal('intercept', mu=0, sd=1)

    # Let's do sigmoid
    alpha_sigmoid = pm.Exponential('alpha_sigmoid', lam=1)
    x_sigmoid = community_coef*community_numeral + action_coef*action_numeral + score_coef*score_numeral + intercept
    p_sigmoid = 1 / (1 + pm.math.exp(-alpha_sigmoid * x_sigmoid))

    # eps is model error
    eps = pm.HalfCauchy('eps', 1)
    output = pm.Normal('output', mu=p_sigmoid, sd=eps, observed=data[:,-1])

    trace = pm.sample(1000)

    mean_error = np.mean(trace['eps'][-100:])


mean_community_weight = np.mean(trace['community_weight'][-100:], axis=0)
mean_action_weight = np.mean(trace['action_weight'][-100:], axis=0)
mean_community_coef = np.mean(trace['community_coef'][-100:], axis=0)
mean_action_coef = np.mean(trace['action_coef'][-100:], axis=0)
mean_score_coef = np.mean(trace['score_coef'][-100:], axis=0)

written_data = {'mean_error': [mean_error], 'mean_community_weight': [mean_community_weight], 'mean_action_weight':[mean_action_weight],
     'mean_community_coef':[mean_community_coef], 'mean_action_coef':[mean_action_coef], 'mean_score_coef':[mean_score_coef]}

if not os.path.exists("analysis"):
    os.mkdir("analysis")

df = pd.DataFrame(data=written_data)
df.to_csv('analysis/output' + user + '.csv')
