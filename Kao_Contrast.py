import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

df = pd.read_csv('https://raw.githubusercontent.com/justinek/metaphors/master/Data/FeaturePriorExp/featurePriors-set.csv', index_col=0)

animals = {k : j for j, k in enumerate(df['animal'].unique())}

priors = []

for j in animals:
    x = ([df['normalizedProb'].loc[df['animal'].isin([j]) & df['type'].isin(['animal'])]], [df['normalizedProb'].loc[df['animal'].isin([j]) & df['type'].isin(['person'])]])
    priors.append(x)

np.reshape(priors, [32, 2, 8])

animal_feat = np.array([[1, 1, 1],
                        [1, 1, 0],
                        [1, 0, 1],
                        [1, 0, 0],
                        [0, 1, 1],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]]).reshape(-1, 3)


def Sn(utterance, weights, animal_feat, lam=3):
    delta = weights @ animal_feat.T
    U = delta * priors

    # softmax decision rule
    S = torch.softmax(torch.FloatTensor(lam*U), dim=0).numpy()
    return S[utterance]


def Ln(utterance, goal_weights, p_animal_human, animal_feat, lam=3):
    u = animals[utterance]
    s = Sn(u, goal_weights, animal_feat, lam)
    L1 = priors[u] * s
    L1 = np.squeeze(p_animal_human.reshape(-1,1,1) * L1,1)
    L1 = L1/L1.sum()
    return L1

ASD_Contrast = pd.DataFrame(data=None, columns=['Animal', 'NT', 'ASD', 'P(a)', 'P(h)', 'Lambda', 'Value'])

nt_pig = Ln('pig',  # the utterance used
    np.array([.6, .2, .2]),  # Priors on goals in order of f1,f2,f3 as reported in Kao 2014
    np.array([.01, .99]),  # Priors on category, in order ANIMAL, HUMAN
    animal_feat,  # The set of worlds everything uses
    3  # the value for lambda
    )

for i in np.arange(0, 500, 0.1):
    p = Ln('pig',  # the utterance used
    np.array([.6, .2, .2]),  # Priors on goals in order of f1,f2,f3 as reported in Kao 2014
    np.array([.01, .99]),  # Priors on category, in order ANIMAL, HUMAN
    animal_feat,  # The set of worlds everything uses
    i  # the value for lambda
    )
    x = nt_pig[1][3]/p[1][3]
    if x > 2.22 and x < 2.29:
        ASD_Contrast = ASD_Contrast.append({'Animal': 'pig', 'NT': nt_pig[1][3], 'ASD': p[1][3], 'P(a)': 0.01, 'P(h)': 0.99, 'Lambda': i, 'Value': x}, ignore_index=True)

nt_fox = Ln('fox',  # the utterance used
    np.array([.6, .2, .2]),  # Priors on goals in order of f1,f2,f3 as reported in Kao 2014
    np.array([.01, .99]),  # Priors on category, in order ANIMAL, HUMAN
    animal_feat,  # The set of worlds everything uses
    3  # the value for lambda
    )

for i in np.arange(0, 500, 0.1):
    p = Ln('fox',  # the utterance used
    np.array([.6, .2, .2]),  # Priors on goals in order of f1,f2,f3 as reported in Kao 2014
    np.array([.01, .99]),  # Priors on category, in order ANIMAL, HUMAN
    animal_feat,  # The set of worlds everything uses
    i  # the value for lambda
    )
    x = nt_fox[1][3]/p[1][3]
    if x > 2.22 and x < 2.29:
        ASD_Contrast = ASD_Contrast.append({'Animal': 'fox', 'NT': nt_fox[1][3], 'ASD': p[1][3], 'P(a)': 0.01, 'P(h)': 0.99, 'Lambda': i, 'Value': x}, ignore_index=True)

nt_whale = Ln('whale',  # the utterance used
    np.array([.6, .2, .2]),  # Priors on goals in order of f1,f2,f3 as reported in Kao 2014
    np.array([.01, .99]),  # Priors on category, in order ANIMAL, HUMAN
    animal_feat,  # The set of worlds everything uses
    3  # the value for lambda
    )

for i in np.arange(0, 500, 0.1):
    p = Ln('whale',  # the utterance used
    np.array([.6, .2, .2]),  # Priors on goals in order of f1,f2,f3 as reported in Kao 2014
    np.array([.01, .99]),  # Priors on category, in order ANIMAL, HUMAN
    animal_feat,  # The set of worlds everything uses
    i  # the value for lambda
    )
    x = nt_whale[1][3]/p[1][3]
    if x > 2.22 and x < 2.29:
        ASD_Contrast = ASD_Contrast.append({'Animal': 'whale', 'NT': nt_whale[1][3], 'ASD': p[1][3], 'P(a)': 0.01, 'P(h)': 0.99, 'Lambda': i, 'Value': x}, ignore_index=True)

nt_shark = Ln('shark',  # the utterance used
    np.array([.6, .2, .2]),  # Priors on goals in order of f1,f2,f3 as reported in Kao 2014
    np.array([.01, .99]),  # Priors on category, in order ANIMAL, HUMAN
    animal_feat,  # The set of worlds everything uses
    3  # the value for lambda
    )

for i in np.arange(0, 500, 0.1):
    p = Ln('shark',  # the utterance used
    np.array([.6, .2, .2]),  # Priors on goals in order of f1,f2,f3 as reported in Kao 2014
    np.array([.01, .99]),  # Priors on category, in order ANIMAL, HUMAN
    animal_feat,  # The set of worlds everything uses
    i  # the value for lambda
    )
    x = nt_shark[1][3]/p[1][3]
    if x > 2.22 and x < 2.29:
        ASD_Contrast = ASD_Contrast.append({'Animal': 'shark', 'NT': nt_shark[1][3], 'ASD': p[1][3], 'P(a)': 0.01, 'P(h)': 0.99, 'Lambda': i, 'Value': x}, ignore_index=True)

nt_monkey = Ln('monkey',  # the utterance used
    np.array([.6, .2, .2]),  # Priors on goals in order of f1,f2,f3 as reported in Kao 2014
    np.array([.01, .99]),  # Priors on category, in order ANIMAL, HUMAN
    animal_feat,  # The set of worlds everything uses
    3  # the value for lambda
    )

for i in np.arange(0, 500, 0.1):
    p = Ln('monkey',  # the utterance used
    np.array([.6, .2, .2]),  # Priors on goals in order of f1,f2,f3 as reported in Kao 2014
    np.array([.01, .99]),  # Priors on category, in order ANIMAL, HUMAN
    animal_feat,  # The set of worlds everything uses
    i  # the value for lambda
    )
    x = nt_monkey[1][3]/p[1][3]
    if x > 2.22 and x < 2.29:
        ASD_Contrast = ASD_Contrast.append({'Animal': 'monkey', 'NT': nt_monkey[1][3], 'ASD': p[1][3], 'P(a)': 0.01, 'P(h)': 0.99, 'Lambda': i, 'Value': x}, ignore_index=True)

ASD_Contrast.to_csv(r'/Users/macke/Downloads/ASD_Contrast.csv', index=False)
