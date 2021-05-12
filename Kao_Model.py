import numpy as np
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

df = pd.read_csv('https://raw.githubusercontent.com/justinek/metaphors/master/Data/FeaturePriorExp/featurePriors-set.csv', index_col=0)

animals = [j for j in df['animal'].unique()]

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

weights = np.array([.7, .1, .2])


def Sn(utterance, weights, animal_feat):
    Lo = priors[utterance]
    U = weights @ animal_feat.T * Lo  # need to transpose things for dot products

    # softmax decision rule
    S = np.exp(U)
    S = S/S.sum()

    return S


human_animal_weight = priors[29]

p_animal_human = np.array([.01, .99])  # important to keep the same order as previous step


def Ln(utterance, weights, human_animal_weight, p_animal_human, animal_feat):
    s = Sn(utterance, weights, animal_feat)
    L1 = human_animal_weight * s
    L1 = p_animal_human.reshape(-1, 1) * L1
    L1 = L1/L1.sum(axis=-1).reshape(-1, 1)  # normalization by column so we can interpret data better
    return L1