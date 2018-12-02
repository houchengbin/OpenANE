#!/usr/bin/env python
# coding: utf-8

import os
import re

import numpy as np
import pandas as pd

DIR = os.path.dirname(os.path.realpath(__file__))

data = pd.DataFrame(columns=["Samples", "AUC", "Micro-F1", "Macro-F1", "Time"], dtype="float")
data.set_index(pd.MultiIndex.from_tuples((), names=("dataset", "method")), inplace=True)
for fname in os.listdir(DIR):
    if fname.endswith(".log"):
        try:
            name, method = fname.split(".")[0].split('-')[0:2]
            with open(os.path.join(DIR, fname)) as f:
                l = np.array(
                    re.findall(
                        r"STEP3: end learning embeddings; time cost: (\d+\.\d+)s.*roc= (\d+\.\d+).*{'micro': (\d+\.\d+), 'macro': (\d+\.\d+), 'samples': \d+\.\d+, 'weighted': \d+\.\d+}",
                        f.read(),
                        re.DOTALL
                    )[-1],
                    dtype="float"
                )[[1,2,3,0]]
                if (name, method) not in data.index:
                    data.loc[(name, method), :] = 0
                n = data.loc[(name, method)][0]
                l = (n * np.array(data.loc[(name, method)][1:5]) + l)/(n + 1)
                data.loc[(name, method)][0] += 1
                data.loc[(name, method), 1:5] = l
        except Exception as e:
            print(f"failed with {fname}: {e}")
data.sort_index(inplace=True)

data.to_csv(os.path.join(DIR, "result.csv"))
