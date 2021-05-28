import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper import DFKeys, ETA_DOFS, NU_DOFS


df = pd.read_csv("data/synthetic/random-1.csv")
timesteps = df[DFKeys.TIME.value].to_numpy()
y_measured = df[ETA_DOFS + NU_DOFS].to_numpy()

for i, dof in enumerate(ETA_DOFS + NU_DOFS):
    plt.plot(timesteps, y_measured[:, i])
    plt.title(dof)
    plt.savefig("results/synthetic_data/random-1/%s.jpg" % dof)
    plt.close()
