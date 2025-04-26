import numpy as np, glob, os
sample = np.load(glob.glob("project3_fingerprint_fvc2000/data/processed/DB1_B/*.npy")[0])
print(sample.min(), sample.max(), sample.dtype)
