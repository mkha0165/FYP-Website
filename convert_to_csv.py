import pandas as pd
from scipy.io import loadmat

# === load the MAT file ===
mat = loadmat("data/mat/FaultyCase6.mat")   # adjust path

# === define your variable names ===
cols = [
    'PT312','PT401','PT408','PT403','PT501','PT408_diff','PT403_diff',
    'FT305','FT104','FT407','LI405','FT406','FT407_density','FT406_density',
    'FT104_density','FT407_temp','FT406_temp','FT104_temp','LI504',
    'VC501','VC302','VC101','PO1','PT417'
]

# === convert each matrix into DataFrame ===
if "Set6_1" in mat:
    df1 = pd.DataFrame(mat["Set6_1"], columns=cols)
    df1.to_csv("Set6_1.csv", index=False)
else:
    print("Set5_1 not found in the MAT file.")

if "Set6_2" in mat:
    df2 = pd.DataFrame(mat["Set6_2"], columns=cols)
    df2.to_csv("Set6_2.csv", index=False)

if "Set6_3" in mat:
    df3 = pd.DataFrame(mat["Set6_3"], columns=cols)
    df3.to_csv("Set6_3.csv", index=False)

print("✅ Exported: Set3_1.csv, Set3_2.csv, Set3_3.csv")
