import pickle as pkl
import pandas as pd
import sys

filename = sys.argv[1]
day = sys.argv[2]
with open(filename, "rb") as file:
    object = pkl.load(file)
df = pd.DataFrame(object)
df['day'] = df['timestamp'].dt.day
df = df.loc[df['day'] == int(day)]
df = df.drop(['day'], axis=1)
df = df.reset_index(drop=True)
df.to_csv(filename.split(".")[0]+"_day_"+str(day)+".csv")
