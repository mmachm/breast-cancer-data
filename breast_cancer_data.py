import pandas as pd
from kmapper import KeplerMapper
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(os.path.join(dir_path, "data.csv"))

del df["Unnamed: 32"]
del df["id"]

diagnosis = df["diagnosis"]
data_wo_diagnosis = df.drop("diagnosis", axis=1)

named_columns = data_wo_diagnosis.columns


scaler = StandardScaler()
data_wo_diagnosis = pd.DataFrame(scaler.fit_transform(data_wo_diagnosis))
data_wo_diagnosis.columns = named_columns


kmeans = KMeans(n_clusters=3)
kmeans.fit(data_wo_diagnosis)

# Find which cluster each data-point belongs to
clusters = kmeans.predict(data_wo_diagnosis)
data_wo_diagnosis["cluster"] = clusters

data = pd.concat([diagnosis, data_wo_diagnosis], axis=1, join='inner')
