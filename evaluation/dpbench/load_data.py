import requests
import io
import json
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import zipfile

#memory cap for Datasets (largest from UCI: bank with 6148824 bytes)
MEM_CAP = 7148824


def retrieve_dataset(dataset):
	sep = dataset['sep']
	index_col = False
	names = dataset['columns'].split(',')
	r = requests.post(dataset['url']) if dataset['zipped'] == 'f' else requests.get(dataset['url'])
	z = None if dataset['zipped'] == 'f' else zipfile.ZipFile(io.BytesIO(r.content))
	skiprow = 1 if dataset['header'] == "t" else 0

	if r.ok:
			data = io.StringIO(r.content.decode('utf8')) if dataset['zipped'] == 'f' else z.open(dataset['zip_name'])
			df = pd.read_csv(data, names=names, sep=sep, index_col=index_col, skiprows=skiprow)
	else:
		raise "Unable to retrieve dataset: " + dataset
	
	return df

def select_column(scol):
	return scol.split(',')

def encode_categorical(df,dataset):
	encoders = {}
	if dataset['categorical_columns'] != "":
		for column in select_column(dataset['categorical_columns']):
			encoders[column] = LabelEncoder()
			df[column] = encoders[column].fit_transform(df[column])

	df = df.apply(pd.to_numeric, errors='ignore')
	data_mem = df.memory_usage(index=True).sum()
	print("Memory consumed by " + dataset['name'] + ":" + str(data_mem))

	if data_mem > MEM_CAP:
		print("Memory use too high with " + dataset['name'] + ", subsampling to:" + str(MEM_CAP))
		reduct_ratio = MEM_CAP / data_mem
		subsample_count = int(len(df.index) * reduct_ratio)
		df = df.sample(n=subsample_count)
		print("Memory consumed by " + dataset['name'] + ":" + str(df.memory_usage(index=True).sum()))

	return {"data": df, "target": dataset['target'], "name": dataset['name'], "imbalanced": dataset['imbalanced'], "categorical_columns": dataset['categorical_columns']}

def load_data(datasets):

    with open('datasets.json') as j:
        dsets = j.read()
    archive = json.loads(dsets)

    loaded_datasets = {}
    for d in datasets:
    	df = retrieve_dataset(archive[d])
    	encoded_df_dict = encode_categorical(df, archive[d])
    	loaded_datasets[d] = encoded_df_dict

    return loaded_datasets