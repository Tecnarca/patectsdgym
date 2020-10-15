import time
from joblib import Parallel, delayed
import mlflow
import os

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import conf

def run_synthesis(synthesis_args):
	n, s, synth_args, d, epsilons, datasets, cat_cols, save_models_path, run_name = synthesis_args
	res = []
	with mlflow.start_run(nested=True):
		synth = s(epsilon=float(epsilons[0]), **synth_args)
		for i, e in enumerate(epsilons):
			start_time = time.time()		

			#Need to save: Epochs, Epsilons traversed, Loss 
			sampled = synth.fit_sample(datasets[d]["data"],categorical_columns=cat_cols.split(','), update_epsilon=float(e), verbose=conf.VERBOSE, mlflow=True)
			end_time = time.time()

			mlflow.set_tags({"synthesizer": type(synth),
				"args": str(synth_args),
				"dataset": str(d),
				"epsilon": str(e),
				"duration_seconds": str(end_time - start_time)})

			res.append((n, d, str(e), sampled))
			print("Epsilon " + str(e) + " finished for Synthesizer " + n + " in " + str(end_time - start_time) + "s")

						
			datapath = os.path.join(save_models_path, n + "_" + str(e) + "_" + d + "_" + run_name + "_dataset.csv")
			modelpath = os.path.join(save_models_path, n + "_" + str(e) + "_" + d + "_" + run_name + "_model.ckpt")

			with open(datapath, 'wb') as f:
				sampled.to_csv(datapath)
			mlflow.log_artifact(datapath)

			synth.save(modelpath)
			mlflow.log_artifact(modelpath)

	return res

	


#@profile
def run_all_synthesizers(datasets, epsilons, save_models_path, run_name):

	synthesizer_runs = []
	epsilons.sort()

	for d in datasets:
		print('Now running synths on: ', d)
		for n, s in conf.SYNTHESIZERS:
			datasets[d][n] = {}

			if d in conf.SYNTH_SETTINGS[n]:
				synth_args = conf.SYNTH_SETTINGS[n][d]
			else:
				print("Unspecified configuration for dataset {} on Synthesizer {}. Using default...".format(d,n))
				synth_args = conf.SYNTH_SETTINGS[n]['default']

			a_run = (n, s, synth_args, d, epsilons, datasets, datasets[d]["categorical_columns"], save_models_path, run_name)
			synthesizer_runs.append(a_run)

		job_num = len(conf.SYNTHESIZERS)
		
		start = time.time()

		results = Parallel(n_jobs=job_num, verbose=True)(map(delayed(run_synthesis), synthesizer_runs))

		end = time.time() - start

		print('Synthesis for ' + str(n) + ' finished in ' + str(end))

		for l in results:
			for n, d, e, sampled in l:
				datasets[d][n][e] = sampled

	return datasets