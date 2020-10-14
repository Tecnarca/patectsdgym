import time
from joblib import Parallel, delayed
import mlflow

import conf

def run_synthesis(synthesis_args):
	n, s, synth_args, d, epsilons, datasets, cat_cols = synthesis_args

	with mlflow.start_run(nested=True):
		synth = s(epsilon=float(epsilons[0]), **synth_args)
		for i, e in enumerate(epsilons):
			start_time = time.time()		
			sampled = synth.fit_sample(datasets[d]["data"],categorical_columns=cat_cols.split(','), update_epsilon=float(e))
			end_time = time.time()

			#i need to log: #Iterations, Epsilon, Args, Synth, Dataset, Time_Took
			#mlflow.log_metrics()

			print("Epsilon " + str(e) + " finished for Synthesizer " + n + " in " + str(end_time - start_time)+"s")
			datasets[d][n][str(e)] = sampled

#@profile
def run_all_synthesizers(datasets, epsilons):

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

			a_run = (n, s, synth_args, d, epsilons, datasets, datasets[d]["categorical_columns"])
			synthesizer_runs.append(a_run)

		job_num = len(conf.SYNTHESIZERS)
		
		start = time.time()

		Parallel(n_jobs=job_num, verbose=True, prefer="threads")(map(delayed(run_synthesis), synthesizer_runs))

		end = time.time() - start

		print('Synthesis for ' + str(n) + ' finished in ' + str(end))

	return datasets