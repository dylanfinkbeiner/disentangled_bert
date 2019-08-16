import time
import subprocess
import math
import sys
import copy
import os
import pickle

import numpy as np

JOBS_DIR = './jobs_tuning'


def main():

    pickled_job_name = ''
    #pickled_job_name = 'sst_stress'
    #pickled_job_name = 'cola_stress'

    #n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_iter = 50
    #conf_name = 'cola_single_trials'
    #conf_name = 'multi_trials'
    conf_name = 'mnli_cola_adv'
    #conf_name = 'qqp_single_trials'
    #conf_name = 'mnli_single_trials'
    #conf_name = 'rte_single_trials'
    #conf_name = 'qnli_single_trials'
    #conf_name = 'mrpc_single_trials'
    #conf_name = 'stsb_single_trials'
    #conf_name = 'stsb_single_trials'
    #conf_name = 'subsampling_bug'
    #conf_name = 'rte_single_trials'
    #conf_name = 'new_side_by_side'
    #run_name_prefix = 'triplet-'
    #run_name_prefix = 'fraction_stability_restof025_'
    #run_name_prefix = 'fraction_fourth_tune_'
    #run_name_prefix = 'repeat_tenth_tuning_again'
    #run_name_prefix = 'test4_discriminatormetrics_'
    #run_name_prefix = 'retrial_crazytuning20_'
    #run_name_prefix = '150_stress_testing_'
    #run_name_prefix = 'crazy02_repeatscalesharedofcrazy20'
    run_name_prefix = 'explore_orthogonality_cuda0_'
    #run_name_prefix = 'whatdoesbiggergradnormdo_'
    #run_name_prefix = 'noAdvPlusDiscHidden_'
    triplets = False

    param_names = {}

    param_names['evaluate_final'] = [True]
    param_names['pickle_records'] = [True]
    param_names['max_grad_norm'] = [5.]
    #param_names['max_grad_norm'] = unif_samp(sample_space=[5, 10], n_samps=n_iter)
    param_names['tokenizer'] = ['bert-base-uncased']
    param_names['input_module'] = ['bert-base-uncased']
    #if 'tarski' in run_name_prefix:
    #param_names['tokenizer'] = ['bert-large-uncased']
    #param_names['input_module'] = ['bert-large-uncased']

    do_pretrain = [1] # Should normally be 1
    param_names['do_pretrain'] = do_pretrain

    param_names['random_baseline'] = [False]

    #target_train_data_fraction = [0.1, 0.25, 0.5, 0.75]
    #pretrain_data_fraction = [0.025]
    ##param_names['target_train_data_fraction'] = target_train_data_fraction
    #param_names['pretrain_data_fraction'] = pretrain_data_fraction


    cola_adv = unif_samp(sample_space=['cola-adv-128', 'cola-adv-256'], n_samps=n_iter)
    mnli_adv = unif_samp(sample_space=['mnli-adv-128', 'mnli-adv-256'], n_samps=n_iter)
    #cola_adv = ['cola-adv-128']
    #mnli_adv = ['mnli-adv-128']
    discriminator_hidden = unif_samp(sample_space=[64, 128], n_samps=n_iter)
    #discriminator_hidden = [32]
    param_names['cola-adv'] = cola_adv
    param_names['mnli-adv'] = mnli_adv
    param_names['discriminator_hidden'] = discriminator_hidden

    #batch_size = [16, 32]
    #batch_size = unif_samp(sample_space=[8, 16, 24], n_samps=n_iter)
    #batch_size = unif_samp(sample_space=[32], n_samps=n_iter)
    batch_size = unif_samp(sample_space=[16, 32, 24], n_samps=n_iter)
    #batch_size = [8]
    #batch_size = [8, 32, 16]
    #batch_size = [32]
    val_batch_size = [96]
    #batch_size = [24]
    #val_batch_size = [24]
    #batch_size = [16, 32]
    #batch_size = [24]
    param_names['batch_size'] = batch_size
    param_names['val_batch_size'] = val_batch_size

    #lr = log_samp(e_lo=-1, n_samps=n_iter)
    #lr = unif_samp(sample_space=[3e-5], n_samps=n_iter)
    #lr = [1.25e-5, 1.5e-5, 1.75e-5]
    #lr = [2e-5, 2.5e-5, 3e-5]
    lr = unif_samp(sample_space=[2e-5, 3e-5, 4e-5], n_samps=n_iter)
    #lr = [2e-5, 2.5e-5] #5e-5]
    #lr = [2e-5, 3e-5]
    #lr = [2e-5]
    #lr = [3e-5]
    #lr = unif_samp(sample_space=[2e-5,2.5e-5, 3e-5], n_samps=n_iter)
    #lr = unif_samp(sample_space=[2.5e-5], n_samps=n_iter)
    #lr = unif_samp(sample_space=[2e-5, 2.5e-5, 3e-5], n_samps=n_iter)
    #lr = unif_samp(sample_space=[2e-5, 2.5e-5], n_samps=n_iter)
    #lr = unif_samp(sample_space=[3e-5], n_samps=n_iter)
    param_names['lr'] = lr

    random_seed = np.random.randint(2000, size=n_iter) #x
    #random_seed = [47]
    #random_seed = [1234]
    #random_seed = [8998, 2341, 2134, 2543, 2222]
    #random_seed = [241, 311, 533, 204, 301, 688, 966, 645, 366, 894] # From HIDDEN
    #random_seed = [894]
    #random_seed = [11, 11, 30, 30]
    #random_seed = [19]
    #random_seed = [21]
    param_names['random_seed'] = random_seed

    #max_epochs = unif_samp(sample_space=[2,3], n_samps=n_iter)
    #max_epochs = [3]
    #max_epochs = [1]
    max_epochs = [3]
    #max_epochs = [3,4,5]
    param_names['max_epochs'] = max_epochs



    #val_interval = unif_samp(sample_space=[500], n_samps=n_iter)
    #val_interval = [2000]
    val_interval = [500]
    #val_interval = [4000]
    param_names['val_interval'] = val_interval

    #max_seq_len = unif_samp(sample_space=[100, 128, 80], n_samps=n_iter)
    max_seq_len = [256]
    param_names['max_seq_len'] = max_seq_len

    cuda = [1]
    param_names['cuda'] = cuda

    #scale = log_samp(e_lo=-4, e_hi=-2, n_samps=n_iter)
    scale_mnli_adv = log_samp(e_lo=-4, e_hi=-1, n_samps=n_iter)
    scale_cola_adv = log_samp(e_lo=-4, e_hi=-1, n_samps=n_iter)
    #scale_mnli_adv = unif_samp(sample_space=[0.005, 0.002, .0001], n_samps=n_iter)
    #scale_cola_adv = unif_samp(sample_space=[0.005, 0.002, 0.001], n_samps=n_iter)
    #scale_cola_adv = [.598948]
    #scale_mnli_adv = [.003848]
    #scale_mnli_adv = [.002841]
    #scale_cola_adv = [.002841]
    #scale_mnli_adv = scale
    #scale_cola_adv = scale
    #scale_cola_adv = [1.]
    #scale_mnli_adv = [1.]
    scale_shared = log_samp(e_lo=-4, e_hi=-2, n_samps=n_iter)
    scale_orthogonality = log_samp(e_lo=-8, e_hi=-5, n_samps=n_iter)
    #scale_shared = [.000307]
    #scale_shared = [.01, .001, .0001, .00001]
    #scale_shared = [.045425]
    #scale_shared = [.000307]
    #scale_shared = [0.]
    param_names['scale_mnli_adv'] = scale_mnli_adv
    param_names['scale_cola_adv'] = scale_cola_adv
    param_names['scale_shared'] = scale_shared
    param_names['scale_orthogonality'] = scale_orthogonality


    #k = unif_samp(sample_space=[128, 256], n_samps=n_iter)
    k = unif_samp(sample_space=[128, 256], n_samps=n_iter)
    special_task = [True]
    #k_syn = unif_samp(sample_space=[128, 256], n_samps=n_iter)
    #k_syn = [128, 256]
    #k_sem = [128, 256]
    #k_syn = [256]
    #k_sem = [256]
    k_syn = k
    k_sem = k
    #k_shared = unif_samp(sample_space=[128, 256, 384], n_samps=n_iter)
    k_shared = k
    #k_shared = [128]
    #special_task = [False]
    #k_syn = [0]
    #k_sem = [0]
    param_names['special_task'] = special_task
    param_names['k_syn'] = k_syn
    param_names['k_sem'] = k_sem
    param_names['k_shared'] = k_shared

    exp_name = [conf_name.replace('_', '-')]
    run_name = [f'{run_name_prefix}{i:02d}' for i in range(n_iter)]
    #tasks = [ ['rte', 'sst', 'cola', 'qqp'] ]
    #tasks = [ ['rte', 'cola', 'rte-adv', 'cola-adv'] ]
    #tasks = [ ['rte', 'cola'] ]
    #tasks = [ ['mrpc'] ]
    #tasks = [ ['cola'] ]
    #tasks = [ ['mnli', 'cola', 'mnli-adv', 'cola-adv'] ]
    tasks = [ ['mnli', 'cola', 'mnli-adv', 'cola-adv', 'mnli-discriminator', 'cola-discriminator'] ]
    #tasks = [ ['mnli', 'cola', 'mnli-discriminator', 'cola-discriminator'] ]
    #tasks = [ ['cola'] ]
    #tasks = [ ['rte'] ]
    #tasks = [ ['mnli'] ] 
    #tasks = [ ['mnli', 'cola'] ] 
    #tasks = [ ['sst'] ] 
    #tasks = [ ['sts-b'] ] 
    #tasks = [ ['qqp'] ] 
    param_names['exp_name'] = exp_name
    param_names['run_name'] = run_name
    param_names['tasks'] = tasks

    use_amp = [True]
    param_names['use_amp'] = use_amp

    #weighting_method = ['power_0.75', 'proportional']
    weighting_method = ['power_0.75']
    param_names['weighting_method'] = weighting_method

    #pretrain_mnli_fraction = unif_samp(sample_space=[0.1, 0.25], n_samps=n_iter)
    #pretrain_mnli_fraction = [0.1]
    pretrain_mnli_fraction = [0.025]
    pretrain_mnli_adv_fraction = [0.025]
    pretrain_mnli_discriminator_fraction = [0.025]
    param_names['pretrain_mnli_fraction'] = pretrain_mnli_fraction
    param_names['pretrain_mnli_adv_fraction'] = pretrain_mnli_adv_fraction
    param_names['pretrain_mnli_discriminator_fraction'] = pretrain_mnli_discriminator_fraction
    param_names['target_train_mnli_fraction'] = pretrain_mnli_fraction
    param_names['target_train_mnli_adv_fraction'] = pretrain_mnli_fraction
    param_names['target_train_mnli_discriminator_fraction'] = pretrain_mnli_discriminator_fraction

    val_data_limit = [5000]
    param_names['val_data_limit'] = val_data_limit

    #write_preds = [ ['test'] ]
    #write_preds = [ ['val'] ]
    write_preds = [0]
    param_names['write_preds'] = write_preds

    experiments = list_of_dicts(param_names, n_iter=n_iter, grid_search=False)
    #experiments = list_of_dicts(param_names, n_iter=-1, grid_search=True, run_name_prefix=run_name_prefix)

    if pickled_job_name:
        pickled_job_path = os.path.join(JOBS_DIR, 'pickled_for_later', pickled_job_name + '.pkl')
        if not os.path.exists(pickled_job_path):
            with open(pickled_job_path, 'wb') as f:
                pickle.dump(experiments, f)
            print('Pickled for later! Exiting.')
            exit()
        else:
            with open(pickled_job_path, 'rb') as f:
                print('Overwriting experiments with those from a pickled job.')
                experiments = pickle.load(f)

                cuda_setting = cuda.pop()
                for exp in experiments:
                    exp['cuda'] = cuda_setting

    if triplets:
        e_new = []
        for exp in experiments:
            e_new.append(exp)
            no_adv = copy.deepcopy(exp)
            no_adv['tasks'] = ['mnli', 'cola']
            no_adv['run_name'] = exp['run_name'] + 'no_adv'
            e_new.append(no_adv)
            no_proj = copy.deepcopy(no_adv)
            no_proj['special_task'] = False
            no_proj['k_syn'] = 0
            no_proj['k_sem'] = 0
            no_proj['run_name'] = exp['run_name'] + 'no_proj'
            e_new.append(no_proj)
        experiments = e_new

    # Write human-readable summaries of the experiments out to JOBS_DIR
    with open(os.path.join(JOBS_DIR, conf_name), 'a') as f:
        for exp in experiments:
            f.write('\n')
            exp_str = ''
            for k, v in sorted(exp.items()):
                exp_str += f'{k} : {v}\n'
            f.write(exp_str)
            f.write('\n' * 2)


    start_time = time.time()
    for exp in experiments:
        command = get_command(exp, exp['exp_name'])
        print(command)
        subprocess.run(command, shell=True, check=True)
    end_time = time.time()

    # Clean up
    if pickled_job_name != '' and os.path.exists(pickled_job_path) and (end_time - start_time > 120):
        os.remove(pickled_job_path)
        print('Deleted pickled job.')

    print('Done with tuning!')


def get_command(experiment, exp_name):
        conf_name = exp_name.replace('-', '_')
        command = f'python main.py --config_file config/{conf_name}.conf'
        
        command += ' --overrides \"'

        for k, v in experiment.items():
            if k == 'tasks':
                task_list = '\\\"' + ','.join(v) + '\\\"'
                command += f'pretrain_tasks = {task_list}, '
                command += f'target_tasks = {task_list}, '
            elif k == 'write_preds' and v != 0:
                preds = '\\\"' + ','.join(v) + '\\\"'
                command += f'write_preds = {preds}, '
            elif k == 'cola-adv' or k == 'mnli-adv':
                v = '\${' + v + '}'
                command += k + ' = ' + v + ', '
            else:
                command += f'{k} = {v}, '

        command += '\"'
        return command


def list_of_dicts(param_names, n_iter=-1, grid_search=False, run_name_prefix=None):
    if grid_search:
        models = []
        if 'run_name' in param_names:
            param_names.pop('run_name')
        if 'exp_name' in param_names:
            exp_name = param_names.pop('exp_name')[0]

        keys = list(param_names.keys())
        values = list(param_names.values())

        num_models = 1
        for v in values:
            num_models *= len(v)

        temp = cartesian_product(values[0], values[1])
        for v in values[2:]:
            temp = cartesian_product(temp, v)
        products = temp

        for i, p in enumerate(products):
            d = {}
            for j, k in enumerate(keys):
                d[k] = p[j]
                d['run_name'] = f'{run_name_prefix}{i:02d}'
                d['exp_name'] = exp_name
            models.append(d)

    else:
        # Check list sizes
        for k, v in param_names.items():
            if not (len(v) == 1 or len(v) == n_iter):
                print(f'{k} has length {len(v)}. Should be 1 or n_iter.')
                raise Exception

        for k, v in param_names.items():
            if len(v) == 1:
                param_names[k] = v * n_iter

        models = [dict() for _ in range(n_iter)]
        for k, v in param_names.items():
            for m, val in zip(models, v):
                m[k] = val

    return models


def cartesian_product(list1, list2):
    product = []

    if type(list1[0]) != list:
        product = [ [x1, x2] for x1 in list1 for x2 in list2 ]
    else:
        product = [ copy.deepcopy(x1) + [x2] for x1 in list1 for x2 in list2 ]

    return product


def unif_samp(sample_space=None, n_samps=None):
    sample = np.random.multinomial(1, [1 / len(sample_space)] * len(sample_space), size=n_samps)
    sample = np.argmax(sample, axis=1)

    return np.array(sample_space).take(sample)


#def log_samp(exponent=None, n_samps=None):
def log_samp(e_lo=None, e_hi=0, n_samps=None):
    # e <-> exponent
    if e_lo >= 0 or e_hi > 0 or e_lo >= e_hi:
        raise Exception
    sample = ((e_lo - e_hi) * np.random.rand(n_samps)) + e_hi
    sample = 10 ** sample
    sample = sample.tolist()

    return sample


def logsample_space(low, high, pts, rnd=None):
    a = np.logspace(np.log10(low), np.log10(high), pts)
    a = rnd * np.round(a / rnd)
    a = np.unique(a)

    return a


# Needed so that main function has access to functions in this file
if __name__ == '__main__':
    main()
