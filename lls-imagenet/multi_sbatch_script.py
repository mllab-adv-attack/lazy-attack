import os
import subprocess
import itertools

file_name = 'main.py'
param_dict={
  '--loss_func': ['cw'],
  '--img_index_start': [0],
  '--epsilon': [0.01],
}
param_keys = [str(v) for v in param_dict.keys()]

# Create param list
param_list = [v for v in itertools.product(*tuple([param_dict[key] for key in param_keys]))]
nkey = len(param_keys)

# Run each process
for i in range(0, len(param_list)):
  param = param_list[i]
  cmd = ''.join(['{} {} '.format(param_keys[key_idx], param[key_idx]) for key_idx in range(nkey)])
  cmd = "'{}'".format(cmd)
  cmd = "python {} {}".format(file_name, cmd)
  sbatch_cmd = "sbatch ./single_srun_script.sh {}".format(cmd)
  print(sbatch_cmd)
  subprocess.check_call(sbatch_cmd, shell=True) 







