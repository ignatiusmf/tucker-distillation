import subprocess
import os
from pathlib import Path

testing = False if os.name == 'posix' else True 
total = 0

def generate_pbs_script(python_cmd, experiment_name):
    global total 
    total += 1
    if total > 10: exit()
    if testing: return

    pbs_script = f"""#!/bin/sh
#PBS -N {experiment_name}
#PBS -q gpu_1
#PBS -P CSCI1166
#PBS -l select=1:ncpus=10:mpiprocs=10:mem=32gb:ngpus=1
#PBS -l walltime=03:00:00
#PBS -o /mnt/lustre/users/iferreira/tucker-distillation/experiments/{experiment_name}/logs
#PBS -e /mnt/lustre/users/iferreira/tucker-distillation/experiments/{experiment_name}/errors
#PBS -m abe -M u25755422@tuks.co.za

ulimit -s unlimited
module load chpc/python/anaconda/3-2021.11
source /mnt/lustre/users/iferreira/myenv/bin/activate

date
echo -e 'Running {python_cmd}\\n'

start_time=$(date +%s) 

cd /mnt/lustre/users/iferreira/tucker-distillation
{python_cmd}

echo -e "\\nTotal execution time: $(( $(date +%s) - start_time)) seconds"
"""

    temp_file = "temp_pbs_script.sh"
    with open(temp_file, 'w') as f:
        f.write(pbs_script)
    try:
        result = subprocess.run(['qsub', temp_file], capture_output=True, text=True)
        print(f"Job submitted: {result.stdout}")
        if result.stderr:
            print(f"Errors: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def generate_python_cmd(distillation, experiment_name, ranks=None, recomp_target=None):
    python_cmd = ''
    if distillation == 'featuremap':
        python_cmd += f"python test2.py --distillation {distillation}"
    elif distillation == 'tucker':
        python_cmd += f"python test2.py --distillation {distillation} --ranks {ranks}"
    elif distillation == 'tucker_recomp':
        python_cmd += f"python test2.py --distillation {distillation} --ranks {ranks} --recomp_target {recomp_target}"

    python_cmd += f' --experiment_name {experiment_name}'
    return python_cmd


def check_path_and_skip(experiment_path):
    if experiment_path.exists():
        return True
    else:
        experiment_path.mkdir(parents=True)
        return False

runs = 5 
            
distillation = 'featuremap'

for run in range(runs):
    en = f'{distillation}/{run}'
    experiment_path = Path(f'experiments/{en}')

    if check_path_and_skip(experiment_path): continue

    python_cmd = generate_python_cmd(distillation, en)
    print(python_cmd)
    generate_pbs_script(python_cmd, en)

ranks_list = ['BATCH_SIZE,32,8,8', 'BATCH_SIZE,24,8,8', 'BATCH_SIZE,16,8,8', 'BATCH_SIZE,8,8,8']

distillation = 'tucker'
for rank in ranks_list:
    for run in range(runs):
        en = f'{distillation}/{rank}/{run}'
        experiment_path = Path(f'experiments/{en}')

        if check_path_and_skip(experiment_path): continue

        python_cmd = generate_python_cmd(distillation, en, rank)
        print(python_cmd)
        generate_pbs_script(python_cmd, en)

recomp_target_list = ['teacher', 'student', 'both']

distillation = 'tucker_recomp'
for target in recomp_target_list:
    for rank in ranks_list:
        for run in range(runs):
            en = f'{distillation}/{target}/{rank}/{run}'
            experiment_path = Path(f'experiments/{en}')

            if check_path_and_skip(experiment_path): continue

            python_cmd = generate_python_cmd(distillation, en, rank, target)
            print(python_cmd)
            generate_pbs_script(python_cmd, en)