import subprocess
import os



def generate_pbs_script(python_cmd, experiment_name):
    python_cmd = f'{python_cmd} --experiment_name {experiment_name}'
    print(python_cmd)
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


def generate_python_cmd(distillation, ranks, recomp_target):
    return f"python test2.py --distillation {distillation} --ranks {ranks} --recomp_target {recomp_target}"


dists = ['tucker_recomp', 'tucker', 'featuremap']
ranks_list = ['BATCH_SIZE,32,8,8', 'BATCH_SIZE,24,6,6']
recomp_target_list = ['teacher', 'student', 'both']

for distillation in dists:
    if distillation == 'featuremap':
        python_cmd = generate_python_cmd(distillation, ranks, recomp_target)
        generate_pbs_script(python_cmd, 'featuremap')
    else:
        for i, ranks in enumerate(ranks_list):
            if distillation == 'tucker_recomp':
                for k, recomp_target in enumerate(recomp_target_list):
                    python_cmd = generate_python_cmd(distillation, ranks, recomp_target)
                    generate_pbs_script(python_cmd, f'recomp_{recomp_target_list[k]}_{i}')
            else:
                python_cmd = generate_python_cmd(distillation, ranks, recomp_target)
                generate_pbs_script(python_cmd, f'tucker_{i}')
                
            

    



