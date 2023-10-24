import os
import subprocess
import sys

# Specify the directory path
directory_path = "config"
configs = []
# List all files in the directory
for entry in os.listdir(directory_path):
    entry_path = os.path.join(directory_path, entry)
    if os.path.isfile(entry_path):
        parent_dir_name = os.path.basename(directory_path)
        file_name = entry
        configs.append(os.path.join(parent_dir_name, file_name))
print(configs)

commands = []
mode = sys.argv[1]
for idx, config in enumerate(configs):
    if mode == 'train':
        commands.append(f"sincFold -d cuda:0 -c {config} {mode} data/train-partition-0.csv data/all_repr_archiveii.pt --valid-file data/valid-partition-0.csv -o ./results-emb-0-{idx} -r {idx}")
    elif mode == 'test':
        commands.append(f"sincFold -d cuda:0 -c {config} {mode} data/test-partition-0.csv data/all_repr_archiveii.pt -w ./results-emb-0-{idx}/weights.pmt -o ./results-emb-0-{idx}-test -r {idx}test")
    else:
        print('you must specify a mode')
for command in commands:
    subprocess.call(command, shell=True)
