import os
import subprocess

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
for idx, config in enumerate(configs):
    commands.append(f"sincFold -d cuda:0 -c {config} train data/train-partition-0.csv data/all_repr_archiveii.pt --valid-file data/valid-partition-0.csv -o ./results-emb-0-{idx} -r {idx}")

for command in commands:
    subprocess.call(command, shell=True)
