import os

path_to_bin_dir = os.path.dirname(os.path.realpath(__file__))
path_to_launch_sh = os.path.join(path_to_bin_dir, 'launch-prod.sh')

with open('mirle-vision.service', 'w') as f:
    f.write('[Unit]\n')
    f.write('Description=mirle-vision\n')
    f.write('\n')
    f.write('[Service]\n')
    f.write(f'ExecStart=/bin/bash {path_to_launch_sh}\n')
    f.write('\n')
    f.write('[Install]\n')
    f.write('WantedBy=multi-user.target')
