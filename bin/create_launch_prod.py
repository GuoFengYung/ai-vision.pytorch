import os
import stat

path_to_bin_dir = os.path.dirname(os.path.realpath(__file__))
path_to_launch_sh = os.path.join(path_to_bin_dir, 'launch-prod.sh')

with open(path_to_launch_sh, 'w') as f:
    path_to_launcher_py = os.path.join(os.path.dirname(path_to_bin_dir), 'webservice', 'launcher.py')
    path_to_envs_dir = os.path.join(os.path.dirname(path_to_bin_dir), 'envs')
    path_to_flask = os.path.join(path_to_envs_dir, 'mvision', 'bin', 'flask')
    path_to_activate = os.path.join(path_to_envs_dir, 'mvision', 'bin', 'activate')
    f.write('#!/usr/bin/env bash\n')
    f.write(f'source {path_to_activate}\n')
    f.write(f'env FLASK_ENV=production FLASK_APP={path_to_launcher_py} {path_to_flask} run --host 0.0.0.0 --port 5000')

st = os.stat(path_to_launch_sh)
os.chmod(path_to_launch_sh, st.st_mode | stat.S_IEXEC)
