#!/usr/bin/env bash

# init conda dirs to avoid filling up $HOME
module load python3
conda config --add envs_dirs $WORK/conda/envs
conda config --add pkgs_dirs $WORK/conda/pkgs
conda config --set auto_activate_base false
eval "$(conda shell.bash activate base)"

# install env
mamba env create -n impfilt -f environment.yml

# install implicit filter in env
eval "$(conda shell.bash activate impfilt)"
python -m pip install .

# install jupyter kernel
python -m ipykernel install --user --name impfilt --display-name="impfilt"

# override kernel def with one that ensures proper activation of the env
cat > $HOME/.local/share/jupyter/kernels/impfilt/kernel.json <<EOF
{
 "argv": [
  "conda",
  "run",
  "-n",
  "impfilt",
  "python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "impfilt",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
EOF