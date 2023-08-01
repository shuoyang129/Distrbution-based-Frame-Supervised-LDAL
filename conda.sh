conda create -n DBFS python=3.7 -y
source activate
conda activate DBFS
conda install pytorch=1.10.0 cudatoolkit=11.3.1 -c pytorch -y
pip install numpy scipy pyyaml tqdm matplotlib