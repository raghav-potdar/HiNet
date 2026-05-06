#!/bin/bash -l

##   Select a cluster, partition, qos and account that is appropriate for your use case
##   Available options and more details are provided in README
#SBATCH --cluster=ub-hpc
#SBATCH --partition=class
#SBATCH --qos=class
#SBATCH --account=cse676

##   Job runtime limit, the job will be canceled once this limit is reached. Format- hr:min:sec 
#SBATCH --time=24:00:00

##   Number of "tasks" (for parallelism). Refer to DOCUMENTATION for more details
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

##   Specify real memory required per node. Default units are megabytes
#SBATCH --mem=20G

##   Number of nodes
#SBATCH --nodes=1

##   Number of gpus per node. Refer to snodes output for breakdown of node capabilities
#SBATCH --gpus-per-node=1

##   Let's start some work

batch="Noise-50-95-LR-2e-6-Resume2000"

folder_name=$(date +"%Y-%m-%d_%H-%M-%S")_"$batch"

scratch_path="/scratch/rpotdar/$folder_name"

hinet_path="$scratch_path/HiNet"

cd /scratch

if [ ! -d "/scratch/rpotdar" ]; then
    echo "rpotdar doesn't exist, Creating...."
    mkdir -p rpotdar
fi

cd rpotdar

mkdir -p "$folder_name"

cp -rp /projects/academic/courses/cse676s26/rpotdar/DataCopy/HiNet "$folder_name"/.

cd "$hinet_path"

source .venv/bin/activate

cd datasets

unzip DIV2K_train_HR.zip
unzip DIV2K_valid_HR.zip

cd ..

python -c "import sys; exit(0 if sys.prefix != sys.base_prefix else 1)" && {
    # your code here
    echo "Running inside venv..."

    module load gcc/11.2.0 openmpi/4.1.1

    module load pytorch/0.13.1-CUDA-11.8.0 torchvision/0.14.1-CUDA-11.8.0 pillow/9.2.0 jax/0.4.16-CUDA-11.8.0

    module load tqdm

    nvidia-smi

    python3 IsCUDAAvailable.py
    
    #python3 main.py --sanity

    #python3 main.py --overfit_one_batch --batch_size 16

    #python3 main.py --epochs 2000 --batch_size 16
    #python3 main.py --epochs 2000 --batch_size 16 --checkpoint_every 10 --val_freq 10 --lr 3.16e-5

    #Added noise
    #python3 main.py --epochs 1000 --batch_size 16 --checkpoint_every 10 --val_freq 10 --resume checkpoints/hinet_best.pth --lr 1e-6 --start_epoch 0 --noise --jpeg_quality_min 70 --jpeg_quality_max 95
    python3 main.py --epochs 1000 --batch_size 16 --checkpoint_every 10 --val_freq 10 --resume model/hinet_trained_till_2000.pth --lr 2e-6 --start_epoch 0 --noise --jpeg_quality_min 50 --jpeg_quality_max 95

    cd /projects/academic/courses/cse676s26/rpotdar/run
    mkdir "$folder_name"

    cp -rp "$hinet_path"/results "$hinet_path"/checkpoints "$folder_name/"

    rm -rf "$scratch_path"

    echo "Task Completed"

} || echo "Not in venv"

##   Let's finish some work
