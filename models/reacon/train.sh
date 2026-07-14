#!/bin/bash
#SBATCH -J reacon
#SBATCH -o sbatch_run/%j.out
#SBATCH -e sbatch_run/%j.err
#SBATCH -p 5000_ada             # partition name: 5000_ada or 6000_ada or cpu_only
#SBATCH -N 1                    # total number of nodes requested (DO NOT MODIFY)
#SBATCH -n 1                    # Max 24 for GPU-included jobs (per GPU), Max 32 for CPU-only jobs
#SBATCH --ntasks-per-node=1     # Same as -n above
#SBATCH --cpus-per-task=24
#SBATCH --mem=35G
#SBATCH --gres=gpu:5000ada:1
#SBATCH --time=72:00:00         # Max 72hrs. CPU-only jobs Max 48hrs

######################## Conda Environment Name Setting #######################
ENVNAME="tar"
CONDA_ENV_BIN="$HOME/.conda/envs/${ENVNAME}/bin"
############################# Predefined Functions ############################
print_time () { echo "[$(date)] $1"; }

module purge
module load cuda/12.8.1 miniconda/25.7.0

mkdir -p sbatch_run

cd "$SLURM_SUBMIT_DIR" || exit 1

# Use the bundled (patched) chemprop so --patience and CSV logging are available
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
echo "=============================================="
echo "SUBMIT_DATE           = "`date`
echo "SLURM_JOBID           = "$SLURM_JOBID
echo "SLURM_JOB_NAME        = "$SLURM_JOB_NAME
echo "SLURM_JOB_PARTITION   = "$SLURM_JOB_PARTITION
echo "SLURM_JOB_NODELIST    = "$SLURM_JOB_NODELIST
echo "SLURM_NNODES          = "$SLURM_NNODES
echo "SLURM_NTASKS          = "$SLURM_NTASKS
echo "SLURM_NTASKS_PER_NODE = "$SLURM_NTASKS_PER_NODE
echo "SLURMTMPDIR           = "$SLURMTMPDIR
echo "working directory     = "$SLURM_SUBMIT_DIR
echo "=============================================="
echo ""

DATA=../../../Data/MPNN_data
LABEL_DIR=../../../Data/labels

# Read number of classes from label files (NR-1 subtracts the header row)
NUM_CAT=$(awk  'END{print NR-1}' $LABEL_DIR/cat_labels.csv)
NUM_SOLV=$(awk 'END{print NR-1}' $LABEL_DIR/solv_labels.csv)
NUM_REAG=$(awk 'END{print NR-1}' $LABEL_DIR/reag_labels.csv)

echo "Label vocab sizes — cat: $NUM_CAT  solv: $NUM_SOLV  reag: $NUM_REAG"
echo ""

print_time "Training cat ($NUM_CAT classes) ..."
"${CONDA_ENV_BIN}/chemprop_train" \
    --target_columns cat \
    --data_path $DATA/GCN_data_train.csv \
    --separate_val_path $DATA/GCN_data_val.csv \
    --separate_test_path $DATA/GCN_data_test.csv \
    --dataset_type multiclass --multiclass_num_classes $NUM_CAT \
    --save_dir ./GCN_cat \
    --reaction --extra_metrics accuracy --epochs 35 --patience 10 --num_workers 8 \
    --no_cache_mol --batch_size 2000

print_time "Training solv0 ($NUM_SOLV classes) ..."
"${CONDA_ENV_BIN}/chemprop_train" \
    --target_columns solv0 \
    --data_path $DATA/GCN_data_train.csv \
    --separate_val_path $DATA/GCN_data_val.csv \
    --separate_test_path $DATA/GCN_data_test.csv \
    --dataset_type multiclass --multiclass_num_classes $NUM_SOLV \
    --save_dir ./GCN_solv0 \
    --reaction --extra_metrics accuracy --epochs 35 --patience 10 --num_workers 8 \
    --no_cache_mol --batch_size 2000

print_time "Training solv1 ($NUM_SOLV classes) ..."
"${CONDA_ENV_BIN}/chemprop_train" \
    --target_columns solv1 \
    --data_path $DATA/GCN_data_train.csv \
    --separate_val_path $DATA/GCN_data_val.csv \
    --separate_test_path $DATA/GCN_data_test.csv \
    --dataset_type multiclass --multiclass_num_classes $NUM_SOLV \
    --save_dir ./GCN_solv1 \
    --reaction --extra_metrics accuracy --epochs 35 --patience 10 --num_workers 8 \
    --no_cache_mol --batch_size 2000

print_time "Training reag0 ($NUM_REAG classes) ..."
"${CONDA_ENV_BIN}/chemprop_train" \
    --target_columns reag0 \
    --data_path $DATA/GCN_data_train.csv \
    --separate_val_path $DATA/GCN_data_val.csv \
    --separate_test_path $DATA/GCN_data_test.csv \
    --dataset_type multiclass --multiclass_num_classes $NUM_REAG \
    --save_dir ./GCN_reag0 \
    --reaction --extra_metrics accuracy --epochs 35 --patience 10 --num_workers 8 \
    --no_cache_mol --batch_size 2000

print_time "Training reag1 ($NUM_REAG classes) ..."
"${CONDA_ENV_BIN}/chemprop_train" \
    --target_columns reag1 \
    --data_path $DATA/GCN_data_train.csv \
    --separate_val_path $DATA/GCN_data_val.csv \
    --separate_test_path $DATA/GCN_data_test.csv \
    --dataset_type multiclass --multiclass_num_classes $NUM_REAG \
    --save_dir ./GCN_reag1 \
    --reaction --extra_metrics accuracy --epochs 35 --patience 10 --num_workers 8 \
    --no_cache_mol --batch_size 2000

print_time "All done."
