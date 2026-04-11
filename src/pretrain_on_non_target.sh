#!/bin/bash
#PBS -N pretrain_on_non_target
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=60gb:gpu_cap=sm_70
#PBS -l walltime=2:00:00

PROJECT=KNN
PROJECT_DIR=/storage/brno2/home/$USER/$PROJECT

echo "$PBS_JOBID running on `hostname -f` scratch $SCRATCHDIR" >> $PROJECT_DIR/jobs_info.txt

test -n "$SCRATCHDIR" || { echo "SCRATCHDIR not set"; exit 1; }

module purge
source $PROJECT_DIR/mer_env/bin/activate

echo "mer_env activated" >> $PROJECT_DIR/jobs_info.txt

export HF_HOME=$SCRATCHDIR/hf_cache
export TRANSFORMERS_CACHE=$SCRATCHDIR/hf_cache
export HF_DATASETS_CACHE=$SCRATCHDIR/hf_cache

mkdir -p $SCRATCHDIR/$PROJECT
cp -r $PROJECT_DIR/* $SCRATCHDIR/$PROJECT

cd $SCRATCHDIR/$PROJECT

echo "learning started" >> $PROJECT_DIR/jobs_info.txt

python src/pretrain_on_non_target.py \
    --batch_size 16 \
    --lr 5e-5 \
    --epochs 10 \
    --project_dir $SCRATCHDIR/$PROJECT \
    --output_dir $PROJECT_DIR/results \
    --freeze_encoder \
    --num_beams 4 \
    --max_length 256 \
    --early_stopping \
    --warmup_ratio 0.1

echo "learning ended" >> $PROJECT_DIR/jobs_info.txt

rm -rf $SCRATCHDIR/*