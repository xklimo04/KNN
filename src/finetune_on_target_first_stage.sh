#!/bin/bash
#PBS -N finetune_on_target
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=60gb:gpu_cap=sm_70
#PBS -l walltime=01:00:00
#PBS -j oe

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
cp -r $PROJECT_DIR/src $SCRATCHDIR/$PROJECT/
cp -r $PROJECT_DIR/results $SCRATCHDIR/$PROJECT/
cp -r $PROJECT_DIR/target $SCRATCHDIR/$PROJECT/

cd $SCRATCHDIR/$PROJECT

echo "learning started" >> $PROJECT_DIR/jobs_info.txt

python -m src.finetune_on_target \
    --batch_size 8 \
    --lr 1e-5 \
    --epochs 10 \
    --project_dir $SCRATCHDIR/$PROJECT \
    --output_dir $PROJECT_DIR/results_finetuned_first_stage \
    --num_beams 4 \
    --max_length 256 \
    --early_stopping \
    --target_path $SCRATCHDIR/$PROJECT/target \
    --model_dir results/best_model

echo "learning ended" >> $PROJECT_DIR/jobs_info.txt

rm -rf $SCRATCHDIR/*