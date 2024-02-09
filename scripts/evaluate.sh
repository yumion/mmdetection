work_dir=$1
config=$work_dir/vis_data/config.py
checkpoint=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../tools/test.py \
    $config \
    $checkpoint \
    --work-dir $work_dir \
    --out $work_dir/vis_data/pred.pkl \
    --show-dir $work_dir/vis_data/pred \
    --cfg-options "test_evaluator.classwise=True"
