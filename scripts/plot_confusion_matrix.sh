work_dir=$1
config=$work_dir/vis_data/config.py
predict_pkl=$work_dir/vis_data/pred.pkl


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../tools/analysis_tools/confusion_matrix.py \
    $config \
    $predict_pkl \
    $work_dir/vis_data \
    --color-theme Blues \
    --font-color k

