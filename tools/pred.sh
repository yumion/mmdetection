CHECKPOINT=$1
NB_GPU=$2
TARGET_DIR="$3"
WORK_DIR=$(dirname $CHECKPOINT)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/inference.py \
    $CHECKPOINT \
    $NB_GPU \
    --target-dir "$TARGET_DIR" \
    --out $WORK_DIR/endovis2023
    ${@:4}
