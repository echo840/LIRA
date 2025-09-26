CHECKPOINT=${1}
PTH_MODEL=${2}


# vqa
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  vqa-vizwiz-val  --dynamic --max-num 6 --min-num 4 ${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  vqa-gqa-testdev  --dynamic --max-num 6 --min-num 4 ${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  vqa-vqav2-val --dynamic --max-num 6 --min-num 4 ${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  vqa-ai2d-test --dynamic --max-num 6 --min-num 4 ${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  vqa-okvqa-val --dynamic --max-num 6 --min-num 4 ${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  scienceqa --dynamic --max-num 6 --min-num 1  ${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  refcoco_attri --dynamic --max-num 6 --min-num 4 ${@:3}


# bench
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  pope --dynamic --max-num 8 --min-num 4 ${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  chair  --dynamic --max-num 6 --min-num 1 ${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  mmbench-test-en --dynamic --max-num 6 --min-num 4 ${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  mmbench-test-cn --dynamic  --max-num 6 --min-num 1${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  tiny_lvlm --dynamic --max-num 8 --min-num 6 ${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  mme --dynamic --max-num 12 --min-num 4 ${@:3}
GPUS=1 bash internvl/evaluate.sh ${CHECKPOINT} ${PTH_MODEL}  seed --dynamic --max-num 6 --min-num 4 ${@:3}


