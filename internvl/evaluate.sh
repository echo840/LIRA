set -x

CONFIG=${1}
PTH_MODEL=${2}
DATASET=${3}
# CONFIG="$(pwd)/${CONFIG}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CONFIG: ${CONFIG}"

MASTER_PORT=${MASTER_PORT:-63665}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}


if  [ ${DATASET} == "mme" ]; then
  DIRNAME=`basename ${CONFIG}`
  python internvl/eval/mme/eval.py --config ${CONFIG} --pth_model  ${PTH_MODEL} ${@:4}
  python internvl/eval/mme/calculation.py --results_dir ./internvl/eval/mme/output_files
  cd ../../
fi

if  [ ${DATASET} == "caption" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/caption/evaluate_caption.py --config ${CONFIG} --pth_model  ${PTH_MODEL} ${@:4}
fi

if  [ ${DATASET} == "caption-coco" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/caption/evaluate_caption.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets coco ${@:4}
fi

if  [ ${DATASET} == "caption-flickr30k" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/caption/evaluate_caption.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets flickr30k ${@:4}
fi

if  [ ${DATASET} == "caption-nocaps" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/caption/evaluate_caption.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets nocaps ${@:4}
fi

if [ ${DATASET} == "vqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} ${@:4}
fi

if [ ${DATASET} == "chair" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/chair/evaluate_chair.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets chair_test ${@:4}
    python ./internvl/eval/chair/chair.py --cap_file ./internvl/results/chair_results.jsonl --image_id_key image_id --caption_key caption --coco_path ./data/chair/annotations   --save_path ./data/chair/results/chair_saved.jsonl
fi

if [ ${DATASET} == "vqa-okvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets okvqa_val ${@:4}
fi

if [ ${DATASET} == "vqa-textvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets textvqa_val ${@:4}
fi

if [ ${DATASET} == "vqa-textvqa-val-ocr" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets textvqa_val_ocr ${@:4}
fi

if [ ${DATASET} == "vqa-vizwiz-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets vizwiz_val ${@:4}
fi

if [ ${DATASET} == "refcoco_attri" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets refcoco_attri ${@:4}
fi

if [ ${DATASET} == "refcocog_attri" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets refcocog_attri ${@:4}
fi

if [ ${DATASET} == "vqa-vizwiz-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets vizwiz_test ${@:4}
fi

if [ ${DATASET} == "vqa-vqav2-testdev" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets vqav2_testdev ${@:4}
fi

if [ ${DATASET} == "vqa-ai2d-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets ai2diagram_test ${@:4}
fi

if [ ${DATASET} == "vqa-vqav2-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets vqav2_val ${@:4}
fi

if [ ${DATASET} == "vqa-gqa-testdev" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets gqa_testdev_llava ${@:4}
fi

if [ ${DATASET} == "vqa-docvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets docvqa_val ${@:4}
fi

if [ ${DATASET} == "vqa-docvqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets docvqa_test ${@:4}
fi

if [ ${DATASET} == "vqa-chartqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets chartqa_test_human,chartqa_test_augmented ${@:4}
fi

if [ ${DATASET} == "vqa-infovqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets infographicsvqa_val ${@:4}
fi

if [ ${DATASET} == "vqa-infovqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets infographicsvqa_test ${@:4}
fi

if [ ${DATASET} == "vqa-chartqa-test-human" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets chartqa_test_human ${@:4}
fi

if [ ${DATASET} == "vqa-chartqa-test-augmented" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets chartqa_test_augmented ${@:4}
fi

if [ ${DATASET} == "vqa-ocrvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets ocrvqa_val ${@:4}
fi

if [ ${DATASET} == "vqa-ocrvqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/vqa/evaluate_vqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets ocrvqa_test ${@:4}
fi

if [ ${DATASET} == "refcoco" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/refcoco/evaluate_grounding.py --config ${CONFIG} --pth_model  ${PTH_MODEL} ${@:4}
fi

if [ ${DATASET} == "refcoco-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/refcoco/evaluate_grounding.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets refcoco_val ${@:4}
fi

if [ ${DATASET} == "llava-bench" ]; then
    rm -rf results/llava_bench_results_review.jsonl
    python internvl/eval/llava_bench/evaluate_llava_bench.py --config ${CONFIG} --pth_model  ${PTH_MODEL} ${@:4}
    python -u internvl/eval/llava_bench/eval_gpt_review_bench.py \
      --question data/llava-bench-in-the-wild/questions.jsonl \
      --context data/llava-bench-in-the-wild/context.jsonl \
      --rule internvl/eval/llava_bench/rule.json \
      --answer-list \
          data/llava-bench-in-the-wild/answers_gpt4.jsonl \
          results/llava_bench_results.jsonl \
      --output \
          results/llava_bench_results_review.jsonl
    python -u internvl/eval/llava_bench/summarize_gpt_review.py -f results/llava_bench_results_review.jsonl
fi

if [ ${DATASET} == "pope" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/pope/evaluate_pope.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets pope ${@:4}
fi

if [ ${DATASET} == "tiny_lvlm" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/eval/tiny_lvlm/evaluate_lvlm.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets updated_datasets ${@:4}
fi

if [ ${DATASET} == "mmvet" ]; then
    python internvl/eval/mmvet/evaluate_mmvet.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets mmvet ${@:4}
fi

if [ ${DATASET} == "cmmmu" ]; then
  CUDA_VISIBLE_DEVICES=0 python internvl/eval/cmmmu/evaluate_cmmmu.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets art_and_design ${@:4} &
  CUDA_VISIBLE_DEVICES=1 python internvl/eval/cmmmu/evaluate_cmmmu.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets business ${@:4} &
  CUDA_VISIBLE_DEVICES=2 python internvl/eval/cmmmu/evaluate_cmmmu.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets health_and_medicine ${@:4} &
  CUDA_VISIBLE_DEVICES=3 python internvl/eval/cmmmu/evaluate_cmmmu.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets humanities_and_social_sciences ${@:4} &
  CUDA_VISIBLE_DEVICES=4 python internvl/eval/cmmmu/evaluate_cmmmu.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets science ${@:4} &
  CUDA_VISIBLE_DEVICES=5 python internvl/eval/cmmmu/evaluate_cmmmu.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets technology_and_engineering ${@:4} &
  wait
fi

if [ ${DATASET} == "mmbench-dev-en" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/mmbench/evaluate_mmbench.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets mmbench_dev_20230712 ${@:4}
fi

if [ ${DATASET} == "mmbench-dev-cn" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/mmbench/evaluate_mmbench.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets mmbench_dev_cn_20231003 ${@:4}
fi

if [ ${DATASET} == "mmbench-test-en" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/mmbench/evaluate_mmbench.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets mmbench_test_en_20231003 ${@:4}
fi

if [ ${DATASET} == "mmbench-test-cn" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/mmbench/evaluate_mmbench.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets mmbench_test_cn_20231003 ${@:4}
fi

if [ ${DATASET} == "ccbench-dev" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/mmbench/evaluate_mmbench.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets ccbench_dev_cn ${@:4}
fi

if [ ${DATASET} == "scienceqa" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/scienceqa/evaluate_scienceqa.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets sqa_test ${@:4}
fi


if [ ${DATASET} == "mmmu-dev" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/mmmu/evaluate_mmmu.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets MMMU_dev ${@:4}
fi

if [ ${DATASET} == "mmmu-val" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/mmmu/evaluate_mmmu.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets MMMU_validation ${@:4}
fi

if [ ${DATASET} == "mmmu-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/mmmu/evaluate_mmmu.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets MMMU_test ${@:4}
fi


if [ ${DATASET} == "mmvp" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/mmvp/evaluate_mmvp.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets MMVP ${@:4}
fi


if [ ${DATASET} == "mathvista-testmini" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/mathvista/evaluate_mathvista.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets MathVista_testmini ${@:4}
fi


if [ ${DATASET} == "mathvista-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/mathvista/evaluate_mathvista.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets MathVista_test ${@:4}
fi

if [ ${DATASET} == "seed" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/eval/seed/evaluate_seed.py --config ${CONFIG} --pth_model  ${PTH_MODEL} --datasets SEEDv1 ${@:4}
fi
