#!/usr/bin/env bash

# an example to do pre-training
# > cd /path/to/MDM
# > bash scripts/pretrain.sh experiment_name

####### template begins #######
SCRIPTS_DIR=/path/to/MDM
cd "${SCRIPTS_DIR}"
MDM_DIR=/path/to/MDM/experiments
echo "MDM_DIR=${MDM_DIR}"

shopt -s expand_aliases
alias python=python3
alias to_scripts_dir='cd "${SCRIPTS_DIR}"'
alias to_MDM_DIR='cd "${MDM_DIR}"'
alias print='echo "$(date +"[%m-%d %H:%M:%S]") (exp.sh)=>"'
function mkd() {
  mkdir -p "$1" >/dev/null 2>&1
}
####### template ends #######


EXP_NAME=$1

EXP_DIR="${MDM_DIR}/output_${EXP_NAME}"


print "===================== Args ====================="
print "EXP_NAME: ${EXP_NAME}"
print "[other_args sent to launch.py]: ${*:2}"
print "================================================"
print ""


print "============== Pretraining starts =============="
to_scripts_dir
touch ~/wait1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=4,5,6,7 python launch.py \
--main_py_relpath main.py \
--exp_name "${EXP_NAME}" \
--exp_dir "${EXP_DIR}" \
--num_nodes=1 \
--ngpu_per_node=4 \
--node_rank=0 \
--master_address=128.0.1.4 \
--master_port=1019 \
--data_path=/path/to/MDM/dataset/adni_mni152_affine_clean.txt \
--atlas_path=/path/to/MDM \
--model=unet \
--opt=adamw \
--bs=48 \
--ep=100 \
--wp_ep=10 \
--base_lr=2e-4 \
--dataloader_workers=8 \
--input_size=96 \
--mask=0.6
print "============== Pretraining ends =============="
rm ~/wait1