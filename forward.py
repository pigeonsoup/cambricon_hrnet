#!/bin/bash

function forward()
{
	device=$1
	batch_size=$2
	core_num=$3
	model_name=$4
	onnx_name=$5
	firstconv=$6
	weight_dir=$7
	
	python3 tools/mlu_forward.py --cfg experiments/cls_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml \
								--weight_dir=$weight_dir --image_list="images.list" \
								--color_mode="rgb" --mean="0.485,0.456,0.406" --std="0.229,0.224,0.225" \
								--batch_size=$batch_size --core_num=$core_num --device=$device --quantized_mode="int8" \
								--use_half=True --epoch=1 --model_name=$model_name --firstconv=$firstconv \
								--logfile="" --onnx_name=$onnx_name --debug=False 
}

export TFU_ENABLE=1
export TFU_NET_FILTER=0
export MLU_VISIBLE_DEVICES=0
python3 -c "import torch;print(torch.__version__,torch.__file__)"
python3 -c "import torch_mlu"

export ROOT=`pwd`
export name=`basename $ROOT`

WEIGTHDIR="./hrnet_w18_small_model_v2.pth"
#forward cpu 4 4 "hrnet_w18_small_v2" "hrnet_w18_small_v2" False $WEIGTHDIR | grep -v "CN*"
forward mlu 4 4 "hrnet_w18_small_v2" "" True $WEIGTHDIR | grep -v "CN*"
#/tmp/cnrtexec $name 0 hrnet_w18_small_v2_intx_4_4 subnet0 512 8 1 output_hrnet_w18_small_v2_mlu.txt output_hrnet_w18_small_v2_cpu.txt
#rm -f output_hrnet_w18_small_v2_mlu.txt output_hrnet_w18_small_v2_cpu.txt

