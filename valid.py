#source /home/torch/env_pytorch.sh
#source /home/torch/pytorch/src/catch/venv/pytorch/bin/activate
#python tools/valid.py --cfg experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel hrnet_w18_small_model_v1.pth
#python tools/mlu_valid.py --cfg=experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel=hrnet_w18_small_model_v1.pth --image_list="images.list" --batch_size=4 --model_name="hrnet_w18_small_v1" --firstconv=True --core_num=4 --use_half=True --quantized_mode="int8" --device=mlu
#python tools/mlu_valid.py --cfg=experiments/cls_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel=hrnet_w18_small_model_v2.pth --image_list="images.list" --batch_size=4 --model_name="hrnet_w18_small_v2" --firstconv=True --core_num=4 --use_half=True --quantized_mode="int8" --device=mlu
function valid()
{
	device=$1
	batch_size=$2
	core_num=$3
	model_name=$4
	onnx_name=$5
	firstconv=$6
	weight_dir=$7
	
	python3 tools/mlu_valid.py --cfg=experiments/cls_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml \
								--data_dir=/data/imagenet/images/val/ \
								--weight_dir=$weight_dir --image_list="images.list" \
								--color_mode="rgb" --mean="0.485,0.456,0.406" --std="0.229,0.224,0.225" \
								--batch_size=$batch_size --core_num=$core_num --device=$device --quantized_mode="int8" \
								--use_half=True --epoch=1 --model_name=$model_name --firstconv=$firstconv \
								--logfile="" --onnx_name=$onnx_name --debug=False 
}

export TFU_ENABLE=1
export TFU_NET_FILTER=0
export MLU_VISIBLE_DEVICES="0"
python3 -c "import torch;print(torch.__version__,torch.__file__)"
python3 -c "import torch_mlu"

export ROOT=`pwd`
export name=`basename $ROOT`

WEIGTHDIR="./hrnet_w18_small_model_v2.pth"
#valid cpu 4 4 "hrnet_w18_small_v2" "hrnet_w18_small_v2" False $WEIGTHDIR | grep -v "CN*"
valid mlu 4 4 "hrnet_w18_small_v2" "" True $WEIGTHDIR | grep -v "CN*"
# /tmp/cnrtexec $name 0 hrnet_w18_small_v2_intx_4_4 subnet0 512 8 1 output_hrnet_w18_small_v2_mlu.txt output_hrnet_w18_small_v2_cpu.txt
# rm -f output_hrnet_w18_small_v2_mlu.txt output_hrnet_w18_small_v2_cpu.txt

