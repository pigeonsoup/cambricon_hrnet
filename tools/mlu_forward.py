# coding=utf-8
import os
os.environ['CNRT_PRINT_INFO']='false'  #开启CNRT打印
os.environ['CNRT_GET_HARDWARE_TIME']='false' #打印硬件时间
os.environ['CNML_PRINT_INFO']='false'  #开启CNML打印
os.environ['ATEN_CNML_COREVERSION']='MLU200' #平台选择 MLU200
os.environ['TFU_ENABLE']='1'
os.environ['TFU_NET_FILTER']='0'

import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
# import cv2
from PIL import Image

import torch
import torchvision.transforms as transforms

import _init_paths
import models
from config import config
from config import update_config
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger
from my_transform import ToTensorNoDiv255

 
parser = argparse.ArgumentParser(description='')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
      
parser.add_argument('--cfg', type=str, help='experiment configure file name', required=True)

parser.add_argument('--device', default='cpu', type=str, help='(cpu,mlu)')
parser.add_argument('--core_num', default=1, type=int, help='core num')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--use_half', default=False, type=str2bool, help='use_half')
parser.add_argument('--quantized_mode', default='int8', type=str, help='(int8,int16)')

parser.add_argument('--epoch', default=1, type=int, help='use_half')
parser.add_argument('--weight_dir', default="", type=str, help="directory of weight file, or empty for random value")
parser.add_argument('--model_name', default='', type=str, help='offline_model')
parser.add_argument('--logfile', default='', type=str, help='logfile')
parser.add_argument('--debug', default=False, type=str2bool, help='debug')
parser.add_argument('--firstconv', default=False, type=str2bool, help='firstconv')

parser.add_argument('--color_mode', default='', type=str, help='color_mode[rgb bgr gray]')
parser.add_argument('--image_list', default='', type=str, help='image_list')
parser.add_argument('--mean', default='', type=str, help='mean')
parser.add_argument('--std', default='', type=str, help='std')
# parser.add_argument('--image_size', default='', type=str, help='image_size(w,h)')

parser.add_argument('--onnx_name', default='', type=str, help='save_onnx')

args = parser.parse_args()
update_config(config, args)


# must added
torch.set_grad_enabled(False)

if args.device=="mlu":
    import torch_mlu
    import torch_mlu.core.mlu_model as ct 
    import torch_mlu.core.mlu_quantize as mlu_quantize

################################# AREA TO BE CHANGED ##############################
def create_model():
    # net = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)

    if args.device == "cpu":
        net = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)
    elif args.device == "mlu":
        net = eval("models.cls_hrnet_gai.get_cls_net")(config)

    if len(args.weight_dir) > 0:
        state_dict = torch.load(args.weight_dir, map_location='cpu')
        net.load_state_dict(state_dict, True)
    else:
        assert False  # shouldn't run here
        torch.manual_seed(2)
        for m in net.modules():
            if isinstance(m,torch.nn.Conv2d):
                # torch.nn.init.normal_(m.weight.data, mean=0, std=1)
                torch.nn.init.xavier_normal_(m.weight.data, gain=0.1)
                # torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m,torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=0, std=1)
        torch.save(net.state_dict(), "hrnet_rand.pth")  # Save for later use

    return net.eval().to("cpu")

def dump_output_tensor(net_output):
    outputs = net_output.to("cpu").type(torch.FloatTensor).numpy().reshape(-1)
    dump_file_name="output_{}_{}.txt".format(args.model_name, args.device)
    with open(dump_file_name,"w") as f:
        for i in outputs:
            f.write("%8.8f\n"%i)

################################# END ##############################

def intx_quantification(input_data, model, intx_pth_path, mean, std):
    data_scale = 1
    avg_mode = False
    iteration = 1
    
    quantized_mode=args.quantized_mode
    save_model_path="."

    quantized_model = mlu_quantize.quantize_dynamic_mlu(model,
                                    {
                                        'iteration':iteration,
                                        'use_avg':avg_mode, 
                                        'data_scale':data_scale,
                                        'mean': mean, 
                                        'std':std, 
                                        'per_channel':False,
                                        'firstconv':args.firstconv
                                    }, 
                                    dtype=quantized_mode,
                                    inplace=True,
                                    gen_quant=True)
    quantized_model.to("cpu").eval().float()
    for _ in range(iteration):
        outputs = quantized_model(input_data)

    checkpoint = quantized_model.state_dict()
    torch.save(checkpoint, os.path.join(save_model_path, intx_pth_path))


def prepare_data(images_list, means, stds, batch_size, is_firstconv):
    image_tensor_list = []
    for image_file in images_list:
        img = Image.open(image_file).convert('RGB')
        if is_firstconv:
            # firstconv data
            normalize = transforms.Normalize(mean=means, std=stds)

            preprocess_transforms = transforms.Compose([
                    transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
                    transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
                    ToTensorNoDiv255()])
            image = preprocess_transforms(img).to('cpu')
            # image = torch.repeat_interleave(image, batch_size, dim=0)
        else:
            # normalized data
            normalize = transforms.Normalize(mean=means, std=stds)

            preprocess_transforms = transforms.Compose([
                    transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
                    transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
                    transforms.ToTensor(),
                    normalize])
            image = preprocess_transforms(img).to('cpu')
            # image = torch.repeat_interleave(image, batch_size, dim=0)
        image_tensor_list.append(image)
    return torch.stack(image_tensor_list, dim=0)


def main():
    # image_size = [int(x) for x in args.image_size.split(",")]
    mean = np.array([float(x) for x in args.mean.split(",")])
    std = np.array([float(x) for x in args.std.split(",")])

    if args.debug:
        print("image_size:{}".format(image_size))
        print("mean:{}".format(mean))
        print("std:{}".format(std))
        print("logfile:{}".format(args.logfile))
        print("intx_pth_path:{}".format(intx_pth_path))
        print("offline_model:{}".format(offline_model))
        print("firstconv:{}".format(args.firstconv))

    # build model
    net = create_model()
    net = net.to('cpu')

    with open(args.image_list,"r") as f:
        image_list=[x.strip() for x in f.readlines()]

    if len(image_list) < args.batch_size:
        print("error image_list count:{} < batch_size:{}".format(len(args.image_list), args.batch_size))
        raise ValueError

    # data preparation
    normalized_input_data = prepare_data(image_list, mean, std, args.batch_size, False)
    firstconv_input_data = prepare_data(image_list, mean, std, args.batch_size, True)

    # save onnx model
    if len(args.onnx_name)>0: 
        torch_out = torch.onnx._export(net, normalized_input_data.to("cpu"), args.onnx_name+".onnx", export_params=True, verbose=False, opset_version=11)
    
    if args.device == "cpu":
        print("---------- CPU MODE ----------")
        net_output=net(normalized_input_data)
        #net_traced = torch.jit.trace(net, normalized_input_data, check_trace=False)
        #net_output = net_traced(normalized_input_data)
        
    elif args.device == "mlu":
        print("---------- MLU MODE ----------")
        intx_pth_path="{}_intx.pth".format(args.model_name)
        offline_model="{}_intx_{}_{}".format(args.model_name, args.core_num, args.batch_size)
    
        intx_quantification(normalized_input_data.to("cpu"), net, intx_pth_path, mean, std)

        ct.set_core_number(args.core_num)
        ct.set_core_version('MLU270')
        ct.save_as_cambricon(offline_model)
        ct.set_input_format(0)
        
        net = create_model()
        
        quantized_model = mlu_quantize.quantize_dynamic_mlu(net.to("cpu"))
        quantized_model.load_state_dict(torch.load(intx_pth_path, map_location='cpu'), strict=False)
        quantized_model = quantized_model.eval().float().to(ct.mlu_device())

        if args.firstconv:
            input_data = firstconv_input_data
        else:
            input_data = normalized_input_data

        if args.use_half:
            input_mlu_data = input_data.type(torch.HalfTensor).to(ct.mlu_device())
        else:
            input_mlu_data = input_data.type(torch.FloatTensor).to(ct.mlu_device())

        net_traced = torch.jit.trace(quantized_model, input_mlu_data, check_trace=False)
        net_output = net_traced(input_mlu_data)
        # net_output = quantized_model(input_mlu_data)

    dump_output_tensor(net_output)

    return
    
if __name__ == "__main__":
    main()
    


