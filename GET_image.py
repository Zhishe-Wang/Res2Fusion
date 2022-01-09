# test phase
import torch
from torch.autograd import Variable
from Net import Rse2Net_atten_fuse
from utils import list_images,make_floor
import utils
from args_fusion import args
import numpy as np
import os

def load_model(path,  output_nc):

	model = Rse2Net_atten_fuse(output_nc)
	model.load_state_dict(torch.load(path,map_location=lambda storage, loc:storage))
	para = sum([np.prod(list(p.size())) for p in model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

	model.eval()
	model.cpu()

	return model


def _generate_fusion_image(model,img_ir,img_vis,strategy_type):
	en_ir = model.encoder(img_ir)
	en_vis = model.encoder(img_vis)
	feat = model.fusion_atten(en_ir, en_vis, strategy_type)
	img_fusion = model.decoder(feat)
	return img_fusion


def run_demo(model, infrared_path, visible_path, output_path_root, index,  network_type, strategy_type, mode):
	# prepare data
	ir_img = utils.get_test_images(infrared_path, height=None, width=None, mode=mode)
	vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode)
	if args.cuda:
		ir_img = ir_img.cpu()
		vis_img = vis_img.cpu()
	ir_img = Variable(ir_img, requires_grad=False)
	vis_img = Variable(vis_img, requires_grad=False)

	# fuse images
	img_fusion = _generate_fusion_image(model, ir_img, vis_img,strategy_type)

	# save images
	if args.cuda:
		img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()
	if index<=9:
		file_name1 =network_type + "_" + strategy_type +'_100' + str(index)+  '.png'
		output_path1 = output_path_root + '/'+ file_name1
		utils.save_images(output_path1, img)
		print(output_path1)
	else:
		file_name2 = network_type + "_" + strategy_type + '_10' + str(index) + '.png'
		output_path2 = output_path_root +'/'+ file_name2
		utils.save_images(output_path2, img)
		print(output_path2)

def main():
	# run demo
	test_path_ir = "images/TNO/thermal"
	test_path_vi = "images/TNO/visual"
	infrared_paths = list_images(test_path_ir)
	visible_paths = list_images(test_path_vi)

	network_type = 'Res2Fusion'
	strategy_type_list = ['add' , 'atten']
	strategy_type = strategy_type_list[1]
	output_root = 'outputs'
	output_path = make_floor(make_floor(os.getcwd(), output_root), strategy_type)

	in_c = 1
	out_c = in_c
	mode = 'L'
	model_path = "./models/1e3/Final_epoch_4_1e3.model" # ssim weight is 1

	with torch.no_grad():
		print('SSIM weight ----- ' + args.ssim_path[0])
		model = load_model(model_path, out_c)
		for i in range(25):
			index =1+i
			run_demo(model, infrared_paths[i], visible_paths[i], output_path, index, network_type,strategy_type,  mode)
		print('Done......')

if __name__ == '__main__':
	main()
