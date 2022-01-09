# Training DenseFuse network
# auto-encoder

import os
import sys
import time
import numpy as np
from tqdm import trange
import scipy.io as scio
import random
import torch

from torch.optim import Adam
from torch.autograd import Variable
import utils
from Net import Rse2Net_atten_fuse
from args_fusion import args
import pytorch_msssim
from utils import make_floor

def main():
	original_imgs_path = utils.list_images(args.dataset)
	train_num = len(original_imgs_path)
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	for i in range(4):
		train(i, original_imgs_path)


def train(i, original_imgs_path):

	batch_size = args.batch_size

	# load network model
	in_c = 1 # 1 - gray
	img_model = 'L'
	output_nc = in_c
	Rse2Net_atten_model = Rse2Net_atten_fuse(output_nc)
	print(Rse2Net_atten_model)

	ssim_loss = pytorch_msssim.msssim
	mse_loss = torch.nn.MSELoss()


	if args.cuda:
		Rse2Net_atten_model.cuda()


	tbar = trange(args.epochs)
	print('Start training.....')

	temp_path_model = make_floor(make_floor(os.getcwd(), args.save_model_dir), args.ssim_path[i])
	temp_path_loss = make_floor(temp_path_model, 'loss')


	Data_SSIM = []
	Loss_pixel= []
	Loss_ssim = []
	Loss_all = []
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	args.lr = 1e-4
	for e in tbar:
		epch = e+1
		print('Epoch %d.....' % epch)
		if epch  != 1:
			args.lr =args.lr /10
		print( "\nlearning_rate:",args.lr)
		print()
		optimizer = Adam(Rse2Net_atten_model.parameters(), args.lr)
		image_set, batches = utils.load_dataset(original_imgs_path, batch_size)
		Rse2Net_atten_model.train()
		count = 0
		for batch in range(batches):
			image_paths = image_set[batch * batch_size:(batch * batch_size + batch_size)]
			img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model)
			count += 1
			optimizer.zero_grad()
			img = Variable(img, requires_grad=False)
			if args.cuda:
				img = img.cuda()

			# get fusion image
			en = Rse2Net_atten_model.encoder(img)
			output = Rse2Net_atten_model.decoder(en)

			# loss
			x = Variable(img.data.clone(), requires_grad=False)
			pixel_loss_value = mse_loss(output,x)
			ssim_loss_temp = ssim_loss(output, x, normalize=True)
			ssim_loss_value = (1 - ssim_loss_temp)
			# total loss
			total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value
			total_loss.backward()
			optimizer.step()

			all_ssim_loss += ssim_loss_value.item()
			all_pixel_loss += pixel_loss_value.item()
			if (batch + 1) % args.log_interval == 0: # args.log_interval = 50
				mesg = "{}\tEpoch {}:\t[{}/{}]\t  pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  all_ssim_loss / args.log_interval,
								  (all_pixel_loss + args.ssim_weight[i] * all_ssim_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((all_pixel_loss + args.ssim_weight[i] * all_ssim_loss) / args.log_interval)
				Data_SSIM.append(1-(all_ssim_loss / args.log_interval))

				all_ssim_loss = 0.
				all_pixel_loss = 0
		Rse2Net_atten_model.eval()
		Rse2Net_atten_model.cpu()
		save_model_filename =  "Epoch_" + str(e) + "_iters_" + str(count) + "_" +  args.ssim_path[i] + ".model"
		save_model_path = os.path.join(temp_path_model, save_model_filename)
		torch.save(Rse2Net_atten_model.state_dict(), save_model_path)
		# save loss data
		# pixel loss
		loss_data_pixel_part = np.array(Loss_pixel)
		loss_filename_path = "loss_pixel_epoch_" + str(e) + "_iters_" + str(count) + "_" + args.ssim_path[i] + ".mat"
		save_loss_path = os.path.join(temp_path_loss, loss_filename_path)
		scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel_part})
		# SSIM loss
		loss_data_ssim_part = np.array(Loss_ssim)
		loss_filename_path =  "loss_ssim_epoch_" + str(e) + "_iters_" + str(count) + "_" + args.ssim_path[i] + ".mat"
		save_loss_path = os.path.join(temp_path_loss, loss_filename_path)
		scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim_part})
		# Data SSIM
		Data_SSIM_part = np.array(Data_SSIM)
		loss_filename_path = "Data_SSIM_epoch_" + str(e) + "_iters_" + str(count) + "_" + args.ssim_path[i] + ".mat"
		save_loss_path = os.path.join(temp_path_loss, loss_filename_path)
		scio.savemat(save_loss_path, {'Data_SSIM': Data_SSIM_part})
		# all loss
		loss_data_total_part = np.array(Loss_all)
		loss_filename_path = "loss_total_epoch_" + str(e) + "_iters_" + str(count) + "_" + args.ssim_path[i] + ".mat"
		save_loss_path = os.path.join(temp_path_loss, loss_filename_path)
		scio.savemat(save_loss_path, {'loss_total': loss_data_total_part})

		Rse2Net_atten_model.train()
		Rse2Net_atten_model.cuda()
		tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# pixel loss
	loss_data_pixel = np.array(Loss_pixel)
	loss_filename_path =  "Final_loss_pixel_epoch_" + str(args.epochs) + args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(temp_path_loss, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_pixel':loss_data_pixel})
	# SSIM loss
	loss_data_ssim = np.array(Loss_ssim)
	loss_filename_path =  "Final_loss_ssim_epoch_" + str(args.epochs) + args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(temp_path_loss, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
	# Data SSIM
	Data_SSIM_path = np.array(Data_SSIM)
	loss_filename_path = "Final_Data_SSIM_epoch_" + str(args.epochs) + args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(temp_path_loss, loss_filename_path)
	scio.savemat(save_loss_path, {'Data_SSIM': Data_SSIM_path})
	# all loss
	loss_data_total = np.array(Loss_all)
	loss_filename_path =  "Final_loss_total_epoch_" + str(args.epochs) + args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(temp_path_loss, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_total': loss_data_total})
	# save model
	Rse2Net_atten_model.eval()
	Rse2Net_atten_model.cpu()
	save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(temp_path_model, save_model_filename)
	torch.save(Rse2Net_atten_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
	main()
