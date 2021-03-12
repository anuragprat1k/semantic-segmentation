import os 
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.sdc_net2d import *

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to trained video reconstruction checkpoint')
parser.add_argument('--flownet2_checkpoint', default='', type=str, metavar='PATH', help='path to flownet-2 best checkpoint')
parser.add_argument('--source_dir', default='', type=str, help='directory for data (default: Cityscapes root directory)')
parser.add_argument('--target_dir', default='', type=str, help='directory to save augmented data')
parser.add_argument('--sequence_length', default=2, type=int, metavar="SEQUENCE_LENGTH",
                    help='number of interpolated frames (default : 2)')
parser.add_argument("--rgb_max", type=float, default = 255.)
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--propagate', type=int, default=3, help='propagate how many steps')
parser.add_argument('--vis', action='store_true', default=False, help='augment color encoded segmentation map')

def get_model():
	model = SDCNet2DRecon(args)
	torch.cuda.set_device(0)
	checkpoint = torch.load(args.pretrained)
	args.start_epoch = 0 if 'epoch' not in checkpoint else checkpoint['epoch']
	state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
	model.load_state_dict(state_dict, strict=False)
	print("Loaded checkpoint '{}' (at epoch {})".format(args.pretrained, args.start_epoch))
	return model

from PIL import Image

def get_data(img1_dir, img2_dir, img3_dir, gt2_color_dir):

	img1_rgb = cv2.imread(img1_dir)
	img2_rgb = cv2.imread(img2_dir)
	img3_rgb = cv2.imread(img3_dir)
	# gt2_rgb = cv2.imread(gt2_color_dir)
	print("seg path {}".format(gt2_color_dir))
	img = Image.open(gt2_color_dir)
	gt2_rgb = np.array(img, dtype='uint8')
	gt2_rgb = np.reshape(gt2_rgb, gt2_rgb.shape + (1,))
	# gt2_rgb = cv2.cvtColor(gt2_rgb,  cv2.COLOR_GRAY2BGR)
	print("{} gt2_rgb shape {}".format(gt2_color_dir, gt2_rgb.shape))
	
	# img1_rgb = cv2.resize(img1_rgb, (1024, 2048))
	# img2_rgb = cv2.resize(img2_rgb, (1024, 2048))																																															
	# img3_rgb = cv2.resize(img3_rgb, (1024, 2048))
	# gt2_rgb =  cv2.resize(gt2_rgb, (1024, 2048))

	# print("gt2_rgb shape {}".format(gt2_rgb.shape))
	# cv2.imwrite("test_reading.jpg", img1_rgb)
	# cv2.imwrite("test_reading_gt.jpg", gt2_rgb)

	img1_rgb = img1_rgb.transpose((2,0,1))
	img2_rgb = img2_rgb.transpose((2,0,1))
	img3_rgb = img3_rgb.transpose((2,0,1))
	gt2_rgb = gt2_rgb.transpose((2,0,1))

	img1_rgb = np.expand_dims(img1_rgb, axis=0)
	img2_rgb = np.expand_dims(img2_rgb, axis=0)
	img3_rgb = np.expand_dims(img3_rgb, axis=0)
	gt2_rgb = np.expand_dims(gt2_rgb, axis=0)

	img1_rgb = torch.from_numpy(img1_rgb.astype(np.float32))
	img2_rgb = torch.from_numpy(img2_rgb.astype(np.float32))
	img3_rgb = torch.from_numpy(img3_rgb.astype(np.float32))	
	gt2_rgb = torch.from_numpy(gt2_rgb.astype(np.float32))

	return img1_rgb, img2_rgb, img3_rgb, gt2_rgb

prop_length = 10
next_start = 60
def one_step_augmentation(model, rgb_prefix, mask_prefix, sequence_prefix, split, mode, reverse):
	global prop_length
	global next_start
	split_dir = os.path.join(args.source_dir, rgb_prefix, split)
	# scenes in root/rgb_prefix/test
	scenes = os.listdir(split_dir)
	scenes.sort()
	for scene in scenes:
		# if scene != "stuttgart_00":
		if scene != "rgb":
			continue
		print("Augmenting %s for mode %s" % (scene, mode))
		scene_dir = os.path.join(split_dir, scene)
		# root/rgb_prefix/test/berlin
		frames = os.listdir(scene_dir)
		frames.sort()
		
		for frame in frames:
			def get_frames_dg_vid1(frame):
				seq_info = frame.split("_")
				idx = 3
				seq_id2 = seq_info[idx]
				print("seq_info {}, seq_id2 {}".format(seq_info, seq_id2))
				if int(seq_id2) < 2:
					return None, None, None, None
				seq_id1 = "%06d" % (int(seq_id2) - 1)
				seq_id3 = "%06d" % (int(seq_id2) + 1)
				im1_name = "_".join(seq_info[:idx]) + "_" + seq_id1 + "_" + "_".join(seq_info[idx+1:])
				im3_name = "_".join(seq_info[:idx]) + "_" + seq_id3 + "_" + "_".join(seq_info[idx+1:])
				color_gt_name = "_".join(seq_info[:-1]) + "_gtFine.jpg"
				color_gt_name3 = "_".join(seq_info[:idx]) + "_" + seq_id3 + "_" + "_".join(seq_info[idx+1:-1]) + "_gtFine.jpg"
				return im1_name, im3_name, color_gt_name, color_gt_name3
			
			def get_frames_rgb(frame):
				seq_info = frame.split(".")
				idx = 0
				seq_id2 = seq_info[idx]
				print("seq_info {}, seq_id2 {}".format(seq_info, seq_id2))
				if int(seq_id2) < next_start:
					return None, None, None, None, None
				seq_id1 = "%05d" % (int(seq_id2) - 1)
				seq_id3 = "%05d" % (int(seq_id2) + 1)
				im1_name = seq_id1+'.jpg'
				im3_name = seq_id3+'.jpg' 
				color_gt_name = seq_id2 + ".png"
				color_gt_name3 = seq_id3 + ".png"
				return im1_name, im3_name, color_gt_name, color_gt_name3, int(seq_id3)

			im1_name, im3_name, color_gt_name, color_gt_name3, seq_id3 = get_frames_rgb(frame)
			if im1_name is None:
				continue
			
			source_im1 = os.path.join(args.source_dir, sequence_prefix, split, scene, im1_name)
			source_im2 = os.path.join(scene_dir, frame)
			source_im3 = os.path.join(args.source_dir, sequence_prefix, split, scene, im3_name)
			
			source_color = os.path.join(args.source_dir, mask_prefix, split, scene, color_gt_name)
			print("source_im1 {} \nsource_im2 {} \nsource_im3 {}, \nsource_color {}".format(source_im1, source_im2, source_im3, source_color))

			if not os.path.isfile(source_im1):
				print("%s does not exist" % (source_im1))
				sys.exit()
			if not os.path.isfile(source_im3):
				print("%s does not exist" % (source_im3))
				sys.exit()

			if not reverse:
				img1_rgb, img2_rgb, img3_rgb, gt2_rgb = get_data(source_im1, source_im2, source_im3, source_color)
			else:
				img1_rgb, img2_rgb, img3_rgb, gt2_rgb = get_data(source_im3, source_im2, source_im1, source_color)

			img1_rgb = Variable(img1_rgb).contiguous().cuda()
			img2_rgb = Variable(img2_rgb).contiguous().cuda()
			img3_rgb = Variable(img3_rgb).contiguous().cuda()
			gt2_rgb = Variable(gt2_rgb).contiguous().cuda()
			input_dict = {}
			input_dict['image'] = [img1_rgb, img2_rgb, img3_rgb]

			print("img shape {}".format(img1_rgb.size()))
			# print("gt2 min, max ", np.max(gt2_rgb, 0), np.min(gt2_rgb, 0))
			
			def infer_w(model, input_dict, label_image, fname):
				_, pred_g, _ = model(input_dict, label_image=label_image)
				pred = pred_g.clone().detach()
				pred = ( pred.data.cpu().numpy().squeeze().transpose(1,2,0)).astype(np.uint8)
				cv2.imwrite(fname, pred)
				sys.exit()

			# if mode == "rgb_image":
			# 	infer_w(model, input_dict, img2_rgb, "img_pred.jpg")
			
			# elif mode == "color_segmap":
			# 	infer_w(model, input_dict, gt2_rgb, "map_pred.jpg")


			if mode == "rgb_image":
				_, pred3_rgb, _ = model(input_dict, label_image=img2_rgb)
				pred3_rgb_img = ( pred3_rgb.data.cpu().numpy().squeeze().transpose(1,2,0) ).astype(np.uint8)
				
				if not os.path.exists(os.path.join(args.target_dir, rgb_prefix, split, scene)):
					os.makedirs(os.path.join(args.target_dir, rgb_prefix, split, scene))
				target_im1 = os.path.join(args.target_dir, rgb_prefix, split, scene, im1_name)
				target_im2 = os.path.join(args.target_dir, rgb_prefix, split, scene, frame)
				target_im3 = os.path.join(args.target_dir, rgb_prefix, split, scene, im3_name)
				
				# if not reverse:
				# 	shutil.copyfile(source_im2, target_im2)

				if not reverse:
					print("writing to {}".format(target_im3))
					cv2.imwrite(target_im3, pred3_rgb_img)
				else:
					print("writing to {}".format(target_im1))
					cv2.imwrite(target_im1, pred3_rgb_img)
			elif mode == "color_segmap":
				_, pred3_colormap, _ = model(input_dict, label_image=gt2_rgb)
				print("np shape ", pred3_colormap.data.cpu().numpy().shape)
				pred3_colormap_img = ( pred3_colormap.data.cpu().numpy().squeeze()).astype(np.uint8)
				prop_length -= 1

				if prop_length == 0:
					prop_length = 10
					next_start += 120

				# print(pred3_colormap_img)
				# if not os.path.exists(os.path.join(args.target_dir, mask_prefix, split, scene)):
				# 	os.makedirs(os.path.join(args.target_dir, mask_prefix, split, scene))

				# target_color = os.path.join(args.source_dir, mask_prefix, split, scene, color_gt_name)
				
				# if not reverse:
				# 	shutil.copyfile(source_color, target_color)
				# color_gt_name = "_".join(seq_info[:-1]) + "_gtFine.jpg"
				# color_gt_name3 = "_".join(seq_info[:idx]) + "_" + seq_id3 + "_" + "_".join(seq_info[idx+1:])

				import png
				if not reverse:
					# Write if pred not in whitelisted ids
					if seq_id3 % 60 == 0:
						continue # don't write
					print("Writing predicted mask to {}".format(color_gt_name3))
					target_color3 = os.path.join(args.source_dir, mask_prefix, split, scene, color_gt_name3)
					# pred3_colormap_img = cv2.resize(pred3_colormap_img, (512, 512))
					print("predicted mask shape {}".format(pred3_colormap_img.shape))
					png.from_array(pred3_colormap_img, 'L').save(target_color3)
					# cv2.imwrite(target_color3, pred3_colormap_img)
					# sys.exit()
				else:
					color_gt_name1 = seq_info[0] + "_" + seq_info[1] + "_" + seq_id1 + "_gtFine.jpg"
					target_color1 = os.path.join(args.target_dir, mask_prefix, split, scene, color_gt_name1)
					cv2.imwrite(target_color1, pred3_colormap_img)
			else:
				print("Mode %s is not supported." % (mode))
				sys.exit()



if __name__ == '__main__':
	global args
	args = parser.parse_args()

	# Load pre-trained video reconstruction model
	net = get_model()
	net.eval()
	net = net.cuda()

	# Config paths
	if not os.path.exists(args.target_dir):
		os.makedirs(args.target_dir)

	rgb_prefix = "leftImg8bit_blurred"
	sequence_prefix = "leftImg8bit_blurred"
	mask_prefix = "masks"
	split = "demoVideo"

	# Generate augmented dataset
	# create +-1 data
	# one_step_augmentation(net, rgb_prefix, mask_prefix, sequence_prefix, split, 'rgb_image', False)
	one_step_augmentation(net, rgb_prefix, mask_prefix, sequence_prefix, split, 'color_segmap', False)
	# one_step_augmentation(net, rgb_prefix, sequence_prefix, split, 'rgb_image', True)
	
