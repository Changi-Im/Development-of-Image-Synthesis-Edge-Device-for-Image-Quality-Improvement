# test phase
import torch
from torch.autograd import Variable
from net_small import DenseFuse_net
import utils
from args_fusion import args
import numpy as np
import time
import cv2
import os
import homography_warp
from binary_fractions import Binary

def binary_weight(weights):
    np.set_printoptions(precision=32)
    torch.set_printoptions(precision=32)
    
    for name in weights.keys():
        precision = 15  # set the precision of Binary to float
        l = np.array([])  # numpy array for float weight
        ls = np.array([])  # numpy array for binary str weight
        a, b, c, d = weights[name].shape  # shape of each original tensor
        
        l = np.append(l, weights[name].numpy())
       
        for i in range(len(l)):
            tmp = Binary.from_float(l[i])
            if tmp[0] == '-':
                tmp = Binary.to_float(tmp[:precision])
                if tmp < 0:
                    ls = np.append(ls, tmp)
                else:
                    ls = np.append(ls, (-1)*tmp)
            else:
                tmp = Binary.to_float(tmp[:precision-1])
                ls = np.append(ls, tmp)
        
        ls = ls.reshape(a, b, c, d)
        weights[name] = torch.Tensor(ls)
    
    return weights

def binary_pixel(img):
    np.set_printoptions(precision=32)
    torch.set_printoptions(precision=32)
    
    r, c = img.squeeze().shape
    img = img.reshape(r*c).numpy()
    l = np.array([])
    precision = 15  # set the precision of Binary to float
    for i in range(len(img)):
        tmp = Binary.from_float(float(img[i]))
        tmp = Binary.to_float(tmp[:precision-1])
        
        l = np.append(l, tmp)
    
    l = l.reshape(1,1,r,c)
    l = torch.Tensor(l)
    
    return l

def load_model(path, input_nc, output_nc):

	nest_model = DenseFuse_net(input_nc, output_nc)
    
	weights = torch.load(path)
	weights = binary_weight(weights)
	nest_model.load_state_dict(weights)
     
	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()

	return nest_model

def export_pixel(img1, img2):
    patch1 = img1.reshape(-1).astype("float64")
    patch2 = img2.reshape(-1).astype("float64")
    
    source1 = np.array([])
    source2 = np.array([])
    final = np.array([])
    
    for i in range(1600):
        i1 = Binary(patch1[i]).components()[1]
        i2 = Binary(patch2[i]).components()[1]
        
        f1 = Binary(patch1[i]).components()[2]
        f2 = Binary(patch2[i]).components()[2]
        
        if f1 == "":
            f1 = "000000000000"
            
        if f2 == "":
            f2 = "000000000000"
            
        a = "00000000000000000"+i1
        b = "00000000000000000"+i2
        source1 = np.append(source1, "storage1[{}] = 14'b{}_{};".format(i, a[-9:],"0000"))
        source2 = np.append(source2, "storage1[{}] = 14'b{}_{};".format(1600+i, b[-9:],"0000"))
        
    source1 = np.insert(source1,0,"// image 1")
    source2 = np.insert(source2,0,"// image 2")
    
    final = np.concatenate((source1, source2))
    
    
    np.savetxt("./pixel(14bit).txt", final, fmt='%s')

def _generate_fusion_image(model, strategy_type, img1, img2, ep):
	if ep:
		export_pixel(img1.numpy().squeeze(), img2.numpy().squeeze())
    
	img1 = binary_pixel(img1)
	img2 = binary_pixel(img2)
    
	# encoder
	en_r = model.encoder1(img1)
	en_v = model.encoder2(img2)
	
	# fusion
	f = model.fusion(en_r, en_v, strategy_type=strategy_type)
	
	# decoder
	img_fusion = model.decoder(f)
	return img_fusion[0]


def run_demo(model, unexp_path, ovexp_path, output_path_root, index, fusion_type, network_type, strategy_type, model_type, ssim_weight_str, mode, set_align, ep, ps):    
	start = time.time()
	if mode == 'LAB':
		if set_align:
			un_img, ov_img, a1, a2, b1, b2 = homography_warp(unexp_path, ovexp_path)
		else:
			un_img, a1, b1 = utils.get_test_images(unexp_path, height=None, width=None, mode=mode)
			ov_img, a2, b2 = utils.get_test_images(ovexp_path, height=None, width=None, mode=mode)
	elif mode == 'L':
		un_img = utils.get_test_images(unexp_path, height=None, width=None, mode=mode)
		ov_img = utils.get_test_images(ovexp_path, height=None, width=None, mode=mode)
    
	cnt = 0
    
	if ps:
		res = np.zeros((540,720,3))
		un_img1 = un_img
		ov_img1 = ov_img
        
		a11 = a1
		b11 = b1
		a22 = a2
		b22 = b2
		for i in range(0,18):
			for j in range(0,14):
				cnt = cnt + 1
				if j == 4:
					un_img = un_img1[:,:,j*40:,i*40:(i+1)*40]
					ov_img = ov_img1[:,:,j*40:,i*40:(i+1)*40]
					
					a1 = a11[j*40:,i*40:(i+1)*40]
					a2 = a22[j*40:,i*40:(i+1)*40]
					
					b1 = b11[j*40:,i*40:(i+1)*40]
					b2 = b22[j*40:,i*40:(i+1)*40]
				else:
					un_img = un_img1[:,:,j*40:(j+1)*40,i*40:(i+1)*40]
					ov_img = ov_img1[:,:,j*40:(j+1)*40,i*40:(i+1)*40]
                    
					a1 = a11[j*40:(j+1)*40,i*40:(i+1)*40]
					a2 = a22[j*40:(j+1)*40,i*40:(i+1)*40]
					
					b1 = b11[j*40:(j+1)*40,i*40:(i+1)*40]
					b2 = b22[j*40:(j+1)*40,i*40:(i+1)*40]

				if args.cuda:
					un_img = un_img.to('cuda',dtype=torch.half)
					ov_img = ov_img.to('cuda',dtype=torch.half)
					
				un_img = Variable(un_img, requires_grad=False)
				ov_img = Variable(ov_img, requires_grad=False)
				dimension = un_img.size()
				img_fusion = _generate_fusion_image(model, strategy_type, un_img, ov_img, ep)
                
				############################ multi outputs ##############################################
				file_name = str(index) + model_type + str(cnt) + '.jpg'
				output_path = output_path_root + file_name

				if args.cuda:
					img = img_fusion.clamp(0, 255).data[0].cpu().numpy()
				else:
					img = img_fusion.clamp(0, 255).data[0].numpy()
											
				img = img.transpose(1, 2, 0).astype('uint8')

				if mode == 'LAB':
					if img.shape[2] == 1:
						img = img.reshape([img.shape[0], img.shape[1]])
					a = a1/2 + a2/2
					b = b1/2 + b2/2
					
					lab = cv2.merge([img,a.astype('uint8'),b.astype('uint8')])
					rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
					
					if ps:
						if j == 4:
							res[j*40:,i*40:(i+1)*40,:] = rgb
						else:
							res[j*40:(j+1)*40,i*40:(i+1)*40] = rgb
					
					cv2.imwrite(output_path, res)
				else:
					utils.save_images(output_path, img)
				print(output_path)

	else:   
		if args.cuda:
			un_img = un_img.to('cuda',dtype=torch.half)
			ov_img = ov_img.to('cuda',dtype=torch.half)
            
		img_fusion = _generate_fusion_image(model, strategy_type, un_img, ov_img, ep)
		
		############################ multi outputs ##############################################
		file_name = str(index) + model_type + str(cnt) + '.jpg'
		output_path = output_path_root + file_name

		if args.cuda:
			img = img_fusion.clamp(0, 255).data[0].cpu().numpy()
		else:
			img = img_fusion.clamp(0, 255).data[0].numpy()
									
		img = img.transpose(1, 2, 0).astype('uint8')
		
		if mode == 'LAB':
			if img.shape[2] == 1:
				img = img.reshape([img.shape[0], img.shape[1]])
			a = a1/2 + a2/2
			b = b1/2 + b2/2
			
			lab = cv2.merge([img,a.astype('uint8'),b.astype('uint8')])
			rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
			cv2.imwrite(output_path, rgb)
			
		else:
			utils.save_images(output_path, img)
		
		print("elapsed time:",round(time.time()-start,4))
		print(output_path)


def vision_features(feature_maps, img_type):
	count = 0
	for features in feature_maps:
		count += 1
		for index in range(features.size(1)):
			file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
			output_path = 'outputs/feature_maps/' + file_name
			map = features[:, index, :, :].view(1,1,features.size(2),features.size(3))
			map = map*255
			# save images
			utils.save_image_test(map, output_path)


def main():
	# run demo
	# test_path = "images/test-RGB/"
	test_path = "images/"
	network_type = 'densefuse'
	fusion_type = 'auto'  # auto, fusion_layer, fusion_all
	strategy_type_list = ['addition', 'attention_weight']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask
	model_type = '_HyperP_MB'
	set_align = False	# if you don't want to use homography warp to align source images, set this value to 'False'
	ep = True	# if you don't want to export binary pixel data, set this value to 'False'
	ps = False


	output_path = './outputs/'
	strategy_type = strategy_type_list[0]

	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	in_c = 1
	if in_c == 1:
		out_c = in_c
		mode = 'LAB'
		model_path = args.model_path_gray

	with torch.no_grad():
		print('SSIM weight ----- ' + args.ssim_path[3])
		ssim_weight_str = args.ssim_path[3]
		model = load_model(model_path, in_c, out_c)
		for i in range(5,6):
			index = i
			unexp_path = test_path + str(index) + '_UN' + '.jpg'
			ovexp_path = test_path + str(index) + '_OV' + '.jpg'
			run_demo(model, unexp_path, ovexp_path, output_path, index, fusion_type, network_type, strategy_type, model_type, ssim_weight_str, mode, set_align, ep, ps)
	print('Done......')

if __name__ == '__main__':
	main()
