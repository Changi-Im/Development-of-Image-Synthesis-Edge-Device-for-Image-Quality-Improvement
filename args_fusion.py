
class args():

	# training args
	epochs = 200 #"number of training epochs, default is 2"
	batch_size = 1 #"batch size for training, default is 4"

	dataset1 = "../UR_2/DB/input1" # source 1
	dataset2 = "../UR_2/DB/input2" # source 2
	datasett = "../UR_2/DB/target" # target
	HEIGHT = 256
	WIDTH = 256

	save_model_dir = "models" #"path to folder where trained model will be saved."
	save_loss_dir = "models/loss"  # "path to folder where trained model will be saved."

	image_size = 256 #"size of training images, default is 256 X 256"
	cuda = 0 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"
	ssim_weight = [1,10,100,1000,10000]
	tv_weight = [1,10,100,1000,10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 0.008 #"learning rate, default is 0.001"
	lr_light = 0.008  # "learning rate, default is 0.001"
	log_interval = 5 #"number of images after which the training loss is logged, default is 500"
	resume = "./models/1e3_resume.model"
	resume_auto_en = None
	resume_auto_de = None
	resume_auto_fn = None

	# for test Final_cat_epoch_9_Wed_Jan__9_04_16_28_2019_1.0_1.0.model
	# model_path_gray = "./models/densefuse_gray.model"
	# model_path_rgb = "./models/densefuse_rgb.model"
    
	model_path_gray = "./models/small_best_hdr/1e3_best.model"