import os

def list_files(directory):
	fileList = [f for f in sorted(os.listdir(directory))]
	# filePath = [os.path.join(directory, f) for f in os.listdir(directory)]
	return fileList

data_folder = '../GOPRO_dataset/test_modified'
# folders = list_files(data_folder)
# folders = [os.path.join(data_folder, f) for f in folders]
folder_a = os.path.join(data_folder, 'blur')
folder_b = os.path.join(data_folder, 'sharp')
files = list_files(folder_a)
file_num = 500
# os.system("source /usr/local/torch3/bin/activate")
arg = ' --fold_A  {}  --fold_B  {}  --fold_AB  ../GOPRO_dataset/test_combined  --num_imgs  {}'.format(folder_a, folder_b, file_num)
os.system('python ./datasets/combine_A_and_B.py' + arg)
pass
