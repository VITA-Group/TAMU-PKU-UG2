gpu_id = 0

% Set caffe mode
if exist('gpu_id', 'var') && gpu_id
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();
end

model_dir = './SRCNN_model/';
net_model = [model_dir 'deploy_SR_6L.prototxt'];
net_weights = [model_dir 'SR_ImageNet_6L_iter_300000.caffemodel'];

phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
    error('Please download CaffeNet from Model Zoo before you run this demo');
end

net = caffe.Net(net_model, net_weights, phase);

name = 'SRCNN_6L_upsample';
scale = 3;


input_dir = './image/';
suffix = 'png';
output_dir = './result/';
        
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

files=dir([input_dir, '*.', suffix]);
m = size(files,1);
avg_psnr = 0;

for i=1:m
    [filepath, filename, ext] = fileparts(files(i).name);
    image_name = [input_dir, files(i).name];
    fprintf('%s %d/%d \n', image_name, i, m);
    image = imread(image_name);
    [hei, wid, ch] = size(image);

    if ch==3
        image = rgb2ycbcr(image);
        image_cb = image(:,:,2);
        image_cr = image(:,:,3);
    end

    image = double(image(:,:,1));

    [input_height, input_width, ~] = size(image);

    image = imresize(image, scale, 'bicubic');

    im_lr = image/255;

    [tmp_result] = matcaffe_SRCNN_joint_one_direction(net, im_lr);

    im_final = imresize(tmp_result, 1/scale, 'bicubic');

    save_path = [output_dir, filename, '.png'];
    
    if ch==3
        im_final = cat(3, uint8(im_final*255), image_cb, image_cr);
        im_final = ycbcr2rgb(im_final);
    else
        im_final = cat(3, uint8(im_final*255), uint8(im_final*255), uint8(im_final*255));
    end

    imwrite(uint8(im_final), save_path);

end

