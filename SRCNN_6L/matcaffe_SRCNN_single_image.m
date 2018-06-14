function [result] = matcaffe_SRCNN_single_image(net, im)
input_data = permute(im,[1,2,3,4]);
batch_size = 1;
[h1,h2,h3,h4] = size(input_data);

input_final = input_data;
[height,width] = size(input_final);

net.blobs('data').reshape([height width 1 1]);
scores = net.forward({input_final});

result = scores{1};
