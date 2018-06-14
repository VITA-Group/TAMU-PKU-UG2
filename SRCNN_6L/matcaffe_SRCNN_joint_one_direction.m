function [ im_final2 ] = matcaffe_rain_joint_one_direction(net, rain_image)

    pad = 80;
    im_lr = rain_image;
    im_lr = padarray(im_lr,[pad, pad],'replicate','both');

    [input_height, input_width, ch] = size(im_lr);

    im_final  = zeros(input_height, input_width, ch);

    crop_step = 80;
    move_step = 60;
    border = 10;

    [rh,rw,~] = size(im_final);

    im_weight = double(zeros(input_height, input_width));
    im_final2 = zeros(size(im_final));

    for ii = 1:move_step:rh
        if ii+crop_step-1 <= rh
            tmpi = ii;
        else
            tmpi = rh - crop_step + 1;
        end

        for jj=1:move_step:rw
            if jj+crop_step-1 <= rw
                tmpj = jj;
            else
                tmpj = rw - crop_step +1;
            end

            tmp_lr = im_lr(tmpi:tmpi+crop_step-1,tmpj:tmpj+crop_step-1,:);


            [ temp_res ] = matcaffe_SRCNN_joint(net, tmp_lr);

            im_final2(tmpi+border:tmpi+crop_step-1-border,tmpj+border:tmpj+crop_step-1-border,:) = im_final2(tmpi+border:tmpi+crop_step-1-border,tmpj+border:tmpj+crop_step-1-border,:) + temp_res(1+border:end-border, 1+border:end-border, :);

        end
    end

im_final2 = im_final2(pad+1:end-pad, pad+1:end-pad, :);
