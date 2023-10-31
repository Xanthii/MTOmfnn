function calculate_CH(filename_load, filename_save, n_elx, n_sample)
    x = load(filename_load);
    x_list = x.x_list;
    CH = zeros(100,3,3);
    for i = 1 : n_sample
        x = reshape(x_list(i,:,:),n_elx,n_elx);
        CH_i = homogenize(1, 1, 2, 4, 90, x);
        CH(i,:,:) = CH_i;
    end
    save(filename_save,'CH');