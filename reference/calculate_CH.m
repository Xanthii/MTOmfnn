function calculate_CH(filename_load, filename_save, n_elx, n_sample)
    x = load(filename_load);
    x_list = x.x_list;
    CH = zeros(100,3,3);
    E = 1;
    v = 0.3;
    lambda_strain = v*E/((1+v)*(1-2*v));
    nu = E/(2*(1+v));
    lambda_stress = 2*nu*lambda_strain/(lambda_strain+2*nu);
    for i = 1 : n_sample
        x = reshape(x_list(i,:,:),n_elx,n_elx);
        CH_i = homogenize(1, 1, lambda_stress, nu, 90, x);
        CH(i,:,:) = CH_i;
    end
    save(filename_save,'CH');