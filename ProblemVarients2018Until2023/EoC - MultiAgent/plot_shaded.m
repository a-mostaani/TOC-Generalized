function plot_shaded(rew,mov_mean_len,point_num,optim_var)
%rew is the singnal to be plotted

%mov_mean_len is the length of moving average applied on the singnal before
%being plotted

%point_num is the number of points that remains from the averaged signal to
%be plotted (number of remaining downsampled points)
%the variance that you get in the case of the centralized Q-learning:

%for n=3,noa=3,goal_point=9 : the recommended value is 1.22: this is the


%is there any optimal variance you may want to reduce from the obtained
%variance? In some cases the variance cannot be avoided. The best variance
%you can get is optim_var. Then we reduce optim_var from the obtained var

    %apply moving average
    aver_rew = movmean(rew,mov_mean_len);
    
    %down sample
    desc_aver_rew = aver_rew(1:length(aver_rew)/point_num:length(aver_rew));
    
    %x axis values
    x = length(aver_rew)/point_num:length(aver_rew)/point_num:length(aver_rew);
    
    %obtain the variance function
    var_vect = zeros(point_num,1);
    for i = 1:point_num
        var_vect(i) = var( rew((i-1)*length(aver_rew)/point_num +1 : (i)*length(aver_rew)/point_num)+1 );        
    end
    
    %reduce the optim_var from the simulation var
    temp_var = var_vect - optim_var; 
    temp_optim_var = heaviside(temp_var) .* (optim_var * ones(point_num,1));
    ploted_var = var_vect - temp_optim_var;
    
%     if length(x) ~=  ploted_var/2
%     end
    figure
    area(x ,[desc_aver_rew - ploted_var/2, ploted_var/2, ploted_var/2])
    hold on
    plot(x , desc_aver_rew)
    
    
    figure
    std = sqrt(ploted_var);
    area(x ,[desc_aver_rew - std/2, std/2, std/2])
    hold on
    plot(x , desc_aver_rew)
    
    
end

