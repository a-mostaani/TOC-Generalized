%generate the random vector with the arbitrary length sam_size:
sam_size=5000;




% load('V_o_1.mat','V_o_1','N_o_n')
% V_o_w=zeros(sam_size+length(V_o_1),1);
% cntr=0;
% 
% 
% for i=1:length(V_o_1)
%     ind_temp = ceil(sam_size * N_o_n(i));
%     V_o_w(cntr+1:cntr+ind_temp) = V_o_1(i)*ones(ind_temp,1);
%     cntr = cntr + ind_temp;
% end
% 
% zero_inds=find(V_o_w==0);
% V_o_w=V_o_w(1:zero_inds(1)-1);
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% %positive listening:
% figure
% z(1,:) = [1.0000    1.5850    2.0000    2.3219].*betarnd(5,0.8,1,4)*0.8*0.8;
% z(2,:) = [1.0000    1.5850    2.0000    2.3219].*betarnd(5,0.5,1,4)*0.9*0.8;
% z(3,:) = [1.0000    1.5850    2.0000    2.3219].*betarnd(5,0.2,1,4)*0.99*0.8;
% z(4,:) = [1.0000    1.5850    2.0000    2.3219].*betarnd(5,0.5,1,4)*0.93*0.8;
% z(5,:) = 0.8*[1.0000    1.0000    1.0000    1.0000];
% z(6,:) = [0.3253    0.3253    0.3253    0.3253];
% plot(transpose(z))
% legend("d=1","d=2","d=3","SAIC","HOC","MI")



%NEW vector
%Adjust length:
ns = 300000;
length_extension = 200000;
extention_type = "strech"; % "strech" or "continue"

avging_length = 110000;

avg_rew = movmean(rew,avging_length);
end_val = avg_rew(end);

new_ns = ns + length_extension;
if extention_type == "continue"
    rew_2 = [rew;end_val*ones(length_extension,1)];
else
    rew_2 = rew;
end
    


%Initialization
vec_length = 30;
max = 1;
%max = 4.79;
achievability = 0.74;



%Discretize:
rew_orig = movmean(rew_2, ns*.2);
x = 1:new_ns/vec_length:new_ns;
rew_orig_desc = rew_orig(1:new_ns/vec_length:new_ns);

%normalize
rew_orig_desc_normal = rew_orig_desc/mean(rew_orig_desc(end-3:end))*max*achievability;

%Plots
figure
plot(x,rew_orig_desc_normal)

figure
plot(x*3,rew_orig_desc_normal)
hold on
plot(3*x,max*ones(1,vec_length))
if extention_type == "strech"
    plot(x*3,rew_orig_desc_normal)
end
