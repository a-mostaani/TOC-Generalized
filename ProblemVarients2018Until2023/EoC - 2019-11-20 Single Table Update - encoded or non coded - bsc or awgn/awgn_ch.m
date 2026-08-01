function [cs] = awgn_ch(cs_bc,snr)
%err=0;


[noa,~,bits]=size(cs_bc);
cs=zeros(noa,1,bits);

%% changing 0s to -1
% We do it to raise the power of the signal to 1
for i=1:noa
    temp=cs_bc(i,1,:)==0;
    cs_bc(i,1,:)= cs_bc(i,1,:)+temp*-1;
end

%% AWGN channel
ch_out=zeros(noa,1,bits);
for i=1:noa
    ch_out(i,1,:)=awgn(cs_bc(i,1,:),snr);
end

%% Detection: Threshold signal=0 above 0 is 1 beneath zero is 0
for i=1:noa
    %temp0=ch_out(i,1,:)<=0;
    temp1=ch_out(i,1,:)>0;
    cs(i,1,:)=temp1;
end

end

