function [cs,err] = bsc_ch_j(cs_bc,bsc_p)
err=0;

%On its current form the function changes
%   Detailed explanation goes here
[bits,noa]=size(cs_bc);
cs=zeros(bits,noa);
for i=1:noa
    for j=1:bits
        if rand<= 1-bsc_p   %if no error
            cs(j,i)=cs_bc(j,i); %if noa increases, this line should be revisited! #noa
        else                %if error ocurres
            cs(j,i)=~cs_bc(j,i);
            err=1;

        end
    end
end

end

