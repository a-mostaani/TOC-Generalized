function [cs,err] = bsc_ch(cs_bc,bsc_p)
err=0;

%On its current form the function changes
%   Detailed explanation goes here
[noa,not_used,bits]=size(cs_bc);
cs=zeros(noa,noa-1,bits);
for i=1:noa
    for j=1:bits
        if rand<= 1-bsc_p
            cs(i,1,j)=cs_bc(i,1,j); %if noa increases, this line should be revisited! #noa
        else
            cs(i,1,j)=~cs_bc(i,1,j);
            err=1;

        end
    end
end

end

