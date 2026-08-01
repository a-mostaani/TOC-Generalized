clear
clc
close all

%initialization
trl  = 10^6;
max_e= 20;
e_1  = 1.0 + 0.1;    %energy consumption for movement + communication
e_2  = 0.6+ 0.1;  %energy consumption for hovering + communication
pr_1 = 0.75;  %probability of e_1
             %pr_2 = 1- pr_1;  %probability of hovering and comm.
min_steps = ceil(max_e/e_1); %minimum number of time-steps rquired to cons-
                             %ume the energy
                             
                             
%matrices



%generating Bernoulli random varialbes as far as your energy allows you to
%go ceil(max_e/e_2)
%we generate this r.v.s for trl number of times to compute a distribution.
temp = double(rand(trl,ceil(max_e/e_2))<pr_1);

%Converting the Bernoulli r.v. to the desired r.v.
%in our desired r.v. instead of 0 and 1 we have e_2 and e_1 where e_1>e_2
rv = temp*(e_1-e_2)+e_2;

%We now count the number of time steps required to consume the energy max_e
counter = ones(trl,1)*min_steps ; %counter is started from the minimum
                                        %number of time steps required to
                                        %consume the available energy max_e
enr_sum = ones(trl,1);
for i=1:trl
    enr_sum(i,1) = 1 * sum(rv(i,1:min_steps));% the energy sum is computed
                                              %  accordingly

    while enr_sum(i,1) < max_e
        enr_sum(i,1) = enr_sum(i,1) + rv(i,counter(i)+1);        
        counter(i) = counter(i)+1;
    end
end


[f,x]=hist(counter,20,'Normalization','probability');
bar(x,f/sum(f));
uniq=unique(counter);

ll=length(uniq);
prob=zeros(ll,1);
for i=1:ll
    prob(i)= length(find(counter==uniq(i)))/length(counter);
end
