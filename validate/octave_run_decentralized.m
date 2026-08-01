%% Reference run: decentralized training (SAIC/ESAIC), for comparison
%% against jax_saic. Caller script around the UNMODIFIED
%% SAIC/EoC_SAIC_3Agents.m, mirroring the initialization pattern of
%% SAIC/parallel_simulator_2phase_encoded_MA.m but with noa/ns/end_learn
%% parameterizable via pre-set workspace variables instead of hardcoded
%% literals. EoC_SAIC_3Agents.m itself is called exactly as written --
%% its internal load('agreggated_states_n3_g9_infbits2_realSAIC', ...) is
%% satisfied by octave_run_centralized.m having saved that exact
%% filename/variable into the current directory first.

clc

if exist('seed','var'); rng(seed); end

scen=3;
n=3;
if ~exist('noa','var'); noa=3; end
bits=2;
inf_bits=2;
goal_set=[9];
best_rew=10;
worst_rew=1;
if ~exist('ns','var'); ns=10000; end
update_tables=1;
if ~exist('policy','var'); policy="ep_greedy"; end
gamma=0.9;
tau_k=0.005;
if ~exist('end_learn','var'); end_learn=0.80; end
bsc_p=1e-10;  % SS0.5: unused in the bits==inf_bits path EoC_SAIC_3Agents.m takes

ca=zeros(noa,bits);
cs=ones(noa,noa-1,inf_bits);
pa=randi(5,noa,1);
ps=randi(n*n-1,noa,1);
ter=0;
rew=zeros(ns,1);
temp_rew=0;
counter=zeros(ns,1);
qc_table=0.02*ones(noa,n^2,2^inf_bits);  % dead plumbing (PORT_NOTES.md SS0.2/SS10.3), passed through unused
qp_table=0.02*ones(noa,n^2,2^((noa-1)*inf_bits),5);

[rew,qp_table,qc_table,counter,NE_table_emerged] = EoC_SAIC_3Agents(scen,n,noa,ns,bits,inf_bits,best_rew,worst_rew,goal_set,gamma,tau_k,ca,cs,pa,ps,ter,rew,temp_rew,counter,qc_table,qp_table,bsc_p,end_learn,update_tables,policy);

if ~exist('out_file','var'); out_file='decentral_reference.mat'; end
save('-v7', out_file, 'rew', 'n', 'noa', 'ns', 'inf_bits', 'best_rew', 'gamma', 'end_learn', 'policy');
printf('Saved decentralized-training results to %s\n', out_file);
