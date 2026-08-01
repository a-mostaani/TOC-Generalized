function ps=ps_calc(main_ps,n,noa)
% Verbatim from benchmark_perfectcom_MultiAgent.m -- split into its own
% file for the same Octave-compatibility reason as pa_calc.m.
ps=zeros(2,1);
counter = 0;
    for kk = noa:-1:1
        main_ps = main_ps - counter;
        ps(kk) = ceil((main_ps)/(n^2^(kk-1)))  ;
        counter = (ps(kk)-1)*(n^2^(kk-1));
    end
end
