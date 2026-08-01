function pa=pa_calc(main_pa,noa)
% Verbatim from benchmark_perfectcom_MultiAgent.m -- split into its own
% file only because this Octave build doesn't support MATLAB-style local
% functions at the end of a script file (confirmed with an isolated
% minimal test, not specific to this script).
pa=zeros(noa,1);
counter = 0;
    for kk = noa:-1:1
        main_pa = main_pa - counter;
        pa(kk) = ceil((main_pa)/(5^(kk-1)))  ;
        counter = (pa(kk)-1)*(5^(kk-1));
    end
end
