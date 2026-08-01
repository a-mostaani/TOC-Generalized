function main_ps = mps_calc(ps,n)
% Verbatim from benchmark_perfectcom_MultiAgent.m -- split into its own
% file for the same Octave-compatibility reason as pa_calc.m. Unused by
% the trimmed driver (ps_calc/pa_calc cover what's actually called) but
% kept for completeness/fidelity to the original file's contents.
    main_ps= (ps(1)-1)*n^2+ps(2);
end
