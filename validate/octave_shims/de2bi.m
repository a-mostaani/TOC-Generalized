function b = de2bi(d, n)
  % Minimal MATLAB Communications-Toolbox-compatible de2bi, 'right-msb'
  % convention (MATLAB's default): b(1) is the LEAST significant bit.
  % Only the (D, N) two-argument form is implemented -- that's all
  % EoC_SAIC_3Agents.m ever calls (bits==inf_bits path, PORT_NOTES.md SS0.5).
  d = d(:);
  b = zeros(numel(d), n);
  v = d;
  for k = 1:n
    b(:, k) = mod(v, 2);
    v = floor(v / 2);
  end
end
