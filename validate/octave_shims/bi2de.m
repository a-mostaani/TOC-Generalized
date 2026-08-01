function d = bi2de(b)
  % Minimal MATLAB Communications-Toolbox-compatible bi2de, 'right-msb'
  % convention (MATLAB's default): b(:,1) is the LEAST significant bit.
  [rows, n] = size(b);
  powers = 2 .^ (0:n-1);
  d = b * powers(:);
end
