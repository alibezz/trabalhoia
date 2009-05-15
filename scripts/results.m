function results

files = dir('*p*');

for i = 1:length(files)
  tmp = struct2cell(carac(files(i).name));
  vetor = [tmp{:}];
  save('results', 'vetor', '-ASCII', '-append');
end