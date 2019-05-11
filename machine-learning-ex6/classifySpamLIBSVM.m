function classifySpamLIBSVM = classifySpamLIBSVM(filename, svmModel)
  file_contents = readFile(filename);
  word_indices  = processEmail(file_contents);
  x             = emailFeatures(word_indices)';
  y=zeros(1,1);
  classifySpamLIBSVM  = svmpredict(y, x, svmModel, '-q');
endfunction
