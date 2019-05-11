function classifySpam = classifySpam(filename, svmModel)
  file_contents = readFile(filename);
  word_indices  = processEmail(file_contents);
  x             = emailFeatures(word_indices);
  classifySpam  = svmPredict(svmModel, x);
endfunction
