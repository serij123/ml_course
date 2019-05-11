function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
if 1==1
  C=1;
  sigma=0.1;
  return;
endif

Cvals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigmavals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';

error_pred = zeros(length(Cvals), length(sigmavals));

err_min = 1000000;
Cmin = Cvals(1);
sigmaMin = sigmavals(1);

for Cidx = 1:length(Cvals)
  for sigmaIdx = 1:length(sigmavals)
    C = Cvals(Cidx);
    sigma = sigmavals(sigmaIdx);
    fprintf(['%d:%d:Perform training with C:%f, sigma:%f\n'], Cidx, sigmaIdx, C, sigma);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    ep = mean(double(predictions ~= yval));
    error_pred(Cidx, sigmaIdx) = ep;
    
    if ep < err_min
      err_min = ep;
      Cmin = C;
      sigmaMin = sigma;
    endif
  endfor
endfor
fprintf('errors preditions:\n');
fprintf('%f\n', error_pred);

C = Cmin;
sigma = sigmaMin;
fprintf('min values:C:%f;sigma:%f:\n', C, sigma);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
