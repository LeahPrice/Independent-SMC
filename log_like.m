function loglike = log_like(theta,modeldim,y)
% Gets the log likelihood for the factor analysis example

p = length(theta); % number of parameters
n = length(y);
num_vars = 6;
der_loglike = zeros(1,p);

% Finding which parameters need to be exponentiated
diags = 7;
temp = 7;
for j=1:modeldim-1
    temp = temp + 6-j+1;
    diags = [diags temp];
end
logged = [1:6 diags];
theta(logged) = exp(theta(logged));

% Getting the beta matrix
beta = tril(ones(6,modeldim));
beta(beta==1) = theta(7:end);

% Getting the covariance
covar = diag(theta(1:6)) + beta*beta';

% log of mvnpdf(y,0,covar)
loglike = -6*n/2*log(2*pi)-n/2*log(det(covar))-trace(y*inv(covar)*y')/2; 

end
