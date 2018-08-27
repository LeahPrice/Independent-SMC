function logprior = log_prior(theta,modeldim)
% Gets the log prior (taking into account log transforms) for the factor analysis example.

% Dimension of theta
p = length(theta);

% Parameters for prior
C0 = 1;
IG1 = 2.2/2;
IG2 = (0.1/2)^(-1);

logprior = 0;
for j=1:6
    logprior = logprior + theta(j) - 1/IG2./exp(theta(j))-log(gamma(IG1))-IG1*log(IG2)-theta(j)*(IG1+1); %Inverse gamma with log taken.
end

if modeldim>0
    % Getting the location of the beta matrix diagonals (which have truncated
    % normal prior)
    diags = 7;
    temp = 7;
    for j=1:modeldim-1
        temp = temp + 6-j+1;
        diags = [diags temp];
    end
    
    % Getting the off-diagonal locations (standard normal prior)
    temp2 = 7:p;
    offdiag = temp2(ismember(temp2,diags)==0);
    
    for j=diags
        logprior = logprior + theta(j) + log_normpdf(exp(theta(j)),0,C0)-log(1-normcdf(0,0,C0)); %Truncated normal with log taken
    end
    for j=offdiag
        logprior = logprior + log_normpdf(theta(j),0,C0); %Normal with no logs
    end
end

end

