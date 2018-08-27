function [theta, loglike, logprior, gammavar, log_evidence, count_loglike] = SMC_RW(N,modeldim,y)
% SMC with a multivariate normal random walk

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% N              -  Size of population of particles
%
% modeldim       -  Number of factors
%
% y              -  The 144x6 data matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% theta          -  Final N samples from each temperature.
%
% loglike        -  Log likelihood corresponding to above thetas.
%
% logprior       -  Log prior corresponding to above thetas.
%
% gammavar       -  The temperatures for the likelihood annealing schedule
%
% log_evidence 	 -  The logged evidence estimate (not necessarily unbiased when
%                   adapting the temperatures and proposals adaptively
%                   online)
%
% count_loglike  -  The total log likelihood computations required for the
%                   method.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dimension of theta and data
p = 6*(modeldim+1)-sum(1:(modeldim-1));
n = length(y);
h = 2.38 / sqrt(p);

% Initialising
c = 0.01; % For choosing number of MCMC repeats. We choose so that particles have probability of at least 1-c of move.
log_evidence = 0;
gammavar = 0;
t = 1;

% Prior parameters
C0 = 1;
IG1 = 2.2/2;
IG2 = (0.1/2)^(-1);

% Inverse gamma priors for diagonals of Sigma and standard normal priors for the beta (except for diagonals which are truncated standard normal)
theta = [-1*log(gamrnd(IG1,IG2,N,6)) normrnd(0,C0,[N,6*modeldim-sum(1:(modeldim-1))])];

% Locations of beta matrix diagonals
diags = 7;
temp = 7;
for j=1:modeldim-1
    temp = temp + 6-j+1;
    diags = [diags temp];
end

% Truncating the normal prior for beta diagonals and taking log
theta(:,diags) = log(abs(theta(:,diags)));

% Calculating the log likelihood and log prior for all initial draws
loglike = zeros(N,1);
logprior = zeros(N,1);
parfor i=1:N
    loglike(i) = log_like(theta(i,:),modeldim,y);
    logprior(i) = log_prior(theta(i,:),modeldim);
end

while gammavar(t)~=1
    % Testing gammavar=1
    logw = (1-gammavar(t))*loglike(:,t);
    logw = logw - logsumexp(logw);
    ESS1 = exp(-logsumexp(2*logw));
    
    % Choosing the new temperature
    if (ESS1 >= N/2)
        gammavar(t+1) = 1;
    else
        gammavar(t+1) = bisection(@(thing)compute_ESS_diff(thing,gammavar(t),loglike(:,t),N),gammavar(t),1);
    end
    
    fprintf('The new temperature is %d.\n',gammavar(t+1));
    
    % Reweighting
    logw = (gammavar(t+1)-gammavar(t))*loglike(:,t);
    log_evidence = log_evidence + logsumexp(logw) - log(N); % updating the evidence estimate
    w = exp(logw-logsumexp(logw));
    
    %Sampling with replacement according to weights
    r = randsample(1:N,N,true,w);
    tplus1 = t + 1;
    theta(:,:,tplus1) = theta(r,:,t);
    loglike(:,tplus1) = loglike(r,t);
    logprior(:,tplus1) = logprior(r,t);
    
    % Estimating the parameters of the move kernel (covariance for RW)
    cov_rw = cov(theta(:,:,tplus1));
    
    % Performing a single MCMC repeat
    accept = zeros(N,1);
    acc_probs = zeros(N,1);
    parfor i=1:N
        theta_prop = mvnrnd(theta(i,:,tplus1),h^2*cov_rw);
        
        loglike_prop = log_like(theta_prop,modeldim,y);
        logprior_prop = log_prior(theta_prop,modeldim);
        
        log_mh = gammavar(t+1)*loglike_prop - gammavar(t+1)*loglike(i,tplus1) + logprior_prop - logprior(i,tplus1);
        
        acc_probs(i) = exp(log_mh);
        
        if (rand < acc_probs(i))
            theta(i,:,tplus1) = theta_prop;
            loglike(i,tplus1) = loglike_prop;
            logprior(i,tplus1) = logprior_prop;
            accept(i) = accept(i)+1;
        end
    end
    
    %Working out remaining number of MCMC repeats
    expected_acc_probs = mean(min(acc_probs,1));
    
    R(tplus1) = ceil(log(c)./log(1-expected_acc_probs));
    fprintf('The new value of R is %d.\n',R(tplus1));
    
    % Performing remaining repeats
    parfor i=1:N
        for j=1:R(tplus1)-1
            theta_prop = mvnrnd(theta(i,:,tplus1),h^2*cov_rw);
            
            loglike_prop = log_like(theta_prop,modeldim,y);
            logprior_prop = log_prior(theta_prop,modeldim);
            
            log_mh = gammavar(t+1)*loglike_prop - gammavar(t+1)*loglike(i,tplus1) + logprior_prop - logprior(i,tplus1);
            
            if (rand < exp(log_mh))
                theta(i,:,tplus1) = theta_prop;
                loglike(i,tplus1) = loglike_prop;
                logprior(i,tplus1) = logprior_prop;
                accept(i) = accept(i)+1;
            end
        end
    end
    
    t = tplus1;
    
end

count_loglike = N+N*sum(R+1);

end
