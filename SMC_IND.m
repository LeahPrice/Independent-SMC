function [theta, loglike, logprior, gammavar, log_evidence, count_loglike, theta_all, loglike_all, logprior_all, gammavar_all, indprop_all, GMModel, GMModel_marg] = SMC_IND(N,modeldim,y)
% Independent proposal
% Adaptive temperature schedule to maintain CESS at rho*N. Resample-move
% only when ESS drops below alpha*N
% For factor analysis example: assuming lower diagonal beta matrix, not
% block lower triangular. Gives same number of parameters as paper says.

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
%
% theta_all      -  All proposed thetas
%
% loglike_all    -  Log likelihood of all proposed thetas
%
% logprior_all   -  Log prior of all proposed thetas
%
% gammavar_all   -  Temperature associated with all proposed thetas
%
% indprop_all    -  Proposal density of all proposed thetas
%
% GMModel        -  Fitted Gaussian mixture models that is fitted to
%                   transformed marginals
%
% GMModel_marg   -  Gaussian mixture model fitted to marginals

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dimension of theta and data
p = 6*(modeldim+1)-sum(1:(modeldim-1));
n = length(y);

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

% Storing all candidates
theta_all = theta;
loglike_all = loglike';
logprior_all = logprior';
indprop_all = logprior';
gammavar_all = zeros(1,N);
addedindex = N;

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
    
    % Transforming the parameters to approximately N(0,1) and starting the proposal density calculations
    all = theta(:,:,tplus1);
    log_marg = zeros(N,1);
    log_Z = zeros(N,1);
    for j=1:p
        GMModel_marg{tplus1}{j} = fitgmdist(all(:,j),5,'RegularizationValue',0.01,'Replicates',5,'Options',optimset('MaxIter',1000));
        U(:,j) = cdf(GMModel_marg{tplus1}{j},all(:,j));
        U(U(:,j)==0,j) = 10^(-15);
        U(U(:,j)==1,j) = 1-10^(-15);
        Z(:,j) = norminv(U(:,j),0,1);
        log_Z = log_Z + log_normpdf(Z(:,j),0,1);
        log_marg = log_marg + log(pdf(GMModel_marg{tplus1}{j},all(:,j))); % for the proposal density
    end
    
    % Estimating the model for the dependence
    GMModel{tplus1} = fitgmdist(Z,6,'RegularizationValue',0.01,'Replicates',10,'Options',optimset('MaxIter',1000));
    
    % Proposal density of current particles for MH ratio
    curr_propprob = log(pdf(GMModel{tplus1},Z)) + log_marg - log_Z;
    
    acc_probs = zeros(N,1);
    parfor i=1:N
        % Drawing from the joint model and transforming back to the original scale
        z_prop = random(GMModel{tplus1});
        u_prop = normcdf(z_prop,0,1);
        theta_prop = zeros(1,p);
        log_marg = 0;
        log_Z = sum(log_normpdf(z_prop,0,1));
        for j = 1:p
            theta_prop(j) = quantile_fun_gaussmix(u_prop(j),GMModel_marg{tplus1}{j},all(:,j));
            log_marg = log_marg + log(pdf(GMModel_marg{tplus1}{j},theta_prop(j)));
        end
        
        % Calculating the proposal density, log likelihood and log prior
        indprop_prop =  log(pdf(GMModel{tplus1},z_prop)) + log_marg - log_Z;
        loglike_prop = log_like(theta_prop,modeldim,y);
        logprior_prop = log_prior(theta_prop,modeldim);
        
        log_mh = gammavar(t+1)*loglike_prop - gammavar(t+1)*loglike(i,tplus1) + logprior_prop - logprior(i,tplus1) + curr_propprob(i) - indprop_prop;
        
        acc_probs(i) = exp(log_mh);
        
        theta_all(addedindex+i,:) = theta_prop;
        loglike_all(addedindex+i) = loglike_prop;
        logprior_all(addedindex+i) = logprior_prop;
        indprop_all(addedindex+i) = indprop_prop;
        gammavar_all(addedindex+i) = gammavar(t+1);
        
        if (rand < acc_probs(i))
            theta(i,:,tplus1) = theta_prop;
            loglike(i,tplus1) = loglike_prop;
            logprior(i,tplus1) = logprior_prop;
            curr_propprob(i) = indprop_prop;
            Z(i,:) = z_prop;
        end
    end
    
    % Working out remaining number of MCMC repeats
    expected_acc_probs = mean(min(acc_probs,1));
    
    R(tplus1) = ceil(log(c)./log(1-expected_acc_probs));
    fprintf('The new value of R is %d.\n',R(tplus1));
    
    parfor i=1:N
        for k=1:R(tplus1)-1
        % Drawing from the joint model and transforming back to the original scale
            z_prop = random(GMModel{tplus1});
            u_prop = normcdf(z_prop,0,1);
            theta_prop = zeros(1,p);
            log_marg = 0;
            log_Z = sum(log_normpdf(z_prop,0,1));
            for j = 1:p
                theta_prop(j) = quantile_fun_gaussmix(u_prop(j),GMModel_marg{tplus1}{j},all(:,j));
                log_marg = log_marg + log(pdf(GMModel_marg{tplus1}{j},theta_prop(j)));
            end
            
            % Calculating the proposal density, log likelihood and log prior
            indprop_prop =  log(pdf(GMModel{tplus1},z_prop)) + log_marg - log_Z;
            loglike_prop = log_like(theta_prop,modeldim,y);
            logprior_prop = log_prior(theta_prop,modeldim);
            
            log_mh = gammavar(t+1)*loglike_prop - gammavar(t+1)*loglike(i,tplus1) + logprior_prop - logprior(i,tplus1) + curr_propprob(i) - indprop_prop;
            
            theta_temp{i}(k,:) = theta_prop;
            loglike_temp{i}(k) = loglike_prop;
            logprior_temp{i}(k) = logprior_prop;
            indprop_temp{i}(k) = indprop_prop;
            gammavar_temp{i}(k) = gammavar(t+1);
            
            if (rand < exp(log_mh))
                theta(i,:,tplus1) = theta_prop;
                loglike(i,tplus1) = loglike_prop;
                logprior(i,tplus1) = logprior_prop;
                curr_propprob(i) = indprop_prop;
                Z(i,:) = z_prop;
            end
        end
    end
    
    % Storing all candidates
    theta_all = [theta_all; cell2mat(theta_temp')];
    loglike_all = [loglike_all cell2mat(loglike_temp)];
    logprior_all = [logprior_all cell2mat(logprior_temp)];
    indprop_all = [indprop_all cell2mat(indprop_temp)];
    gammavar_all = [gammavar_all cell2mat(gammavar_temp)];
    
    addedindex = addedindex+R(tplus1)*N;
    
    t = tplus1;
    
end

count_loglike = N+N*sum(R);

end
