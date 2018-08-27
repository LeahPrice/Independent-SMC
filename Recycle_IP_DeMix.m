function [new_weight_ip_amis, log_evidence_ip_amis, ess_ip_amis] = Recycle_IP_DeMix(theta_all,loglike_all,logprior_all,indprop_all,gammavar_all,GMModel,GMModel_marg,modeldim)
% This function can be used to recycle all MCMC candidates using DeMix.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% theta_all      -  All candidates from all move steps
%
% loglike_all    -  Corresponding log likelihoods
%
% logprior_all   -  Corresponding values of the log prior
%
% indprop_all    -  Proposal density (not really necessary as just
%                   recalculated)
%
% gammavar_all   -  Temperature the independent proposal was based on
%
% GMModel        -  Gaussian mixture models that are fitted to the
%                   transformed marginals
%
% GMModel_marg   -  Gaussian mixture model fitted to marginals
%
% modeldim       -  Number of factors

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% new_weight_ip_amis -  Weights assigned when reweighting all particles to
%                   target the posterior
%
% log_evidence_ip_amis -  Log estimate of the evidence based on DeMix identity
%
% ess_ip_amis       -  Effective sample size targeting the posterior after
%                   recycling.


[numparts, d] = size(theta_all);
uniqtemps = unique(gammavar_all);
T = length(uniqtemps);
log_w_t = zeros(numparts,1);
N_l = zeros(1,T);
indprop_crossover = zeros(numparts,T);

% Finding out which proposal each candidate was drawn from
for t=1:T
    gammavar_curr = uniqtemps(t);
    inds = find(gammavar_all==gammavar_curr);
    N_l(t) = length(inds);
end

% The density of each candidate at the first proposal (the prior)
indprop_crossover(:,1) = logprior_all';

% Getting what the proposal density would be for each other temperature. T proposal densities per theta.
parfor t=2:T
    Z = zeros(numparts,d);
    U = zeros(numparts,d);
    log_marg = zeros(numparts,1);
    log_Z = zeros(numparts,1);
    for j=1:d
        U(:,j) = cdf(GMModel_marg{t}{j},theta_all(:,j));
        U(U(:,j)==0,j) = 10^(-15);
        U(U(:,j)==1,j) = 1-10^(-15);
        Z(:,j) = norminv(U(:,j),0,1);
        log_Z = log_Z + log(normpdf(Z(:,j),0,1));
        log_marg = log_marg + log(pdf(GMModel_marg{t}{j},theta_all(:,j)));
    end
    indprop_crossover(:,t) = log(pdf(GMModel{t},Z)) + log_marg - log_Z;
end
indprop_crossover(isnan(indprop_crossover(:))==1) = -inf;

%Working out the denominator (the mixture) for the DeMix method
denominator=zeros(1,numparts);
for i=1:numparts
    denominator(i) = -log(sum(N_l)) + logsumexp(log(N_l)+indprop_crossover(i,:));
end

%Getting the new weights
for t=1:T
    gammavar_curr = uniqtemps(t);
    inds = find(gammavar_all==gammavar_curr);
    
    log_w_t(inds) = loglike_all(inds)+logprior_all(inds)-denominator(inds);
end

new_weight_ip_amis = log_w_t - logsumexp(log_w_t);
ess_ip_amis = exp(-logsumexp(2*new_weight_ip_amis));
new_weight_ip_amis = exp(new_weight_ip_amis);

log_evidence_ip_amis = logsumexp(log_w_t)-log(numparts);
end

