function [new_weight_ip_cis, log_evidence_ip_cis, ess_ip_cis] = Recycle_IP_CIS(loglike_all,logprior_all,indprop_all,gammavar_all)
% This function can be used to recycle all MCMC candidates using CIS.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_all    -  Log likelihoods of all candidates
%
% logprior_all   -  Log prior of all candidates
%
% indprop_all    -  Proposal density of all candidates
%
% gammavar_all   -  Temperature the independent proposal was based on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% new_weight_ip_cis -  Weights assigned when reweighting all particles to
%                   target the posterior
%
% log_evidence_ip_cis -  Log estimate of the evidence based on version given
%                   in independent SMC paper
%
% ess_ip_cis        -  Effective sample size targeting the posterior after
%                   recycling.

numparts = length(loglike_all);
uniqtemps = unique(gammavar_all);
T = length(uniqtemps);
log_w_t = zeros(numparts,1);
log_sum_w = zeros(T,1);
log_ESS = zeros(T,1);
log_evidence_t = zeros(T,1);
new_weight_ip_cis=zeros(numparts,1);

% Finding the ESS if each independent proposal was used in IS to target the posterior
for t=1:T
    gammavar_curr = uniqtemps(t);
    inds = find(gammavar_all==gammavar_curr);    
    log_w_t(inds) = loglike_all(inds)+logprior_all(inds)-indprop_all(inds);
    
    log_sum_w(t) = logsumexp(log_w_t(inds)); % calculating sum of these weights. Used in normalising weights and calculating ESS.
    
    log_ESS(t) = 2*log_sum_w(t) - logsumexp(2*log_w_t(inds)); % calculating the ESS of t-th collection of importance weights
end

logLambda = log_ESS - logsumexp(log_ESS); % calculating the contribution to the total ESS.

% Getting the combined weights and evidence estimate
for t=1:T
    gammavar_curr = uniqtemps(t);
    inds = find(gammavar_all==gammavar_curr);
    %the weighting for the recycling is the normalised weights times lambda.
    new_weight_ip_cis(inds) = log_w_t(inds)-log_sum_w(t)+logLambda(t);
	
	%The log evidence calculation should use unnormalised weights.
    log_evidence_t(t) = logsumexp(log_w_t(inds)) - log(length(inds));
end

ess_ip_cis = exp(-logsumexp(2*new_weight_ip_cis));
new_weight_ip_cis = exp(new_weight_ip_cis);

log_evidence_ip_cis = logsumexp(log_evidence_t + logLambda); %Weighted sum of all invidiual log evidence estimates

end

