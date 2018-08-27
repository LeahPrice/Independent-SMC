function [new_weight_pp_cis, log_evidence_pp_cis, ess_pp_cis] = Recycle_PP_CIS(loglike,gammavar)
% This function can be used to recycle all power posterior samples using CIS.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike        -  Log likelihoods of all power posterior samples
%
% logprior	     -  Log prior of all power posterior samples
%
% gammavar       -  Temperatures

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% new_weight_pp_cis -  Weights assigned when reweighting all particles to
%                   target the posterior
%
% log_evidence_pp_cis -  Log estimate of the evidence based on version given
%                   in independent SMC paper
%
% ess_pp_cis        -  Effective sample size targeting the posterior after
%                   recycling.

N = size(loglike,1);
T = length(gammavar);
log_Z = zeros(T,1);
log_w_t = zeros(N,T);
log_sum_w = zeros(T,1);
log_ESS = zeros(T,1);
log_evidence_t = zeros(T,1);
new_weight_pp_cis=zeros(N,T);

log_Z(1) = 0;
log_w_t(:,1) = loglike(:,1);
log_sum_w(1) = logsumexp(log_w_t(:,1)); % calculating sum of these weights. Used in normalising weights and calculating ESS.
log_ESS(1) = 2*log_sum_w(1) - logsumexp(2*log_w_t(:,1)); % calculating the ESS of t-th collection of importance weights

% Getting the intermediate estimates of the normalising constants and the ESS from each set of power posterior samples
for t=2:T
    curr = logsumexp((gammavar(t)-gammavar(t-1))*loglike(:,t-1))-log(N);
    log_Z(t) = log_Z(t-1) + curr;
    
    log_w_t(:,t) = (1-gammavar(t))*loglike(:,t);
    log_w_t(:,t) = log_w_t(:,t) + log_Z(t);
    log_sum_w(t) = logsumexp(log_w_t(:,t)); % calculating sum of these weights. Used in normalising weights and calculating ESS.
    log_ESS(t) = 2*log_sum_w(t) - logsumexp(2*log_w_t(:,t)); % calculating the ESS of t-th collection of importance weights
end

log_lambda_t = log_ESS - logsumexp(log_ESS); % calculating the contribution to the total ESS.

% Getting the combined weights and evidence estimate
for t=1:T
    new_weight_pp_cis(:,t) = log_w_t(:,t)-log_sum_w(t)+log_lambda_t(t);
    
    %The log evidence calculation should use unnormalised weights.
    log_evidence_t(t) = logsumexp(log_w_t(:,t)) - log(N);
end

new_weight_pp_cis = new_weight_pp_cis(:) - logsumexp(new_weight_pp_cis(:));
ess_pp_cis = exp(-logsumexp(2*new_weight_pp_cis));
new_weight_pp_cis = exp(new_weight_pp_cis);

log_evidence_pp_cis = logsumexp(log_evidence_t + log_lambda_t); %Weighted sum of all invidiual log evidence estimates

end

