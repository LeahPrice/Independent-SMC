function [new_weight_pp_amis, log_evidence_pp_amis, ess_pp_amis] = Recycle_PP_DeMix(loglike,logprior,gammavar)
% This function can be used to recycle all power posterior samples using DeMix.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike        -  Log likelihoods of all accepted
%
% logprior	     -  Log prior of all accepted
%
% gammavar       -  Temperatures

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% new_weight_pp_amis -  Weights assigned when reweighting all particles to
%                   target the posterior
%
% log_evidence_pp_amis -  Log estimate of the evidence based on version given
%                   in independent SMC paper
%
% ess_pp_amis        -  Effective sample size targeting the posterior after
%                   recycling.

N = size(loglike,1);
T = length(gammavar);
log_Z = zeros(T,1);
log_w_t = zeros(N,T);

% Calculating the intermediate normalising constants.
log_Z(1) = 0;
for t=2:T
    curr = logsumexp((gammavar(t)-gammavar(t-1))*loglike(:,t-1))-log(N);
    log_Z(t) = log_Z(t-1) + curr;
end

% Getting the normalised density under each of the other power posteriors
for t=1:T
    extra_dens{t} = zeros(N,T);
    for t2 = 1:T
    extra_dens{t}(:,t2) = gammavar(t2)*loglike(:,t) + logprior(:,t) - log_Z(t2);
    end
end

% Calculating the mixture of proposal densities.
denom = zeros(N,T);
for i=1:N
    for t=1:T
        denom(i,t) = -log(N*T) + logsumexp(log(N)+extra_dens{t}(i,:));
    end
end

% The new (unnormalised) log weights
for t = 1:T    
    log_w_t(:,t) = loglike(:,t) + logprior(:,t) - denom(:,t);
end

new_weight_pp_amis = log_w_t(:) - logsumexp(log_w_t(:));
ess_pp_amis = exp(-logsumexp(2*new_weight_pp_amis));

new_weight_pp_amis = exp(new_weight_pp_amis);

log_evidence_pp_amis = logsumexp(log_w_t(:))-log(N*T); % Weighted sum of all individual log evidence estimates

end

