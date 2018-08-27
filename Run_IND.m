% Code for the Factor Analysis example in Section 4.1 of the paper

load('Data.mat'); % Loading the data
N = 5000; % Choosing the number of particles
modeldim = 1; % Choosing the number of factors in the FA model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INDEPENDENT MCMC PROPOSALS WITH RECYCLING

[theta, loglike, logprior, gammavar, log_evidence, count_loglike, theta_prop, loglike_prop, logprior_prop, gammavar_prop, indprop_prop, GMModel, GMModel_marg] = SMC_IND(N,modeldim,Y);

% Recycling the power posterior samples.
% The pair of theta_pp and new_weight_pp_demix (or new_weight_pp_cis) represent a weighted approximation of the posterior using all samples.
splitA = num2cell(theta, [1 2]); % preparing the power posterior samples for the quantile calculations
theta_pp = vertcat(splitA{:});
[new_weight_pp_cis, log_evidence_pp_cis, ess_pp_cis] = Recycle_PP_CIS(loglike,gammavar);
[new_weight_pp_demix, log_evidence_pp_demix, ess_pp_demix] = Recycle_PP_DeMix(loglike,logprior,gammavar);

% Recycling all candidates (i.e. the prior and all MCMC candidates)
[new_weight_ip_cis, log_evidence_ip_cis, ess_ip_cis] = Recycle_IP_CIS(loglike_prop,logprior_prop,indprop_prop,gammavar_prop);
[new_weight_ip_demix, log_evidence_ip_demix, ess_ip_demix] = Recycle_IP_DeMix(theta_prop,loglike_prop,logprior_prop,indprop_prop,gammavar_prop,GMModel,GMModel_marg,modeldim);

% Example quantile calculations (2.5% quantile)
lower =  quantile_weighted(theta(:,:,end),0.025,1/N*ones(N,1));
lower_pp_cis =  quantile_weighted(theta_pp,0.025,new_weight_pp_cis);
lower_pp_demix =  quantile_weighted(theta_pp,0.025,new_weight_pp_demix);
lower_ip_demix =  quantile_weighted(theta_prop,0.025,new_weight_ip_demix);
lower_ip_cis =  quantile_weighted(theta_prop,0.025,new_weight_ip_cis);

% Printing out some basic results for the normalising constant
fprintf('The logged evidence estimates are:\nStandard SMC: %.2f\nCIS PP: %.2f\nDeMix PP: %.2f\nCIS IP: %.2f\nDeMix IP: %.2f',...
    log_evidence, log_evidence_pp_cis, log_evidence_pp_demix, log_evidence_ip_cis, log_evidence_ip_demix);
