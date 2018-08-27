% Code for the Factor Analysis example in Section 4.1 of the paper

load('Data.mat'); % Loading the data
N = 5000; % Choosing the number of particles
modeldim = 1; % Choosing the number of factors in the FA model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MVN RANDOM WALK MCMC PROPOSALS WITH RECYCLING

[theta, loglike, logprior, gammavar, log_evidence, count_loglike] = SMC_RW(N,modeldim,Y);

% Recycling the power posterior samples.
% The pair of theta_pp and new_weight_pp_amis (or new_weight_pp_cis) represent a weighted approximation of the posterior using all samples.
splitA = num2cell(theta, [1 2]); % preparing the power posterior samples for the quantile calculations
theta_pp = vertcat(splitA{:});
[new_weight_pp_cis, log_evidence_pp_cis, ess_pp_cis] = Recycle_PP_CIS(loglike,gammavar);
[new_weight_pp_amis, log_evidence_pp_amis, ess_pp_amis] = Recycle_PP_DeMix(loglike,logprior,gammavar);

% Example quantile calculations (2.5% quantile)
lower =  quantile_weighted(theta(:,:,end),0.025,1/N*ones(N,1));
lower_pp_cis =  quantile_weighted(theta_pp,0.025,new_weight_pp_cis);
lower_pp_amis =  quantile_weighted(theta_pp,0.025,new_weight_pp_amis);

% Printing out some basic results for the normalising constant
fprintf('The logged evidence estimates are:\nStandard SMC: %.2f\nCIS PP: %.2f\nDeMix PP: %.2f',...
    log_evidence, log_evidence_pp_cis, log_evidence_pp_demix);
