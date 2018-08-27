function f = quantile_fun_gaussmix(u,GMModel,x)
%%
% Inputs:
%   u - quantile
%   GMModel - fitted Gaussian mixture model
%   x - set of trial points that can help provide quick bounds for most quantiles
%
% Outputs:
%   f - an approximate u-quantile from Gaussian mixture model GMModel
%%
tol = 1e-10;

cdfs = cdf(GMModel,x);

[cdfs,ix] = sort(cdfs);
x = x(ix);
lower = find(cdfs<u,1,'last');
if (isempty(lower))
   lower_x = -1000;
   while(1)
       ul = cdf(GMModel,lower_x);
       if (ul<u)
           break;
       end
       lower_x = lower_x - 1000;
   end
else
    lower_x = x(lower);
end
    

upper = find(cdfs>u,1,'first');
if (isempty(upper))
   upper_x = 1000;
   while(1)
       uu = cdf(GMModel,upper_x);
       if (uu>u)
           break;
       end
       upper_x = upper_x + 1000;
   end
else
   upper_x = x(upper);
end

% Bisection method to estimate the quantile
count = 0;
while(1)
	count = count+1;
    f = (lower_x + upper_x)/2;
    utrial = cdf(GMModel,f);
    error = upper_x-lower_x;
    if (error < tol)
        break; % finished
    end
    if (utrial < u)
        lower_x = f;
    else
        upper_x = f;
    end
    if count>1000 && mod(count,1000)==0
        fprintf('In quantile fun gaussmix the count is %d\n',count);
    end
end

end