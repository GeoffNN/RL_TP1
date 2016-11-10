function [xnew,reward] = simu_transition(x,a,M,K,h,c,p,D)
d=simu(D); % simulate the number of customers that arrive this week
xnew=max(0,min((x+a),M)-d); % compute the new state
reward= -K*(a>0) - c*max(0,min(x+a,M) - x) - h*x + p*min([d , x+a, M]); % compute the associated reward
end


