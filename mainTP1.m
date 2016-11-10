%% Problem setting 

M = 15; % size of the stock 
K = 0.8; % delivery cost
h = 0.3; % maintenance cost
c = 0.5; % buying price
p = 1; % selling price

gamma = 0.98; % discount factor (inflation rate)
 

% D represents the distribution of the customers (truncated geometric)
D=zeros(1,M+1); 
q = 0.1; % parameter of the geometric distribution
D(1+(0:(M-1)))=q*(1-q).^(0:(M-1));
D(M+1)=1-sum(D);

x0=M; % initial stock

%% Visualisation of a single trajectory

pi=2*ones(1,M+1);%politique always buying two machines
%pi=15:-1:0; %policy always filing the stock 

n=1000;

[X,R] = trajectory(n,x0,pi,M,K,h,c,p,D);


% discounted profit up to time n 
plot(cumsum(gamma.^((1:n)-1).*R))


