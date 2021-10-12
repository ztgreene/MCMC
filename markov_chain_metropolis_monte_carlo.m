%% Markov Chain Metropolis Monte Carlo Method using spherical Gaussian proposal distribution

% This is the target distribution
mu = [0 0];
rho = 0.998;
sigma = [1 rho; rho 1];
P = @(X) mvnpdf(X, mu, sigma);
x1 = linspace(-1, 1);
x2 = x1;
[x1, x2] = meshgrid(x1, x2);
Pcontour = reshape(P([x1(:), x2(:)]), 100, 100);
figure(1); clf; [M,c] = contour(x1, x2, Pcontour, [1.6 3], 'k'); axis square
c.LineWidth = 3;
hold on;

%% Metropolis Algorithm
%% Initialise values
T = 10000;

X_stored = zeros(T,2);
X_term = zeros(T+1, 2);

X = [0 0]; % starting X value
X_stored(1,:) = X;
X_term = X;

EPSILON = [0.1 0; 0 0.1];
MU2 = [0 0];

accepted = 0;

for i = 1:T
    test = mvnrnd(X, EPSILON);

    A = P(test)/P(X);
    
    if A >= 1
        accept = 1;
    else
        a = rand();
        if a <= A
            accept = 1;
        else
            accept = 0;
        end
    end
    if accept
        X = test;
    end
    accepted = accepted + accept;
    X_stored(i, :) = X;
    X_term(i+1, :) = X;
end

figure(1);
plot(X_stored(:,1), X_stored(:,2), '-x')
hold on; plot(X_term(:,1), X_term(:,2), 'ksq')
acceptance_rate = accepted/T


%% Analysis of results (ie, independence of sample)
% Calling the autocorrelation function (acf()) on either column vector of 
% the X_term arrays leads to very similar (almost identical) results, 
% so I am addressing only one column vector (that for X1).

%% This calculates the 'rule of thumb for T to generate an effectively
%% independent sample - the result is rather large:
%% anywhere from around 450,000, to orders of magnitude higher (1.2 BILLION!)
X1sum = sum(X_term(:,1))^2;
L = sqrt(X1sum);
ruleOfThumb = (L/EPSILON(1,1))^2

%% I have commented the autocorrelate function out, so it doesn't overwrite
%% plots for previous section. Uncomment it for results.

%acf(X_term(:,1), T);

% According to the acf function, a value of 0 was acheived with a lag (ie
% T) of 10,000. However, it fluctuates between above and below the zero
% line. When setting Lag,T to 50,000, the value still fluctuates around the
% zero-line, however, it remains very close to zero after T = 30,000:
% <0.05 after T=30,000, and < 0.01 after T=47,000, and is effectively equal
% to zero by T = 50,000. 
% The acf results would seem to suggest an essentially-independent sample
% could be generated in around 30-50 thousand samples... which is still
% quite a large number - however, it is orders of magnitude fewer than the
% 'rule of thumb' would suggest.
% 

