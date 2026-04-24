function x = prox_l1(b,lambda)
q=1;
x = max(0,b-lambda*q)+min(0,b+lambda*q);
x = 1/(1+lambda*(1-q))*x;
end