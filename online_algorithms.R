### projection for ONS
generalized_projection <- function(P,theta,D) {
  x <- CVXR::Variable(d)
  objective <- CVXR::Minimize(CVXR::matrix_frac(x-theta, P)) # min (x-theta) P^-1 (x-theta)
  constraint <- list(CVXR::norm2(x) <= D)                    # s.t. norm(x) <= D
  prob <- CVXR::Problem(objective, constraint)
  result <- CVXR::solve(prob)
  if (result$status != 'optimal')
    result <- CVXR::solve(prob, solver = 'SCS')
  if (result$status != 'optimal') {
    print(paste('Generalized projection error: optimization is not optimal and ends with status',
                result$status))
  }
  result$getValue(x)
}


### instantaneous likelihood, gradient and hessian
instgrad <- function(t, t0, hat_T, delta, X, beta, R_t) {
  N <- length(t0)
  d <- dim(X)[2]
  grad <- matrix(0,d,1)
  hess <- matrix(0,d,d)
  lik <- matrix(0,1,1)
  for (i in R_t) {
    lik <- lik - ((t-1 < hat_T[i] )*( hat_T[i] <= t )* delta[i]*t(beta)%*%X[i,]+ exp(t(beta)%*%X[i,])*max(0, min(t,hat_T[i]) - max(t0[i],t-1)))/N
    hess <- hess + X[i,]%*%t(X[i,])*exp(beta*X[i,])[1]*
      max(0, min(t,hat_T[i]) - max(t0[i],t-1)) / N
    grad <- grad + X[i,] * exp(crossprod(beta,X[i,])[1]) * 
      max(0, min(t,hat_T[i]) - max(t0[i],t-1))  / N
    if (t-1 < hat_T[i] & hat_T[i] <= t & delta[i])
      grad <- grad - X[i,] / N
  }
  list(grad = grad, hess = hess, lik = lik)
}


###complete likelihood
complete_likelihood <- function(t0,hat_T,delta,X,n,beta){
  N <- length(t0)
  d <- dim(X)[2]
  likelihood <- matrix(0,n,1)
  
  for (t in 1:n){
    for (i in 1:N){
      likelihood[t] <- delta[i]*t(beta[t,])%*%X[i,]/N+ exp(t(beta[t,])%*%X[i,])*(hat_T[i]-t0[i])/N
    }
  }
  list(likelihood = likelihood)
}


### original ONS
ons <- function(t0, hat_T, delta, X, D, gamma, n, epsilon, R) {
  N <- length(t0)
  d <- dim(X)[2]
  beta <- matrix(0,d,1)
  beta_arr <- matrix(0,n,d)
  A_inv <- (1/epsilon)*diag(d)
  mu_temp <- matrix(0,n,1)
  gamma_temp <- matrix(0,n,1)
  grad_ons <- matrix(0,n,d)
  
  for (t in 1:n) {
    beta_arr[t,] <- beta
    ### compute gradient
    inst <- instgrad(t, t0, hat_T, delta, X, beta, R[[t]])
    grad <- inst$grad
    hess <- inst$hess
    norm_grad <- sqrt(sum(grad^2))
    grad_ons[t,] <- grad
    
    ### compute the optimal constants
    mu <- as.numeric(t(grad)%*%hess%*%grad/max(10^-9,norm_grad^4))
    gamma_t <-  as.numeric(2*((-1/mu)*log(1 + mu*norm_grad*D) + norm_grad*D)/max(10^-9,norm_grad*D)^2)
    gamma_temp[t] <- gamma_t
    mu_temp[t] <- mu
    
    ### original ONS step
    A_inv <- A_inv - tcrossprod(A_inv %*% grad) / (1 + crossprod(grad, A_inv %*% grad)[1])
    beta <- beta - gamma^-1 * A_inv %*% grad
    if (sqrt(crossprod(beta)[1]) > D)
      beta <- generalized_projection(A_inv,beta,D)
  }
  list(beta_arr=beta_arr, gamma_temp = gamma_temp, mu_temp = mu_temp, grad_ons = grad_ons)
}


### gradient descent
grad_descent <- function(t0,hat_T,delta, X,D,gamma,n, R){
  N <- length(t0)
  d <- dim(X)[2]
  beta <- matrix(0,d,1)
  beta_arr <- matrix(0,n,d)
  like_arr <- matrix(0,n,1)
  I <- diag(d)
  
  for (t in 1:n) {
    beta_arr[t,] <- beta
    ### compute gradient
    grad <- instgrad(t, t0, hat_T, delta, X, beta, R[[t]])$grad
    like_arr[t] <- instgrad(t, t0, hat_T, delta, X, beta, R[[t]])$lik
    ### gradient descent step
    beta <- beta - (gamma/sqrt(t)) * grad
    if (sqrt(crossprod(beta)[1]) > D)
      beta <- generalized_projection(I,beta,D)
  }
  list(beta_arr=beta_arr, like_arr = like_arr)
}


###original BOA
ons_boa <- function(t0, hat_T, delta, X, D, GAMMA, n, EPSILON, R) {
  N <- length(t0)
  d <- dim(X)[2]
  K <- length(GAMMA)
  
  BETA <- matrix(0,d,K)
  BETA_ARR <- array(0,dim=c(n,d,K))
  A_INV_ARR <- array(0,dim=c(d,d,K))
  for (i in 1:K)
    A_INV_ARR[,,i] <- (1/EPSILON[i])*diag(d)
  
  pi_boa <- matrix(1/K, K, 1)
  L <- matrix(0, K, 1)
  renorm <- matrix(0, K, 1)
  
  BETA_BOA_ARR <- matrix(0, n, d)
  PI_BOA_ARR <- matrix(0, n, K)
  LIK_BOA <- matrix(0,n,1)
  
  for (t in 1:n) {
    for (k in 1:K) {
      BETA_ARR[t,,k] <- BETA[,k]
      ### compute gradient
      grad <- instgrad(t, t0, hat_T, delta, X, BETA[,k], R[[t]])$grad
      
      ### ONS step
      A_inv <- matrix(A_INV_ARR[,,k], d, d)
      A_inv <- A_inv - tcrossprod(A_inv %*% grad) / (1 + crossprod(grad, A_inv %*% grad)[1])
      A_INV_ARR[,,k] <- A_inv
      BETA[,k] <- BETA[,k] - GAMMA[k]^-1 * A_inv %*% grad
      if (sqrt(crossprod(BETA[,k])[1]) > D)
        BETA[,k] <- generalized_projection(A_inv,BETA[,k],D)
    }
    
    ### BOA aggregation
    beta_boa <- BETA %*% pi_boa
    inst <- instgrad(t, t0, hat_T, delta, X, beta_boa, R[[t]])
    LIK_BOA[t] <- inst$lik
    grad_boa <- inst$grad
    term1 <- as.numeric(crossprod(grad_boa, BETA-tcrossprod(beta_boa, matrix(1,K,1))))
    renorm <- renorm + 2.2*term1^2
    if (sum(renorm)>0) { 
      eta <- 1/sqrt(renorm) 
      L <- L - term1 - eta * term1^2
      pi_boa <- eta * exp(pmax(-100, pmin(100, eta * L - max(eta * L))))
      pi_boa <- pi_boa / sum(pi_boa)
    }
    if (t < n) {
      BETA_BOA_ARR[t+1,] <- BETA %*% pi_boa
      PI_BOA_ARR[t+1,] <- pi_boa
    }
  }
  list(BETA_ARR=BETA_ARR, BETA_BOA_ARR=BETA_BOA_ARR, PI_BOA_ARR=PI_BOA_ARR, LIK_BOA = LIK_BOA)
}




### SurvONS
ons_boa_max <- function(t0, hat_T, delta, X, D, GAMMA, n, EPSILON, R, max0 = FALSE) {
  N <- length(t0)
  d <- dim(X)[2]
  K <- length(GAMMA)
  
  BETA <- matrix(0,d,K)
  BETA_ARR <- array(0,dim=c(n,d,K))
  A_INV_ARR <- array(0,dim=c(d,d,K))
  for (i in 1:K){
    A_INV_ARR[,,i] <- (1/EPSILON[i])*diag(d)
  }
  pi_boa <- matrix(1/K, K, 1)
  pi_boa2 <- pi_boa
  L <- matrix(0, K, 1)
  renorm <- matrix(0, K, 1)
  
  BETA_BOA_ARR <- matrix(0, n, d)
  GRAD_BOA <- matrix(0,n,d)
  GRAD_BOA_HAT <- array(0, dim = c(n,d,K))
  PI_BOA_ARR <- matrix(0, n, K)
  LIK_BOA <- matrix(0,n,1)
  gamma_temp <- matrix(0,n,1)
  
  for (t in 1:n) {
    
    ###update beta_boa
    beta_boa <- BETA %*% pi_boa
    
    ###observe gradients
    inst <- instgrad(t, t0, hat_T, delta, X, beta_boa, R[[t]])
    grad_boa <- inst$grad
    GRAD_BOA[t,] <- grad_boa
    hess_boa <- inst$hess
    LIK_BOA[t] <-inst$lik
    
    ###compute the constants
    norm_grad_boa <- sqrt(sum(grad_boa^2))
    mu <- as.numeric(t(grad_boa)%*%hess_boa%*%grad_boa/max(10^-9,norm_grad_boa^4))
    gamma_t <- as.numeric(2*((-1/mu)*log(1 + mu*norm_grad_boa*D) + norm_grad_boa*D)/max(10^-9,norm_grad_boa*D)^2)
    if (max0)
      gamma_t <- 0
    gamma_temp[t] <- gamma_t
    
    for (k in 1:K) {
      BETA_ARR[t,,k] <- BETA[,k]
      
      ###condition for gamma
      gamma_max <- max(gamma_t/4,GAMMA[k])
      
      ### compute gradient and hessian
      grad_hat <- grad_boa * as.numeric(1+gamma_max*crossprod(grad_boa, BETA[,k]- beta_boa))
      GRAD_BOA_HAT[t,,k] <- grad_hat
      
      ### ONS step
      A_inv <- matrix(A_INV_ARR[,,k], d, d)
      A_inv <- A_inv - tcrossprod(A_inv %*% grad_hat) / (1 + crossprod(grad_hat, A_inv %*% grad_hat)[1])
      A_INV_ARR[,,k] <- A_inv
      BETA[,k] <- BETA[,k] - GAMMA[k]^{-1} * A_inv %*% grad_hat
      
      if (sqrt(crossprod(BETA[,k])[1]) > D)
        BETA[,k] <- generalized_projection(A_inv, BETA[,k], D)
    }
    
    ### BOA aggregation step
    term1 <- GAMMA*as.numeric(crossprod(grad_boa, BETA-tcrossprod(beta_boa, matrix(1,K,1))))
    pi_boa2 <- exp(pmax(-100,pmin(100, log(pi_boa2) -term1 - term1^2)))
    pi_boa2 <- pi_boa2/sum(pi_boa2)
    pi_boa <- (GAMMA*pi_boa2)/sum(GAMMA*pi_boa2)
    
    if (t < n) {
      BETA_BOA_ARR[t+1,] <- BETA %*% pi_boa
      PI_BOA_ARR[t+1,] <- pi_boa
    }
    
  }
  list(BETA_ARR=BETA_ARR, BETA_BOA_ARR=BETA_BOA_ARR, PI_BOA_ARR=PI_BOA_ARR, LIK_BOA = LIK_BOA, gamma_temp = gamma_temp,GRAD_BOA=GRAD_BOA, GRAD_BOA_HAT=GRAD_BOA_HAT )
}



