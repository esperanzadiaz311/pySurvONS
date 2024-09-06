###import the algorithms
source('online_algorithms.R')

###excel library
library('openxlsx')


#=============== Fixed Parameters for the data simulation ================

###number of individuals
N <- 10^4
###number of iterations of ONS algorithm 
n_it <- 10^3
###Maximum arrival time
max_T <- 10^3
###covariates dimension
d <- 4
###covariates parameter
eta <- matrix(c(0,0,0), 3, 1)
sigma <- diag(c(1,1,1))



# ============== Monte Carlo simualtions grid 1 ============================

# GRID 1

#Learning rates up to 1/4GD (environ 50) (1.2)
GAMMA <- seq(log(1/n_it), log(50), by=1.2)
GAMMA <- exp(GAMMA)
#Length of the grid
K<- length(GAMMA)


M <-5

beta_real <- matrix(0,d,M)
grid_bound <- matrix(0,M,1)

beta_boa <- array(0, dim = c(n_it,d,M))
beta_surv <- array(0, dim = c(n_it,d,M))
beta_ogd <- array(0, dim = c(n_it,d,K,M))
beta_ons <- array(0, dim = c(n_it,d,K,M))

like_boa <- array(0, dim = c(n_it,M))
like_surv <- array(0, dim = c(n_it,M))
like_ons <- array(0, dim = c(n_it,K,M))
like_ogd <- array(0, dim = c(n_it,K,M))
like_real <-array(0, dim = c(n_it,M))

gamma_t <- array(0, dim = c(n_it,M))

print("Monte Carlo simulations grid 1")
for (m in 1:M){
  print("simulation:")
  print(m)
  
  # ===== sample dataset ====
  
  #real parameter
  beta_real[,m] <- rnorm(d,0,1)
  #diameter
  D <- 1.1*sqrt(crossprod(beta_real[,m]))[1,]
  #epsilon grid
  EPSILON <- 1/(GAMMA*D)^2
  #covariates
  X <- mvtnorm::rmvnorm(N, eta, sigma)
  X1 <- cbind(1,X)
  #arrival time
  arrival_time = runif(N, min=0, max = max_T)
  #event time
  Time_indiv <- arrival_time + sapply(1:N, function(i) rexp(1,rate=exp(crossprod(beta_real[,m], X1[i,])[1])))
  #censor time
  Censor_indiv <- arrival_time + sapply(1:N, function(i) rexp(1,rate=exp(crossprod(beta_real[,m], X1[i,])[1])))
  #observed time
  hat_T <- sapply(1:N, function(i) {min(Time_indiv[i], Censor_indiv[i])})
  #status indicator
  delta <- (Time_indiv < Censor_indiv)
  # R[[t]] at risk at time t
  R <- list()
  for (t in 1:n_it)
    R[[t]] <- c(1)
  for (i in 2:N) {
    t1 <- max(1,floor(arrival_time[i])-1)
    t2 <- min(n_it,floor(hat_T[i])+1)
    for (t in t1:t2)
      R[[t]] <- c(R[[t]], i)
  }
  
  # ====================== algorithms =====================

  ###BOA
  #boa_ons <- ons_boa(arrival_time, hat_T, delta, X1, D, GAMMA, n_it, EPSILON, R)
  #beta_boa[,,m] <- boa_ons$BETA_BOA_ARR
  #beta_ons[,,,m] <- boa_ons$BETA_ARR
  ###SurvONS
  survons <- ons_boa_max(arrival_time, hat_T, delta, X1, D, GAMMA, n_it, EPSILON, R)
  beta_surv[,,m] <- survons$BETA_BOA_ARR
  ###OGD
  for (k in 1:K){
    ogd <- grad_descent(arrival_time,hat_T,delta,X1,D,GAMMA[k],n_it,R)
    beta_ogd[,,k,m] <- ogd$beta_arr
    like_ogd[,k,m] <- ogd$like_arr
  }

  #gradient
  G <- max(sapply(1:n_it, function(t) {sqrt(crossprod(survons$GRAD_BOA[t,]))}))
  grid_bound[m] <- 1/(4*G*D)
  
  gamma_t[,m] <- survons$gamma_temp
  
  
  # =================== likelihood ========================
  
  #aggregation
  #like_boa[,m] <- boa_ons$LIK_BOA
  like_surv[,m] <- survons$LIK_BOA
  
  #real
  for (t in 1:n_it){
    like_real[t,m] <- instgrad(t,arrival_time,hat_T,delta,X1,beta_real[,m],R[[t]])$lik
  }
  
  #ons
  #for (k in 1:K){
  #  for (t in 1:n_it){
  #    like_ons[t,k,m] <- instgrad(t,arrival_time,hat_T,delta,X1,beta_ons[t,,k,m],R[[t]])$lik
  #  }
  #}
}


# ====================== Results ========================


###compute the averages

BETA_REAL_MEAN <- c(mean(beta_real[1,]), mean(beta_real[2,]), mean(beta_real[3,]),mean(beta_real[4,]))

#beta_boa_mean <- array(0, dim = c(n_it,d))
beta_surv_mean <- array(0, dim = c(n_it,d))

#beta_ogd_mean <- array(0, dim = c(n_it,d,K))
#beta_ons_mean <- array(0, dim = c(n_it,d,K))

for (idx in 1:d){
  for (t in 1:n_it){
    #beta_boa_mean[t,idx]<- mean(beta_boa[t,idx,])
    beta_surv_mean[t,idx] <- mean(beta_surv[t,idx,])
  }
}

#for (k in 1:K){
#  for (idx in 1:d){
#    for (t in 1:n_it){
#      beta_ogd_mean[t,idx,k]<- mean(beta_ogd[t,idx,k,])
#      beta_ons_mean[t,idx,k] <- mean(beta_ons[t,idx,k,])
#    }
#  }
#}


###save results
#write.xlsx(data.frame(beta_boa_mean),'grid1_beta_boa.xlsx')
write.xlsx(data.frame(beta_surv_mean),'grid1_beta_surv.xlsx')
#write.xlsx(data.frame(BETA_REAL_MEAN),'grid1_beta_real.xlsx')


# =================== Estimation Error =================

###chose the best ONS and OGD to save

BETA_REAL_ARR = t(matrix(rep(as.numeric(BETA_REAL_MEAN),n_it), nrow = 4))

#error <- matrix(0,K,1)
#for (k in 1:K){
#  error[k] <- cumsum(apply((beta_ogd_mean[,,k] - BETA_REAL_ARR)^2, 1, sum))[n_it]
#}
#idx_ogd <- which.min(error)


#error <- matrix(0,K,1)
#for (k in 1:K){
#  error[k] <- cumsum(apply((beta_ons_mean[,,k] - BETA_REAL_ARR)^2, 1, sum))[n_it]
#}

#idx_ons <- which.min(error)


###save results
#write.xlsx(data.frame(beta_ons_mean[,,idx_ons]),'grid1_beta_ons.xlsx')
#write.xlsx(data.frame(beta_ogd_mean[,,idx_ogd]),'grid1_beta_ogd.xlsx')

# ====================== Likelihood =================

###compute the averages
#like_boa_mean <- matrix(0,n_it,1)
like_surv_mean <- matrix(0,n_it,1)
#like_ons_mean <- matrix(0,n_it,K)
#like_ogd_mean <- matrix(0,n_it,K)
#like_real_mean <- matrix(0,n_it,1)

"for (k in 1:K){
  for (t in 1:n_it){
    like_ogd_mean[t,k] <- mean(like_ogd[t,k,])
    like_ons_mean[t,k] <- mean(like_ons[t,k,])
  }
}"


for (t in 1:n_it){
  #like_real_mean[t] <- mean(like_real[t,])
  #like_boa_mean[t]<- mean(like_boa[t,])
  like_surv_mean[t] <- mean(like_surv[t,])
}



#save results
#write.xlsx(data.frame(like_boa_mean),'grid1_like_boa.xlsx')
write.xlsx(data.frame(like_surv_mean),'grid1_like_surv.xlsx')
#write.xlsx(data.frame(like_ons_mean[,idx_ons]),'grid1_like_ons.xlsx')
#write.xlsx(data.frame(like_ogd_mean[,idx_ogd]),'grid1_like_ogd.xlsx')
#write.xlsx(data.frame(like_real_mean),'grid1_like_real.xlsx')



# =================== gamma_t ===================

###compute the average optimal learning rate
gamma_mean <- matrix(0,n_it,1)
gamma_std <- matrix(0,n_it,1)
for (t in 1:n_it){
  gamma_mean[t] <- mean(gamma_t[t,])
  gamma_std[t] <- sd(gamma_t[t,])
}

###save results
write.xlsx(data.frame(gamma_t),'grid1_gamma.xlsx')

















# ========================== Monte Carlo simulations grid 2 ============================

# GRID 2

#Big learning rates
GAMMA <- seq(log(200), log(2000), by=.25)
GAMMA <- exp(GAMMA)
#Length of the grid
K<- length(GAMMA)


M <-5

beta_real <- matrix(0,d,M)
grid_bound <- matrix(0,M,1)

#beta_boa <- array(0, dim = c(n_it,d,M))
beta_surv <- array(0, dim = c(n_it,d,M))
#beta_ogd <- array(0, dim = c(n_it,d,K,M))
#beta_ons <- array(0, dim = c(n_it,d,K,M))

#like_boa <- array(0, dim = c(n_it,M))
like_surv <- array(0, dim = c(n_it,M))
#like_ons <- array(0, dim = c(n_it,K,M))
#like_ogd <- array(0, dim = c(n_it,K,M))
#like_real <-array(0, dim = c(n_it,M))

gamma_t <- array(0, dim = c(n_it,M))



print("Monte Carlo simulations grid 2")
for (m in 1:M){
  print("simulation:")
  print(m)
  
  # ====================== sample dataset ===============
  
  #real parameter
  beta_real[,m] <- rnorm(d,0,1)
  #diameter
  D <- 1.1*sqrt(crossprod(beta_real[,m]))[1,]
  #epsilon grid
  EPSILON <- 1/(GAMMA*D)^2
  #covariates
  X <- mvtnorm::rmvnorm(N, eta, sigma)
  X1 <- cbind(1,X)
  #arrival time
  arrival_time = runif(N, min=0, max = max_T)
  #event time
  Time_indiv <- arrival_time + sapply(1:N, function(i) rexp(1,rate=exp(crossprod(beta_real[,m], X1[i,])[1])))
  #censor time
  Censor_indiv <- arrival_time + sapply(1:N, function(i) rexp(1,rate=exp(crossprod(beta_real[,m], X1[i,])[1])))
  #observed time
  hat_T <- sapply(1:N, function(i) {min(Time_indiv[i], Censor_indiv[i])})
  #status indicator
  delta <- (Time_indiv < Censor_indiv)
  # R[[t]] at risk at time t
  R <- list()
  for (t in 1:n_it)
    R[[t]] <- c(1)
  for (i in 2:N) {
    t1 <- max(1,floor(arrival_time[i])-1)
    t2 <- min(n_it,floor(hat_T[i])+1)
    for (t in t1:t2)
      R[[t]] <- c(R[[t]], i)
  }
  
  # ====================== algorithms =====================
  
  ###BOA
  "boa_ons <- ons_boa(arrival_time, hat_T, delta, X1, D, GAMMA, n_it, EPSILON, R)
  beta_boa[,,m] <- boa_ons$BETA_BOA_ARR
  beta_ons[,,,m] <- boa_ons$BETA_ARR"
  ###SurvONS
  survons <- ons_boa_max(arrival_time, hat_T, delta, X1, D, GAMMA, n_it, EPSILON, R)
  beta_surv[,,m] <- survons$BETA_BOA_ARR
  ###OGD
  "for (k in 1:K){
    ogd <- grad_descent(arrival_time,hat_T,delta,X1,D,GAMMA[k],n_it,R)
    beta_ogd[,,k,m] <- ogd$beta_arr
    like_ogd[,k,m] <- ogd$like_arr
  }"
  
  #gradient
  G <- max(sapply(1:n_it, function(t) {sqrt(crossprod(survons$GRAD_BOA[t,]))}))
  grid_bound[m] <- 1/(4*G*D)
  
  gamma_t[,m] <- survons$gamma_temp
  
  
  # =================== likelihood ========================
  
  #aggregation
  #like_boa[,m] <- boa_ons$LIK_BOA
  like_surv[,m] <- survons$LIK_BOA
  
  #real
  "for (t in 1:n_it){
    like_real[t,m] <- instgrad(t,arrival_time,hat_T,delta,X1,beta_real[,m],R[[t]])$lik
  }"
  
  #ons
  "for (k in 1:K){
    for (t in 1:n_it){
      like_ons[t,k,m] <- instgrad(t,arrival_time,hat_T,delta,X1,beta_ons[t,,k,m],R[[t]])$lik
    }
  }"
  
  
}



# ====================== Results ========================

###compute the averages
BETA_REAL_MEAN <- c(mean(beta_real[1,]), mean(beta_real[2,]), mean(beta_real[3,]),mean(beta_real[4,]))

#beta_boa_mean <- array(0, dim = c(n_it,d))
beta_surv_mean <- array(0, dim = c(n_it,d))

#beta_ogd_mean <- array(0, dim = c(n_it,d,K))
#beta_ons_mean <- array(0, dim = c(n_it,d,K))

for (idx in 1:d){
  for (t in 1:n_it){
    #beta_boa_mean[t,idx]<- mean(beta_boa[t,idx,])
    beta_surv_mean[t,idx] <- mean(beta_surv[t,idx,])
  }
}

"for (k in 1:K){
  for (idx in 1:d){
    for (t in 1:n_it){
      beta_ogd_mean[t,idx,k]<- mean(beta_ogd[t,idx,k,])
      beta_ons_mean[t,idx,k] <- mean(beta_ons[t,idx,k,])
    }
  }
}"


#save results
#write.xlsx(data.frame(beta_boa_mean),'grid2_beta_boa.xlsx')
write.xlsx(data.frame(beta_surv_mean),'grid2_beta_surv.xlsx')
#write.xlsx(data.frame(BETA_REAL_MEAN),'grid2_beta_real.xlsx')



# =================== Estimation Error =================

###chose the best ONS and OGD
"BETA_REAL_ARR = t(matrix(rep(as.numeric(BETA_REAL_MEAN),n_it), nrow = 4))


error <- matrix(0,K,1)
for (k in 1:K){
  error[k] <- cumsum(apply((beta_ogd_mean[,,k] - BETA_REAL_ARR)^2, 1, sum))[n_it]
}
idx_ogd <- which.min(error)

error <- matrix(0,K,1)
for (k in 1:K){
  error[k] <- cumsum(apply((beta_ons_mean[,,k] - BETA_REAL_ARR)^2, 1, sum))[n_it]
}

idx_ons <- which.min(error)"

#save results
#write.xlsx(data.frame(beta_ons_mean[,,idx_ons]),'grid2_beta_ons.xlsx')
#write.xlsx(data.frame(beta_ogd_mean[,,idx_ogd]),'grid2_beta_ogd.xlsx')



# ====================== Likelihood =================

###compute the averages
#like_boa_mean <- matrix(0,n_it,1)
like_surv_mean <- matrix(0,n_it,1)
#like_ons_mean <- matrix(0,n_it,K)
#like_ogd_mean <- matrix(0,n_it,K)
#like_real_mean <- matrix(0,n_it,1)

"for (k in 1:K){
  for (t in 1:n_it){
    like_ogd_mean[t,k] <- mean(like_ogd[t,k,])
    like_ons_mean[t,k] <- mean(like_ons[t,k,])
  }
}"


for (t in 1:n_it){
  #like_real_mean[t] <- mean(like_real[t,])
  #like_boa_mean[t]<- mean(like_boa[t,])
  like_surv_mean[t] <- mean(like_surv[t,])
}


#save results
#write.xlsx(data.frame(like_boa_mean),'grid2_like_boa.xlsx')
write.xlsx(data.frame(like_surv_mean),'grid2_like_surv.xlsx')
#write.xlsx(data.frame(like_ons_mean[,idx_ons]),'grid2_like_ons.xlsx')
#write.xlsx(data.frame(like_ogd_mean[,idx_ogd]),'grid2_like_ogd.xlsx')
#write.xlsx(data.frame(like_real_mean),'grid2_like_real.xlsx')



# =================== gamma_t ===================

###save the average learning rate
gamma_mean <- matrix(0,n_it,1)
gamma_std <- matrix(0,n_it,1)
for (t in 1:n_it){
  gamma_mean[t] <- mean(gamma_t[t,])
  gamma_std[t] <- sd(gamma_t[t,])
}


write.xlsx(data.frame(gamma_t),'grid2_gamma.xlsx')


