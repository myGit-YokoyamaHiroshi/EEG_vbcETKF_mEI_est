# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:15:10 2024

@author: H.Yokoyama
"""

import numpy as np
from scipy.linalg import sqrtm, lu
from scipy.special import digamma, gammaln
from numpy.linalg import slogdet

class vbETKF_JansenRit:
    def __init__(self, X, P, Q, R, dt, Npar):
        ###############
        self.X       = X
        self.P       = P
        self.Q       = Q
        self.R       = R
        self.dt      = dt
        self.Npar    = Npar
        self.Nstate  = 6
        self.a0      = 1E-3
        self.b0      = 1E-3
        
        self.a       = self.a0 + 1/2
        self.b       = self.b0
        self.H       = np.array([[0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    def Sigm(self, v):
        v0   = 6
        vmax = 5
        r    = 0.56
        sigm = vmax / (1 + np.exp(r * ( v0 - v )))
        
        return sigm
    
    def postsynaptic_potential_function(self, y, z, A, a, Sgm):
        dy = z
        dz = A * a * Sgm - 2 * a * z - a**2 * y
        
        f_out = np.vstack((dy, dz)).T
        return f_out
    
    def JansenRit_model(self, x, par):
        A       = par[:,0]
        a       = par[:,1]
        B       = par[:,2]
        b       = par[:,3]
        u       = par[:,4]
        
        dx      = np.zeros(x.shape)
        C       = 135
        c1      = 1.0  * C
        c2      = 0.8  * C
        c3      = 0.25 * C
        c4      = 0.25 * C
        
        Sgm_12  = self.Sigm(x[:,1] - x[:,2]);
        Sgm_p0  = u + c2 * self.Sigm(c1*x[:,0]);
        Sgm_0   = c4 * self.Sigm(c3*x[:,0]);
            
        dx_03   = self.postsynaptic_potential_function(x[:,0], x[:,3], A, a, Sgm_12);
        dx_14   = self.postsynaptic_potential_function(x[:,1], x[:,4], A, a, Sgm_p0);
        dx_25   = self.postsynaptic_potential_function(x[:,2], x[:,5], B, b, Sgm_0);
        
        # sort order of dy
        dx[:,0] = dx_03[:,0]
        dx[:,3] = dx_03[:,1]
        
        dx[:,1] = dx_14[:,0]
        dx[:,4] = dx_14[:,1]
        
        dx[:,2] = dx_25[:,0]
        dx[:,5] = dx_25[:,1]
        
        dX      = np.hstack((dx, np.zeros(par.shape)))
        
        return dX
    
    def state_func(self, x, par):
        dt     = self.dt
        X_now  = np.hstack((x, par))
        # X_next = X_now + dt * self.JansenRit_model(x, par)
        
        k1   = self.JansenRit_model(X_now[:,:6], X_now[:,6:])

        X_k2 = X_now + (dt/2)*k1
        k2   = self.JansenRit_model(X_k2[:,:6], X_k2[:,6:])
        
        X_k3 = X_now + (dt/2)*k2
        k3   = self.JansenRit_model(X_k3[:,:6], X_k3[:,6:])
        
        X_k4 = X_now + dt*k3
        k4   = self.JansenRit_model(X_k4[:,:6], X_k4[:,6:])
        
        X_next = X_now + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        return X_next
    
    
    def inv_lu(self, X):
        p,l,u = lu(X, permute_l = False)
        l     = np.dot(p,l) 
        l_inv = np.linalg.inv(l)
        u_inv = np.linalg.inv(u)
        X_inv = np.dot(u_inv,l_inv)
        
        return X_inv

    ###################
    def predict(self):
        X       = self.X
        P       = self.P
        Q       = self.Q
        Npar    = self.Npar
        Nstate  = self.Nstate
        x_sgm   = np.random.multivariate_normal(mean=X, cov=P, size=Npar)
        v       = np.random.multivariate_normal(mean=np.zeros(len(X)), cov=Q, size=Npar)
        X_sgm   = self.state_func(x_sgm[:,:Nstate], x_sgm[:,Nstate:]) + v
        
        XPred   = np.mean(X_sgm, axis=0)  
        
        self.X      = XPred
        self.X_sgm  = X_sgm
    
    def update(self):
        z     = self.z
        X     = self.X
        X_sgm = self.X_sgm
        R     = np.array([[self.R]])
        Npar  = self.Npar
        a      = self.a 
        b      = self.b
        H      = self.H
        
        eta    = b/a
        ####### parameter settings for constraint algorithm
        D     = np.array([
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                         ])
        ub    = np.array([100.00,  200, 100.00,  200, 320])
        lb    = np.array([  0.01,    5,   0.01,    5, 120])
        c     = np.zeros(ub.shape)
        ######## update state X based on ETKF algorithm
        zPred = H @ X
        dX    = X_sgm.T - X[:, np.newaxis] # Nstate x Npar
        dY    = (H@dX) # Nroi x Npar
        
        C     = dY.T @ np.linalg.inv(eta*R) # Npar x Nroi
        Pa    = np.linalg.inv((Npar-1)*np.eye(Npar) + C @ dY) # Npar x Npar
        T     = np.real(sqrtm((Npar-1)*Pa)) # Npar x Npar
        TT    = T @ T.T # Npar x Npar
        w_a   = Pa @ C @ (z - zPred[:,np.newaxis]) # Npar x 1
        T_a   = w_a + T # Npar x Npar
        
        X_sgm_new = X[:,np.newaxis] + dX @ T_a # Nstate x Npar
        X_new     = X_sgm_new.mean(axis=1)
        P_new     = (dX @ TT @ dX.T)/(Npar-1)
        
        ######## update gamma prior parameters
        a     = a + 1/2
        b     = b + 1/2 * np.sum(((z - (H@X_new))**2)/np.diag(R)) + 1/2 * np.trace((H@P_new@H.T/np.diag(R)))  
            
        ##### inequality constraints ##########################################
        ###   Constraint would be applied only when the inequality condition is not satisfied.
        W_inv = np.linalg.inv(P_new)
        L     = W_inv @ D.T @ np.linalg.inv(D @ W_inv @ D.T)
        value = D @ X_new 
        for i in range(len(value)):
            if (value[i] > ub[i]) | (value[i] < lb[i]):
                if (value[i] > ub[i]): 
                    c[i] = ub[i]
                elif (value[i] < lb[i]):
                    c[i] = lb[i]
        ## Calculate state variables with interval contraints
        X_c   = X_new - L @ (D @ X_new - c)
        for i in range(len(value)):
            if (value[i] > ub[i]) | (value[i] < lb[i]):
                X_new[i+6] = X_c[i+6]
        ##### inequality constraints ##########################################
        
        ####### elbo ##########################################################
        s, logdetR = slogdet(R)
        s, logdetP = slogdet(P_new)
        P_inv      = np.linalg.inv(P_new)
        err        = (z - H @ X_new)
        N          = len(zPred)
        Nstate     = len(X)
        
        ll_state   = 1/2 * (-Nstate * np.log(2*np.pi) - logdetP - np.trace(P_inv@P_new))
        ll_obs     = 1/2 * (-N * np.log(2*np.pi) +  (digamma(a) - np.log(b)) - logdetR - a/b * (np.sum((err**2)/np.diag(R)) + np.trace((H@P_new@H.T)/np.diag(R))))
        ll_gamma   = (self.a-1)* (digamma(a)-np.log(b))-gammaln(self.a)+self.a*np.log(self.b)-self.b*(a/b)

        Hx         = Nstate/2 * (1 + np.log(2*np.pi)) + 1/2 * logdetP
        Heta       = a - np.log(b) + gammaln(a) + (1-a) * digamma(a)
        
        ELBO       = ll_state + ll_obs + ll_gamma + Hx + Heta
        
        self.X     = X_new
        self.P     = P_new
        self.zPred = zPred
        self.elbo  = ELBO
        self.a     = a
        self.b     = b
    
    def vbetkf_estimation(self, z):   
        self.z      = z
        
        # Prediction step (estimate state variable)
        self.predict()
        
        # Update state (Update parameters)
        self.update()
