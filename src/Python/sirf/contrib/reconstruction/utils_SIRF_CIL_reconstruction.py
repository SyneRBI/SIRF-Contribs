#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:05:07 2019

@author: sirfuser
"""
from ccpi.optimisation.operators import LinearOperator
from ccpi.optimisation.functions import KullbackLeibler
from sirf.Utilities import check_status, assert_validity
import sirf.pystir as pystir
import pSTIR as pet
import numpy as np
import scipy

def run_CIL_SIRF_utils():

    # For the PDHG algorithm we need a norm method for the AcquisitionModel of SIRF
    def norm(self):
        return LinearOperator.PowerMethod(self, 50)[0]
          
    # No need to add negative sign
    def SIRF_KL_call(self, x):
        
        return self.get_value(x) 
#        tmp = self.b.as_array() - self.b.as_array() * np.log(np.maximum(self.b.as_array(),1e-6))
#        return self.get_value(x) - tmp.sum()
    
#    def Kullback_CIL(self, x):
#        
#        ind = x.as_array()>0
#        tmp = scipy.special.kl_div(self.b.as_array()[ind], x.as_array()[ind]) 
#        tmp1 = self.b.as_array() - self.b.as_array() * np.log(np.maximum(self.b.as_array(),1e-6))
#        return (np.sum(tmp) + tmp1.sum()
        
    # SIRF: Maximize -objective
    # CIL: Minimize objective
    # need a minus for the gradient
    
    def gradient(self, image, subset = -1, out = None):
        
        assert_validity(image, pet.ImageData)
        grad = pet.ImageData()
        grad.handle = pystir.cSTIR_objectiveFunctionGradient\
            (self.handle, image.handle, subset)
        check_status(grad.handle)
        
        if out is None:
            return -1*grad  
        else:
            out.fill(-1*grad)
            
    def KL_convex_conjugate(self, x):
        
        '''Convex conjugate of KullbackLeibler at x'''
        
        xlogy = - scipy.special.xlogy(self.b.as_array(), 1 - x.as_array())
        return np.sum(xlogy)        
            
    def KL_proximal_conjugate(self, x, tau, out=None):
            
            r'''Proximal operator of the convex conjugate of KullbackLeibler at x:
               
               .. math::     prox_{\tau * f^{*}}(x)
            '''        
                    
            if out is None:
                z = x + tau * self.bnoise
                return 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.b).sqrt())
            else:
                
                tmp = tau * self.bnoise
                tmp += x
                tmp -= 1
                
                self.b.multiply(4*tau, out=out)    
                
                out.add(tmp.power(2), out=out)
                out.sqrt(out=out)
                out *= -1
                tmp += 2
                out += tmp
                out *= 0.5        
    
    setattr(pet.ObjectiveFunction, '__call__', SIRF_KL_call)
    setattr(pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData, 'gradient', gradient)
    setattr(pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData, 'proximal_conjugate', KL_proximal_conjugate)
    setattr(pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData, 'convex_conjugate', KL_convex_conjugate)
    setattr(pet.AcquisitionModelUsingRayTracingMatrix, 'norm', norm)
#    setattr(KullbackLeibler, '__call__', Kullback_CIL)