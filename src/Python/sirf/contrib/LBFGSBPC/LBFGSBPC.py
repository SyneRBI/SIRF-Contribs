#
# SPDX-License-Identifier: Apache-2.0
#
# Class implementing the LBFGSB-PC algorithm in sirf.STIR
#
# Authors:  Kris Thielemans
#
# Copyright 2025 University College London

import numpy as np
import numpy.typing as npt
import sirf.STIR as STIR
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

class LBFGSBPC:
    ''' Implementation of the LBFGSB-PC Algorithm
    
    See
    #sai et al,
    Fast Quasi-Newton Algorithms for Penalized Reconstruction in Emission Tomography and Further Improvements via Preconditioning
    IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. 37, NO. 4, APRIL 2018
    DOI: 10.1109/TMI.2017.2786865
    
    This implementation is NOT a CIL.Algorithm, but it behaves somewhat as one.
    '''
    def __init__(self, objfun: STIR.ObjectiveFunction, initial: STIR.ImageData, update_objective_interval: int = 0):
        self.objfun = objfun
        self.initial = initial
        self.shape = initial.shape
        self.output = None
        self.update_objective_interval = update_objective_interval
        precon = objfun.multiply_with_Hessian(initial, initial.get_uniform_copy(1))*-1
        self.Dinv = 1 / (np.sqrt(np.maximum(10, precon.as_array())).ravel() + 1e-0) # TODO remove arbitrary factors
        #precon.show()
        plt.imshow(np.reshape(self.Dinv, self.shape)[0,:,:])
        self.tmp_for_value = initial.clone()
        self.tmp_for_gradient = initial.clone()

    def precond_objfun_value(self, z: npt.ArrayLike) -> float:
        self.tmp_for_value.fill(np.reshape(z * self.Dinv, self.shape))
        return -self.objfun(self.tmp_for_value)

    def precond_objfun_gradient(self, z: npt.ArrayLike) -> np.ndarray:
        self.tmp_for_gradient.fill(np.reshape(z * self.Dinv, self.shape))
        return self.objfun.gradient(self.tmp_for_gradient).as_array().ravel() * self.Dinv * -1

    def callback(self, x):
        if self.update_objective_interval > 0 and self.iter % self.update_objective_interval == 0:
            self.loss.append(self.precond_objfun_value(x))
            self.iterations.append(self.iter)
        self.iter += 1

    def process(self) -> None:
        precond_init = self.initial.as_array().ravel() * self.Dinv
        bounds = precond_init.size * [(0, None)]
        self.iter = 0
        self.iterations = [0]
        self.loss = [self.precond_objfun_value(precond_init)] if self.update_objective_interval > 0 else []
        res = fmin_l_bfgs_b(
            self.precond_objfun_value,
            precond_init,
            self.precond_objfun_gradient,
            maxiter=50, # TODO use kwargs
            bounds=bounds,
            m=20,
            callback=self.callback,
            factr=0,
            pgtol=0,
        )
        # store result (use name "x" for CIL compatibility)
        self.x = self.tmp_for_value.get_uniform_copy(0)
        self.x.fill(np.reshape(res[0] * self.Dinv, self.shape))

    def run(self) -> None: # CIL alias, would need to support iterations keyword?
        self.process()
        
    def get_output(self) -> STIR.ImageData:
        return self.x