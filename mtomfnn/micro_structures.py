# -*- coding: utf-8 -*-

import torch
import numpy

class MicroStruct:
    def __init__(self) -> None:
        pass
    def getCE_and_CEDiff(self):
        raise NotImplementedError('get_CE must be defined')


class Crossed_2D(MicroStruct):
    def __init__(self, model_L, model_Linear, model_noLinear) -> None:
        self.model_L = torch.load(model_L) 
        self.model_Linear = torch.load(model_Linear)  
        self.model_noLinear = torch.load(model_noLinear) 
        self.alpha = 0.5

    

    def getCE_and_CEDiff(self, x):
        x = numpy.asanyarray(x).reshape((-1,1))
        x = torch.from_numpy(x.reshape((-1, 1))).float().requires_grad_()
        alpha = self.alpha
        pred_L = self.model_L(x)
        pred_H = alpha * self.model_noLinear(torch.cat((x, pred_L), 1)) +\
              (1 - alpha) * self.model_Linear(torch.cat((x, pred_L), 1))

        dyh_dx = torch.ones_like(pred_H)
        for i in range(3):
            dx_i = torch.autograd.grad(outputs=pred_H[:,i], inputs=x,
                        grad_outputs=torch.ones_like(pred_H[:,i]),
                          retain_graph=True)[0]
            dyh_dx[:,i] = dx_i[:,0]

        
        return pred_H.detach().numpy(), dyh_dx.detach().numpy()


class SIMP(MicroStruct):
    def __init__(self, E0=1, nu=0.3, penal=3.0, Emin=1e-9) -> None:
        self.E0 = E0
        self.nu = nu
        self.penal = penal
        self.Emin = Emin

    

    def getCE_and_CEDiff(self, x):
        E0 = self.E0
        nu = self.nu
        penal = self.penal
        Emin = self.Emin
        
        CE_com = numpy.array([1/(1-nu**2), nu/(1-nu**2), 1/((1+nu)*2)])
        
        
        CE = numpy.kron((Emin + (E0 - Emin) * numpy.power(x, penal)).reshape((-1,1)),
                        numpy.ones((1,3))) * CE_com
        
        CE_Diff =  numpy.kron(((E0 - Emin)*penal*numpy.power(x, penal-1)).reshape((-1,1)),
                              numpy.ones((1,3))) * CE_com
        
        
        
        return CE, CE_Diff
        
        
if __name__ == '__main__':
    micro = SIMP()
    x = numpy.array([0,0.1,0.5,1])
    print(micro.getCE_and_CEDiff(x))
        

    
    
