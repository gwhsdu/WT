from typing import List
import torch
from torch.nn.functional import conv1d, conv2d, pad

##################################### 2D #####################################

def permute_y2x(attr_y):
    attr_x = []
    for i in attr_y:
        if i is None:
            attr_x.append(i)
        elif isinstance(i,tuple):
            tmp=(j.permute(0,1,3,2) for j in i)
            attr_x.append(tmp)
        else:
            attr_x.append(i.permute(0,1,3,2))
    return attr_x

class operator_base2D(object):
    def __init__(self, accuracy = 2, device='cpu') -> None:
        self.accuracy = accuracy
        self.device = device

        self.centralfilters_y_1 = [None,None,
            torch.tensor([[[[-.5, 0, 0.5]]]],device=self.device),
            None,
            torch.tensor([[[[1/12, -2/3, 0, 2/3, -1/12]]]],device=self.device),
        ]
        
        self.centralfilters_y_2nd_derivative = [None,None,
            torch.tensor(
                [[[[1., -2., 1.]]]],device=self.device),
            None,
            torch.tensor(
                [[[[-1/12, 4/3, -5/2, 4/3, -1/12]]]],device=self.device),
            None,
            torch.tensor(
                [[[[1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]]]],device=self.device),
            None,
            torch.tensor(
                [[[[-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]]]],device=self.device)
        ]
        self.centralfilters_y_4th_derivative = [None,None,
            torch.tensor([[[[1., -4., 6., -4., 1.]]]],device=self.device),
            None,
            torch.tensor([[[[-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6]]]],device=self.device),
            None,
            torch.tensor([[[[7/240, -2/5, 169/60, -122/15, 91/8, -122/15, 169/60, -2/5, 7/240]]]],device=self.device),
        ]
        self.forwardfilters_y:list(torch.Tensor) = [None,
            torch.tensor(
                [[[[-1., 1.]]]],device=self.device),
            torch.tensor(
                [[[[-3/2, 2., -1/2]]]],device=self.device),
            torch.tensor(
                [[[[-11/6, 3, -3/2, 1/3]]]],device=self.device),    
        ]

        self.backwardfilters_y:list(torch.Tensor) = [None,
            torch.tensor(
                [[[[-1., 1.]]]],device=self.device),
            torch.tensor(
                [[[[1/2, -2., 3/2]]]],device=self.device),
            torch.tensor(
                [[[[-1/3, 3/2, -3., 11/6]]]],device=self.device),
        ]

        self.centralfilters_x_1:list(torch.Tensor) = permute_y2x(self.centralfilters_y_1)
        self.centralfilters_x_2nd_derivative:list(torch.Tensor) = permute_y2x(self.centralfilters_y_2nd_derivative)
        self.centralfilters_x_4th_derivative:list(torch.Tensor) = permute_y2x(self.centralfilters_y_4th_derivative)
        self.forwardfilters_x:list(torch.Tensor) = permute_y2x(self.forwardfilters_y)
        self.backwardfilters_x:list(torch.Tensor) = permute_y2x(self.backwardfilters_y)

        self.xschemes = {'Central1':self.centralfilters_x_1,
                        'Central2':self.centralfilters_x_2nd_derivative,
                        'Central4':self.centralfilters_x_4th_derivative,
                        'Forward1':self.forwardfilters_x,
                        'Backward1':self.backwardfilters_x}
        self.yschemes = {'Central1':self.centralfilters_y_1,
                        'Central2':self.centralfilters_y_2nd_derivative,
                        'Central4':self.centralfilters_y_4th_derivative,
                        'Forward1':self.forwardfilters_y,
                        'Backward1':self.backwardfilters_y}

        self.yschemes['Upwind1'] =  [(None,None),
            (torch.tensor([[[[0, -1., 1.]]]],device=self.device),
             torch.tensor([[[[-1., 1., 0.]]]],device=self.device)),
            (torch.tensor([[[[0, 0, -3/2, 2., -1/2]]]],device=self.device),
             torch.tensor([[[[1/2, -2., 3/2, 0, 0]]]],device=self.device)),
            (torch.tensor([[[[0, 0, 0, -11/6, 3, -3/2, 1/3]]]],device=self.device),
             torch.tensor([[[[-1/3, 3/2, -3., 11/6, 0, 0, 0]]]],device=self.device)), 
        ]

        self.xschemes['Upwind1'] = {}
        for i,v in enumerate(self.yschemes['Upwind1']):
            self.xschemes['Upwind1'][i] = permute_y2x(v)

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class d2udx2_2D(operator_base2D):
    def __init__(self, scheme='Central2', accuracy = 2, device='cpu') -> None:
        super().__init__(accuracy, device)
        assert accuracy%2 == 0, 'diffusion operator precision must be even number'
        assert scheme == 'Central2' or scheme == 'Central4' or scheme == 'Forward1' or scheme == 'Backward1', 'scheme must be one of the following: Central2, Central4, Forward1, Backward1'
        
        self.filter  = self.xschemes[scheme][accuracy]

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        return conv2d(u, self.filter)

class d2udy2_2D(operator_base2D):
    def __init__(self, scheme='Central2', accuracy = 2, device='cpu') -> None:
        super().__init__(accuracy, device)
        assert accuracy%2 == 0, 'diffusion operator precision must be even number'
        assert scheme == 'Central2' or scheme == 'Central4' or scheme == 'Forward1' or scheme == 'Backward1', 'scheme must be one of the following: Central2, Central4, Forward1, Backward1'
        self.filter  = self.yschemes[scheme][accuracy]

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        return conv2d(u, self.filter)

class dudx_2D(operator_base2D):
    def __init__(self, scheme='Upwind1', accuracy = 1, device='cpu') -> None:
        super().__init__(accuracy, device)
        self.xscheme = scheme
        self.filter = self.xschemes[scheme][accuracy]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        if self.xscheme == 'Upwind1':
            return (u[:,:,self.accuracy:-self.accuracy]<=0)*conv2d(u, self.filter[0]) +\
                 (u[:,:,self.accuracy:-self.accuracy]>0)*conv2d(u, self.filter[1])
        else:
            return conv2d(u, self.filter)

class dudy_2D(operator_base2D):
    def __init__(self, scheme='Upwind1', accuracy = 1, device='cpu') -> None:
        super().__init__(accuracy, device)
        self.yscheme = scheme
        self.filter = self.yschemes[scheme][accuracy]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        if self.yscheme == 'Upwind1':
            return (u[:,:,:,self.accuracy:-self.accuracy]<=0)*conv2d(u, self.filter[0]) +\
                 (u[:,:,:,self.accuracy:-self.accuracy]>0)*conv2d(u, self.filter[1])
        else:
            return conv2d(u, self.filter)
        


