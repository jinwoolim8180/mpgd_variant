from abc import ABC, abstractmethod
import torch
import numpy as np
import os

import sys
from omegaconf import OmegaConf
from .util import instantiate_from_config
from util.svd_operators import jacobian_of_f
import glob

__CONDITIONING_METHOD__ = {}

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.train()
    # model.eval()
    return model

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])
    return model, global_step

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, create_graph=False)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
    
    def tangent_norm(self, x_prev, x_next, point, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = point - x_next
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, create_graph=False)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
    
    # def tangent_norm(self, x_0_hat, measurement, V, **kwargs):
    #     b, x, m = V.shape
    #     b, c, h, w = x_0_hat.shape
    #     # AV = []
    #     # for i in range(m):
    #     #     AV.append(self.operator.forward(V[:,:,i].reshape(b, c, h, w)).reshape(b, -1))
    #     # AV = torch.stack(AV, dim=-1)
    #     # AV_operator = torch.linalg.pinv(AV)
    #     V_operator = torch.linalg.pinv(V)
    #     if self.noiser.__name__ == 'gaussian':
    #         difference = self.operator.forward(x_0_hat, **kwargs) - measurement
    #         # b = difference.shape[0]
    #         temp = self.operator.transpose(difference, **kwargs).reshape(b, -1)
    #         # c = torch.einsum('bij, bj -> bi', AV_operator, difference.reshape(b, -1))
    #         c = torch.einsum('bij, bj -> bi', V_operator, temp.reshape(b, -1))
    #     else:
    #         raise NotImplementedError
        
    #     return c
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='mpgd_wo_proj')
class WoProjSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_0_hat, measurement, at, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_0_hat = x_0_hat.detach()
        x_0_hat -= norm_grad * self.scale / at.sqrt()
        return x_0_hat, norm, 1
    
@register_conditioning_method(name='mpgd_w_proj')
class WProjSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_0_hat, measurement, at, **kwargs):
        t = kwargs.get('t', 0.5)
        if 0.8 > t > 0.2:
            # norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            # x_0_hat -= norm_grad * self.scale / at.sqrt()
            x_0_hat = x_0_hat + self.operator.transpose(measurement - self.operator.forward(x_0_hat))
            norm = torch.linalg.norm(x_0_hat)
        else:
            norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            x_0_hat -= norm_grad * self.scale / at.sqrt()
        
        return x_0_hat, norm, 1
    
@register_conditioning_method(name='ddnm')
class DDNM(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_0_hat, measurement, at, **kwargs):
        t = kwargs.get('t', 0.5)
        sigma_t = (1 - at).mean().sqrt()
        gamma_t = 1
        sigma_y = 0.05
        if 1.0 > t > 0.1:
            # norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            if sigma_t.mean() >= at.mean().sqrt()*sigma_y: 
                lambda_t = 1
                gamma_t = (sigma_t**2 - at * (lambda_t*sigma_y)**2).sqrt()
            else:
                lambda_t = sigma_t/(at*sigma_y)
                gamma_t = 0
            # x_0_hat -= norm_grad * self.scale / at.sqrt()
            x_0_hat = x_0_hat + lambda_t*self.operator.transpose(measurement - self.operator.forward(x_0_hat))
            norm = torch.linalg.norm(x_0_hat)
        else:
            norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            x_0_hat -= norm_grad * self.scale / at.sqrt()
        
        return x_0_hat, norm, gamma_t

@register_conditioning_method(name='mpgd_ae')
class AESampling(ConditioningMethod):
    def __init__(self, operator, noiser, resume, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        ckpt = None
        if os.path.isfile(resume):
            try:
                resumedir = '/'.join(resume.split('/')[:-1])
            except ValueError:
                paths = resume.split("/")
                idx = -2
                resumedir = "/".join(paths[:idx])
            ckpt = resume
        else:
            assert os.path.isdir(resume), f"{resume} is not a directory"
            resumedir = resume.rstrip("/")
            ckpt = os.path.join(resumedir, "model.ckpt")

        base_configs = sorted(glob.glob(os.path.join(resumedir, "config.yaml")))

        configs = [OmegaConf.load(cfg) for cfg in base_configs]
        cli = OmegaConf.from_dotlist([])
        config = OmegaConf.merge(*configs, cli)

        gpu = True
        eval_mode = True

        self.ldm_model, _ = load_model(config, ckpt, gpu, eval_mode)
        
    def conditioning(self, x_0_hat, measurement, at, **kwargs):
        t = kwargs.get('t', 0.5)
        if 0.5 > t > 0.3:
            E_x0_t = self.ldm_model.encode_first_stage(x_0_hat)
            D_x0_t = self.ldm_model.decode_first_stage(E_x0_t)
            # v = self.operator.transpose(self.operator.forward(D_x0_t) - measurement)
            # norm_grad = torch.autograd.functional.vjp(ae, x_0_hat, v=v, create_graph=False)[1]
            # norm = torch.linalg.norm(norm_grad)
            norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=D_x0_t, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            x_0_hat -= norm_grad * self.scale*0.0075 / at.sqrt()
        else:
            norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            x_0_hat -= norm_grad * self.scale / at.sqrt()
        return x_0_hat, norm, 1
    
@register_conditioning_method(name='mpgd_variant')
class VariantSampling(ConditioningMethod):
    def __init__(self, operator, noiser, resume, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        ckpt = None
        if os.path.isfile(resume):
            try:
                resumedir = '/'.join(resume.split('/')[:-1])
            except ValueError:
                paths = resume.split("/")
                idx = -2
                resumedir = "/".join(paths[:idx])
            ckpt = resume
        else:
            assert os.path.isdir(resume), f"{resume} is not a directory"
            resumedir = resume.rstrip("/")
            ckpt = os.path.join(resumedir, "model.ckpt")

        base_configs = sorted(glob.glob(os.path.join(resumedir, "config.yaml")))

        configs = [OmegaConf.load(cfg) for cfg in base_configs]
        cli = OmegaConf.from_dotlist([])
        config = OmegaConf.merge(*configs, cli)

        gpu = True
        eval_mode = True

        self.ldm_model, _ = load_model(config, ckpt, gpu, eval_mode)

    def get_basis(self, f, z):
        print("get basis")
        dx_dz = jacobian_of_f(f, z, create_graph=False)
        print("got basis")
        Q, R = torch.linalg.qr(dx_dz.permute(0, 2, 1))
        return Q
        
    def conditioning(self, x_0_hat, measurement, at, **kwargs):
        t = kwargs.get('t', 0.5)
        input_shape = x_0_hat.shape
        if 0.7 > t > 0.2:
            E_x0_t = self.ldm_model.encode_first_stage(x_0_hat)
            V = self.get_basis(self.ldm_model.decode_first_stage, E_x0_t)
            c = self.tangent_norm(x_0_hat, measurement, V, **kwargs)
            x_0_hat = x_0_hat.detach()
            # x_0_hat -= V @ c * self.scale*0.0075 / at.sqrt()
            grad = torch.einsum('bij, bj -> bi', V, c)
            grad = grad.reshape(input_shape)
            norm = torch.linalg.norm(grad)
            x_0_hat -= grad
            x_0_hat = self.operator.project(x_0_hat, measurement)
        else:
            norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            x_0_hat -= norm_grad * self.scale / at.sqrt()
        return x_0_hat, norm, 1
    
@register_conditioning_method(name='mpgd_admm')
class ADMMSampling(ConditioningMethod):
    def __init__(self, operator, noiser, resume, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.rho = 0.08
        self.steps = 2
        ckpt = None
        if os.path.isfile(resume):
            try:
                resumedir = '/'.join(resume.split('/')[:-1])
            except ValueError:
                paths = resume.split("/")
                idx = -2
                resumedir = "/".join(paths[:idx])
            ckpt = resume
        else:
            assert os.path.isdir(resume), f"{resume} is not a directory"
            resumedir = resume.rstrip("/")
            ckpt = os.path.join(resumedir, "model.ckpt")

        base_configs = sorted(glob.glob(os.path.join(resumedir, "config.yaml")))

        configs = [OmegaConf.load(cfg) for cfg in base_configs]
        cli = OmegaConf.from_dotlist([])
        config = OmegaConf.merge(*configs, cli)

        gpu = True
        eval_mode = True

        self.ldm_model, _ = load_model(config, ckpt, gpu, eval_mode)

    def conditioning(self, x_0_hat, measurement, at, **kwargs):
        t = kwargs.get('t', 0.5)
        input_shape = x_0_hat.shape
        # norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)

        # if self.z is None or self.u is None:
        #     self.z = x_0_hat.clone().detach()
        #     self.u = torch.zeros_like(x_0_hat).to(x_0_hat.clone())

        # x = (1 - self.rho) * x_0_hat + self.rho * self.z - self.rho * self.u
        # self.z = self.operator.project(x + self.u, measurement)
        # self.u += x - self.z

        if 0.5 > t > 0.2:
            # Step 1
            x_0_hat = x_0_hat.detach()
            z = x_0_hat + self.operator.transpose(measurement - self.operator.forward(x_0_hat))
            u = x_0_hat - z
            
            # Step 2
            x = (1 - self.rho) * x_0_hat + self.rho * z - self.rho * u
            z = x + u + self.operator.transpose(measurement - self.operator.forward(x + u))
            u += x - z
            norm = torch.linalg.norm(x - x_0_hat)

            # Step 3 - 
            x_prev = x_0_hat
            x_prev.requires_grad = True
            for _ in range(self.steps):
                x = x.detach()
                x.requires_grad = True
                E_x0_t = self.ldm_model.encode_first_stage(x_prev)
                D_x0_t = self.ldm_model.decode_first_stage(E_x0_t)
                grad, norm = self.tangent_norm(x_prev, x, D_x0_t, **kwargs)
                grad = grad.reshape(input_shape)
                inn = torch.sum(grad * (x - x_prev)) / norm
                x = x_prev - abs(inn) * grad
                x_prev = x
                x -= self.rho * (x - z + u)
                z = x + u + self.operator.transpose(measurement - self.operator.forward(x + u))
                u += x - z
            # x = z
        else:
            norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x = x_0_hat.detach()
            x -= norm_grad * self.scale / at.sqrt()
        return x, norm, 1
    
@register_conditioning_method(name='mpgd_admmp')
class ADMMSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, resume, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.rho = 0.1
        self.steps = 3
        ckpt = None
        if os.path.isfile(resume):
            try:
                resumedir = '/'.join(resume.split('/')[:-1])
            except ValueError:
                paths = resume.split("/")
                idx = -2
                resumedir = "/".join(paths[:idx])
            ckpt = resume
        else:
            assert os.path.isdir(resume), f"{resume} is not a directory"
            resumedir = resume.rstrip("/")
            ckpt = os.path.join(resumedir, "model.ckpt")

        base_configs = sorted(glob.glob(os.path.join(resumedir, "config.yaml")))

        configs = [OmegaConf.load(cfg) for cfg in base_configs]
        cli = OmegaConf.from_dotlist([])
        config = OmegaConf.merge(*configs, cli)

        gpu = True
        eval_mode = True

        self.ldm_model, _ = load_model(config, ckpt, gpu, eval_mode)

    def conditioning(self, x_0_hat, measurement, at, **kwargs):
        t = kwargs.get('t', 0.5)
        input_shape = x_0_hat.shape
        # norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)

        # if self.z is None or self.u is None:
        #     self.z = x_0_hat.clone().detach()
        #     self.u = torch.zeros_like(x_0_hat).to(x_0_hat.clone())

        # x = (1 - self.rho) * x_0_hat + self.rho * self.z - self.rho * self.u
        # self.z = self.operator.project(x + self.u, measurement)
        # self.u += x - self.z

        sigma_t = kwargs.get('sigma', 1.0)
        gamma_t = 1
        sigma_y = 0.05
        if 0.5 > t > 0.2:
            # norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            if sigma_t.mean() >= t*sigma_y:
                lambda_t = 1
                gamma_t = sigma_t**2 - (t*lambda_t*sigma_y)**2
            else:
                lambda_t = sigma_t/(at*sigma_y)
                gamma_t = 0
            z = x_0_hat + lambda_t*self.operator.transpose(measurement - self.operator.forward(x_0_hat))
            u = x_0_hat - z
            
            # Step 2
            x = (1 - self.rho) * x_0_hat + self.rho * z - self.rho * u
            z = x + u + lambda_t*self.operator.transpose(measurement - self.operator.forward(x + u))
            u += x - z
            norm = torch.linalg.norm(x - x_0_hat)

            # Step 3 - 
            x_prev = x_0_hat
            x_prev.requires_grad = True
            for _ in range(self.steps):
                x = x.detach()
                x.requires_grad = True
                E_x0_t = self.ldm_model.encode_first_stage(x_prev)
                D_x0_t = self.ldm_model.decode_first_stage(E_x0_t)
                grad, norm = self.tangent_norm(x_prev, x, D_x0_t, **kwargs)
                inn = torch.sum(grad * (x - x_prev)) / norm
                x = x_prev - inn * grad
                x_prev = x
                x -= self.rho * (x - z + u)
                z = x + u + lambda_t*self.operator.transpose(measurement - self.operator.forward(x + u))
                u += x - z
            # x = z
        else:
            norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x = x_0_hat.detach()
            x -= norm_grad * self.scale / at.sqrt()
        return x, norm, 1

@register_conditioning_method(name='mpgd_z')
class LatentSampling(ConditioningMethod):
    def __init__(self, operator, noiser, resume, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        ckpt = None
        if os.path.isfile(resume):
            try:
                resumedir = '/'.join(resume.split('/')[:-1])
            except ValueError:
                paths = resume.split("/")
                idx = -2
                resumedir = "/".join(paths[:idx])
            ckpt = resume
        else:
            assert os.path.isdir(resume), f"{resume} is not a directory"
            resumedir = resume.rstrip("/")
            ckpt = os.path.join(resumedir, "model.ckpt")

        base_configs = sorted(glob.glob(os.path.join(resumedir, "config.yaml")))

        configs = [OmegaConf.load(cfg) for cfg in base_configs]
        cli = OmegaConf.from_dotlist([])
        config = OmegaConf.merge(*configs, cli)

        gpu = True
        eval_mode = True

        self.ldm_model, _ = load_model(config, ckpt, gpu, eval_mode)
        

    def conditioning(self, x_0_hat, measurement, at, **kwargs):
        t = kwargs.get('t', 0.5)
        if 0.5 > t > 0.3:
            E_x0_t = self.ldm_model.encode_first_stage(x_0_hat)
            E_x0_t = E_x0_t.detach()
            E_x0_t.requires_grad = True
            D_x0_t = self.ldm_model.decode_first_stage(E_x0_t)
            v = self.operator.transpose(self.operator.forward(D_x0_t) - measurement)
            norm_grad = torch.autograd.functional.vjp(self.ldm_model.decode_first_stage, E_x0_t, v=v, create_graph=False)[1]
            norm = torch.linalg.norm(norm_grad)
            # norm_grad, norm = self.grad_and_value(x_prev=E_x0_t, x_0_hat=D_x0_t, measurement=measurement, **kwargs)
            diff = x_0_hat.detach() - D_x0_t.detach()
            E_x0_t = E_x0_t.detach()
            E_x0_t -= self.scale*0.01*norm_grad / at[0,0,0,0].sqrt()
            D_x0_t = self.ldm_model.decode_first_stage(E_x0_t)
            x_0_hat = D_x0_t.detach() + diff
            x_0_hat = x_0_hat.detach()
        else:
            norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            x_0_hat -= norm_grad * self.scale / at.sqrt()
        return x_0_hat, norm, 1
