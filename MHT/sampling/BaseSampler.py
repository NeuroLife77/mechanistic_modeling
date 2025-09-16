import numpy as np
from pystorm import mnt
from datetime import datetime as dtm
from MHT.utils import dcp,broadcast_like
from scipy.stats.qmc import Sobol, MultivariateNormalQMC
import warnings
warnings.simplefilter("ignore", UserWarning)

class BaseSampler:

    def __init__(
                    self,
                    param_min, param_range = None, param_max = None,
                    min_norm = 0.0,width = 1.0, init_center = None,
                    random_seed = None, sample_distribution = 'full_cube',
                    use_sobol = False,  
                    return_numpy = True,
                    **kwargs
        ):
        
        self.param_min  = mnt.ensure_numpy(param_min)
        self.param_dim = param_min.shape[0]
        self.min_norm = min_norm

        if param_range is None and param_max is not None:
            self.param_max = mnt.ensure_numpy(param_max)
            self.param_range = self.param_max - self.param_min
        elif param_max is None and param_range is not None:
            self.param_range = mnt.ensure_numpy(param_range)  
            self.param_max = self.param_min + self.param_range
        elif param_max is not None and param_range is not None:
            self.param_range = mnt.ensure_numpy(param_range)  
            self.param_max = mnt.ensure_numpy(param_max)
        else:
            print("BaseSampler: No range")
            self.param_range = self.param_min * 2
            self.param_max = self.param_min + self.param_range

        if init_center is None:
            self.center = self.param_min + self.param_range/2
        else:
            self.center = init_center
        self.width = width

        if random_seed is None:
            self.noise_seed = np.random.Generator(np.random.PCG64(dtm.now().second)).integers(2**32)
        else:
            self.noise_seed = random_seed
        self.sample_generator = np.random.Generator(np.random.PCG64(self.noise_seed))

        self.sampling_method = None
        self.sample_distribution = sample_distribution
        self.set_sampling_method(self.sample_distribution) 
        self.use_sobol = use_sobol
        self.sobol_sampler = Sobol(self.param_dim, seed=self.sample_generator)
        self.normalQMC_sampler = MultivariateNormalQMC(np.zeros_like(self.param_min),np.eye(self.param_dim),engine=self.sobol_sampler,seed=self.sample_generator)
        self.ensure_fn = mnt.ensure_torch
        self.return_numpy = return_numpy
        if self.return_numpy:
            self.ensure_fn = mnt.ensure_numpy

        self._edge_behavior = self._wall_hug
        self._edge_behavior_kwargs = {}
        if "edge_behavior" in kwargs:
            self._set_edge_behavior(kwargs["edge_behavior"])
        if "edge_behavior_kwargs" in kwargs:
            self._edge_behavior_kwargs = kwargs["edge_behavior_kwargs"]
    # <-------------------------- init end --------------------------    
    
    # <----------- Bounds Related Helper Functions ----------->
    def _clip_bounds_range(self,sample_space_bounds):
        sample_space_bounds_ = dcp(sample_space_bounds)
        sample_space_bounds_[0] = np.where(sample_space_bounds_[0]<self.param_min,self.param_min,sample_space_bounds_[0])
        sample_space_bounds_[1] = np.where(sample_space_bounds_[1]>self.param_max,self.param_max,sample_space_bounds_[1])
        return sample_space_bounds_

    def _wall_pass(self,samples, sample_space_bounds = None, **kwargs):
        return self._wall_hug(samples)
    
    def _wall_hug(self,samples, sample_space_bounds = None, **kwargs):
        if sample_space_bounds is None:
            sample_space_bounds = [self.param_min,self.param_max]
        elif sample_space_bounds == "constrained":
            sample_space_bounds = [self.center + (-self.width/2 * self.param_range),self.center + (-self.width/2 * self.param_range)]
        sample_space_bounds = self._clip_bounds_range(sample_space_bounds)
        samples_ = np.where(samples < sample_space_bounds[0], sample_space_bounds[0], samples)
        samples_ = np.where(samples_ > sample_space_bounds[1], sample_space_bounds[1], samples_)
        return samples_
    
    def _bounce_wall(
            self,
            samples, sample_space_bounds = None, 
            bouncyness = 0.1, bounce_decay = 0.5, max_bounces = 5,
            **kwargs
        ):
        if sample_space_bounds is None:
            sample_space_bounds = [self.param_min,self.param_max]
        elif sample_space_bounds == "constrained":
            sample_space_bounds = [self.center + (-self.width/2 * self.param_range),self.center + (-self.width/2 * self.param_range)]
        sample_space_bounds = self._clip_bounds_range(sample_space_bounds)
        if isinstance(bouncyness, list):
            bouncyness_min = bouncyness[0]
            bouncyness_max = bouncyness[1]
        else:
            bouncyness_min = bouncyness
            bouncyness_max = bouncyness
        bounced_samples = np.where(samples < sample_space_bounds[0], sample_space_bounds[0] + (np.abs(sample_space_bounds[0] - samples) * bouncyness_min), samples)
        bounced_samples = np.where(bounced_samples > sample_space_bounds[1], sample_space_bounds[1] - (np.abs(bounced_samples-sample_space_bounds[1]) * bouncyness_max), bounced_samples)
        if max_bounces>1 and bounce_decay>0:
            n_bounces = 1
            bounce_scale = bounce_decay
            if bounce_decay>1:
                bounce_scale = 1/bounce_decay
            while np.logical_or(bounced_samples < sample_space_bounds[0],bounced_samples > sample_space_bounds[1]).any() and n_bounces < max_bounces:
                bounced_samples = np.where(bounced_samples < sample_space_bounds[0], sample_space_bounds[0] + (np.abs(sample_space_bounds[0] - bounced_samples) * bouncyness_min * bounce_scale), bounced_samples)
                bounced_samples = np.where(bounced_samples > sample_space_bounds[1], sample_space_bounds[1] - (np.abs(bounced_samples-sample_space_bounds[1]) * bouncyness_max * bounce_scale), bounced_samples)
                bounce_scale = bounce_scale*bounce_scale
                n_bounces += 1
        return self._wall_hug(bounced_samples,sample_space_bounds)

    def _set_edge_behavior(self,edge_behavior):
        if edge_behavior == "WallHug":
            self.edge_behavior_type = edge_behavior
            self._edge_behavior = self._wall_hug
        elif edge_behavior == "WallBounce":
            self.edge_behavior_type = edge_behavior
            self._edge_behavior = self._bounce_wall
        elif callable(edge_behavior):
            self.edge_behavior_type = "custom fn"
            self._edge_behavior = edge_behavior
        else:
            self.edge_behavior_type = "None"
            self._edge_behavior = self._wall_pass

    def edge_behavior(self, samples, **kwargs):
        samples_ = mnt.ensure_numpy(samples)
        _edge_behavior_kwargs  = dcp(self._edge_behavior_kwargs)
        if "edge_behavior_kwargs" in kwargs:
            for key in kwargs["edge_behavior_kwargs"]:
                _edge_behavior_kwargs[key] = kwargs[key]
        return self._edge_behavior(
            samples_, 
            **_edge_behavior_kwargs
        )
    # <----------- Bounds Related Helper Functions -----------

    # ----------- Base Sampling Wrappers ----------->
    def _check_sobol_based_sampler(self, size, sampling_type):
        if size[-1] != self.param_dim:
            temp_sobol_instance = Sobol(size[-1], seed=self.sample_generator)
            if isinstance(sampling_type,MultivariateNormalQMC):
                temp_sampler =  MultivariateNormalQMC(np.zeros((size[-1],)),np.eye(size[-1]),engine=temp_sobol_instance,seed=self.sample_generator)
            else:
                temp_sampler = temp_sobol_instance
            return temp_sampler
        else:
            return sampling_type

    def _sobol_based_sampling_wrapper(self,size,sampling_type,**kwargs):
        sampling_type_ = self._check_sobol_based_sampler(size,sampling_type)
        n_dims = len(size)
        total_n_samples = 0
        for d in range(n_dims-1):
            total_n_samples += size[d]
        return sampling_type_.random(total_n_samples,**kwargs).reshape(size)
    
    def sample_uniform(self, low=0,high=1,size=(1,1),**kwargs):
        if self.use_sobol:
            return self._sobol_based_sampling_wrapper(
                                                        size,
                                                        self.sobol_sampler,
                                                        **kwargs
                    ) * (high-low) + low
        else:
            return self.sample_generator.uniform(low=low,high=high,size=size)
        
    def sample_normal(self,mean,std,size=(1,1), **kwargs):
        if self.use_sobol:
            return self._sobol_based_sampling_wrapper(
                                                        size,
                                                        self.normalQMC_sampler,
                                                        **kwargs
                    ) * std + mean
        else:
            return self.sample_generator.normal(mean, std, size = size) 
    # <----------- Base Sampling Wrappers ----------- 

    # ----------- Sampler State Setting ----------->
    def set_sample_center(self, center, **kwargs):
        self.center = mnt.ensure_numpy(center)
    
    def set_sample_width(self,width, **kwargs):
        if isinstance(width, list) or isinstance(width, mnt._Tensor):
            width = mnt.ensure_numpy(width)
        self.width = width
        
    def set_sample_vector_norm(self,min_norm = 0, **kwargs):
        self.min_norm = min_norm
        
    def set_state(self, center = None, width = None, min_norm = None, **kwargs):
        if center is not None:
            self.set_sample_center(center)
        if width is not None:
            self.set_sample_width(width)
        if min_norm is not None:
            self.set_sample_vector_norm(min_norm = min_norm)
    
    def _project_to_param_space(self,nondimensional_samples):
        nondimensional_samples_ndims = [
            i for i in range(len(nondimensional_samples.shape)-1)
        ]
        center_reshaped = broadcast_like(
            self.center, nondimensional_samples_ndims,
            missing_dims=nondimensional_samples_ndims
        )
        param_range_reshaped = broadcast_like(
                self.param_range, nondimensional_samples_ndims,
                missing_dims = nondimensional_samples_ndims
            )
        width_reshaped = self.width
        if isinstance(self.width, mnt._ndarray):
            width_reshaped = broadcast_like(
                self.param_range, nondimensional_samples_ndims,
                missing_dims = nondimensional_samples_ndims
            )
        return (
                nondimensional_samples * width_reshaped * param_range_reshaped
            ) + center_reshaped
    
    def set_sampling_method(self, method):
        if method == "cube":
            self.sampling_method = self._sample_cube_around
        elif method == "sphere":
            self.sampling_method = self._sample_uniform_sphere_around
        elif method == "sphere_bias":
            self.sampling_method = self._sample_parametrized_bias_sphere_around
        elif method == "sparse_sphere":
            self.sampling_method = self._sample_sparse_sphere_around
        elif method == "sparse_sphere_bias":
            self.sampling_method = self._sample_sparse_parametrized_bias_sphere_around
        elif method == "univariate_gaussian":
            self.sampling_method = self._sample_univariate_gaussian
        elif method == "full_cube":
            self.sampling_method = self._sample_in_cube
        elif method == "custom":
            self.sampling_method = self._sample_custom
        self.sample_distribution = method
    # <----------- Sampler State Setting -----------

    # ----------- Sampling functions ----------->
    def sample(self, sample_shape, method = "default", **kwargs):
        sample_shape = mnt.ensure_numpy(sample_shape).tolist()
        if method == "default":
            return self.sampling_method(sample_shape, **kwargs)
        elif method == "cube":
            return self._sample_cube_around(sample_shape, **kwargs)
        elif method == "sphere":
            return self._sample_uniform_sphere_around(sample_shape, **kwargs)
        elif method == "sphere_bias":
            return self._sample_parametrized_bias_sphere_around(sample_shape, **kwargs)
        elif method == "sparse_sphere":
            return self._sample_sparse_sphere_around(sample_shape, **kwargs)
        elif method == "sparse_sphere_bias":
            return self._sample_sparse_parametrized_bias_sphere_around(sample_shape, **kwargs)
        elif method == "univariate_gaussian":
            return self._sample_univariate_gaussian(sample_shape, **kwargs)
        elif method == "full_cube":
            return self._sample_in_cube(sample_shape, **kwargs)
        elif method == "custom":
            return self._sample_custom(sample_shape=sample_shape,**kwargs)
    
    def _sample_in_cube(self, sample_shape, **kwargs):
        return self.ensure_fn(
                    self.sample_uniform(
                                            low=self.param_min,
                                            high=self.param_max, 
                                            size = (*sample_shape,self.param_dim)
                    )
                )

    def _sample_cube_around(self, sample_shape, **kwargs):
        samples = self.sample_uniform(
                                        low=-0.5,
                                        high=0.5, 
                                        size = (*sample_shape,self.param_dim)
                    ) 
        samples = self._project_to_param_space(samples)
        samples = self.edge_behavior(samples,**kwargs)
        return self.ensure_fn(samples)

    def _sample_uniform_sphere_around(self, sample_shape, **kwargs):
        samples_surface = self.sample_normal(0,1, size = (*sample_shape,self.param_dim))  
        samples_surface = samples_surface / np.linalg.norm(samples_surface, axis= -1)[..., None]
        random_vector_norm = np.sqrt(
                                    self.sample_uniform(
                                                        low=self.min_norm, 
                                                        high=1.0, 
                                                        size = (*sample_shape, 1)
                                    )
                            )
        samples = random_vector_norm/2 * samples_surface
        samples = self._project_to_param_space(samples)
        samples = self.edge_behavior(samples,**kwargs)
        return self.ensure_fn(samples)

    def _sample_parametrized_bias_sphere_around(
                                                    self, 
                                                    sample_shape, 
                                                    power = 7/12, 
                                                    **kwargs
        ):
        samples_surface = self.sample_normal(0,1, size = (*sample_shape,self.param_dim)) 
        samples_surface = samples_surface / np.linalg.norm(samples_surface, axis= -1)[..., None]
        random_vector_norm = (
                                self.sample_uniform(
                                                    low=self.min_norm, high=1.0, 
                                                    size = (*sample_shape, 1)
                                )
                            ) ** power
        samples = random_vector_norm/2 * samples_surface 
        samples = self._project_to_param_space(samples)
        samples = self.edge_behavior(samples,**kwargs)
        return self.ensure_fn(samples)

    def _sample_sparse_sphere_around(
                                        self, 
                                        sample_shape, 
                                        exp_val = 1, 
                                        **kwargs
    ):
        return self._sample_sparse_parametrized_bias_sphere_around(
                                                                    sample_shape,
                                                                    exp_val=exp_val,
                                                                    power = 1, **kwargs
                )
    
    def _sample_sparse_parametrized_bias_sphere_around(
                                                        self, 
                                                        sample_shape, 
                                                        exp_val = 1, power = 7/12, 
                                                        **kwargs
        ):
        rank = self.sample_uniform(
            size = (*sample_shape, self.param_dim)
        ).argsort(-1).argsort(-1)
        magnitude = self.sample_generator.exponential(
                        exp_val,
                        size=(*sample_shape,self.param_dim)
                    ) 
        magnitude = (
                        mnt.ensure_numpy(
                                        mnt.ensure_torch(
                                                            magnitude
                                        ).sort(dim = -1, descending=True)[0]
                        )*(
                            self.sample_uniform(
                                low=0, high=1.0,
                                size = (*sample_shape, self.param_dim)
                            ).round() * 2 
                            - 1
                        )
            )
        samples_surface = magnitude/ np.linalg.norm(magnitude, axis= -1)[..., None]
        samples_surface = np.take_along_axis(samples_surface,rank,-1)
        random_vector_norm = (
                                self.sample_uniform(
                                                    low=self.min_norm, high=1.0,
                                                    size = (*sample_shape, 1)
                                )
                            ) ** power 
        samples = random_vector_norm/2 * samples_surface 
        samples = self._project_to_param_space(samples)
        samples = self.edge_behavior(samples,**kwargs)
        return self.ensure_fn(samples)
    

    def _sample_univariate_gaussian(self, sample_shape, **kwargs):
        samples = self.sample_normal(0, 1, size = (*sample_shape,self.param_dim)) 
        samples = self._project_to_param_space(samples)
        #* self.param_range + self.center[None,...]
        samples = self.edge_behavior(samples,**kwargs)
        return self.ensure_fn(samples)

    def _sample_custom(
                        self, 
                        custom_fn,  
                        sample_shape, 
                        custom_output_is_in_param_space = False, 
                        custom_fn_needs_attr = [], 
                        apply_edge_behavior=True,
                        **kwargs
        ):
        needed_attr = {}
        for attr in custom_fn_needs_attr:
            if "BaseSampler" == attr:
                needed_attr["BaseSampler"] = self
            else:
                needed_attr[attr] = getattr(self,attr)
        samples = custom_fn(sample_shape, **kwargs, **needed_attr)
        if not custom_output_is_in_param_space:
            samples = self._project_to_param_space(samples)
        if apply_edge_behavior:
            samples = self.edge_behavior(samples,**kwargs)
        return samples
    # <----------- Sampling functions -----------
    def get_samples(self,
                        n_param_samples,
                        **kwargs):
        raise NotImplementedError("'.get_samples()' was not overloaded by child class and inherited from BaseSampler which does not implement it.")
    