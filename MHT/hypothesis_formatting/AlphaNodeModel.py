from MHT.utils import *
class AlphaNodeModel:
    def __init__(
            self,
            bounds,
            free_parameters,
            template,
            param_name_index_mapping,
            verbose = 1
    ):
        self.verbose = verbose
        self.bounds = bounds
        free_parameters_ = dcp(free_parameters)
        self.param_name_index_mapping = dcp(param_name_index_mapping)
        if isinstance(free_parameters_,dict):
            self.free_parameters = mnt.ensure_numpy([
                                                        self.param_name_index_mapping[p] 
                                        for p in free_parameters_ if free_parameters_[p]
                                    ])
        elif isinstance(free_parameters_,list) or isinstance(mnt.ensure_numpy(free_parameters_),mnt._ndarray):
            self.free_parameters = np.zeros((len(free_parameters_),),dtype=int)
            for p,param in enumerate(free_parameters_):
                value = param
                if isinstance(value,str):
                    value = self.param_index_name_mapping[value]
                
                self.free_parameters[p] = int(value)

        else:
            raise ValueError(f"{self.__class__}.__init__(): \n\tIncorrect 'free_parameter' input type ({type(free_parameters)}), must be dict with param names as keys and boolean as values or a list/array/tensor of param indices or param names specifying free parameters.")
        
        self.total_free_parameters = self.free_parameters.shape[0]
        self.param_index_name_mapping = {i:n for i,n in enumerate(list(self.param_name_index_mapping.keys()))}
        # Define coupling params
        self.sample_params_correspondance = np.full(
                                                        (self.total_free_parameters,),
                                                        fill_value=-3,
                                                dtype = object
                                            )
        self.sample_dim = self.sample_params_correspondance.shape[0]
        self.params_names = np.empty_like(
                                            self.sample_params_correspondance, 
                                            dtype = object
                            )
        # For each local free params
        for i, index in enumerate(self.free_parameters):
            # Find the index of the free parameter within the template parameter format
            local_param = self.param_index_name_mapping[index]
            
            self.sample_params_correspondance[i] = {
                                "template_index":index,
                                "sampler_index":i,
                                "name":local_param,
            }
            self.params_names[i] = f"{local_param}"
        # Make a dict format for the template
        template_ = mnt.ensure_numpy(dcp(template))
        try:
            template_shape = template_.shape
        except:
            template_shape = []
        if len(template_shape) == 1:
            self.params_template = {
                                    "parameters": template_
            }
        elif len(template_shape) == 2:
            self.params_template = [{
                                        "parameters": template_[i]
            } for i in range(template_.shape[0])]
        else:
            raise ValueError(f"{self.__class__}.__init__(): \n\tParameter template has incorrect format (type={type(template)},inferred shape={template_shape}). Can only take a vector or a set of vectors in list/array/tensor form.")
        self.dict_bounds = dcp(bounds)
        self.formatted_min,self.formatted_max = self._get_min_max_from_dict(self.dict_bounds)

    def _get_min_max_from_dict(self, parameters_min_max):
        """Takes in a directory defining the parameter ranges and converts it into the sampled-format for the free parameters"""
        parameters_min_max_ = dcp(parameters_min_max)
        if isinstance(parameters_min_max_,dict) and len(parameters_min_max_)==1 and ("local_params" in parameters_min_max_ or "parameters" in parameters_min_max_):
            parameters_min_max_ = parameters_min_max_[list(parameters_min_max_.keys())[0]]
        shaped_min = np.zeros(self.sample_params_correspondance.shape[0:1])
        shaped_max = np.zeros_like(shaped_min)
        use_free_index = False
        use_full_index = False
        if isinstance(mnt.ensure_numpy(parameters_min_max_),mnt._ndarray):
            if self.verbose > 0:
                print("Min-Max range is not name-defined, assuming that the order with respect to the free parameter is maintained")
            n_parameters_specified = len(parameters_min_max_)
            use_free_index = True
            use_full_index = False
        
            if n_parameters_specified > len(self.free_parameters):
                if self.verbose>1:
                    print("Min-Max range containes more parameters ranges than the total number of free parameters defined, assuming that it is defined over the full space and will index accordingly.")
                use_full_index = True
                use_free_index = False
            
            for p_index in range(self.sample_params_correspondance.shape[0]):
                free_param_info = self.sample_params_correspondance[p_index]
                i = free_param_info["sampler_index"]
                template_i = free_param_info["template_index"]
                p = free_param_info["name"]

                if use_full_index:
                    param_select = template_i
                elif use_free_index:
                    param_select = i
                else:
                    param_select = p
                
                min_max_values = mnt.ensure_numpy(parameters_min_max_[param_select])
                min_max_values = min_max_values.squeeze()
                shaped_min[p_index] = min_max_values[0]
                shaped_max[p_index] = min_max_values[1]
                
        return shaped_min, shaped_max
    
    
    def _get_formatted_bound(self, parameters_bound):
        """Does basically the same as 'self._get_min_max_from_dict()' but returns min and max grouped together"""
        min_,max_ = self._get_min_max_from_dict(parameters_min_max=parameters_bound)
        shaped_bound = np.concatenate([min_[...,None],max_[...,None]],axis = -1)
        return shaped_bound
            

    def __call__(self, param_samples, **kwargs):
        """Converts a parameter sample from the sampled-format to the template-format"""
        template_indexing = np.zeros((len(param_samples,)),dtype=int)
        if "template_indexing" in kwargs:
            template_indexing = kwargs["template_indexing"]
            if isinstance(template_indexing,int) or isinstance(template_indexing,float):
                template_indexing = np.zeros((len(param_samples,)),dtype=int) + int(template_indexing)
            elif isinstance(mnt.ensure_numpy(template_indexing),mnt._ndarray):
                template_indexing = mnt.ensure_numpy(template_indexing).astype(int)
                if template_indexing.shape[0] != len(param_samples):
                    raise ValueError(f"{self.__class__}.__call__(): \n\tNumber of samples ({len(param_samples)}) does not match the number of template_indexing ({template_indexing.shape[0]}), if you want each sample to be applied to all templates you will have to loop, this is not implemented here.")
        sample_dicts = []
        for s, samp in enumerate(param_samples):
            if isinstance(self.params_template, list):
                ti=template_indexing[s]
                if ti>=len(self.params_template):
                    raise ValueError(f"{self.__class__}.__call__(): \n\t Template index from input keyword argument 'template_indexing' for sample {s} ({ti}) exceeds the size of 'self.params_template' ({len(self.params_template)}).")
                sample_dict = dcp(self.params_template[ti])
            else:
                sample_dict = dcp(self.params_template)
            for index, val in enumerate(samp):
                param_info = self.sample_params_correspondance[index]
                if not isinstance(param_info,int):
                    sample_dict["parameters"][param_info["template_index"]] = val
            sample_dicts.append(sample_dict)
        return sample_dicts

    def _generate_param_sample_of_template(self, *args, **kwargs):
        """Generate a parameter sample in the sampled-format from the template values"""
        sample_copy = np.zeros((self.sample_dim,))
        if isinstance(self.params_template, list):
            if "template_index" in list(kwargs.keys()):
                template_sel = kwargs["template_index"]
            else:
                template_sel = 0
            sample_dict = dcp(self.params_template[template_sel])
            for index, _ in enumerate(sample_copy):
                param_info = self.sample_params_correspondance[index]
                if not isinstance(param_info,int):
                    local_p = sample_dict["parameters"][param_info["template_index"]]
                    sample_copy[index] = local_p
        else:
            sample_dict = dcp(self.params_template)
            for index, _ in enumerate(sample_copy):
                param_info = self.sample_params_correspondance[index]
                if not isinstance(param_info,int):
                    local_p = sample_dict["parameters"][param_info["template_index"]]
                    sample_copy[index] = local_p
        return sample_copy

