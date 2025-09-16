from MHT.utils import *
class AlphaCoupledNetworkModel:
    def __init__(
            self,
            n_nodes,
            bounds,
            local_free_parameters,
            coupling_free_parameters,
            local_template,
            coupling_template,
            param_name_index_mapping,
            verbose = 1
    ):
        self.n_nodes = n_nodes
        self.verbose = verbose
        self.bounds = bounds
        self.local_free_parameters = dcp(local_free_parameters)
        for p in self.local_free_parameters:
            
            if isinstance(self.local_free_parameters[p],list):
                if isinstance(self.local_free_parameters[p][0],int) or isinstance(self.local_free_parameters[p][0],bool):
                    if self.verbose > 0:
                        print("only 1 group")
                    self.local_free_parameters[p] = [self.local_free_parameters[p]]
                mask = mnt.ensure_numpy(self.local_free_parameters[p])
                if isinstance(mask,list) or mask.max()>1 or mask.shape[-1]<self.n_nodes:
                    if self.verbose > 0:
                        print("Local parameter grouping is not boolean, assuming it's indexing")
                    mask = np.zeros((len(self.local_free_parameters[p]),self.n_nodes), dtype = bool)
                    for g in range(mask.shape[0]):
                        indexing = mnt.ensure_numpy(self.local_free_parameters[p][g])
                        if self.verbose > 1:
                            print(indexing,mask.shape,g)
                        mask[g,indexing] = True
                else:
                    mask = mask.astype(bool)

                self.local_free_parameters[p] = mask
        
        self.n_groupings_for_local_free_params = np.zeros((len(self.local_free_parameters),),dtype=int)
        for p ,free_param in enumerate(self.local_free_parameters):
            self.n_groupings_for_local_free_params[p] = len(self.local_free_parameters[free_param])
        self.total_nodal_free_parameters = int(self.n_groupings_for_local_free_params.sum().item())
        
        self.local_free_parameter_indices = np.zeros((len(self.local_free_parameters),),dtype=int)
        
        for i, local_param in enumerate(self.local_free_parameters):
            # Find the index of the free parameter within the template parameter format
            try:
                self.local_free_parameter_indices[i] = param_name_index_mapping[local_param]
            except:
                raise ValueError(f"Invalid local parameter {local_param}, if you are passing a list instead of a dict, you must match alpha.local_params")
        # Keep a mapping to go from parameter-name to parameter-index and back
        self.param_name_index_mapping = param_name_index_mapping
        self.param_index_name_mapping = {i:n for i,n in enumerate(list(self.param_name_index_mapping.keys()))}
        
        # Define coupling params
        self.coupling_free_parameters = coupling_free_parameters
        self.sample_params_correspondance = np.full(
                (
                    len(self.coupling_free_parameters) 
                    +self.total_nodal_free_parameters
                    , 
                ), fill_value=-3,
                dtype = object
            )
        self.sample_dim = self.sample_params_correspondance.shape[0]
        
        self.params_names = np.empty((
                                        len(self.coupling_free_parameters) 
                                        +self.total_nodal_free_parameters,
                                    ), 
                                dtype = object
                            )

        # "populate" the parameter vector in the sampled format
        param_counter = 0 # To index sample_params_correspondance and params_names
        for coupl_par in self.coupling_free_parameters: # Coupling pars have negative indices to be able to ignore them when indexing local params
            if coupl_par == "a" or coupl_par == "coupling":
                self.sample_params_correspondance[param_counter] = -1
                self.params_names[param_counter] = "a"
            elif coupl_par == "K" or coupl_par == "conduction_speed":
                self.sample_params_correspondance[param_counter] = -2
                self.params_names[param_counter] = "K"
            else:
                raise ValueError(f"Invalid coupling parameter {coupl_par}")
            param_counter+=1

        # For each local free params
        for i, local_param in enumerate(self.local_free_parameters):
            # Find the index of the free parameter within the template parameter format
            index = self.local_free_parameter_indices[i]
            for grouping in range(self.local_free_parameters[local_param].shape[0]):
                # And keep track of the groupings for each the local free parameter
                self.sample_params_correspondance[param_counter] = {
                                "template_index":index,
                                "sampler_index":i,
                                "name":local_param,
                                "grouping":self.local_free_parameters[local_param][grouping],
                                "grouping_id":grouping,
                                "total_n_groupings":self.local_free_parameters[local_param].shape[0]
                }
                self.params_names[param_counter] = f"{local_param}_G{grouping}"
                param_counter+=1

        # Make a dict format for the template
        if len(local_template.shape) == 1:
            loc_template = np.concatenate([local_template[None,...] for _ in range(self.n_nodes)])
            self.params_template = {
                                    "coupling": coupling_template[0],
                                    "conduction_speed": coupling_template[1],
                                    "parameters": loc_template
            }
        elif len(local_template.shape) == 3:
            loc_template = local_template
            self.params_template = [{
                                        "coupling": coupling_template[0],
                                        "conduction_speed": coupling_template[1],
                                        "parameters": loc_template[i]
            } for i in range(loc_template.shape[0])]
        else:
            loc_template = local_template
            self.params_template = {
                                    "coupling": coupling_template[0],
                                    "conduction_speed": coupling_template[1],
                                    "parameters": loc_template
            }
        self.dict_bounds = bounds
        self.formatted_min,self.formatted_max = self._get_min_max_from_dict(bounds)
    def _get_min_max_from_dict(self, parameters_min_max):
        """Takes in a directory defining the parameter ranges and converts it into the sampled-format for the free parameters"""
        shaped_min = np.zeros(self.sample_params_correspondance.shape[0:1])
        shaped_max = np.zeros_like(shaped_min)
        param_counter = 0
        # If there are coupling free params
        if parameters_min_max["coupling_params"] is not None:
            coupling_pars_min_max = mnt.ensure_numpy(parameters_min_max["coupling_params"])
            for i in range(coupling_pars_min_max.shape[0]):
                if i==0 and "a" in self.coupling_free_parameters or "coupling" in self.coupling_free_parameters :
                    shaped_min[param_counter] = coupling_pars_min_max[i,0] # Min
                    shaped_max[param_counter] = coupling_pars_min_max[i,1] # Max
                    param_counter+=1
                if i==1 and "K" in self.coupling_free_parameters or "conduction_speed" in self.coupling_free_parameters :
                    shaped_min[param_counter] = coupling_pars_min_max[i,0] # Min
                    shaped_max[param_counter] = coupling_pars_min_max[i,1] # Max
                    param_counter+=1
        # If there are local free params
        use_free_index = False
        use_full_index = False
        if parameters_min_max["local_params"] is not None:
            if isinstance(parameters_min_max["local_params"],list) or isinstance(parameters_min_max["local_params"],mnt._ndarray) or isinstance(parameters_min_max["local_params"],mnt._Tensor):
                if self.verbose > 0:
                    print("Local Min-Max range is not name-defined, assuming that the order with respect to the local free parameter is maintained")
                local_min_max = dcp(parameters_min_max["local_params"])
                min_max_node_specific = False
                if len(mnt.ensure_numpy(local_min_max).shape) == 3:
                    min_max_node_specific = True
                    n_local_par_specified = len(local_min_max[0])
                else:
                    n_local_par_specified = len(local_min_max)
                use_free_index = True
                use_full_index = False
            
                if n_local_par_specified > len(self.local_free_parameters):
                    if not min_max_node_specific and self.verbose>1:
                        print("Local Min-Max range containes more parameters ranges than the total number of local free parameters defined, assuming that it is defined over the full space and will index accordingly.")
                    use_full_index = True
                    use_free_index = False
            else:
                # TODO: Make sure that the name-defined implementation does work...
                local_min_max = dcp(parameters_min_max["local_params"])
            counter_offset = 0 + param_counter
            for p_index in range(counter_offset,self.sample_params_correspondance.shape[0]):
                free_param_info = self.sample_params_correspondance[p_index]
                i = free_param_info["sampler_index"]
                template_i = free_param_info["template_index"]
                p = free_param_info["name"]
                grouping = free_param_info["grouping"]
                grouping_id = free_param_info["grouping_id"]
                n_groupings = free_param_info["total_n_groupings"]
                if use_full_index:
                    nodal_param_select = template_i
                elif use_free_index:
                    nodal_param_select = i
                else:
                    nodal_param_select = p
                if min_max_node_specific:
                    min_max_values_ = mnt.ensure_numpy(local_min_max)[grouping,nodal_param_select]
                    if len(min_max_values_.shape) > 1:
                        if not (min_max_values_ == min_max_values_.mean(0,keepdims=True)).all() and self.verbose>1:
                            print(f"Local Min-Max range specified for the nodes in grouping {grouping_id} ({grouping}) for parameter {p} are not identical. Taking the extremeties of the bounds.")
                        min_max_values = mnt.ensure_numpy([min_max_values_.min(),min_max_values_.max()])
                    else:
                        min_max_values =  mnt.ensure_numpy(local_min_max)[:,nodal_param_select]
                else:
                    min_max_values = mnt.ensure_numpy(local_min_max[nodal_param_select])

                if len(min_max_values.squeeze().shape)==1:
                    if n_groupings>1 and not min_max_node_specific and self.verbose>1:
                        print(f"Local Min-Max range containes less bounds than the defined number of grouping for parameter {p}, assuming that they all have the same bounds.")
                    min_max_values = min_max_values.squeeze()
                    shaped_min[param_counter] = min_max_values[0]
                    shaped_max[param_counter] = min_max_values[1]
                else:
                    if use_full_index:
                        grouping_select = grouping
                    elif use_free_index:
                        grouping_select = template_i
                    else:
                        grouping_select = grouping_id
                    shaped_min[param_counter] = min_max_values[grouping_select][0]
                    shaped_max[param_counter] = min_max_values[grouping_select][1]
                param_counter += 1 
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
                if isinstance(param_info,int) and param_info == -1:
                    sample_dict["coupling"] = val
                elif isinstance(param_info,int) and  param_info == -2:
                    sample_dict["conduction_speed"] = val
                elif not isinstance(param_info,int):
                    sample_dict["parameters"][param_info["grouping"], param_info["template_index"]] = val
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
                if isinstance(param_info,int) and param_info == -1:
                    sample_copy[index] = sample_dict["coupling"]
                elif isinstance(param_info,int) and  param_info == -2:
                    sample_copy[index] = sample_dict["conduction_speed"]
                elif not isinstance(param_info,int):
                    local_p = sample_dict["parameters"][param_info["grouping"], param_info["template_index"]]
                    if local_p.shape[0] >= 1:
                        local_p = local_p[0] 
                    sample_copy[index] = local_p
        else:
            sample_dict = dcp(self.params_template)
            for index, _ in enumerate(sample_copy):
                param_info = self.sample_params_correspondance[index]
                if isinstance(param_info,int) and param_info == -1:
                    sample_copy[index] = sample_dict["coupling"]
                elif isinstance(param_info,int) and  param_info == -2:
                    sample_copy[index] = sample_dict["conduction_speed"]
                elif not isinstance(param_info,int):
                    local_p = sample_dict["parameters"][param_info["grouping"], param_info["template_index"]]
                    if local_p.shape[0] >= 1:
                        local_p = local_p[0] 
                    sample_copy[index] = local_p
        return sample_copy
    def _generate_param_sample_from_full(self, full_set_param, *args, **kwargs):
        """Generate a parameter sample in the sampled-format from the template values"""
        sample_copy = np.zeros((self.sample_dim,))
        sample_dict = dcp(full_set_param)
        for index, _ in enumerate(sample_copy):
            param_info = self.sample_params_correspondance[index]
            if isinstance(param_info,int) and param_info == -1:
                sample_copy[index] = sample_dict["coupling"]
            elif isinstance(param_info,int) and  param_info == -2:
                sample_copy[index] = sample_dict["conduction_speed"]
            elif not isinstance(param_info,int):
                local_p = sample_dict["parameters"][param_info["grouping"], param_info["template_index"]]
                if local_p.shape[0] >= 1:
                    local_p = local_p[0] 
                sample_copy[index] = local_p
        return sample_copy