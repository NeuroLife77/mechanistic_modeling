import numpy as np
from pystorm import mnt
from copy import deepcopy as dcp
from datetime import datetime as dtm
from joblib import Parallel, delayed
from MHT.sampling.BaseSampler import BaseSampler
from MHT.utils import id_transform, gof_template, round_width_cosine_decay, DataSample
from MHT.hypothesis_formatting.AlphaCoupledNetworkModel import AlphaCoupledNetworkModel

class CoupledNetworkModelSampler(BaseSampler):
    def __init__(
                    self, model,connectivity,
                    data_sample_set:DataSample, 
                    alpha:AlphaCoupledNetworkModel,
                    dt = 0.25, length = 10, max_jobs = 1,
                    use_template_as_init_center=False,
                    gof = gof_template, model_kwargs = {},
                    random_seed = None,combine_gofs=id_transform,
                    verbose =  0,
                    **kwargs
        ):
        

        # Set up the model function and it's settings args or other keyword-args that are constant across param_samples
        self.model = model
        self.model_kwargs = model_kwargs
        self.verbose = verbose
        self.use_template_as_init_center = use_template_as_init_center
        self.max_jobs = max_jobs
        # Define and format the data_sample_set to optimize for
        self.data_sample_set = data_sample_set
        self.n_data_sample_set = self.data_sample_set.n_data_samples
        self.n_features = 0
        for T_feature in self.data_sample_set.T_feature_shapes:
            T = self.data_sample_set.T_feature_shapes[T_feature]
            if len(T)>2:
                self.n_features += T[0]
            else:
                self.n_features += 1
        self.data_sample_set_features = [self.data_sample_set.data_sample_formatting[T](self.data_sample_set.T_features[T],**model_kwargs) for T in self.data_sample_set.T_feature_names]

        # Query the model for its output format and state-variable range
        self._get_model_info()
        # Define connectivity 
        self.connectivity = mnt.ensure_numpy(connectivity)    
        self.n_nodes = self.connectivity.shape[-1]    
        
        # Define free parameters
        # Local free params
        self.alpha = alpha
        
        # Simulation settings
        self.dt = dt
        self.length = length
        # Compute control
        self.parallelize_loss_compute = False
        self.parallelize_sim = False
        if "parallelize_sim" in kwargs:
            self.parallelize_sim = kwargs["parallelize_sim"]
        # Debug/eval info
        self.time_steps = False

        # Define gof metrics and combine_gofs function
        self.gof_metric = gof
        self.combine_gofs = combine_gofs

        
        
        BaseSampler.__init__(
            self,
            param_min=self.alpha.formatted_min,
            param_max=self.alpha.formatted_max,
            random_seed=random_seed,
            **kwargs
        )
        # Define sampler for initial conditions
        self._init_cond_sampler = self._sample_initial_conditions_beta
        self.initialization_sampling_method = "full_cube"

    def _get_min_max_from_dict(self, parameters_min_max):
        """Takes in a directory defining the parameter ranges and converts it into the sampled-format for the free parameters"""
        
        return self.alpha._get_min_max_from_dict(parameters_min_max)
    
    
    def _get_formatted_bound(self, parameters_bound):
        """Does basically the same as 'self._get_min_max_from_dict()' but returns min and max grouped together"""
        return self.alpha._get_formatted_bound(parameters_bound)
            
    def _generate_param_sample_of_template(self, sample, **kwargs):
        """Generate a parameter sample in the sampled-format from the template values"""
        
        return self.alpha._generate_param_sample_of_template(sample,**kwargs)
    def _generate_from_template(self, param_samples, **kwargs):
        """Converts a parameter sample from the sampled-format to the template-format"""
        return self.alpha(param_samples,**kwargs)

    def _get_model_info(self):
        """Query the model for its state variable, their range, and its output format"""
        info = self.model(parameters=None,connectivity=None)
        self.n_state_vars = info["n_state_var"]
        self.state_var_init_range = np.asarray(info["state_var_init_range"],dtype=float).T
        self.model_output = info["output"]
        is_output_ok = sum([self.model_output[i]==self.data_sample_set.T_feature_names[i] for i in range(len(self.model_output))]) == len(self.model_output)
        if not is_output_ok:
            still_not_ok = False
            for data_sample_set_feature_name in self.data_sample_set.T_feature_names:
                has_matching_subset = False
                for model_feature in self.model_output:
                    if model_feature in data_sample_set_feature_name.split("_"):
                        has_matching_subset = True
                if not has_matching_subset:
                    still_not_ok = True
            if still_not_ok:
                raise AttributeError(f"CoupledNetworkModel_sampler._get_model_info(): \t data_sample_set and Model features don't match \n \t data_sample_set Features = {self.data_sample_set.T_feature_names} \n \t Model Features = {self.model_output}")
            else:
                print("CoupledNetworkModel_sampler._get_model_info(): \t Found that some model feature names are subsets of multiple data_sample_set feature names, assuming that it is intentional. Will crash if not.")
                
        
        
    def get_samples(
                        self,
                        n_param_samples,
                        center=None, width=None, 
                        return_best = False, 
                        sample_template = None,
                        verbose=1, debug = False, 
                        **kwargs
        ):
        """
            Run a search iteration around a center at a given width within a certain range.
        """
        if width is None:
            width =self.width
        
        if center is None: # Running an initialization step instead of an iteration step
            # if self.use_template_as_init_center is not None:
            #     center_template = self._generate_param_sample_of_template(param_samples[i],template_index=ti)

            
            self.set_state(center = center, width = width, **kwargs)
            param_samples = self.sample(
                                                                    sample_shape = (n_param_samples,),
                                                                    method = self.initialization_sampling_method,
                                                                    **kwargs
                            )
        else: # Running an iteration step
            self.set_state(center = center, width = width, **kwargs)
            param_samples = self.sample(
                                                    sample_shape = (n_param_samples,),
                                                    **kwargs
                            )

        
        if sample_template is not None: # Force one of the param_samples to be the template(s)
            for i, ti in enumerate(sample_template):
                if i >= param_samples.shape[0]:
                    print(f"ModelFit.run_iteration(): More templates ({len(sample_template)}) than param_samples ({param_samples.shape[0]}), only used the first {i-1} templates.")
                    break
                param_samples[i] = self._generate_param_sample_of_template(param_samples[i],template_index=ti)

        # Convert param_samples from their sampled-format to the template-format
        sample_dicts = self._generate_from_template(param_samples)
        # Sample initial conditions for each parameter sample
        initial_conditions = self._sample_initial_conditions(n_param_samples)
        # Generate a list of job dicts for each sample
        job_dicts = []
        #print("\t\t\tInit",self.data_sample_set.n_data_sample_set)
        for s, sample_dict in enumerate(sample_dicts):
            job_dict = dcp(sample_dict)
            job_dict["connectivity"] = self.connectivity
            job_dict["initial_conditions"] = initial_conditions[s]
            job_dict["noise_seed"] = self.noise_seed
            job_dict["dt"] = self.dt
            job_dict["length"] = self.length
            job_dict["kwargs"] = kwargs
            job_dicts.append(job_dict)
        sim_start = dtm.now()
        # Get simulation results
        if self.parallelize_sim:
            results = Parallel(
                                n_jobs = self.max_jobs,
                                verbose = verbose,
                                backend="loky",
                                max_nbytes=None
                    )(delayed(self.model)(
                                            **job_info,
                                            **self.model_kwargs,
                                            **job_info["kwargs"],
                        ) for _, job_info in enumerate(job_dicts)
                    )
        else:
            results = [
                        self.model(
                                    **job_info,
                                    **self.model_kwargs,
                                    **job_info["kwargs"],
                        ) 
            for _, job_info in enumerate(job_dicts)]
        # Get and print runtime if set
        sim_end = dtm.now()
        sim_runtime = sim_end-sim_start
        if self.time_steps:
            print("Sim runtime: ",str(sim_runtime))
        
        # Group all simulation results along with their meta data
        run_log = {
            "job_dicts": job_dicts,
            "model_kwargs":self.model_kwargs,
            "param_samples_OG_format":param_samples,
            
        }
        run_log["data_sample_set"] = self.data_sample_set._todict()
        run_log["noise_seed"] = self.noise_seed
        run_log["dt"] = self.dt
        run_log["length"] = self.length
        # Count the number of features outputed by the model
        num_features = 0
        for i in range(len(self.model_output)):
            run_log[self.model_output[i]] = mnt.zeros(len(sample_dicts),*results[0][i].shape).double()
            if len(results[0][i].shape)>2:
                num_features += results[0][i].shape[0]
            else:
                num_features += 1
        # Compute GOF/GOFs for each sample
        gof_start = dtm.now()

        run_log["gofs"] = mnt.zeros(len(sample_dicts),max(self.n_data_sample_set,1),self.n_features)
        #print(run_log["gofs"].shape)
        if self.parallelize_loss_compute:
            # Collect model outputs
            for s, sample_dict in enumerate(results):
                for i in range(len(self.model_output)):
                    run_log[self.model_output[i]][s] = mnt.ensure_torch(results[s][i])
            # GOF function wrapper
            def get_gof(model_output,gof_metric,data_sample_features,data_sample_formatting):
                return gof_metric(model_output,data_sample_features,formatting=data_sample_formatting)
            # If n_data_sample_set == 0 there is no GOF
            if self.n_data_sample_set > 0:
                # Temporary fix for some issues with non-writeable numpy array
                data_sample_features_torch = [mnt.ensure_torch(self.data_sample_set_features[i]) for i in range(len(self.data_sample_set_features))]

                # Compute all GOF values
                results_gof = Parallel(n_jobs = min(len(sample_dicts),self.max_jobs), verbose = verbose)(delayed(get_gof)(
                                                                            results[s],
                                                                            gof_metric=self.gof_metric,
                                                                            data_sample_features=data_sample_features_torch,
                                                                            data_sample_formatting=self.data_sample_set.data_sample_formatting,

                                                                        ) for s, sample_dict in enumerate(sample_dicts)
                                                                    )
                # Store all GOF values
                for s, sample_dict in enumerate(results_gof):
                    run_log["gofs"][s] = results_gof[s]
        else: # Do the same as above but not in parallel
            for s, sample_dict in enumerate(sample_dicts):
                for i in range(len(self.model_output)):
                    run_log[self.model_output[i]][s] = mnt.ensure_torch(results[s][i])
                if self.n_data_sample_set > 0:
                    run_log["gofs"][s] = self.gof_metric(results[s],self.data_sample_set_features,formatting=self.data_sample_set.data_sample_formatting)
        # Get and print runtime if set
        gof_end = dtm.now()
        gof_runtime = gof_end-gof_start
        if self.time_steps:
            print("Gof runtime: ",str(gof_runtime))
        # Compute and store combined gof value
        run_log["gof"] = self.combine_gofs(run_log["gofs"])

        # If n_data_sample_set == 0 there is no GOF
        if self.n_data_sample_set > 0:
            merge_start = dtm.now()
            # Define the best fit for each data_sample_set and store it in a ["best"] sub-log (with meta data)
            run_log["best_sample"] = (run_log["gof"]).argmax(0).view(run_log["gof"].shape[1]) # Define best sample for each data_sample_set
            run_log["best"] = {
                    "gof":  mnt.ensure_numpy(mnt.zeros(self.n_data_sample_set)),
                    "gofs": [],
                    "param_samples_OG_format": [],
                    "params":{
                        'coupling': [],
                        'conduction_speed': [],
                        'parameters': [],
                    },
                    "job_dict": []
            }
            # Initialize data array for each model output
            for o in range(len(self.model_output)):
                run_log["best"][self.model_output[o]] = -mnt.ensure_numpy(mnt.ones(self.n_data_sample_set,*run_log[self.model_output[o]].shape[1:]))

            # For each data_sample_set
            for i in range(run_log["gof"].shape[1]):
                # Collect job dict (for simulation replicability)
                run_log["best"]["job_dict"].append(job_dicts[run_log["best_sample"][i]])
                # Collect gof, model output, gofs, and parameters (in both formats)
                run_log["best"]["gof"][i]  = mnt.ensure_numpy(run_log["gof"][run_log["best_sample"][i],i]).item()
                for o in range(len(self.model_output)):
                    run_log["best"][self.model_output[o]][i] = mnt.ensure_numpy(run_log[self.model_output[o]][run_log["best_sample"][i]])[None,...]
                run_log["best"]["gofs"].append(mnt.ensure_torch(run_log["gofs"][run_log["best_sample"][i]][i,...][None,...]))
                run_log["best"]["param_samples_OG_format"].append(mnt.ensure_torch(run_log["param_samples_OG_format"][run_log["best_sample"][i]][None,...]))
                run_log["best"]["params"]["coupling"].append(job_dicts[run_log["best_sample"][i]]["coupling"])
                run_log["best"]["params"]["conduction_speed"].append(job_dicts[run_log["best_sample"][i]]["conduction_speed"])
                run_log["best"]["params"]["parameters"].append(mnt.ensure_torch(job_dicts[run_log["best_sample"][i]]["parameters"][None,...]))

            # Concatenate all data_sample_set into a single tensor
            run_log["best"]["gofs"] = mnt.ensure_numpy(mnt.cat(run_log["best"]["gofs"], 0))
            run_log["best"]["param_samples_OG_format"] = mnt.ensure_numpy(mnt.cat(run_log["best"]["param_samples_OG_format"], 0))
            run_log["best"]["params"]["coupling"] = mnt.ensure_numpy(run_log["best"]["params"]["coupling"])
            run_log["best"]["params"]["conduction_speed"] = mnt.ensure_numpy(run_log["best"]["params"]["conduction_speed"])
            run_log["best"]["params"]["parameters"] = mnt.ensure_numpy(mnt.cat(run_log["best"]["params"]["parameters"] , 0))
            merge_end = dtm.now()
            if self.time_steps:
                print("Merge runtime: ",str(merge_end-merge_start))
            if return_best:
                return run_log["best"]
        else:
            run_log["best"] = None
            run_log["best_sample"] = None
        return run_log  

    def _sample_initial_conditions(self,n_init_samples, **kwargs):
        init_cond_range = (self.state_var_init_range[1]-self.state_var_init_range[0])[None,:,None]
        init_cond_min = (self.state_var_init_range[0])[None,:,None]
        param_samples = self._init_cond_sampler(n_init_samples, **kwargs)
        return (param_samples * init_cond_range)+init_cond_min

    def _sample_initial_conditions_uniform(self, n_init_samples, **kwargs):
        return self.sample_generator.uniform(0,1,size=(n_init_samples,self.n_state_vars,self.connectivity.shape[-1]))
    
    def _sample_initial_conditions_beta(self, n_init_samples, beta_params = [2,3], **kwargs):
        return self.sample_generator.beta(beta_params[0],beta_params[1],size=(n_init_samples,self.n_state_vars,self.connectivity.shape[-1]))
    def _sample_initial_conditions_gaussian(self,n_init_samples, mu=0.5,sigma=0.15, **kwargs):
        return self.sample_generator.normal(mu,sigma,size=(n_init_samples,self.n_state_vars,self.connectivity.shape[-1]))

class SimulatedAnnealingSamplerCoupledNetwork:
    def __init__(self,
                model_sampler:CoupledNetworkModelSampler,
                theta_center = None,
                min_norm = 0.0, sample_distribution = 'full_cube',
                init_sample_ratio = 0.5, n_rounds = 3, 
                search_width=0.1,search_max_width = 0.5,temperature = 0.035, decay = 4, round_width_period = 2,
                patience_threshold = 5,annealing_fn = None, use_sobol = True, round_width_fn = round_width_cosine_decay, return_all = True,
                use_corrector = False, data_sample_index = 0,
                **kwargs
    ):
        
        self.model_sampler = model_sampler
        self.n_samples_included = model_sampler.data_sample_set.n_data_samples
        if self.n_samples_included == 1:
            self.sample_index_select = 0
        else:
            self.sample_index_select = data_sample_index
        self.theta_min  = model_sampler.param_min
        self.theta_dim = model_sampler.param_dim
        self.min_norm = min_norm
        self.theta_max =  model_sampler.param_max
        self.theta_range =  model_sampler.param_range
        if theta_center is None:
            self.center = None
        else:
            self.center = theta_center
        self.search_max_width = search_max_width
        self.sample_distribution = sample_distribution
        self.use_sobol = use_sobol
        self.model_sampler.set_sampling_method(method = sample_distribution)
        self.model_sampler.use_sobol = self.use_sobol
        self.model_sampler.set_state(
            min_norm = min_norm, 
            center = self.center,
            width = self.search_max_width,
            **kwargs
        )
        self.random_seed = self.model_sampler.noise_seed
        self.n_rounds = n_rounds
        self.init_sample_ratio = init_sample_ratio
        self.search_width = search_width
        
        self.temperature = temperature
        self.decay = decay
        self.round_width_period = round_width_period
        self.patience_threshold = max(n_rounds//patience_threshold,1)
        self.annealing_fn = self._default_annealing
        if annealing_fn is not None:
            self.annealing_fn = annealing_fn
        self.round_width_fn = round_width_fn
        self.use_corrector = use_corrector
        self.return_all = return_all
        self.samples_meta_data = []
        #self.sampler.sampling_method = self.sampler._sample_in_cube

    def sample(
                    self, sample_shape, conditional_sampling_method = id_transform,
                    init_sample_kwargs={},round_sample_kwargs={}, 
                    **kwargs
        ):
        n_round_samples = 0
        if self.n_rounds  != 0 :
            n_round_samples = np.floor((1-self.init_sample_ratio)/self.n_rounds * sample_shape[0]).astype(int).item()
        n_init = sample_shape[0] - n_round_samples*self.n_rounds
        init_samples_dict = self.model_sampler.get_samples(n_init,**init_sample_kwargs)
        self.samples_meta_data.append(init_samples_dict)
        init_samples = mnt.ensure_numpy(init_samples_dict["param_samples_OG_format"])
        all_samples = np.zeros((sample_shape[0],*init_samples.shape[1:]))
        all_samples[:n_init] = init_samples
        index_offset = n_init
        init_gofs =  mnt.ensure_numpy(init_samples_dict["gof"])[...,self.sample_index_select]
        gamma_inits = [init_samples_dict[val] for val in self.model_sampler.data_sample_set.data_sample_formatting]
        Z_inits = [self.model_sampler.data_sample_set.data_sample_formatting[T](init_samples_dict[T]) for T in self.model_sampler.data_sample_set.data_sample_formatting]

        best_sample = init_samples[np.argmax(init_gofs)]
        best_sample_gof = init_gofs[np.argmax(init_gofs)]
        overall_best_sample = best_sample
        overall_best_sample_gof = best_sample_gof
        all_gammas = [np.zeros((sample_shape[0],*gamma_inits[i].shape[min(len(gamma_inits[i].shape)-1,1):])) for i in range(len(gamma_inits))]
        all_Zs = [np.zeros((sample_shape[0],*Z_inits[i].shape[min(len(Z_inits[i].shape)-1,1):])) for i in range(len(Z_inits))]
        all_gofs = np.zeros((sample_shape[0],))
        for i,init_sample in enumerate(init_samples):
            for j in range(len(gamma_inits)):
                all_gammas[j][i] = gamma_inits[j][i]
            for j in range(len(Z_inits)):
                all_Zs[j][i] = Z_inits[j][i]
            all_gofs[i] = init_gofs[i]
        corrector = 0
        impatience = 0
        for rd in range(self.n_rounds):
            round_width = self.round_width_fn(
                                                        rd = rd-corrector,
                                                        n_rounds = self.n_rounds,
                                                        min_width=self.search_width, max_width=self.search_max_width,
                                                        period = self.round_width_period,
                                                        decay = self.decay
                        )
            conditional_sampling_method(self.model_sampler,round_width=round_width)
            self.model_sampler.set_sample_center(best_sample)
            self.model_sampler.set_sample_width(round_width)

            round_samples_dict = self.model_sampler.get_samples(n_round_samples,**round_sample_kwargs)
            self.samples_meta_data.append(round_samples_dict)

            round_samples = mnt.ensure_numpy(round_samples_dict["param_samples_OG_format"])
            round_gofs =  mnt.ensure_numpy(round_samples_dict["gof"])[...,self.sample_index_select]
            gamma_round = [round_samples_dict[val] for val in self.model_sampler.data_sample_set.data_sample_formatting]
            Z_round = [self.model_sampler.data_sample_set.data_sample_formatting[T](round_samples_dict[T]) for T in self.model_sampler.data_sample_set.data_sample_formatting]
            for i,round_sample in enumerate(round_samples):
                all_samples[index_offset+i] = round_sample

                for j in range(len(gamma_round)):
                    all_gammas[j][i+index_offset] = gamma_round[j][i]
                for j in range(len(Z_inits)):
                    all_Zs[j][i+index_offset] = Z_round[j][i]
                all_gofs[i+index_offset] = round_gofs[i]

            
            index_offset = index_offset+n_round_samples

            best_round_sample = round_samples[np.argmax(round_gofs)]
            best_round_sample_gof = round_gofs[np.argmax(round_gofs)]
            diff = (overall_best_sample_gof - best_round_sample_gof)
            if overall_best_sample_gof>0:
                diff /= overall_best_sample_gof
            
            exponent_val = -diff * self.annealing_fn(rd+1) / self.temperature
            exponent_sign = np.sign(exponent_val)
            exponent_val = min(50,np.abs(exponent_val))*exponent_sign
            metropolis = np.exp(exponent_val)
            if self.model_sampler.sample_generator.random() < metropolis:
                best_sample = best_round_sample
                best_sample_gof = best_round_sample_gof
                if best_sample_gof>overall_best_sample_gof:
                    overall_best_sample_gof = best_sample_gof
                    overall_best_sample = best_sample
            
            if diff > 0:
                impatience+=1
                if impatience%self.patience_threshold == 0:
                    if self.use_corrector:
                        corrector += 1
                        corrector = min(corrector,self.n_rounds//4)
                    else:
                        impatience = 0
                    best_sample = overall_best_sample
                    best_sample_gof = overall_best_sample_gof
            else:
                impatience = 0

        if not self.return_all:
            return all_samples
        else:
            return all_samples,all_gammas,all_Zs, all_gofs
    def _default_annealing(self,rd):
        return rd/self.n_rounds
    