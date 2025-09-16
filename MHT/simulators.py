import numba
import numpy as np
from copy import deepcopy as dcp
from MHT.utils import dtm
param_ind = {
    "c_ee": 0, "c_ei": 1, "c_ie": 2, "c_ii": 3, 
    "tau_e": 4, "tau_i": 5,
    "theta_e": 6, "theta_i": 7,
    "a_e": 8, "a_i": 9,
    "k_e": 10, "k_i": 11, "r_e": 12, "r_i": 13,
    "noise_E": 14, "noise_I": 15,
}

parameter_names_WC = [par for par in param_ind]

parameter_latex_mapping = {
    "c_ee": "$c_{{{ee}}}$", "c_ei": "$c_{{{ei}}}$", "c_ie": "$c_{{{ie}}}$", "c_ii": "$c_{{{ii}}}$", 
    "tau_e": "$\\tau_e$", "tau_i": "$\\tau_i$",
    "theta_e": "$\\theta_e$", "theta_i": "$\\theta_i$",
    "a_e": "$a_e$", "a_i": "$a_i$",
    "k_e": "$k_e$", "k_i": "$k_i$", "r_e": "$r_e$", "r_i": "$r_i$",
    "noise_E": "$Noise_E$", "noise_I": "$Noise_I$",
}

parameter_latex_mapping["theta_e"] = "$b_e$"
parameter_latex_mapping["theta_i"] = "$b_i$"


def compute_window_fft(x, fs, axis = -1):
    
    hamming_window = (0.54 - 0.46 * np.cos(np.linspace(0,2*np.pi,x.shape[-1])))
    signal = (x - x.mean(axis, keepdims = True))*hamming_window
    win_noise_power_gain = (hamming_window**2).sum()
    scaling_term = np.sqrt(2/(win_noise_power_gain*fs))
    fft_res = np.fft.rfft(signal,axis = axis)
    fft_res[...,1:-1] *= scaling_term
    fft_blocks = fft_res
    
    return fft_blocks


class OutputTypeError(Exception):
    def __init__(self, dims):
        self.msg = f"{len(dims)}d()"
    def __str__(self):
        return "Wrong function used, you need to use compute_fft"+self.msg

@numba.njit
def compute_fft(x, fs):
    with numba.objmode(y='complex128[:]'):
        y = compute_window_fft(x,fs)
    return y
@numba.njit
def compute_fft2d(x, fs):
    with numba.objmode(y='complex128[:,:]'):
        y = compute_window_fft(x,fs)
    return y

@numba.njit
def simulate_WC_node_PSD(
            parameters,
            cutoff = 100,
            length: float = 12,
            dt: float = 0.5,
            initial_conditions = np.array([0.25,0.25]),
            noise_seed: int = 42,
            store_I: bool = False,
            psd_window_size = 2,
            psd_overlap = 0.5,
            psd_transient_size = 2,
            is_noise_log_scale = True,
            verbose = 0
            
    ):
    # Input parameters are of shape (num_nodes, num_parameters) to match parameter estimation output, but we need (num_parameters,num_nodes) to simulate
    params = np.zeros_like(parameters.T) + parameters.T 
    if is_noise_log_scale:
        for i in range(1,3):
            params[-i] = 10**(params[-i])
            params[-i,params[-i]==1] = 0
    # Set seet
    np.random.seed(noise_seed)
    # White noise
    DE, DI = np.sqrt(2*params[-2]* dt), np.sqrt(2*params[-1]* dt)

    num_nodes = parameters.shape[0] 

    # Equivalent to allocating memory
    sim_length = int(1000/dt*length)
    fs = int(1000/dt)
    psd_window_length = int(1000/dt*psd_window_size)
    psd_window_step = int(psd_window_length * (1-psd_overlap))
    psd_transient_length = int(1000/dt*psd_transient_size)

    time_series_E = np.zeros((sim_length+1, int(num_nodes)))
    time_series_I = np.zeros((3,int(num_nodes)))
    if store_I:
        time_series_I = np.empty((sim_length+1,int(num_nodes)))
    time_series_E_temp = np.zeros((1,int(num_nodes)))
    time_series_I_temp = np.zeros((1,int(num_nodes)))
    time_series_E_corr = np.zeros((1,int(num_nodes)))
    time_series_I_corr = np.zeros((1,int(num_nodes)))
    time_series_E_noise = np.zeros((1,int(num_nodes)))
    time_series_I_noise = np.zeros((1,int(num_nodes)))

    # Set initial conditions
    time_series_E[0] = initial_conditions[0]
    time_series_I[0] = initial_conditions[1]

    psd_window_counter = 0
    psd_running_mean = np.zeros((int(num_nodes),int(cutoff)))
    # Heun performed in-place within the time_series_X arrays to maximize speed
    for i in range(int(1000/dt*length)-1):
        # Forward Euler
        j_0 = i
        j_1 = (i+1)
        if not store_I:
            j_0 = i%2
            j_1 = (i+1)%2
        # Calculating input from other nodes
                #               c_ee   *       E          -   c_ei    *        I           +     conn kernel      *    a     *     network input      -  theta_e
        time_series_E[i+1] = params[0] * time_series_E[i] - params[1] * time_series_I[j_0] - params[6]
        #                       c_ie   *       E          -   c_ii    *        I           +     conn kernel      *    a     *     network input      -  theta_i
        time_series_I[j_1] = params[2] * time_series_E[i] - params[3] * time_series_I[j_0] - params[7]
        #                    c_e /  1 +    exp(-  a_e     *    node input E   )      
        time_series_E[i+1] = 1.0 / (1 + np.exp(-params[8] * time_series_E[i+1]))
        #                    c_i /  1 +    exp(-  a_i     *    node input I    )
        time_series_I[j_1] = 1.0 / (1 + np.exp(-params[9] * time_series_I[j_1]))
        #                         (     k_e     -    r_e     *        E       ) *  S_e(input node E) -       E         ) /   tau_e
        time_series_E[i+1] = dt*(((params[10] - params[12] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / params[4]
        #                         (     k_i     -    r_i     *        I       )  *  S_i(input node I)  -       I       )    /   tau_i 
        time_series_I[j_1] = dt*(((params[11] - params[13] * time_series_I[j_0]) * time_series_I[j_1]) - time_series_I[j_0]) / params[5] 
        #time_series_E_noise = np.random.normal(0,1,size=num_nodes) *  DE 
        #time_series_I_noise = np.random.normal(0,1,size=num_nodes) *  DI 
        time_series_E_temp = time_series_E[i] + time_series_E[i+1] #+ time_series_E_noise
        time_series_I_temp = time_series_I[j_0] + time_series_I[j_1] #+ time_series_I_noise
        # Corrector point
        #                       c_ee   *       E          -   c_ei      *        I           +     conn kernel      *    a     *     network input      -  theta_e
        time_series_E_corr = params[0] * time_series_E_temp - params[1] * time_series_I_temp - params[6]
        #                       c_ie   *       E            -   c_ii    *        I           +     conn kernel      *    a     *     network input      -  theta_i
        time_series_I_corr = params[2] * time_series_E_temp - params[3] * time_series_I_temp - params[7]
        #                    c_e /  1 +    exp(-  a_e     *    node input E    )  
        time_series_E_corr = 1.0 / (1 + np.exp(-params[8] * time_series_E_corr))
        #                    c_i /  1 +    exp(-  a_i     *    node input I    )  
        time_series_I_corr = 1.0 / (1 + np.exp(-params[9] * time_series_I_corr))
        #                         (   k_e     -    r_e     *           E       ) *  S_e(input node E)  -       E           ) /   tau_e
        time_series_E_corr = dt*(((params[10] - params[12] * time_series_E_temp) * time_series_E_corr) - time_series_E_temp) / params[4] 
        #                         (   k_i     -    r_i     *           I       ) *  S_i(input node I)  -       I           ) /   tau_i
        time_series_I_corr = dt*(((params[11] - params[13] * time_series_I_temp) * time_series_I_corr) - time_series_I_temp) / params[5]
        # Heun point
        time_series_E_noise = np.random.normal(0,1,size=num_nodes) *  DE 
        time_series_I_noise = np.random.normal(0,1,size=num_nodes) *  DI
        time_series_E[i+1] = time_series_E[i] + (time_series_E[i+1]+time_series_E_corr)/2 + time_series_E_noise
        time_series_I[j_1] = time_series_I[j_0] + (time_series_I[j_1]+time_series_I_corr)/2 + time_series_I_noise
        # Correcting for ceiling and floor of activity
        if time_series_E[i+1].max() > 1.0 or time_series_I[j_1].max() > 1.0:
            time_series_E[i+1] = np.where(time_series_E[i+1] > 1.0, 1.0, time_series_E[i+1])
            time_series_I[j_1] = np.where(time_series_I[j_1] > 1.0, 1.0, time_series_I[j_1])
        if time_series_E[i+1].min() < 0.0 or time_series_I[j_1].min() < 0.0:
            time_series_E[i+1] = np.where(time_series_E[i+1] < 0.0, 0.0, time_series_E[i+1])
            time_series_I[j_1] = np.where(time_series_I[j_1] < 0.0, 0.0, time_series_I[j_1])

        
        # Running mean
        if i > psd_window_length+psd_transient_length and (i-psd_transient_length)%psd_window_step == 0:
            psd_running_mean += (np.abs(compute_fft2d(time_series_E[i+1-psd_window_length:i+1].T,fs=fs))**2)
            psd_window_counter+=1


    psd_ret = psd_running_mean[...,:cutoff]/psd_window_counter
    return psd_ret


def get_WC_node_PSD(
                    parameters,
                    cutoff = 100,
                    length: float = 12,
                    dt: float = 0.5,
                    initial_conditions = np.array([0.25,0.25]),
                    noise_seed: int = 42,
                    store_I: bool = False,
                    is_noise_log_scale = False,
                    psd_window_size = 2,
                    psd_overlap = 0.5,
                    psd_transient_size = 2,
                    verbose = 0,
                    **kwargs
        ):
    params = dcp(parameters)
    if params is None:
        return {"n_state_var":2,"state_var_init_range":[[1e-15,0.5-1e-15],[1e-15,0.5-1e-15]],"output":["psd"]}

    else:
        if len(params.shape) == 1:
            params = params[None,:]
        psd = simulate_WC_node_PSD(
                                                        params,
                                                        cutoff = cutoff,
                                                        length = length,
                                                        dt = dt,
                                                        initial_conditions = initial_conditions,
                                                        noise_seed = noise_seed,
                                                        store_I = store_I,
                                                        is_noise_log_scale = is_noise_log_scale,
                                                        psd_window_size = psd_window_size,
                                                        psd_overlap = psd_overlap,
                                                        psd_transient_size = psd_transient_size,
                                                        verbose = verbose,
            )
        psds = psd[...,:cutoff]
        return [psds]

def get_WC_node_PSD_with_input(parameters=None,S=None,**kwargs):
    if S is None and "connectivity" in kwargs:
        S = kwargs["connectivity"]
    if parameters is None:
        return  get_WC_node_PSD(parameters=parameters,**kwargs)
    params = dcp(parameters)
    params[6] = max(0,params[6] - S)
    return get_WC_node_PSD(parameters=params,**kwargs)


@numba.njit
def simulate_WC_conn_PSD(
            parameters,
            connectivity,
            cutoff = 100,
            coupling = 0.1,
            conduction_speed = 10.0,
            length: float = 12,
            dt: float = 0.5,
            initial_conditions = np.array([0.25,0.25]),
            noise_seed: int = 42,
            input_connections = np.array(([1.0, 0.0])),
            store_I: bool = False,
            is_noise_log_scale = False,
            psd_window_size = 2,
            psd_overlap = 0.5,
            psd_transient_size = 2,
            verbose = 0
            
    ):
    # Set seet
    np.random.seed(noise_seed)
    sim_length = int(1000/dt*length)
    num_nodes = connectivity.shape[1] 
    initial_conditions_ = np.zeros((2,connectivity.shape[-1]))
    initial_conditions_[0,:] = initial_conditions[0]
    initial_conditions_[1,:] = initial_conditions[1]
    
        
    # Input parameters are of shape (num_nodes, num_parameters) to match parameter estimation output, but we need (num_parameters,num_nodes) to simulate
    params = np.zeros_like(parameters.T) + parameters.T  
    if is_noise_log_scale:
        for i in range(1,3):
            params[-i] = 10**(params[-i])
            params[-i,params[-i]==1] = 0
    
    # White noise
    DE, DI = np.sqrt(2*params[-2]* dt), np.sqrt(2*params[-1]* dt)

    #Connectivity delays
    delay = (1/dt*connectivity[1]/conduction_speed).astype("int32") # nxn where delay[i] = [index of delay with self, index of delay with node 1, index of delay with node 2, ...]

    # Equivalent to allocating memory
    fs = int(1000/dt)
    psd_window_length = int(1000/dt*psd_window_size)
    psd_window_step = int(psd_window_length * (1-psd_overlap))
    psd_transient_length = int(1000/dt*psd_transient_size)

    time_series_E = np.zeros((sim_length+1, int(num_nodes)))
    time_series_I = np.zeros((3,int(num_nodes)))
    if store_I:
        time_series_I = np.empty((sim_length+1,int(num_nodes)))
    time_series_E_input = np.zeros((2,int(num_nodes)))
    time_series_E_temp = np.zeros((1,int(num_nodes)))
    time_series_I_temp = np.zeros((1,int(num_nodes)))
    time_series_E_corr = np.zeros((1,int(num_nodes)))
    time_series_I_corr = np.zeros((1,int(num_nodes)))
    time_series_E_noise = np.zeros((1,int(num_nodes)))
    time_series_I_noise = np.zeros((1,int(num_nodes)))

    # Set initial conditions
    time_series_E[0] = initial_conditions_[0]
    time_series_I[0] = initial_conditions_[1]
    input_delayed = np.zeros((int(num_nodes)))
    input_delayed_heun = np.zeros((int(num_nodes)))

    psd_window_counter = 0
    psd_running_mean = np.zeros((int(num_nodes),int(cutoff)))

    # Heun performed in-place within the time_series_X arrays to maximize speed
    for i in range(int(1000/dt*length)-1):
        # Forward Euler
        j_0 = i
        j_1 = (i+1)
        if not store_I:
            j_0 = i%2
            j_1 = (i+1)%2
        # Calculating input from other nodes
        if i > np.min(delay):
            for j in range(num_nodes):
                input_delayed *= 0
                input_delayed_heun *= 0
                temp_delay = i - np.where(i <= delay[j], -1, delay[j])
                input_delayed += np.diag(time_series_E[temp_delay])
                time_series_E_input[0][j] = np.dot(input_delayed, connectivity[0][j])
                temp_delay_hn = temp_delay + 1
                temp_delay = np.where(temp_delay_hn>i, i, temp_delay_hn)
                input_delayed_heun  += np.diag(time_series_E[temp_delay])
                time_series_E_input[1][j] = np.dot(input_delayed_heun, connectivity[0][j])
        else:
            for j in range(num_nodes):
                input_delayed *= 0
                input_delayed_heun *= 0
                temp_delay = i - np.ones_like(np.where(i <= delay[j], -1, delay[j]))
                input_delayed += np.diag(time_series_E[temp_delay])
                time_series_E_input[0][j] = np.dot(input_delayed, connectivity[0][j])
                input_delayed_heun  += np.diag(time_series_E[temp_delay])
                time_series_E_input[1][j] = np.dot(input_delayed_heun, connectivity[0][j])
                #               c_ee   *       E          -   c_ei    *        I           +     conn kernel      *    a     *     network input      -  theta_e
        time_series_E[i+1] = params[0] * time_series_E[i] - params[1] * time_series_I[j_0] + input_connections[0] * coupling * time_series_E_input[0] - params[6]
        #                       c_ie   *       E          -   c_ii    *        I           +     conn kernel      *    a     *     network input      -  theta_i
        time_series_I[j_1] = params[2] * time_series_E[i] - params[3] * time_series_I[j_0] + input_connections[1] * coupling * time_series_E_input[0] - params[7]
        #                    c_e /  1 +    exp(-  a_e     *    node input E   )      
        time_series_E[i+1] = 1.0 / (1 + np.exp(-params[8] * time_series_E[i+1]))
        #                    c_i /  1 +    exp(-  a_i     *    node input I    )
        time_series_I[j_1] = 1.0 / (1 + np.exp(-params[9] * time_series_I[j_1]))
        #                         (     k_e     -    r_e     *        E       ) *  S_e(input node E) -       E         ) /   tau_e
        time_series_E[i+1] = dt*(((params[10] - params[12] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / params[4]
        #                         (     k_i     -    r_i     *        I       )  *  S_i(input node I)  -       I       )    /   tau_i 
        time_series_I[j_1] = dt*(((params[11] - params[13] * time_series_I[j_0]) * time_series_I[j_1]) - time_series_I[j_0]) / params[5] 
        #time_series_E_noise = np.random.normal(0,1,size=num_nodes) *  DE 
        #time_series_I_noise = np.random.normal(0,1,size=num_nodes) *  DI 
        time_series_E_temp = time_series_E[i] + time_series_E[i+1] #+ time_series_E_noise
        time_series_I_temp = time_series_I[j_0] + time_series_I[j_1] #+ time_series_I_noise
        # Corrector point
        #                       c_ee   *       E          -   c_ei      *        I           +     conn kernel      *    a     *     network input      -  theta_e
        time_series_E_corr = params[0] * time_series_E_temp - params[1] * time_series_I_temp + input_connections[0] * coupling * time_series_E_input[1] - params[6]
        #                       c_ie   *       E            -   c_ii    *        I           +     conn kernel      *    a     *     network input      -  theta_i
        time_series_I_corr = params[2] * time_series_E_temp - params[3] * time_series_I_temp + input_connections[1] * coupling * time_series_E_input[1] - params[7]
        #                    c_e /  1 +    exp(-  a_e     *    node input E    )  
        time_series_E_corr = 1.0 / (1 + np.exp(-params[8] * time_series_E_corr))
        #                    c_i /  1 +    exp(-  a_i     *    node input I    )  
        time_series_I_corr = 1.0 / (1 + np.exp(-params[9] * time_series_I_corr))
        #                         (   k_e     -    r_e     *           E       ) *  S_e(input node E)  -       E           ) /   tau_e
        time_series_E_corr = dt*(((params[10] - params[12] * time_series_E_temp) * time_series_E_corr) - time_series_E_temp) / params[4] 
        #                         (   k_i     -    r_i     *           I       ) *  S_i(input node I)  -       I           ) /   tau_i
        time_series_I_corr = dt*(((params[11] - params[13] * time_series_I_temp) * time_series_I_corr) - time_series_I_temp) / params[5]
        # Heun point
        time_series_E_noise = np.random.normal(0,1,size=num_nodes) *  DE 
        time_series_I_noise = np.random.normal(0,1,size=num_nodes) *  DI
        time_series_E[i+1] = time_series_E[i] + (time_series_E[i+1]+time_series_E_corr)/2 + time_series_E_noise
        time_series_I[j_1] = time_series_I[j_0] + (time_series_I[j_1]+time_series_I_corr)/2 + time_series_I_noise
        # Correcting for ceiling and floor of activity
        if time_series_E[i+1].max() > 1.0 or time_series_I[j_1].max() > 1.0:
            time_series_E[i+1] = np.where(time_series_E[i+1] > 1.0, 1.0, time_series_E[i+1])
            time_series_I[j_1] = np.where(time_series_I[j_1] > 1.0, 1.0, time_series_I[j_1])
        if time_series_E[i+1].min() < 0.0 or time_series_I[j_1].min() < 0.0:
            time_series_E[i+1] = np.where(time_series_E[i+1] < 0.0, 0.0, time_series_E[i+1])
            time_series_I[j_1] = np.where(time_series_I[j_1] < 0.0, 0.0, time_series_I[j_1])

        
        # Welford's algorithm for computing the mean and std of the PSDs
        if i > psd_window_length+psd_transient_length and (i-psd_transient_length)%psd_window_step == 0:
            psd_running_mean += (np.abs(compute_fft2d(time_series_E[i+1-psd_window_length:i+1].T,fs=fs))**2)
            psd_window_counter+=1

    return psd_running_mean[:,:cutoff]/psd_window_counter

def get_WC_conn_PSD(
                    parameters,
                    connectivity,
                    cutoff = 100,
                    coupling = 0.1,
                    conduction_speed = 10.0,
                    length: float = 12,
                    dt: float = 0.5,
                    initial_conditions = np.array([0.25,0.25]),
                    noise_seed: int = 42,
                    input_connections = np.array(([1.0, 0.0])),
                    store_I: bool = False,
                    is_noise_log_scale = False,
                    psd_window_size = 2,
                    psd_overlap = 0.5,
                    psd_transient_size = 2,
                    is_debug = False,
                    verbose = 0, print_runtime=False,
                    precompile=True,
                    **kwargs
        ):
    if parameters is None:
        return {"n_state_var":2,"state_var_init_range":[[1e-15,0.5-1e-15],[1e-15,0.5-1e-15]],"output":["psd"]}
    elif is_debug or dt==0 or length == 0:
        return np.ones((connectivity.shape[-1],int(cutoff)))
    else:
        if precompile:
            _ = simulate_WC_conn_PSD(
                parameters[:2],connectivity[:,:2,:2],
                cutoff = cutoff,
                coupling = coupling,
                conduction_speed = conduction_speed,
                length = 5,
                dt = 4,
                noise_seed = noise_seed,
                input_connections = input_connections,
                store_I = store_I,
                is_noise_log_scale = is_noise_log_scale,
                psd_window_size = psd_window_size,
                psd_overlap = psd_overlap,
                psd_transient_size = 0,
                verbose = 0,
            )
        sim_start = dtm.now()
        psd = simulate_WC_conn_PSD(
                                                        parameters,connectivity,
                                                        cutoff = cutoff,
                                                        coupling = coupling,
                                                        conduction_speed = conduction_speed,
                                                        length = length,
                                                        dt = dt,
                                                        initial_conditions = initial_conditions,
                                                        noise_seed = noise_seed,
                                                        input_connections = input_connections,
                                                        store_I = store_I,
                                                        is_noise_log_scale = is_noise_log_scale,
                                                        psd_window_size = psd_window_size,
                                                        psd_overlap = psd_overlap,
                                                        psd_transient_size = psd_transient_size,
                                                        verbose = verbose,
            )
        sim_end = dtm.now()
        sim_runtime = sim_end-sim_start
        if print_runtime:
            print("Single sim runtime: ",str(sim_runtime))
        return [psd]