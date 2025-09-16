import numpy as np 
from pystorm import mnt
from copy import deepcopy as dcp
from joblib import Parallel, delayed
from numba import njit
from functools import partial

def id_transform(x, *args,**kwargs):
    return x
def gof_template(Z,Z_tilde):
    return 0
def broadcast_like(x,y, missing_dims = None, infer_type="forward"):
    x_shape,y_shape = x.shape,y.shape
    if len(x_shape) == len(y_shape):
        return x
    
    if missing_dims is None:
        reversed_shape = False
        if infer_type == "backwards":
            x_shape,y_shape,reversed_shape = x_shape[::-1],y_shape[::-1],True
        x_new_shape = [1 for _ in range(len(y_shape))]
        x_dim_matched = [False for _ in range(len(x_shape))]
        for d,dim  in enumerate(y_shape):
            for xd, xdim in enumerate(x_shape):
                if not x_dim_matched[xd] and dim == xdim:
                    x_dim_matched[xd] = True
                    x_new_shape[d] = xdim
        if reversed_shape:
            x_new_shape = x_new_shape[::-1]
        
    else:
        x_new_shape = [y_shape[s] for s in range(len(y_shape))]
        if isinstance(missing_dims, list):
            for d in missing_dims:
                x_new_shape[d] = 1
        elif isinstance(missing_dims,int):
            x_new_shape[missing_dims] = 1
        else:
            raise ValueError(f"Incorrect missing_dims argument, only takes None, int, or list of ints: {missing_dims}")

    x_new_shape_tuple = tuple(x_new_shape)
    return x.reshape(x_new_shape_tuple)

def zpd(numerator,denominator,zero_behavior = "zero_out",**broadcast_kwargs):
    """Zero-proof divide"""
    denominator_ = dcp(denominator)
    numerator_ = dcp(numerator)
    if isinstance(denominator,float) or isinstance(denominator,int):
        if denominator_==0:
            denominator_ = 1
            if zero_behavior == "max_out":
                numerator_ *= 0
                numerator_ += 1e300
            elif zero_behavior == "zero_out":
                numerator_ *= 0
    else:
        make_like_fn = np.zeros_like
        if isinstance(denominator_,mnt._Tensor):
            make_like_fn = mnt.zeros_like
        zero_denominator_numerator_mask = (
                                                make_like_fn(numerator_) 
                                                * 
                                                broadcast_like(
                                                    (denominator_==0)+0.0, numerator_,
                                                    **broadcast_kwargs
                                                )
                                        ) == 1
        if zero_behavior == "max_out":
            numerator_[zero_denominator_numerator_mask] = 1e300
        elif zero_behavior == "zero_out":
            numerator_[zero_denominator_numerator_mask] = 0
        denominator_[denominator_==0] = 1
    try:
        return numerator_/denominator_
    except:
        return numerator_/broadcast_like(
                    denominator_, numerator_,
                    **broadcast_kwargs
                )




def mean_along_axis(data, axis = 0):
    return data.mean(axis)
def std_along_axis(data, axis = 0):
    return data.std(axis)
def index_along_axis(x,axis=-1,index=0):
    return np.take(x, index, axis=axis)
def sum_along_axis(x,axis):
    return x.sum(axis)
def indices_along_axis(x,axis=-1,indices=[0]):
    indices_format = [1 for _ in range(len(x.shape))]
    indices_format[axis] = len(indices)
    formatted_indices = mnt.ensure_numpy(indices).reshape(indices_format)
    return np.take_along_axis(x, formatted_indices, axis=axis)
def fn_along_indices(x,axis,indices,fn):
    return fn(indices_along_axis(x,axis=axis,indices=indices),axis=axis)



def clip_bounds_change(theta_ranges_new,theta_ranges):
    thetas_ = dcp(theta_ranges_new)
    thetas_[...,0] = np.where(thetas_[...,0]<theta_ranges[...,0],theta_ranges[...,0],thetas_[...,0])
    thetas_[...,1]= np.where(thetas_[...,1]>theta_ranges[...,1],theta_ranges[...,1],thetas_[...,1])
    return thetas_

def round_width_cosine_decay(rd, min_width, n_rounds = 25, max_width = 0.75, decay = 4, period = 3):
    # https://www.desmos.com/calculator/enzh74uava
    rd = mnt.ensure_torch([rd])
    return ((mnt.cos(rd * 2/period * mnt.pi)+1)/2  * (max_width - min_width) * (-rd*decay/n_rounds).exp() + min_width).item()


def parallelize(fn,jobs,n_jobs = 16, verbose = 1,parallelize_kwargs = {"max_nbytes":None}):
    res = Parallel(n_jobs = min(len(jobs),n_jobs), verbose = verbose,**parallelize_kwargs)(delayed(fn)(
                                                **job
                                        ) for job in jobs
                                ) 
    return res

def get_triu(mat, triu_inds):
    return mat[(Ellipsis,triu_inds[0],triu_inds[1])]

def get_min_max(mat, dim = None):
    mat = mnt.ensure_numpy(mat)
    cp_mat = mat.copy()
    cp_mat[np.isinf(mat)] = np.nan
    if dim is None:
        max_val = np.nanmax(cp_mat)
        min_val = np.nanmin(cp_mat)
    elif isinstance(dim, int):
        max_val = np.nanmax(cp_mat,axis = dim, keepdims = True)
        min_val = np.nanmin(cp_mat,axis = dim, keepdims = True)
    elif isinstance(dim, list) or isinstance(dim, tuple):
        max_val = np.nanmax(cp_mat,axis = dim[0], keepdims = True)
        min_val = np.nanmin(cp_mat,axis = dim[0], keepdims = True)
        for dim_val in range(1,len(dim)):
            max_val = np.nanmax(max_val,axis = dim[dim_val], keepdims = True)
            min_val = np.nanmin(min_val,axis = dim[dim_val], keepdims = True)

    return min_val, max_val

def get_upper_lower_quantiles(mat,lq = 0,uq = 1, dim = None):
    mat = mnt.ensure_numpy(mat)
    cp_mat = mat.copy()
    cp_mat[np.isinf(mat)] = np.nan
    if dim is None:
        max_val = np.nanquantile(cp_mat,uq)
        min_val = np.nanquantile(cp_mat,lq)
    elif isinstance(dim, int):
        max_val = np.nanquantile(cp_mat,uq,axis = dim, keepdims = True)
        min_val = np.nanquantile(cp_mat,lq,axis = dim, keepdims = True)
    elif isinstance(dim, list) or isinstance(dim, tuple):
        max_val = np.nanquantile(cp_mat,uq,axis = dim[0], keepdims = True)
        min_val = np.nanquantile(cp_mat,lq,axis = dim[0], keepdims = True)
        for dim_val in range(1,len(dim)):
            max_val = np.nanquantile(max_val,uq,axis = dim[dim_val], keepdims = True)
            min_val = np.nanquantile(min_val,lq,axis = dim[dim_val], keepdims = True)

    return min_val, max_val

def norm_max(mat, dim = None, quantiles = None):
    mat = mnt.ensure_numpy(mat)
    if quantiles is not None:
        min_val, max_val = get_upper_lower_quantiles(mat,lq=quantiles[0], uq=quantiles[1], dim=dim)
        quant_norm = zpd(mat-min_val,max_val-min_val)
        quant_norm[quant_norm<0] = 0
        quant_norm[quant_norm>1] = 1
        return quant_norm
    else:
        min_val, max_val = get_min_max(mat,dim=dim)
        return zpd(mat-min_val,max_val-min_val)


class DataSample:
    def __init__(
                    self,
                    T_features, T_feature_shapes,
                    data_sample_formatting = None,
                    formatting_args = None
        ):
        """ Data Sample model output/empirical data wrapper with T_feature formatting."""

        self.T_features = T_features
        self.n_T_features = len(T_features)
        self.T_feature_names = list(T_features.keys())
        self.n_data_sample_T_feature_fn = []
        self.T_feature_shapes = T_feature_shapes
        self.formatting_args = {}
        self.data_sample_formatting_fns = {}
        self.data_sample_formatting = {}

        # Collect formatting functions and arguments for each T_feature (default is identity with no args)
        for T in self.T_feature_names:
            self.formatting_args[T] = {}
            self.data_sample_formatting_fns[T] = data_sample_identity_fn
            if data_sample_formatting is not None and T in data_sample_formatting:
                self.data_sample_formatting_fns[T] = data_sample_formatting[T]
            if formatting_args is not None and  T in formatting_args:
                self.formatting_args[T] = formatting_args[T] 
            self.data_sample_formatting[T] = partial(self.data_sample_formatting_fns[T],**self.formatting_args[T])
        
        # Get info on number of data samples, number of data samples, T_feature shape, etc.
        if self.T_features[self.T_feature_names[0]] is None:
            self.n_data_samples = 0
        else:
            for i, T in enumerate(self.T_feature_names):
                if len(list(self.T_features[T].shape)) == len(self.T_feature_shapes[T]):
                    self.T_features[T] = self.T_features[T][None,...]
                    self.n_data_sample_T_feature_fn.append(1)
                else:
                    self.n_data_sample_T_feature_fn.append(self.T_features[T].shape[0])
            
            self.n_data_samples = self.n_data_sample_T_feature_fn[0]
            for n in range(1,len(self.n_data_sample_T_feature_fn)):
                if self.n_data_sample_T_feature_fn[n] != self.n_data_samples:
                    raise AttributeError(f"DataSample.__init__(): Incoherent number of data_samples across T_features: \n \t {self.n_data_sample_T_feature_fn} for T_features {self.T_feature_names}, \n \t Inferred from T_feature shapes {self.T_feature_shapes}")
    
    # For saving metadata 
    def _todict(self):
        try:
            from dill.source import getsource
            data_sample_formatting = {T:{"fn":getsource(self.data_sample_formatting_fns[T]),"args":self.formatting_args[T]} for T in self.T_feature_names}
        except:
            try:
                data_sample_formatting = {T:{"fn":self.data_sample_formatting_fns[T].__name__,"args":self.formatting_args[T]} for T in self.T_feature_names}
            except:
                data_sample_formatting = {T:{"fn":type(self.data_sample_formatting_fns[T]),"args":self.formatting_args[T]} for T in self.T_feature_names}
            
        return {
            "T_features":self.T_features,
            "n_T_features":self.n_T_features,
            "T_feature_names":self.T_feature_names,
            "n_data_samples":self.n_data_samples,
            "T_feature_shapes":self.T_feature_shapes,
            "data_sample_formatting":data_sample_formatting,
        }
    
def data_sample_identity_fn(features, **kwargs):
    return features

from pickle import load, dump
from datetime import datetime as dtm
from numpy import load as np_load
from numpy import savez_compressed as np_savez_compressed
from os.path import exists as os_exists
def save_zlog(log, directory, file_name, enforce_replace = False):
    try:
        _ = np_load(f'{directory}/{file_name}.npz', allow_pickle=True)
        if not enforce_replace:
            print("This file already exists. It was saved with date stamp, please select 'enforce_replace = True' to rewrite the original file.")
            np_savez_compressed(f'{directory}/{file_name}_{dtm.now().strftime("%H_%M_%d_%m_%Y")}', log=log)
            return 
    except:
        pass
    _ = np_savez_compressed(f'{directory}/{file_name}', log=log)
    return

def save_zlog_(log, directory, file_name, enforce_replace = False):
    timestamp = dtm.now().strftime("%H_%M_%d_%m_%Y")
    try:
        if os_exists(f'{directory}/{file_name}.npz') and not enforce_replace:
            print("This file already exists. It was saved with date stamp, please select 'enforce_replace = True' to rewrite the original file.")
            np_savez_compressed(f'{directory}/{file_name}_{timestamp}', log=log)
            return directory,f'{file_name}_{timestamp}',"npz"
    except:
        pass
    try:
        _ = np_savez_compressed(f'{directory}/{file_name}', log=log)
    except:
        return None
    return directory,f'{file_name}',"npz"


def _plk_save_wrapper(log,path):
    with open(path, 'wb') as f:
        dump(log,f)
def _dill_save_wrapper(log,path):
    from dill import dump as dill_dump
    with open(path, 'wb') as f:
        dill_dump(log,f)

def save_log_(log, directory, file_name, enforce_replace = False, use_dill = False, use_np_compressed = False, chunkify_log = False):
    timestamp = dtm.now().strftime("%H_%M_%d_%m_%Y")
    if use_np_compressed:
        return save_zlog_(log, directory, file_name, enforce_replace = enforce_replace)
    save_fn = _plk_save_wrapper
    if use_dill:
        save_fn = _dill_save_wrapper
    try:
        if os_exists(f'{directory}/{file_name}.pkl') and not enforce_replace:
            print("This file already exists. It was saved with date stamp, please select 'enforce_replace = True' to rewrite the original file.")
            save_fn(log,f'{directory}{file_name}_{timestamp}.pkl')
            return directory,f'{file_name}_{timestamp}',"pkl"
    except:
        pass
    try:
        save_fn(log,f'{directory}{file_name}.pkl')
    except:
        return None
    return directory,f'{file_name}','pkl'

def save_log_dill(log, directory, file_name, enforce_replace = False):
    from dill import dump as d_dump
    timestamp = dtm.now().strftime("%H_%M_%d_%m_%Y")
    try:
        if os_exists(f'{directory}/{file_name}.pkl') and not enforce_replace:
            print("This file already exists. It was saved with date stamp, please select 'enforce_replace = True' to rewrite the original file.")
            with open(f'{directory}{file_name}_{timestamp}.pkl', 'wb') as f:
                d_dump(log, f)
            return directory,f'{file_name}_{timestamp}',"pkl"
    except:
        pass
    try:
        with open(f'{directory}{file_name}.pkl', 'wb') as f:
            d_dump(log, f)
    except:
        return None
    return directory,f'{file_name}',"pkl"

def get_zlog(directory, file_name, autoload = True, verbose = 1):
    try:
        file_ = np_load(directory+'/'+file_name+".npz",allow_pickle=True)
        
    except:
        if verbose > 0:
            print(f"File '{directory}{file_name}.npz' does not exist.")
        return None
    if autoload:
        try:
            return file_["log"].item()
        except:
            pass
    return file_

def _plk_load_wrapper(path):
    with open(path, 'rb') as f:
        log = load(f)
    return log
def _dill_load_wrapper(path):
    from dill import load as dill_load
    with open(path, 'rb') as f:
        log = dill_load(f)
    return log


def get_log_(directory, file_name, extension, use_dill=False, autoload = True, verbose = 1,use_safe=False):
    from functools import partial
    load_fn = _plk_load_wrapper
    if extension =="npz":
        load_fn = partial(np_load,allow_pickle=True)
    if use_dill:
        load_fn = _dill_load_wrapper
    if use_safe:
        
        try:
            file_ = load_fn(directory+'/'+file_name+"."+extension)
        except:
            if verbose > 0:
                print(f"File '{directory}/{file_name}.{extension}' does not exist. Or just can't load it using {load_fn.__name__}")
            return None
    else:
        file_ = load_fn(directory+'/'+file_name+"."+extension)
    if extension =="npz" and autoload:
        try:
            return file_["log"].item()
        except:
            try:
                return file_["log"]
            except:
                pass
    return file_