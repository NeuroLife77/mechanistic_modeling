import numpy as np
from MHT.utils import norm_max, dcp,zpd,gof_template,id_transform
from MHT.stats import uniform_prior
from pystorm import mnt

def approximate_expected_from_finite_theta_sample_set(theta_samples_dict, verbose = 0):
    theta_samples_info_dict = dcp(theta_samples_dict)
    try:
        normed_gof = norm_max(theta_samples_info_dict["gof"]*theta_samples_info_dict["prior"])
        gof_sum = normed_gof.sum()
    except:
        if verbose > 0:
            from lib.log_utils import log_content
            log_content(theta_samples_info_dict)
        raise
    if gof_sum == 0:
        gof_sum = 1
        approx_param_likelihood = zpd(theta_samples_info_dict["prior"],theta_samples_info_dict["prior"].sum())
    else:
        approx_param_likelihood = zpd(normed_gof,gof_sum)
    theta_samples_info_dict["approx_param_likelihood"] = approx_param_likelihood

    approx_param_likelihood_theta = np.zeros_like(theta_samples_info_dict["theta_samples"]) + approx_param_likelihood.reshape([approx_param_likelihood.shape[0]] + [1]*len(theta_samples_info_dict["theta_samples"].shape[1:]))
    theta_samples_info_dict["expected_theta"] = (approx_param_likelihood_theta * theta_samples_info_dict["theta_samples"]).sum(0)

    approx_param_likelihood_gamma = np.zeros_like(theta_samples_info_dict["gamma_tilde"]) + approx_param_likelihood.reshape([approx_param_likelihood.shape[0]] + [1]*len(theta_samples_info_dict["gamma_tilde"].shape[1:]))
    theta_samples_info_dict["gamma_tilde"] = (theta_samples_info_dict["gamma_tilde"] * approx_param_likelihood_gamma).sum(0)

    approx_param_likelihood_Z = np.zeros_like(theta_samples_info_dict["Z_tilde"]) + approx_param_likelihood.reshape([approx_param_likelihood.shape[0]] + [1]*len(theta_samples_info_dict["Z_tilde"].shape[1:]))
    theta_samples_info_dict["expected_Z_tilde"] = (theta_samples_info_dict["Z_tilde"] * approx_param_likelihood_Z).sum(0)

    theta_samples_info_dict["expected_Y_tilde"] = [None for _ in range(len(theta_samples_info_dict["Y_tilde"]))]
    for i in range(len(theta_samples_info_dict["Y_tilde"])):
        approx_param_likelihood_Y_i = np.zeros_like(theta_samples_info_dict["Y_tilde"][i]) + approx_param_likelihood.reshape([approx_param_likelihood.shape[0]] + [1]*len(theta_samples_info_dict["Y_tilde"][i].shape[1:]))
        theta_samples_info_dict["expected_Y_tilde"][i] = (theta_samples_info_dict["Y_tilde"][i]*approx_param_likelihood_Y_i).sum(0) 

    return theta_samples_info_dict

def approximate_expected_from_finite_theta_sample_set_with_rejection(
                                                                        theta_samples_dict,
                                                                        reference_gof_type = "mean"
    ):
    theta_samples_info_dict = dcp(theta_samples_dict)
    theta_samples_info_dict["pre_rejection_gof"] = dcp(theta_samples_info_dict["gof"])
    pre_rejection_gofs = theta_samples_info_dict["pre_rejection_gof"]
    if reference_gof_type == "max":
        reference_gof = pre_rejection_gofs.max()
    elif reference_gof_type == "mean":
        reference_gof = pre_rejection_gofs.mean()
    elif reference_gof_type == "median":
        reference_gof = np.median(pre_rejection_gofs)
    elif "quantile" in reference_gof_type:
        qantile_val = float(reference_gof_type.split("-")[-1])
        index_of_qantile_val = np.ceil(pre_rejection_gofs.shape[0]*qantile_val).astype(int)
        reference_gof = pre_rejection_gofs[pre_rejection_gofs.argsort()[index_of_qantile_val]]
    else:
        reference_gof = pre_rejection_gofs[np.random.randint(0,pre_rejection_gofs.shape[0])]
    acceptance_ratio = zpd(pre_rejection_gofs,reference_gof)
    rejection_criteria = acceptance_ratio < np.random.uniform(0,1,size=acceptance_ratio.shape)
    theta_samples_info_dict["gof"][rejection_criteria] = 0
    normed_gof = norm_max(theta_samples_info_dict["gof"]*theta_samples_info_dict["prior"])
    gof_sum = normed_gof.sum()
    if gof_sum == 0:
        gof_sum = 1
        approx_param_likelihood = zpd(theta_samples_info_dict["prior"],theta_samples_info_dict["prior"].sum())
    else:
        approx_param_likelihood = zpd(normed_gof,gof_sum)
    theta_samples_info_dict["approx_param_likelihood"] = approx_param_likelihood

    approx_param_likelihood_theta = np.zeros_like(theta_samples_info_dict["theta_samples"]) + approx_param_likelihood.reshape([approx_param_likelihood.shape[0]] + [1]*len(theta_samples_info_dict["theta_samples"].shape[1:]))
    theta_samples_info_dict["expected_theta"] = (approx_param_likelihood_theta * theta_samples_info_dict["theta_samples"]).sum(0)

    approx_param_likelihood_gamma = np.zeros_like(theta_samples_info_dict["gamma_tilde"]) + approx_param_likelihood.reshape([approx_param_likelihood.shape[0]] + [1]*len(theta_samples_info_dict["gamma_tilde"].shape[1:]))
    theta_samples_info_dict["gamma_tilde"] = (theta_samples_info_dict["gamma_tilde"] * approx_param_likelihood_gamma).sum(0)

    approx_param_likelihood_Z = np.zeros_like(theta_samples_info_dict["Z_tilde"]) + approx_param_likelihood.reshape([approx_param_likelihood.shape[0]] + [1]*len(theta_samples_info_dict["Z_tilde"].shape[1:]))
    theta_samples_info_dict["expected_Z_tilde"] = (theta_samples_info_dict["Z_tilde"] * approx_param_likelihood_Z).sum(0)

    theta_samples_info_dict["expected_Y_tilde"] = [None for _ in range(len(theta_samples_info_dict["Y_tilde"]))]
    for i in range(len(theta_samples_info_dict["Y_tilde"])):
        approx_param_likelihood_Y_i = np.zeros_like(theta_samples_info_dict["Y_tilde"][i]) + approx_param_likelihood.reshape([approx_param_likelihood.shape[0]] + [1]*len(theta_samples_info_dict["Y_tilde"][i].shape[1:]))
        theta_samples_info_dict["expected_Y_tilde"][i] = (theta_samples_info_dict["Y_tilde"][i]*approx_param_likelihood_Y_i).sum(0) 

    return theta_samples_info_dict


def expectation_over_parameter_space(
                                S,Z,
                                model,
                                n_samples,
                                theta_sampler,
                                gof_fn=gof_template,
                                F_fn=[id_transform],T_fn=id_transform,
                                prior_fn=uniform_prior, expectation_fn = approximate_expected_from_finite_theta_sample_set,
                                sampler_returns_all = True, compress = True,
                                theta_sampler_kwargs = {}, model_kwargs = {}
    ):
    theta_sampler_instance = theta_sampler(
                                                **theta_sampler_kwargs
                            )
    if sampler_returns_all:
        theta_samples_,gamma_tilde_,Z_tilde_,gof_vals_ = theta_sampler_instance.sample(n_samples,S=S,Z=Z,model_fn=model,T_fn=T_fn,gof_fn=gof_fn,model_kwargs=model_kwargs,**theta_sampler_kwargs)
        theta_samples = mnt.ensure_numpy(theta_samples_)
        gamma_tilde_ = mnt.ensure_numpy(gamma_tilde_)
        Z_tilde_ = mnt.ensure_numpy(Z_tilde_)
        gof_vals_ = mnt.ensure_numpy(gof_vals_)
    else:
        theta_samples = mnt.ensure_numpy(theta_sampler_instance.sample(n_samples,S=S,Z=Z,model_fn=model,T_fn=T_fn,gof_fn=gof_fn,model_kwargs=model_kwargs,**theta_sampler_kwargs))
    prior = prior_fn(theta_samples)
    model_gamma_template = model(S,theta_samples[0],**model_kwargs)
    gamma_shape = model_gamma_template.shape
    Z_shape = T_fn(model_gamma_template).shape
    Y_shapes = [F_fni(model_gamma_template).shape for F_fni in F_fn]
    theta_samples_info_dict = {
        "theta_samples":theta_samples,
        "prior":prior,
        "gamma_tilde":np.zeros((theta_samples.shape[0],*gamma_shape)),
        "Z_tilde":np.zeros((theta_samples.shape[0],*Z_shape)),
        "Y_tilde":[np.zeros((theta_samples.shape[0],*Y_shapes[f])) for f in range(len(F_fn))],
        "gof":np.zeros((theta_samples.shape[0],))
    }

    for s,theta_sample in enumerate(theta_samples):
        if sampler_returns_all:
            gamma_tilde=gamma_tilde_[s]
            Z_tilde=Z_tilde_[s]
            gof_val=gof_vals_[s]
        else:
            gamma_tilde = model(S,theta_sample,**model_kwargs)
            Z_tilde = T_fn(gamma_tilde)
            gof_val = gof_fn(Z,Z_tilde)
        theta_samples_info_dict["gamma_tilde"][s] = gamma_tilde
        theta_samples_info_dict["Z_tilde"][s] = Z_tilde
        theta_samples_info_dict["gof"][s] = gof_val 
        for i in range(len(F_fn)):
            theta_samples_info_dict["Y_tilde"][i][s] = F_fn[i](gamma_tilde) 
    theta_samples_info_dict["gof"][np.logical_or(np.isnan(theta_samples_info_dict["gof"]),np.isinf(theta_samples_info_dict["gof"]))] = 0
    expected = expectation_fn(theta_samples_info_dict)
    if compress:
        expected.pop("gamma_tilde")
    return expected
    
