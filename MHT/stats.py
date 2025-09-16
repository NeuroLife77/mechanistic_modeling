from MHT.utils import *
import statsmodels.api as sm
import warnings
def uniform_prior(samples):
    return np.ones((samples.shape[0],))/samples.shape[0]

def zscore(x, dim = 0, use_std_replacement = False, **kwargs):
    if "axis" in kwargs:
        dim = kwargs["axis"]
    std_ = x.std(dim,keepdims=True)
    if use_std_replacement:
        std_[std_==0] = std_[std_>0].min()/10
    return zpd((x-x.mean(dim,keepdims=True)),std_)
def get_rval(t_vals,df):
    return t_vals/np.sqrt(df+t_vals**2-2)

@njit
def sse_likelihood_fn(Y,Y_model,factor=100):
    if np.isnan(Y_model).any():
        return 0.0
    sum_of_squared_residuals = ((Y-Y_model)**2).sum()
    density_val = (
        np.exp(
            -(sum_of_squared_residuals)*factor
        )
    )
    density_val = density_val
    if np.isnan(density_val):
        return 0.0
    else:
        return density_val
@njit
def ssmre_likelihood_fn(Y,Y_model, factor = 25):
    if np.isnan(Y_model).any():
        return 0.0
    sum_of_squared_residuals = (((Y-Y_model)/(np.abs(Y).max()))**2).sum()
    density_val = (
        np.exp(
            -(sum_of_squared_residuals)*factor
        )
    )
    if np.isnan(density_val):
        return 0.0
    else:
        return density_val

warnings.filterwarnings('ignore')
class StatsModelPolynomialWrapper:
    def __init__(self,use_intercept=True, degree = 1,norm_fn = id_transform, M = None,n_perms=0):
        self.use_intercept = use_intercept
        self.slope = 0
        self.intercept = 0
        self.r_value =  np.nan
        self.p_value = np.nan
        self.model_object = None
        self.model_properties = dict(
            slope=self.slope,
            intercept=self.intercept,
            r_value=self.r_value,
            p_value=self.p_value
        )
        self.degree = degree
        self.norm_fn = norm_fn
        self.coef_attr = None
        self.M=M
        self.model_object_RM = None
        self.model_object_RM_ = None
        self.betas_RM = None
        self.betas_perms = None
        self.n_perms=n_perms
        self.p_value_perms = None
        self.r_square_perms = None
        self.r_square_RM=None
        warnings.filterwarnings('ignore')

    def _prepareX(self,x):
        base_x = []
        if self.use_intercept:
            base_x = [np.ones((x.shape[0],1))]
        if len(x.shape) == 1:
            X = np.concatenate(base_x+[x[...,None]**(deg+1) for deg in range(self.degree)], axis = -1)
            if self.coef_attr is None:
                coef_attr = [f"x^{deg+1}" for deg in range(self.degree)]
        else:
            xs = base_x
            xs_names = []
            for i in range(x.shape[-1]):
                base_name = f"x_{i}"
                for deg in range(self.degree):
                    xs.append((x[...,i]**(deg+1))[...,None])
                    xs_names.append(base_name+f"^{deg+1}")
            X = np.concatenate(xs, axis = -1)
            coef_attr = xs_names
        X[:,1:] = self.norm_fn(X[:,1:])
        if self.use_intercept and self.coef_attr is None:
            coef_attr = ["intercept"] + coef_attr
        if self.coef_attr is None:
            self.coef_attr = coef_attr
        return X
    
    def _prepareY(self,y):
        Y = self.norm_fn(y)
        
        return Y
    
    def fit(self,x,y, weights = None):
        X = self._prepareX(x)
        Y = self._prepareY(y)
        if weights is None:
            if self.M is not None:
                self.model_object_RM_ = sm.RLM(Y,X,M=self.M)
                self.model_object_RM = self.model_object_RM_.fit()
                self.betas_RM = self.model_object_RM.params[int(self.use_intercept):] * (X[:,int(self.use_intercept):].std(0)/Y.std(0))
                self.r_square_RM=1-(self.model_object_RM.resid.std()**2/(Y.std()**2))
            self.model_object_ = sm.OLS(Y,X)
            self.ignored_M = True
        else:
            self.model_object_ = sm.WLS(Y,X, weights=weights)
        self.model_object = self.model_object_.fit()

        

        if self.use_intercept:
            intercept = self.model_object.params[0]
            slopes = self.model_object.params[1:]
            variables_pvalue = self.model_object.pvalues[1:]
            variables_rvalue = get_rval(self.model_object.tvalues[1:],df=self.model_object.df_resid-1)
        else:
            slopes = self.model_object.params
            intercept = 0
            variables_pvalue = self.model_object.pvalues
            
            variables_rvalue = get_rval(self.model_object.tvalues,df=self.model_object.df_resid-1)
        xy_ratio = (X[:,int(self.use_intercept):].std(0)/Y.std(0))
        betas = self.model_object.params[int(self.use_intercept):] * xy_ratio
        if isinstance(self.n_perms,int) and self.n_perms>0:
            betas_perms = np.zeros((self.n_perms,betas.shape[0]))
            pvals_perms = np.zeros((self.n_perms,))
            rsquare_perms = np.zeros((self.n_perms,))
            ordering = np.arange(X.shape[0])
            for p in range(self.n_perms):
                np.random.shuffle(ordering)
                y_perm = Y[ordering,...]
                temp_model = sm.OLS(y_perm,X).fit()
                betas_perms[p] = temp_model.params[int(self.use_intercept):]*xy_ratio
                pvals_perms[p] = temp_model.f_pvalue
                rsquare_perms[p] = temp_model.rsquared_adj
            self.betas_perms = betas_perms.T
            self.p_value_perms = pvals_perms
            self.r_square_perms = rsquare_perms
        elif isinstance(self.n_perms,np.ndarray):
            betas_perms = np.zeros((self.n_perms.shape[0],betas.shape[0]))
            pvals_perms = np.zeros((self.n_perms.shape[0],))
            rsquare_perms = np.zeros((self.n_perms.shape[0],))
            for p in range(self.n_perms.shape[0]):
                ordering = self.n_perms[p]
                y_perm = Y[ordering,...]
                temp_model = sm.OLS(y_perm,X).fit()
                betas_perms[p] = temp_model.params[int(self.use_intercept):]*xy_ratio
                pvals_perms[p] = temp_model.f_pvalue
                rsquare_perms[p] = temp_model.rsquared_adj
            self.betas_perms = betas_perms.T
            self.p_value_perms = pvals_perms
            self.r_square_perms = rsquare_perms
        self.slopes = slopes
        self.intercept = intercept
        self.r_values = variables_rvalue
        self.p_values = variables_pvalue
        self.betas = betas
        self.p_value_prod = self.p_values.prod() ** (1/len(self.p_values))
        self.p_value = self.model_object.f_pvalue
        self.pval_orders = self.p_values.argsort()
        self.r_value =  self.model_object.rsquared#max(0,self.model_object.rsquared_adj)**(1/2)
        self.model_properties = dict(
                slope=self.slopes[self.pval_orders[0]],
                intercept=self.intercept,
                r_value=self.r_value,
                p_value=self.p_value,
                r_values=self.r_values,
                p_values=self.p_values,
                betas=self.betas,
                betas_RM=self.betas_RM,
                betas_perms=self.betas_perms,
                p_value_perms=self.p_value_perms,
                r_square_perms=self.r_square_perms,
                r_square=self.model_object.rsquared,
                r_square_adj=self.model_object.rsquared_adj,
                r_square_RM=self.r_square_RM
            )

    def predict(self,x,model_to_use = "default", **kwargs):
        model_ = self.model_object
        if model_to_use == "RM" and self.M is not None:
            model_ = self.model_object_RM
        X_r = self._prepareX(x)
        return mnt.ensure_numpy(model_.predict(X_r))