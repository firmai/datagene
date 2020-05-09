import pandas as pd
import numpy as np
from deltapy import extract 
import statsmodels.api as sm
from sklearn.metrics.pairwise import * 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorly.decomposition import matrix_product_state, tucker, parafac
from pyts.multivariate.transformation import MultivariateTransformer
from pyts.image import RecurrencePlot, GramianAngularField, MarkovTransitionField
from pyts.multivariate.image import JointRecurrencePlot
import esig.tosig as ts


## utilities

def tensor_to_pandas(arr_3d, cols):
  if cols==False:
    cols = ["ID"] + list("Instance_{}".format(ra) for ra in range(arr_3d.shape[0])) 
  a = arr_3d
  m,n,r = a.shape
  out_arr = np.column_stack((np.repeat(np.arange(m),n),a.reshape(m*n,-1)))
  return pd.DataFrame(out_arr, columns=cols)

class pipe:
    def __init__(self, value, func=None):
        self.value = value
        self.func = func
    def __getitem__(self, func):
        return pipe(self.value, func)        
    def __call__(self, *args, **kwargs):
        return pipe(self.func(self.value, *args, **kwargs))
    def __repr__(self):
        return 'pipe(%s, %s)' % (self.value, self.func)
      
### Tesseract
### ==========================================================================================
def mps_decomp_tess(arr_4d, dim=5):

  features_n =  len(arr_4d[0,0,0,:])
  factors = matrix_product_state(arr_4d, rank=[1, 1,dim,features_n,1]) #rank=[1, 1,1,1,1]
  print("Tensor Shape:")
  mps_dsf_3d = factors[2]; print(mps_dsf_3d.shape) ## tensor to tensor
  mps_odd_3d  = factors[1]; print(mps_odd_3d.shape)
  print("Matrix Shape:")
  mps_dd_2d = mps_odd_3d.mean(axis=0); print(mps_dd_2d.shape)
  mps_ff_2d = factors[3].mean(axis=2); print(mps_ff_2d.shape) #(Preferred Comparison Item)
  return mps_dsf_3d, mps_dd_2d, mps_ff_2d

def mps_decomp_4_to_2(arr_4d, dim=5):
  return (mps_decomp_tess(array, dim=dim)[2] for array in arr_4d)


### Tensor
### ==========================================================================================
def gaf_encode_3_to_4(arr_3d,swap=(2,2)):
  transformer_multi = MultivariateTransformer(GramianAngularField(),flatten=False)
  gramian_isff_4d = (transformer_multi.fit_transform(array.swapaxes(swap[0],swap[1])) for array in arr_3d)
  return gramian_isff_4d

def mrp_encode_3_to_4(arr_3d,percentage=60,swap=(2,2)):

  transformer_multi = MultivariateTransformer(RecurrencePlot(threshold='point',percentage=percentage),flatten=False)
  recplot_isff_4d = (transformer_multi.fit_transform(array.swapaxes(swap[0],swap[1])) for array in arr_3d)
  return recplot_isff_4d

def gaf_encode_3_to_4(arr_3d,swap=(2,2)):
  transformer_multi = MultivariateTransformer(GramianAngularField(),flatten=False)
  gramian_isff_4d = (transformer_multi.fit_transform(array.swapaxes(swap[0],swap[1])) for array in arr_3d)
  return  gramian_isff_4d


def mtf_encode_3_to_4(arr_3d, dim_mult=3):
  dim = arr_3d[0].mean(axis=0).shape[1]*dim_mult
  transformer_multi = MultivariateTransformer(MarkovTransitionField(image_size=dim),flatten=False)
  mtf_fsdd_4d = (transformer_multi.fit_transform(array.T) for array in arr_3d)
  return  mtf_fsdd_4d


def mps_decomp_tens(arr_3d,dim = 5,swap=(1,2)):
  features_n =  len(arr_3d[0,0,:])
  factors = matrix_product_state(arr_3d.swapaxes(swap[0],swap[1]), rank=[1,dim,features_n,1])
  print("Tensor Shape:")
  mps_dsf_3d = factors[1]; print(mps_dsf_3d.shape) ## tensor to tensor
  mps_off_3d  = factors[2]
  print("Matrix Shape:")
  mps_ff_2d = mps_off_3d.mean(axis=2); print(mps_ff_2d.shape) ## The one to be used in comparison
  return mps_dsf_3d, mps_ff_2d, factors


def mps_decomp_3_to_2(arr_3d,dim = 5,swap=(1,2)):
  return (mps_decomp_tens(array,dim = dim,swap=swap)[1] for array in  arr_3d)


def jrp_encode_3_to_3(arr_3d):
  transformer = JointRecurrencePlot()
  jrp_iss_3d = (transformer.transform(array.swapaxes(1,2)) for array in arr_3d)
  return jrp_iss_3d


def mean_3_to_2(arr_3d):
  mean_sf_2d = (array.mean(axis=0) for array in arr_3d)
  return mean_sf_2d


def sum_3_to_2(arr_3d):
  sum_sf_2d = (array.sum(axis=0) for array in arr_3d)
  return sum_sf_2d


def min_3_to_2(arr_3d):
  min_sf_2d = (array.min(axis=0) for array in arr_3d)
  return min_sf_2d


def var_3_to_2(arr_3d):
  var_sf_2d = (array.var(axis=0) for array in arr_3d)
  return var_sf_2d


def tucker_decomp_tens(arr_3d, dim=5):
  features_n =  len(arr_3d[0,0,:])
  tucker_tensor = tucker(arr_3d, rank=[1, features_n, dim], verbose=-2,random_state=1) ## (x,y,z) (x can change)
  print("feature_x_dimension matrix")
  tuck_fd_2d = tucker_tensor[1][2]
  print(tuck_fd_2d.shape)
  print("steps_x_feature matrix")
  tuck_sf_2d = tucker_tensor[1][1]
  print(tuck_sf_2d.shape)
  tuck_fd_2d_2 = tucker_tensor[0][0] ## one to compare with 
  tuck_i_1d = tucker_tensor[1][0]
  
  return tuck_fd_2d, tuck_sf_2d, tuck_fd_2d_2, tuck_i_1d

def tucker_decomp_3_to_2(arr_3d, dim=5):
  return (tucker_decomp_tens(array, dim=dim)[2] for array in arr_3d)


def parafac_decomp_tens(arr_3d,swap=(1,2), dim=5):
  factors = parafac(arr_3d.swapaxes(swap[0],swap[1]), rank=dim,random_state=1)
  print("feature_x_dimension matrix")
  parafac_fd_2d = factors[1][2]; print(parafac_fd_2d.shape)
  print("steps_x_dimension matrix")
  parafac_sd_2d = factors[1][1]; print(parafac_sd_2d.shape) ## this is the one to compe
  parafac_i_1d = factors[1][0]
  return parafac_fd_2d, parafac_sd_2d, parafac_i_1d

def parafac_decomp_3_to_2(arr_3d,swap=(1,2), dim=5):
  return (parafac_decomp_tens(array,swap=swap, dim=dim)[1] for array in arr_3d)


def tensor_to_pandas(arr_3d, cols):
  if cols==False:
    cols = ["ID"] + list("Instance_{}".format(ra) for ra in range(arr_3d.shape[2])) 
  a = arr_3d
  m,n,r = a.shape
  out_arr = np.column_stack((np.repeat(np.arange(m),n),a.reshape(m*n,-1)))
  return pd.DataFrame(out_arr, columns=cols)

def pca_decomp_mat(matrix,components=5):
  pca_r = PCA(n_components=components)
  cols = ["PCA_"+str(r) for r in range(components)]
  df_pca = pd.DataFrame(pca_r.fit_transform(matrix),columns=cols); print(df_pca.shape)
  return df_pca

def pca_decomp_3_to_2(arr_3d,components=3):
  df = (tensor_to_pandas(array.swapaxes(0,2), cols=False) for array in arr_3d)
  pca_sd_2d = (pca_decomp_mat(df_one,components=components) for df_one in df)
  return pca_sd_2d


### Matrix
### ==========================================================================================


def rp_encode_2_to_3(arr_2d,percentage=60):
  recplot = RecurrencePlot(threshold='point', percentage=percentage)
  recplot_sff_3d = (recplot.fit_transform(array) for array in arr_2d)
  return recplot_sff_3d


def gaf_encode_mat(arr_2d):
  transformer = GramianAngularField()
  gramian_sff_3d = transformer.transform(arr_2d)
  gramian_fss_3d = transformer.transform(arr_2d.T) #features
  return gramian_sff_3d, gramian_fss_3d

def gaf_encode_2_to_3(arr_2d):
  return (gaf_encode_mat(array)[0] for array in arr_2d)


def mtf_encode_2_to_3(arr_2d, dim_multiple=3):
  dim = arr_2d[0].shape[1]*dim_multiple
  mtf = MarkovTransitionField(image_size=dim)
  mtf_fdd_3d = (mtf.fit_transform(array.T) for array in arr_2d)
  return mtf_fdd_3d

def pca_decomp_2_to_2(matrix,components=5):
  pca_r = PCA(n_components=components)
  cols = ["PCA_"+str(r) for r in range(components)]
  df_pca = (pd.DataFrame(pca_r.fit_transform(mat),columns=cols) for mat in matrix)
  return df_pca

def svd_decomp_mat(arr_2d):
  u_ss_2d, s, vh_ff_2d = np.linalg.svd(arr_2d, full_matrices=True)
  return u_ss_2d, vh_ff_2d,s #singular values

def svd_decomp_2_to_2(arr_2d):
  return (svd_decomp_mat(array)[2] for array in arr_2d)

def qr_decomp_mat(arr_2d):
  q_sf_2d, r_ff_2d = np.linalg.qr(arr_2d)
  print(q_sf_2d.shape); print(r_ff_2d.shape) #r comparison component
  return q_sf_2d, r_ff_2d

def qr_decomp_2_to_2(arr_2d):
  return (qr_decomp_mat(array)[1] for array in arr_2d)

def lik_kernel_2_to_2(arr_2d):
  return (linear_kernel(array.T) for array in arr_2d)

def cos_kernel_2_to_2(arr_2d):
  return (cosine_similarity(array.T) for array in arr_2d)

def pok_kernel_2_to_2(arr_2d):
  return (polynomial_kernel(array.T) for array in arr_2d)

def lak_kernel_2_to_2(arr_2d):
  return (laplacian_kernel(array.T) for array in arr_2d)

def cov_2_to_2(arr_2d):
  cov_ff_2d = (np.cov(array.T) for array in arr_2d)
  return cov_ff_2d


def corr_2_to_2(arr_2d):
  corr_ff_2d = (pd.DataFrame(array).corr().values for array in arr_2d)
  return corr_ff_2d

def hist_2d_mat(arr_2d, first_ind=4, second_ind=5, dim=5, plot=False):
  hist_2d, xedges, yedges = np.histogram2d(arr_2d[:,first_ind], arr_2d[:,second_ind],bins=dim, density=True)
  print(hist_2d.shape)
  if plot:
    plt.hist2d(arr_2d[:,first_ind], arr_2d[:,second_ind],bins=dim, density=True)
  
  return hist_2d, xedges, yedges

def hist_2d_2_to_2(arr_2d, first_ind=4, second_ind=5, dim=5, plot=False):
  return (hist_2d_mat(array, first_ind=first_ind, second_ind=second_ind, dim=dim, plot=plot)[0] for array in arr_2d)

def pwd_2_to_2(arr_2d,metric="euclidean"):
  pwd_ss_2d = (pairwise_distances(array,metric=metric)  for array in arr_2d)
  return pwd_ss_2d

def prp_encode(arr_2d, eps=None, steps=None):
  if eps==None: eps=0.01
  if steps==None: steps=10
  d = pairwise_distances(arr_2d)
  d = np.floor(d / eps)
  d[d > steps] = steps; print(d.shape)
  return d

def prp_encode_2_to_2(arr_2d, eps=None, steps=None):
  return (prp_encode(array, eps=eps, steps=steps) for array in arr_2d)

# Vector
### ==========================================================================================

def pca_decomp_2_to_1(arr):
  return (pca_decomp_mat(array,components=1) for array in arr)

def GetWindow(x,h_window =10,f_window=3):

    X = np.array(x.iloc[:h_window,]).reshape(1,-1)
   
    for i in range(1,len(x)-h_window+1):
        x_i = np.array(x.iloc[i:i+h_window,]).reshape(1,-1)
        X = np.append(X,x_i, axis=0)
        
    rolling_window = (pd.DataFrame(X)).iloc[:-f_window,]
    return rolling_window

def GetNextMean(x,h_window=10,f_window=3):

    return pd.DataFrame((x.rolling(f_window).mean().iloc[h_window+f_window-1:,]))


def AddTime(X):
    t = np.linspace(0,1,len(X))
    return np.c_[t, X]

def Lead(X):
    
    s = X.shape
    x_0 = X[:,0]
    Lead = np.delete(np.repeat(x_0,2),0).reshape(-1,1)
     
    for j in range(1,s[1]):
        x_j = X[:,j]
        x_j_lead = np.delete(np.repeat(x_j,2),0).reshape(-1,1)
        Lead = np.concatenate((Lead,x_j_lead), axis =1)
     
    return Lead

def Lag(X):
    
    s = X.shape
    x_0 = X[:,0]
    Lag = np.delete(np.repeat(x_0,2),-1).reshape(-1,1)
  
    for j in range(1,s[1]):
        x_j = X[:,j]
        x_j_lag  = np.delete(np.repeat(x_j,2),-1).reshape(-1,1)
        Lag = np.concatenate((Lag,x_j_lag), axis = 1)
        
    return Lag

def sig_encode_vect(arr_1d):
  h_window = 10
  f_window = 3

  h_window = int(len(arr_1d)/2)
  f_window = int(h_window/2)
  sig_level = 2

  close_price = arr_1d


  y = GetNextMean(close_price, h_window = h_window , f_window = f_window)

  X_window = AddTime(GetWindow(close_price, h_window = h_window, f_window = f_window))
  X_window = pd.DataFrame(X_window)


  close_price_slice = close_price.iloc[0:(len(close_price)-(f_window))]
  close_price_array = np.array(close_price_slice).reshape(-1,1)
  lag = Lag(close_price_array)
  lead = Lead(AddTime(close_price_array))

  stream = np.concatenate((lead,lag), axis = 1)
  X_sig = [ts.stream2sig(stream[0:2*h_window-1], sig_level)]

  for i in range(1,(len(close_price)-(f_window)-(h_window)+1)):
      stream_i = stream[2*i: 2*(i+h_window)-1]
      signature_i = [ts.stream2sig(stream_i, sig_level)]
      X_sig = np.append(X_sig, signature_i, axis=0)

  X_sig = pd.DataFrame(X_sig); print(X_sig.shape)
  return X_sig

def sig_encode_1_to_2(arr_1d):
  return (sig_encode_vect(array).values for array in arr_1d)

# def scale(data):
#   scaler = StandardScaler()
#   scaled_data = scaler.fit_transform(data.T)
#   return data

def vect_extract(arr_in):
  dict_vect = {}
  dict_vect["abs_energy"] = extract.abs_energy(arr_in)
  dict_vect["mean_abs_change"] = extract.mean_abs_change(arr_in)
  dict_vect["mean_second_derivative_central"] = extract.mean_second_derivative_central(arr_in)
  dict_vect["partial_autocorrelation"] = extract.partial_autocorrelation(arr_in)[0][1]
  dict_vect["augmented_dickey_fuller"] = extract.augmented_dickey_fuller(arr_in)[0][1]
  dict_vect["gskew"] = extract.gskew(arr_in)
  dict_vect["stetson_mean"] = extract.stetson_mean(arr_in)
  dict_vect["count_above_mean"] = extract.count_above_mean(arr_in)
  dict_vect["longest_strike_below_mean"] = extract.longest_strike_below_mean(arr_in)
  dict_vect["wozniak"] = extract.wozniak(arr_in)[0][1]
  dict_vect["fft_coefficient"] = extract.fft_coefficient(arr_in)[0][1]
  dict_vect["ar_coefficient"] = extract.ar_coefficient(arr_in)[0][1]
  dict_vect["index_mass_quantile"] = extract.index_mass_quantile(arr_in)[0][1]
  dict_vect["number_cwt_peaks"] = extract.number_cwt_peaks(arr_in)[0][1]
  dict_vect["spkt_welch_density"] = extract.spkt_welch_density(arr_in)
  dict_vect["c3"] = extract.c3(arr_in)
  dict_vect["binned_entropy"] = extract.binned_entropy(arr_in)
  #dict_vect["svd_entropy"] = extract.svd_entropy(arr_in)[0][1]
  dict_vect["hjorth_complexity"] = extract.hjorth_complexity(arr_in)
  dict_vect["max_langevin_fixed_point"] =extract.max_langevin_fixed_point(arr_in)
  dict_vect["percent_amplitude"] = extract.percent_amplitude(arr_in)[0][1]
  dict_vect["cad_prob"] = extract.cad_prob(arr_in)[1][1]
  dict_vect["zero_crossing_derivative"] = extract.zero_crossing_derivative(arr_in)[0][1]
  dict_vect["detrended_fluctuation_analysis"] = extract.detrended_fluctuation_analysis(arr_in)
  #dict_vect["fisher_information"] = extract.fisher_information(arr_in)[0][1]
  dict_vect["higuchi_fractal_dimension"] = extract.higuchi_fractal_dimension(arr_in)[0][1]
  dict_vect["hurst_exponent"] = extract.hurst_exponent(arr_in)
  #dict_vect["largest_lyauponov_exponent"] = extract.largest_lyauponov_exponent(arr_in)[0][1]
  dict_vect["whelch_method"] = extract.whelch_method(arr_in)[0][1]
  dict_vect["find_freq"] = extract.find_freq(arr_in)[0][1]
  #dict_vect["flux_perc"] = extract.flux_perc(arr_in)['FluxPercentileRatioMid20']
  dict_vect["range_cum_s"] = extract.range_cum_s(arr_in)['Rcs']
  dict_vect["kurtosis"] = extract.kurtosis(arr_in)
  dict_vect["stetson_k"] = extract.stetson_k(arr_in)
  return pd.DataFrame.from_dict(dict_vect, orient="index",columns=["values"]), dict_vect

def vect_extract_1_to_1(arr_in):
  return (vect_extract(array)[0].values for array in arr_in)

def autocorr_1_to_1(arr_1d,time_deduct=1):
  nlags = len(arr_1d)-time_deduct
  acor = (sm.tsa.stattools.acf(array, unbiased=False, nlags=nlags, fft=True) for array in arr_1d)
  return acor
