from sklearn import metrics
from google.colab.patches import cv2_imshow
import cv2
import skimage
import sys
from PIL import Image
import statistics
import numpy as np
from functools import reduce
import math
import operator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import ecopy as ep
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.nonparametric import bandwidths
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ranksums, mood, fligner, ansari, bartlett, levene, mannwhitneyu, ks_2samp, distributions
from collections import OrderedDict
from scipy.spatial.distance import directed_hausdorff
from scipy import stats
from scipy import special
import warnings
warnings.filterwarnings("ignore")
from datagene import dist_utilities as distu
from datagene.transform import vect_extract
import pandas as pd 
import pandasvault as pv





## Tensor
##Model Measures
##==============================================================================================


def regression_metrics(pred_list=[],name_list=[],valid=None):
  dict_metric = {}
  for pred, name in zip(pred_list,name_list):
    dict_metric[name] = {}
    dict_metric[name]["explained_variance_score"] = metrics.explained_variance_score(pred,valid)
    dict_metric[name]["max_error"]  = metrics.max_error(pred,valid)
    dict_metric[name]["mean_absolute_error"]  = metrics.mean_absolute_error(pred,valid)
    dict_metric[name]["mean_squared_error"]  = metrics.mean_squared_error(pred,valid)
    dict_metric[name]["mean_squared_error"]  = metrics.mean_squared_error(pred,valid)
    dict_metric[name]["mean_squared_log_error"]  = metrics.mean_squared_log_error(pred,valid)
    dict_metric[name]["median_absolute_error"]  = metrics.median_absolute_error(pred,valid)
    dict_metric[name]["r2_score"]  = metrics.r2_score(pred,valid)
  return dict_metric



def boot_stat(arr_mean, arr_mean_org):
  org_mean = statistics.mean(arr_mean_org)
  gen_mean = statistics.mean(arr_mean)
  org_gen_diff = org_mean - gen_mean
  print("Original: {}, Generated: {}, Difference: {}".format(org_mean,gen_mean,org_gen_diff))

  std = np.sqrt((np.array(arr_mean_org).var(ddof=1) + np.array(arr_mean).var(ddof=1))/2)
  N = len(arr_mean_org)
  un_var_t = (np.array(arr_mean_org).mean()-np.array(arr_mean).mean())/(std*np.sqrt(2/N))

  deg = 2*N - 2
  prob = distributions.t.sf(np.abs(un_var_t), deg) * 2
  return un_var_t, prob

def stat_pval(single_org_total,single_gen_total):
  std = np.sqrt((single_org_total.var(ddof=1) + single_gen_total.var(ddof=1))/2)
  N = len(single_org_total) ## p-value deflator
  un_var_t = (single_org_total.mean()-single_gen_total.mean())/(std*np.sqrt(2/N)); 
  deg = 2*N - 2
  pval = stats.distributions.t.sf(np.abs(un_var_t), deg) * 2  # use np.abs to get upper tail
  df_pval =pd.DataFrame(pval, index=un_var_t.index.to_list(),columns=["pvalue"]).round(5)
  return un_var_t, df_pval




## Matrix 
## Image Similarity
def ssim_grey(gray_org,gray_gen_1):
    

  # Compute SSIM between two images
  (score, diff) = skimage.measure.compare_ssim(gray_org, gray_gen_1,full=True)
  print("Image similarity:", score)

  # The diff image contains the actual image differences between the two images
  # and is represented as a floating point data type in the range [0,1] 
  # so we must convert the array to 8-bit unsigned integers in the range
  # [0,255] image1 we can use it with OpenCV
  diff = (diff * 255).astype("uint8")

  cv2_imshow(diff)
  cv2.waitKey()
  return score

## Histogram Similarity

def get_thumbnail(image, size=(128,128), stretch_to_fit=False, greyscale=False):
    " get a smaller version of the image - makes comparison much faster/easier"
    if not stretch_to_fit:
        image.thumbnail(size, Image.ANTIALIAS)
    else:
        image = image.resize(size); # for faster computation
    if greyscale:
        image = image.convert("L")  # Convert it to grayscale.
    return image
  
def image_histogram_similarity(image1, image2):

 
    image1 = get_thumbnail(image1)
    image2 = get_thumbnail(image2)
    
    h1 = image1.histogram()
    h2 = image2.histogram()
 
    rms = math.sqrt(reduce(operator.add,  list(map(lambda a,b: (a-b)**2, h1, h2)))/len(h1) )
    return rms


def _avhash(im):
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    im = im.resize((8, 8), Image.ANTIALIAS).convert('1').convert('L')
    avg = reduce(lambda x, y: x + y, im.getdata()) / 64.
    return reduce(lambda x, yz: x | (yz[1] << yz[0]), enumerate(map(lambda i: 0 if i < avg else 1, im.getdata())),
                  0)
    
def _hamming(h1, h2):
    h, d = 0, h1 ^ h2
    while d:
        h += 1
        d &= d - 1
    return h

def hash_simmilarity(img1,img2):
    
    hash1 = _avhash(img1)
    hash2 = _avhash(img2)
    dist = _hamming(hash1, hash2)
    simm = (64 - dist) * 100 / 64
    return simm



## Traditional Distance Measures

# Running Calculation Distance Calcultion for Sanity
def simple_diff_ratio_1(real, fake, mult=1):
  val = abs(((abs(real+1)-abs(fake+1))/(real+1))*100).mean()
  return val 


def distance_matrix_tests(dist_org,dist_gen):
  pvalue = {}
  stat ={}
  ### The mantel test would actually work very well for shapley values (output that is the same)
  ### Low p-value signifies similarity (r_obs is the statistic)
  mant = ep.Mantel(dist_org,dist_gen )
  pvalue["mantel"] = mant.pval
  stat["mantel"] = mant.r_obs


  ### Conducts permutation procrustes test of relationship between two non-diagonal (raw) matrices
  ### Low p-value signifies similarity (ml2 is the statistic)
  ep_test = ep.procrustes_test(dist_org,dist_gen)
  pvalue["procrustes"] = ep_test.pval
  stat["procrustes"] = ep_test.m12_obs

  # RDA Analysis
  # All values should be 1
  rd = ep.rda(dist_org,dist_gen )

  pvalue["rda"] = round(1- rd.R2,5)
  stat["rda"] = rd.R2a

  return pvalue, stat


## i have to redo this completley do the minus one
def entropy_dissimilarity(df_org_out,df_gen_out):
  d_m_m = OrderedDict()
  df_org_out = pd.DataFrame(df_org_out)
  df_gen_out = pd.DataFrame(df_gen_out)

  d_m_m["incept_multi"] = abs(distu.inception(df_gen_out)/distu.inception(df_org_out)-1)

  d_m_m["cent_multi"] = (abs(distu.centropy(df_org_out,df_org_out)/distu.centropy(df_org_out,df_gen_out)-1))
  place = abs(distu.ctc(df_org_out, df_org_out)/distu.ctc(df_org_out, df_org_out))
  #d_m_m["ctc_multi"] = abs(abs(distu.ctc(df_gen_out, df_org_out)/distu.ctc(df_org_out, df_gen_out))-place)
  d_m_m["corexdc_multi"] = abs(distu.corexdc(df_org_out, df_gen_out)/distu.corexdc(df_org_out, df_org_out)-1)
  d_m_m["ctcdc_mult"] = abs(distu.ctcdc(df_org_out, df_gen_out)/distu.ctcdc(df_org_out, df_org_out)-1)
  hold = abs(distu.mutual_information((df_org_out, df_org_out))/distu.mutual_information((df_org_out, df_org_out)))
  d_m_m["mutual_mult"] = abs(abs(distu.mutual_information((df_gen_out, df_gen_out))/distu.mutual_information((df_org_out, df_gen_out))) - hold)

  #d_m_m["corex"] = abs(distu.corex(df_org_out,df_org_out)/distu.corex(df_org_out,df_gen_out)) #correlation not distance
  hold = distu.mi(df_org_out,df_org_out, base=2, alpha=0)/distu.mi(df_org_out,df_org_out, base=2, alpha=0)
  d_m_m["minfo"] = abs(distu.mi(df_org_out,df_org_out, base=2, alpha=0)/distu.mi(df_org_out,df_gen_out, base=2, alpha=0) - hold) #could work

  return OrderedDict([(key,round(abs(ra),5)) for key, ra in d_m_m.items()])


def matrix_distance(a,b, skip_bhat=False):
  dict_dist = OrderedDict()
  #similarity
  dict_dist["correlation"] = distu.correlation(a, b)-distu.correlation(a,a)
  dict_dist["intersection"] = distu.return_intersection(a, b)-distu.return_intersection(a,a)
  dict_dist["renyi_divergence"] =  abs(distu.renyi_divergence(a,b))-abs(distu.renyi_divergence(a,a))
  dict_dist["pearson_rho"] = np.nanmean(distu.pearson_rho(a,b))-np.nanmean(distu.pearson_rho(a,a))
  dict_dist["jensen_shannon_divergence"] = np.nanmean(distu.jensen_shannon_divergence(a,b))-np.nanmean(distu.jensen_shannon_divergence(a,a))

  dict_dist["ks_statistic_kde"] = distu.ks_statistic_kde(a,b)
  dict_dist["js_metric"] = distu.js_metric(a,b)
  
  #distance
  dict_dist["dice"] = round(abs(distu.dice(a, b)),5) - round(abs(distu.dice(a, a)),5)
  dict_dist["kulsinski"] = round(distu.kulsinski(a, b),5) - round(distu.kulsinski(a, a),5)
  dict_dist["rogerstanimoto"] = round(abs(distu.rogerstanimoto(a, b)),5) - round(abs(distu.rogerstanimoto(a, a)),5)
  dict_dist["russellrao"] = round(abs(distu.russellrao(a, b)),5) - round(abs(distu.russellrao(a, a)),5)
  dict_dist["sokalmichener"] = round(abs(distu.sokalmichener(a, b)),5)- round(abs(distu.sokalmichener(a, a)),5)
  dict_dist["sokalsneath"] = round(abs(distu.sokalsneath(a, b)),5) - round(abs(distu.sokalsneath(a, a)),5)
  dict_dist["yule"] = round(distu.yule(a, b),5) - round(distu.yule(a, a),5)

  dict_dist["braycurtis"] = distu.braycurtis(a, b)
  dict_dist["directed_hausdorff"] = directed_hausdorff(a,b)[0]
  dict_dist["manhattan"] = distu.manhattan(a, b)
  #dict_dist["cosine"] = np.abs(np.nanmedian(cosine(a,b)))
  #dict_dist["sqeuclidean"] = np.abs(sqeuclidean(a, b))
  dict_dist["chi2"] =  distu.chi2_distance(a, b, eps = 1e-10)
  dict_dist["euclidean"] =  distu.euclidean_distance(a,b)
  dict_dist["variational"] =  distu.variational_distance(a,b)
  dict_dist["kulczynski"] = distu.kulSim(a, b)
  dict_dist["bray"] = distu.braySim(a, b, t="1")
  dict_dist["gower"] = distu.gowerSim(a, b)
  dict_dist["hellinger"] = distu.chordDis(a, b, t="1")
  dict_dist["czekanowski"] = distu.charDist(a, b, t="1")
  dict_dist["whittaker"] = np.mean(distu.whitDist(a, b, t="1"))
  dict_dist["canberra"] = distu.canDist(a, b, t="1")

 
  return OrderedDict([(key,round(abs(ra),5)) for key, ra in dict_dist.items()])



### Vectors


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the mean absolute percentage error between y_true and y_pred. Throws ValueError if y_true contains zero values.
    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :return: Mean absolute percentage error (float).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def bootstrapped_frame(probs_org,probs_gen,prop):
  
  scaler = MinMaxScaler()
  vect_gen = {}
  vect_org = {}
  for iteration in range(100):
    choice = np.random.choice(len(probs_org), int(len(probs_org)*prop), replace=False)
    choice.sort()
    arr_prob_org = np.sort(probs_org)[choice]
    arr_prob_gen = np.sort(probs_gen)[choice]
    vect_gen["Iteration_{}".format(iteration)] = vect_extract(arr_prob_gen)[1]
    vect_org["Iteration_{}".format(iteration)] = vect_extract(arr_prob_org)[1]

  vect_gen_df = pd.DataFrame.from_dict(vect_gen).T
  vect_org_df = pd.DataFrame.from_dict(vect_org).T

  scaler = MinMaxScaler()
  vect_org_df_sc = pd.DataFrame(scaler.fit_transform(vect_gen_df), columns=vect_org_df.columns)
  scaler = MinMaxScaler()
  vect_gen_df_sc = pd.DataFrame(scaler.fit_transform(vect_org_df), columns=vect_gen_df.columns)

  return vect_org_df, vect_org_df_sc, vect_gen_df, vect_gen_df_sc

def pca_variance(vect_org_df_sc, vect_gen_df_sc):
    pca_r = PCA(n_components=10)
    pca_f = PCA(n_components=10)

    pca_r.fit_transform(vect_org_df_sc)
    pca_f.fit_transform(vect_gen_df_sc)

    pca_error = mean_absolute_percentage_error(pca_r.explained_variance_, pca_f.explained_variance_); pca_error
    pca_corr, p_value = pearsonr(pca_r.explained_variance_, pca_f.explained_variance_); pca_corr

    return pca_error, pca_corr, p_value

def pca_extract_explain(y_pred_org,y_pred_gen,prop=0.1):
  _, vect_org_df_sc, _, vect_gen_df_sc = bootstrapped_frame(y_pred_org,y_pred_gen,prop)

  # Some Cleaning Operations
  vect_org_df_sc = vect_org_df_sc.dropna(thresh = len(vect_org_df_sc)*0.95, axis = "columns") 
  vect_gen_df_sc = vect_gen_df_sc.dropna(thresh = len(vect_gen_df_sc)*0.95, axis = "columns") 
  qconstant_col_o = pv.constant_feature_detect(data=vect_org_df_sc,threshold=0.9)
  qconstant_col_g = pv.constant_feature_detect(data=vect_gen_df_sc,threshold=0.9)
  constant_cols = set(qconstant_col_o) and set(qconstant_col_g)
  vect_org_df_sc = vect_org_df_sc.drop(constant_cols, axis=1)
  vect_gen_df_sc = vect_gen_df_sc.drop(constant_cols, axis=1)
  vect_org_df_sc = vect_org_df_sc.replace({-np.inf:np.nan,np.inf:np.nan}).ffill().bfill()
  vect_gen_df_sc = vect_gen_df_sc.replace({-np.inf:np.nan,np.inf:np.nan}).ffill().bfill()

  pca_error, pca_corr, p_value = pca_variance(vect_org_df_sc, vect_gen_df_sc)
  print("PCA Error: {}, PCA Correlation: {}, p-value: {}".format(pca_error, pca_corr, p_value))
  return pca_error, pca_corr, p_value 


 # Distance and Similarity measures.
# (All similarity measures, turned into distance measures)
def vector_distance(a,b):
  dict_dist = {}
  dict_dist["braycurtis"] = distu.braycurtis(a, b)
  dict_dist["canberra"] = distu.canberra(a, b)
  dict_dist["correlation"] = distu.correlation(a, b)
  dict_dist["cosine"] = distu.cosine(a, b)
  dict_dist["dice"] = distu.dice(a, b)
  dict_dist["euclidean"] = distu.euclidean(a, b)
  # dict_dist["hamming"] = distu.hamming(a, b) #discrete
  # dict_dist["jaccard"] = distu.jaccard(a, b) #discrete
  dict_dist["kulsinski"] = distu.kulsinski(a, b)
  dict_dist["manhattan"] = distu.manhattan(a, b)
  dict_dist["rogerstanimoto"] = distu.rogerstanimoto(a, b)
  dict_dist["russellrao"] = distu.russellrao(a, b)
  dict_dist["sokalmichener"] = distu.sokalmichener(a, b)
  dict_dist["sokalsneath"] = distu.sokalsneath(a, b)
  dict_dist["sqeuclidean"] = distu.sqeuclidean(a, b)
  dict_dist["yule"] = distu.yule(a, b)
  dict_dist["ks_statistic"] = distu.ks_statistic_vec(a,b) #here

  return dict_dist

def vector_distance_boots(probs_org_1, probs_org_2):
  vect_org_dist = {}
  for iteration in range(100):
    choice = np.random.choice(len(probs_org_1), int(len(probs_org_1)*.01), replace=False)
    choice.sort()
    arr_prob_org_1 = probs_org_1[choice]
    arr_prob_org_2 = probs_org_2[choice]
    vect_org_dist["Iteration_{}".format(iteration)] = vector_distance(arr_prob_org_1,arr_prob_org_2 )

  return pd.DataFrame.from_dict(vect_org_dist).T



def density_arr(dist_org_1,dist_org_2, col):

  kernel_switch = dict(gau=kernels.Gaussian, epa=kernels.Epanechnikov,
                      uni=kernels.Uniform, tri=kernels.Triangular,
                      biw=kernels.Biweight, triw=kernels.Triweight,
                      cos=kernels.Cosine, cos2=kernels.Cosine2)
  bw = bandwidths.select_bandwidth(dist_org_1[col], "normal_reference", kernel_switch["gau"]())


  def kde_func_sans(series, bw ):
    kde = sm.nonparametric.KDEUnivariate(series)
    kde.fit(bw=bw) # Estimate the densities

    return kde.support, kde.density, kde 

  support_org_1, density_org_1, _ = kde_func_sans(dist_org_1[col], bw)

  max_target = dist_org_1[col].astype(int).max()
  min_target = dist_org_1[col].astype(int).min()
  change = dist_org_2[col]
  scaled_array = np.interp(change, (change.min(), change.max()), (min_target, max_target))
  support_org_2, density_org_2, _ = kde_func_sans(scaled_array, bw)
  return density_org_1, density_org_2, support_org_1, support_org_2



def distance_funct_kde(df_1, df_2, X):
    
  for en, col in enumerate(X):
    density_org_1, density_org_2, _, _ = density_arr(df_1, df_2, col)
    vect_org_dens_dist = {}
    for iteration in range(100):
      choice = np.random.choice(len(density_org_1), int(len(density_org_1)*.01), replace=False)
      choice.sort()
      arr_dens_org_1 = density_org_1[choice]
      arr_dens_org_2 = density_org_2[choice]
      vect_org_dens_dist["Iteration_{}".format(iteration)] = vector_distance(arr_dens_org_1,arr_dens_org_2 )
    
    mean_series = pd.DataFrame.from_dict(vect_org_dens_dist).T.mean()
    if en==0:
      frame_all = mean_series.to_frame()
    else:
      frame_all = pd.merge(frame_all,mean_series.to_frame(), left_index=True, right_index=True,how="left" )

  frame_all.columns = X
  return  frame_all

def distribution_distance_map(df_org_out,df_gen_out,X):
  frac = 0.5
  vect_org_dens_dist = distance_funct_kde(df_org_out.sample(frac=frac).astype('double'),df_org_out.sample(frac=frac).astype('double'), X)
  vect_gen_dens_dist = distance_funct_kde(df_gen_out.astype('double'),df_org_out.astype('double'), X)
  vect_gen_dens_dist.loc["canberra",:] = vect_gen_dens_dist.loc["canberra",:]*frac
  vect_gen_dens_dist.loc["sqeuclidean",:] = vect_gen_dens_dist.loc["sqeuclidean",:]*frac
  vect_gen_dens_dist.loc["manhattan",:] = vect_gen_dens_dist.loc["manhattan",:]*frac
  return vect_gen_dens_dist, vect_org_dens_dist


def kde_func(series, bw, additional=[]):
  kde = sm.nonparametric.KDEUnivariate(series)
  kde.fit(bw=bw) # Estimate the densities

  kde.support
  kde.density
  fig = plt.figure(figsize=(12, 5))
  ax = fig.add_subplot(111)

  # Plot the histrogram
  ax.hist(series, bins=20, density=True, label='Histogram from samples',
          zorder=5, edgecolor='k', alpha=0.5)

  # Plot the KDE as fitted using the default arguments
  ax.plot(kde.support, kde.density, lw=3, label='KDE from samples', zorder=10)

  if len(additional)>0:
    for r in additional:
      ax.plot(kde.support, r, lw=3, label='KDE other: {}'.format(r), zorder=10)

  return ax, kde.support, kde.density, kde 

def rescale_interpolate(df_org, df_gen, feat):
  max_target = df_org[feat].astype(int).max()
  min_target = df_org[feat].astype(int).min()
  change = df_gen[feat]
  #change = df_org["Age"].sample(int(len(df_org)*0.5))
  scaled_array = np.interp(change, (change.min(), change.max()), (min_target, max_target))
  return scaled_array

kernel_switch = dict(gau=kernels.Gaussian, epa=kernels.Epanechnikov,
                    uni=kernels.Uniform, tri=kernels.Triangular,
                    biw=kernels.Biweight, triw=kernels.Triweight,
                    cos=kernels.Cosine, cos2=kernels.Cosine2)



# Curves
def curve_metrics(matrix_org,matrix_gen):
  c_dict = {}
  c_dict["Curve Length Difference"] = round(distu.curve_length_measure(matrix_org, matrix_gen),5)
  c_dict["Mean Absolute Difference"] = np.mean(abs(matrix_org[:, 1]-matrix_gen[:, 1])).round(5)
  c_dict["Partial Curve Mapping"] = round(distu.pcm(matrix_gen, matrix_org),5)
  c_dict["Discrete Frechet Distance"] = round(distu.frechet_dist(matrix_gen, matrix_org),5)
  c_dict["Dynamic Time Warping"] = round(distu.dtw(matrix_gen, matrix_org)[0],5)
  c_dict["Area Between Curves"] = round(distu.area_between_two_curves(matrix_org, matrix_gen),5)

  return c_dict



#curve_dictionary(matrix_org, matrix_gen):
# I removed two methods, because they are too costly
def dict_curve(matrix_org_s,matrix_gen_s):
  c_dict = {}
  c_dict["Curve Length Difference"] = round(distu.curve_length_measure(matrix_org_s, matrix_gen_s),5)
  c_dict["Partial Curve Mapping"] = round(distu.pcm(matrix_gen_s, matrix_org_s),5)
  c_dict["Discrete Frechet Distance"] = round(distu.frechet_dist(matrix_gen_s, matrix_org_s),5)
  c_dict["Dynamic Time Warping"] = round(distu.dtw(matrix_gen_s, matrix_org_s)[0],5)
  c_dict["Area Between Curves"] = round(distu.area_between_two_curves(matrix_org_s, matrix_gen_s),5)
  #c_dict["KS Statistic X x Y"] = round(ks_2samp(matrix_org_s[:,1]*matrix_org_s[:,0], matrix_gen_s[:,1]*matrix_gen_s[:,1])[0],5)
  return c_dict

def curve_kde_map(df_1, df_2, X, frac=0.01):

  for en, col in enumerate(X):
    print(col)
    density_org_1, density_org_2, support_org_1, support_org_2 = density_arr(df_1, df_2, col)
    vect_org_dens_curve = {}
    for iteration in range(20):
      choice = np.random.choice(len(density_org_1), int(len(density_org_1)*frac), replace=False)
      choice.sort()

      matrix_dens_org_1 = np.array([support_org_1,density_org_1]).T[choice]
      matrix_dens_org_2 = np.array([support_org_2,density_org_2]).T[choice]

      vect_org_dens_curve["Iteration_{}".format(iteration)] = dict_curve(matrix_dens_org_1,matrix_dens_org_2 )
    
    mean_series = pd.DataFrame.from_dict(vect_org_dens_curve).T.mean()
    if en==0:
      frame_all = mean_series.to_frame()
    else:
      frame_all = pd.merge(frame_all,mean_series.to_frame(), left_index=True, right_index=True,how="left" )

  frame_all.columns = X
  return  frame_all


# hypotheses distances
# low p-value means that the array of values are from a different source (or are distinguisable)

def vector_hypotheses(a,b ):

  dict_stat = {}
  dict_pval = {}
  pea = pearsonr(a, b)
  dict_stat["pearsonr"], dict_pval["pearsonr"] = pea[0], pea[1]
  ran = ranksums(a, b)
  dict_stat["ranksums"], dict_pval["ranksums"] = ran[0], ran[1]
  moo = mood(a, b) 
  dict_stat["mood"], dict_pval["mood"] = moo[0], moo[1]
  fli = fligner(a, b)
  dict_stat["fligner"], dict_pval["fligner"] = fli[0], fli[1]
  ans = ansari(a, b)
  dict_stat["ansari"], dict_pval["ansari"] = ans[0], ans[1]
  bar = bartlett(a, b)
  dict_stat["bartlett"], dict_pval["bartlett"] = bar[0], bar[1]
  lev = levene(a, b)
  dict_stat["levene"], dict_pval["levene"] = lev[0], lev[1]
  man = mannwhitneyu(a, b)
  dict_stat["mannwhitneyu"], dict_pval["mannwhitneyu"] = man[0], man[1]
  return dict_stat, dict_pval

