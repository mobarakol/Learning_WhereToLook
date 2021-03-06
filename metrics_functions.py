'''
Original source: https://github.com/imatge-upc/saliency-2019-SalBCE/blob/master/src/evaluation/metrics_functions.py
'''
from functools import partial
import numpy as np
from numpy import random
import time
from skimage import exposure
from skimage import img_as_float
from skimage.transform import resize
import cv2
try:
	from cv2 import cv
except ImportError:
	cv = None
	# print('please install Python binding of OpenCV to compute EMD')

#from tools import normalize, match_hist

def normalize(x, method='standard', axis=None):
    '''Normalizes the input with specified method.
    Parameters
    ----------
    x : array-like
    method : string, optional
        Valid values for method are:
        - 'standard': mean=0, std=1
        - 'range': min=0, max=1
        - 'sum': sum=1
    axis : int, optional
        Axis perpendicular to which array is sliced and normalized.
        If None, array is flattened and normalized.
    Returns
    -------
    res : numpy.ndarray
        Normalized array.
    '''
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            #print(np.mean(x))
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def match_hist(image, cdf, bin_centers, nbins=256):
    '''Modify pixels of input image so that its histogram matches target image histogram, specified by:
    cdf, bin_centers = cumulative_distribution(target_image)
    Parameters
    ----------
    image : array
        Image to be transformed.
    cdf : 1D array
        Values of cumulative distribution function of the target histogram.
    bin_centers ; 1D array
        Centers of bins of the target histogram.
    nbins : int, optional
        Number of bins for image histogram.
    Returns
    -------
    out : float array
        Image array after histogram matching.
    References
    ----------
    [1] Matlab implementation histoMatch(MTX, N, X) by Simoncelli, 7/96.
    '''
    image = img_as_float(image)
    old_cdf, old_bin = exposure.cumulative_distribution(image, nbins) # Unlike [1], we didn't add small positive number to the histogram
    new_bin = np.interp(old_cdf, cdf, bin_centers)
    out = np.interp(image.ravel(), old_bin, new_bin)
    return out.reshape(image.shape)

def AUC_Judd(saliency_map, fixation_map, jitter=True):
	'''
	AUC stands for Area Under ROC Curve.
	This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
	ROC curve is created by sweeping through threshold values
	determined by range of saliency map values at fixation locations.
	True positive (tp) rate correspond to the ratio of saliency map values above threshold
	at fixation locations to the total number of fixation locations.
	False positive (fp) rate correspond to the ratio of saliency map values above threshold
	at all other locations to the total number of possible other locations (non-fixated image pixels).
	AUC=0.5 is chance level.
	Parameters
	----------
	saliency_map : real-valued matrix
	fixation_map : binary matrix
		Human fixation map.
	jitter : boolean, optional
		If True (default), a small random number would be added to each pixel of the saliency map.
		Jitter saliency maps that come from saliency models that have a lot of zero values.
		If the saliency map is made with a Gaussian then it does not need to be jittered
		as the values vary and there is not a large patch of the same value.
		In fact, jittering breaks the ordering in the small values!
	Returns
	-------
	AUC : float, between [0,1]
	'''
	saliency_map = np.array(saliency_map, copy=False)
	fixation_map = np.array(fixation_map, copy=False) > 0.5

	# If there are no fixation to predict, return NaN
	if not np.any(fixation_map):
		print('no fixation to predict')
		return np.nan
	# Make the saliency_map the size of the fixation_map
	if saliency_map.shape != fixation_map.shape:
		saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='nearest')
	# Jitter the saliency map slightly to disrupt ties of the same saliency value
	if jitter:
		saliency_map += random.rand(*saliency_map.shape) * 1e-7
	# Normalize saliency map to have values between [0,1]
	saliency_map = normalize(saliency_map, method='range')

	S = saliency_map.ravel()
	F = fixation_map.ravel()
	S_fix = S[F] # Saliency map values at fixation locations
	n_fix = len(S_fix)
	n_pixels = len(S)
	# Calculate AUC
	thresholds = sorted(S_fix, reverse=True)
	tp = np.zeros(len(thresholds)+2)
	fp = np.zeros(len(thresholds)+2)
	tp[0] = 0; tp[-1] = 1
	fp[0] = 0; fp[-1] = 1
	for k, thresh in enumerate(thresholds):
		above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
		tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
		fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
	return np.trapz(tp, fp) # y, x


def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
	'''
	This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
	ROC curve created by sweeping through threshold values at fixed step size
	until the maximum saliency map value.
	True positive (tp) rate correspond to the ratio of saliency map values above threshold
	at fixation locations to the total number of fixation locations.
	False positive (fp) rate correspond to the ratio of saliency map values above threshold
	at random locations to the total number of random locations
	(as many random locations as fixations, sampled uniformly from fixation_map ALL IMAGE PIXELS),
	averaging over n_rep number of selections of random locations.
	Parameters
	----------
	saliency_map : real-valued matrix
	fixation_map : binary matrix
		Human fixation map.
	n_rep : int, optional
		Number of repeats for random sampling of non-fixated locations.
	step_size : int, optional
		Step size for sweeping through saliency map.
	rand_sampler : callable
		S_rand = rand_sampler(S, F, n_rep, n_fix)
		Sample the saliency map at random locations to estimate false positive.
		Return the sampled saliency values, S_rand.shape=(n_fix,n_rep)
	Returns
	-------
	AUC : float, between [0,1]
	'''
	saliency_map = np.array(saliency_map, copy=False)
	fixation_map = np.array(fixation_map, copy=False) > 0.5
	# If there are no fixation to predict, return NaN
	if not np.any(fixation_map):
		print('no fixation to predict')
		return np.nan
	# Make the saliency_map the size of the fixation_map
	if saliency_map.shape != fixation_map.shape:
		saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='nearest')
	# Normalize saliency map to have values between [0,1]
	saliency_map = normalize(saliency_map, method='range')

	S = saliency_map.ravel()
	F = fixation_map.ravel()
	S_fix = S[F] # Saliency map values at fixation locations
	n_fix = len(S_fix)
	n_pixels = len(S)
	# For each fixation, sample n_rep values from anywhere on the saliency map
	if rand_sampler is None:
		r = random.randint(0, n_pixels, [n_fix, n_rep])
		S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
	else:
		S_rand = rand_sampler(S, F, n_rep, n_fix)
	# Calculate AUC per random split (set of random locations)
	auc = np.zeros(n_rep) * np.nan
	for rep in range(n_rep):
		thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
		tp = np.zeros(len(thresholds)+2)
		fp = np.zeros(len(thresholds)+2)
		tp[0] = 0; tp[-1] = 1
		fp[0] = 0; fp[-1] = 1
		for k, thresh in enumerate(thresholds):
			tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
			fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
		auc[rep] = np.trapz(tp, fp)
	return np.mean(auc) # Average across random splits


def AUC_shuffled(saliency_map, fixation_map, other_map, n_rep=100, step_size=0.1):
	'''
	This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
	ROC curve created by sweeping through threshold values at fixed step size
	until the maximum saliency map value.
	True positive (tp) rate correspond to the ratio of saliency map values above threshold
	at fixation locations to the total number of fixation locations.
	False positive (fp) rate correspond to the ratio of saliency map values above threshold
	at random locations to the total number of random locations
	(as many random locations as fixations, sampled uniformly from fixation_map ON OTHER IMAGES),
	averaging over n_rep number of selections of random locations.
	Parameters
	----------
	saliency_map : real-valued matrix
	fixation_map : binary matrix
		Human fixation map.
	other_map : binary matrix, same shape as fixation_map
		A binary fixation map (like fixation_map) by taking the union of fixations from M other random images
		(Borji uses M=10).
	n_rep : int, optional
		Number of repeats for random sampling of non-fixated locations.
	step_size : int, optional
		Step size for sweeping through saliency map.
	Returns
	-------
	AUC : float, between [0,1]
	'''
	other_map = np.array(other_map, copy=False) > 0.5
	if other_map.shape != fixation_map.shape:
		raise ValueError('other_map.shape != fixation_map.shape')

	# For each fixation, sample n_rep values (from fixated locations on other_map) on the saliency map
	def sample_other(other, S, F, n_rep, n_fix):
		fixated = np.nonzero(other)[0]
		indexer = list(map(lambda x: random.permutation(x)[:n_fix], np.tile(range(len(fixated)), [n_rep, 1])))
		r = fixated[np.transpose(indexer)]
		S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
		return S_rand
	return AUC_Borji(saliency_map, fixation_map, n_rep, step_size, partial(sample_other, other_map.ravel()))


def NSS(saliency_map, fixation_map):
	'''
	Normalized scanpath saliency of a saliency map,
	defined as the mean value of normalized (i.e., standardized) saliency map at fixation locations.
	You can think of it as a z-score. (Larger value implies better performance.)
	Parameters
	----------
	saliency_map : real-valued matrix
		If the two maps are different in shape, saliency_map will be resized to match fixation_map..
	fixation_map : binary matrix
		Human fixation map (1 for fixated location, 0 for elsewhere).
	Returns
	-------
	NSS : float, positive
	'''
	s_map = np.array(saliency_map, copy=False)
	f_map = np.array(fixation_map, copy=False) > 0.5
	#print(s_map.shape, f_map.shape)
	# if s_map.shape != f_map.shape:
	# 	s_map = resize(s_map, f_map.shape)
	# Normalize saliency map to have zero mean and unit std
	#print('s_map', np.max(s_map))
	s_map = normalize(s_map, method='standard')
	# Mean saliency value at fixation locations
	return np.mean(s_map[f_map])


def CC(saliency_map1, saliency_map2):
	'''
	Pearson's correlation coefficient between two different saliency maps
	(CC=0 for uncorrelated maps, CC=1 for perfect linear correlation).
	Parameters
	----------
	saliency_map1 : real-valued matrix
		If the two maps are different in shape, saliency_map1 will be resized to match saliency_map2.
	saliency_map2 : real-valued matrix
	Returns
	-------
	CC : float, between [-1,1]
	'''
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='nearest') # bi-cubic/nearest is what Matlab imresize() does by default
	# Normalize the two maps to have zero mean and unit std
	map1 = normalize(map1, method='standard')
	map2 = normalize(map2, method='standard')
	# Compute correlation coefficient
	return np.corrcoef(map1.ravel(), map2.ravel())[0,1]


def SIM(saliency_map1, saliency_map2):
	'''
	Similarity between two different saliency maps when viewed as distributions
	(SIM=1 means the distributions are identical).
	This similarity measure is also called **histogram intersection**.
	Parameters
	----------
	saliency_map1 : real-valued matrix
		If the two maps are different in shape, saliency_map1 will be resized to match saliency_map2.
	saliency_map2 : real-valued matrix
	Returns
	-------
	SIM : float, between [0,1]
	'''
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='nearest') # bi-cubic/nearest is what Matlab imresize() does by default
	# Normalize the two maps to have values between [0,1] and sum up to 1
	map1 = normalize(map1, method='range')
	map2 = normalize(map2, method='range')
	map1 = normalize(map1, method='sum')
	map2 = normalize(map2, method='sum')
	# Compute histogram intersection
	intersection = np.minimum(map1, map2)
	return np.sum(intersection)


def EMD(saliency_map1, saliency_map2, sub_sample=1/32.0):
	'''
	Earth Mover's Distance measures the distance between two probability distributions
	by how much transformation one distribution would need to undergo to match another
	(EMD=0 for identical distributions).
	Parameters
	----------
	saliency_map1 : real-valued matrix
		If the two maps are different in shape, saliency_map1 will be resized to match saliency_map2.
	saliency_map2 : real-valued matrix
	Returns
	-------
	EMD : float, positive
	'''
	map2 = np.array(saliency_map2, copy=False)
	map1 = np.array(saliency_map1, copy=False)
	# Reduce image size for efficiency of calculation
	#map2 = resize(map2, np.round(np.array(map2.shape)*sub_sample), order=3, mode='nearest')
	#map1 = resize(saliency_map1, map2.shape, order=3, mode='nearest')
	# Histogram match the images so they have the same mass
	map1 = match_hist(map1, *exposure.cumulative_distribution(map2))
	# Normalize the two maps to sum up to 1,
	# so that the score is independent of the starting amount of mass / spread of fixations of the fixation map
	map1 = normalize(map1, method='sum')
	map2 = normalize(map2, method='sum')
	# Compute EMD with OpenCV
	# - http://docs.opencv.org/modules/imgproc/doc/histograms.html#emd
	# - http://stackoverflow.com/questions/5101004/python-code-for-earth-movers-distance
	# - http://stackoverflow.com/questions/12535715/set-type-for-fromarray-in-opencv-for-python
	r, c = map2.shape
	x, y = np.meshgrid(range(c), range(r))
	signature1 = cv2.CreateImage(r * c, 3, cv2.CV_32FC1)
	signature1 = cv.CreateMat(r*c, 3, cv.CV_32FC1)
	signature2 = cv.CreateMat(r*c, 3, cv.CV_32FC1)
	cv.Convert(cv.fromarray(np.c_[map1.ravel(), x.ravel(), y.ravel()]), signature1)
	cv.Convert(cv.fromarray(np.c_[map2.ravel(), x.ravel(), y.ravel()]), signature2)
	return cv.CalcEMD2(signature2, signature1, cv.CV_DIST_L2)

