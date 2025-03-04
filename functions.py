#functions
import numpy as np
from PIL import Image

#Compute the histogram equalization transformation for an image's luminance matrix
def get_equalization_transform_of_img(img_array):
    #Compute frequencies of each value in the image (histogram)
    bits_per_pixel = img_array.dtype.itemsize*8
    L=2**bits_per_pixel
    hist = np.zeros(L, dtype=int)
    height, width = img_array.shape
    for i in range(height):
        for j in range(width):
            hist[img_array[i, j]] += 1

    num_of_pixels=height*width
    #Compute the cumulative function
    v=np.zeros(L)
    v[0]=hist[0]/num_of_pixels
    for i in range(1, L):
        v[i] = v[i - 1] + hist[i]/num_of_pixels
    
    #Equalization Transform  
    equalization_transform=np.round((v-v[0])/(1-v[0]) * (L - 1))
  
    return equalization_transform

#Apply global histogram equalization to enhance the contrast of an image
def perform_global_hist_equalization(img_array):
  equalization_transform=get_equalization_transform_of_img(img_array)
  height, width = img_array.shape
  equalized_img = np.zeros_like(img_array)
  for i in range(height):
    for j in range(width):
      equalized_img[i, j] = equalization_transform[img_array[i, j]]

  return equalized_img

#Compute local histogram equalization transformations for image regions
def calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w): 
  region_to_eq_transform = {}
  height, width = img_array.shape
  #Iterate over the image array and extract regions
  for i in range(0, height, region_len_h):
    for j in range(0, width, region_len_w):
      subimg_array = img_array[i:i+region_len_h, j:j+region_len_w]
      equalized_region=get_equalization_transform_of_img(subimg_array)
      region_to_eq_transform[(i, j)] = equalized_region

  #If the division height/region_len_h & width/region_len_w is not perfect
  #Find the coordinates of the remaining region
  num_h_regions =height //region_len_h
  num_w_regions =width //region_len_w
  last_region_h = height-num_h_regions*region_len_h
  last_region_w = width-num_w_regions*region_len_w

  if last_region_h !=0:
    for j in range(0, width, region_len_w):
      subimg_array = img_array[last_region_h:height, j:j+region_len_w]
      equalized_region=get_equalization_transform_of_img(subimg_array)
      region_to_eq_transform[(height - last_region_h, j)] = equalized_region
    #Get the last corner
    if last_region_w !=0: 
      subimg_array = img_array[last_region_h:height, last_region_w:width]
      equalized_region=get_equalization_transform_of_img(subimg_array)
      region_to_eq_transform[(height - last_region_h, width-last_region_w)] = equalized_region

  if last_region_w !=0:
    for i in range(0, height, region_len_h):
      subimg_array = img_array[i:i+region_len_h, last_region_w:width]
      equalized_region=get_equalization_transform_of_img(subimg_array)
      region_to_eq_transform[(i, width-last_region_w)] = equalized_region    
  
  return region_to_eq_transform

#Apply adaptive histogram equalization with optional interpolation
def perform_adaptive_hist_equalization(img_array ,region_len_h ,region_len_w, interpolation):
  height, width = img_array.shape
  region_to_eq_transform=calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)
  equalized_img=np.zeros_like(img_array)
  num_regions=len(region_to_eq_transform)
  upper_left=np.zeros((num_regions,2))
  centers=np.zeros((num_regions,2))
  #if the division is not perfect find the number of vertical and horizontal regions
  num_w_regions=width // region_len_w
  num_h_regions=height // region_len_h
  if (height-num_h_regions*region_len_h) !=0:
    num_h_regions+=1
  if (width-num_w_regions*region_len_w) !=0:
    num_w_regions+=1
  #Find the coordinates of upper left corner
  for i, key in enumerate(region_to_eq_transform.keys()):
    upper_left[i] = key
    #Calculate the center coordinates
    centers[i,0] = region_len_h / 2 + upper_left[i,0]
    centers[i,1] = region_len_w / 2 + upper_left[i,1]
    if centers[i,0]>=height:
      centers[i,0]==(height-centers[i-num_w_regions,0])/2
    if centers[i,1]>=width:
      centers[i,1]==(width-centers[i-1,1])/2
  
  #Apply in all the points
  for i in range(num_regions):
    #Get the equalization transformation for the current region
    eq_transform = region_to_eq_transform[tuple(upper_left[i])]
    #For the last row
    if i >= num_regions - num_w_regions: 
      #for the last corner
      if i == num_regions - 1:
        equalized_img[int(upper_left[i, 0]):height, int(upper_left[i, 1]):width] = eq_transform[img_array[int(upper_left[i, 0]):height, int(upper_left[i, 1]):width]]
      else:
        equalized_img[int(upper_left[i, 0]):height, int(upper_left[i, 1]):int(upper_left[i+1, 1])] = eq_transform[img_array[int(upper_left[i, 0]):height, int(upper_left[i, 1]):int(upper_left[i+1, 1])]]
    else:
      #for the last column
      if (i - (num_w_regions - 1)) % num_w_regions == 0:
        equalized_img[int(upper_left[i, 0]):int(upper_left[i+num_w_regions,0]), int(upper_left[i, 1]):width] = eq_transform[img_array[int(upper_left[i, 0]):int(upper_left[i+num_w_regions,0]), int(upper_left[i, 1]):width]]
      else:
        equalized_img[int(upper_left[i, 0]):int(upper_left[i+num_w_regions,0]), int(upper_left[i, 1]):int(upper_left[i+1, 1])] = eq_transform[img_array[int(upper_left[i, 0]):int(upper_left[i+num_w_regions,0]), int(upper_left[i, 1]):int(upper_left[i+1, 1])]]
 
  #Inner points
  #with interpolation
  if interpolation == 1:
    for hp in range(int(centers[0,0]), int(centers[num_regions-1,0])):
      for wp in range(int(centers[0,1]), int(centers[num_regions-1,1])): 
        #for each pixel we find the region it belongs (its center)
        h_center=num_w_regions*(hp//region_len_h)
        w_center=wp//region_len_w
        #index of centers[] for the found center
        i=h_center+w_center
        #for better readability
        idx=num_w_regions
  
        #for the up right corner
        if (hp/region_len_h-hp//region_len_h)<=0.5 and (wp/region_len_w-w_center)>0.5: 
          Tminusminus = region_to_eq_transform[tuple(upper_left[i-idx])]
          T_plus = region_to_eq_transform[tuple(upper_left[i-idx+1])]
          Tplus_= region_to_eq_transform[tuple(upper_left[i])]
          Tplusplus = region_to_eq_transform[tuple(upper_left[i+1])]
          #pixel_transformation(img_pixel,hp,wp,w_,h_,w_plus,h_plus,T__,T_plus,Tplus_,Tplusplus)
          equalized_img[hp,wp]=pixel_transformation(img_array[hp,wp],hp,wp,centers[i,1],centers[i-idx,0],centers[i+1,1],centers[i,0],Tminusminus,T_plus,Tplus_,Tplusplus)
        #for the up left corner
        elif (hp/region_len_h-hp//region_len_h)<=0.5 and (wp/region_len_w-w_center)<=0.5:
          Tminusminus = region_to_eq_transform[tuple(upper_left[i-idx-1])]
          T_plus = region_to_eq_transform[tuple(upper_left[i-idx])]
          Tplus_= region_to_eq_transform[tuple(upper_left[i-1])]
          Tplusplus = region_to_eq_transform[tuple(upper_left[i])]
          #pixel_transformation(img_pixel,hp,wp,w_,h_,w_plus,h_plus,T__,T_plus,Tplus_,Tplusplus)
          equalized_img[hp,wp]=pixel_transformation(img_array[hp,wp],hp,wp,centers[i-1,1],centers[i-idx,0],centers[i,1],centers[i,0],Tminusminus,T_plus,Tplus_,Tplusplus)
        #for the down right corner
        elif (hp/region_len_h-hp//region_len_h)>0.5 and (wp/region_len_w-w_center)>0.5:
          Tminusminus = region_to_eq_transform[tuple(upper_left[i])]
          T_plus = region_to_eq_transform[tuple(upper_left[i+1])]
          Tplus_= region_to_eq_transform[tuple(upper_left[i+idx])]
          Tplusplus = region_to_eq_transform[tuple(upper_left[idx+i+1])]
          #pixel_transformation(img_pixel,hp,wp,w_,h_,w_plus,h_plus,T__,T_plus,Tplus_,Tplusplus)
          equalized_img[hp,wp]=pixel_transformation(img_array[hp,wp],hp,wp,centers[i,1],centers[i,0],centers[i+1,1],centers[i+idx,0],Tminusminus,T_plus,Tplus_,Tplusplus)
        #for the down left corner
        else:
          Tminusminus = region_to_eq_transform[tuple(upper_left[i-1])]
          T_plus = region_to_eq_transform[tuple(upper_left[i])]
          Tplus_= region_to_eq_transform[tuple(upper_left[i+idx-1])]
          Tplusplus = region_to_eq_transform[tuple(upper_left[idx+i])]
          #pixel_transformation(img_pixel,hp,wp,w_,h_,w_plus,h_plus,T__,T_plus,Tplus_,Tplusplus)
          equalized_img[hp,wp]=pixel_transformation(img_array[hp,wp],hp,wp,centers[i-1,1],centers[i,0],centers[i,1],centers[i+idx,0],Tminusminus,T_plus,Tplus_,Tplusplus)
  
  return equalized_img

#Compute new pixel value using interpolation based on neighboring regions
def pixel_transformation(img_pixel,hp,wp,w_,h_,w_plus,h_plus,T__,T_plus,Tplus_,Tplusplus):
  a=(wp-w_)/(w_plus-w_)
  b=(hp-h_)/(h_plus-h_)
  equalized_pixel=(1-a)*(1-b)*T__[img_pixel]+(1-a)*b*Tplus_[img_pixel]+a*(1-b)*T_plus[img_pixel]+a*b*Tplusplus[img_pixel]
  return equalized_pixel