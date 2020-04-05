import scipy
import numpy as np
import seaborn as sns
import fiona
import rasterio
from shapely import geometry
from rasterio.mask import mask



def extractor(frame,points,size=7,normalize=True,labeling=False,verbose=False):
  '''
  The extractor fuction takes in three paramters:
  
    1. frame - The image to be subdevided. Resterio type file
    2. points - fiona collection file with coordinates of points
    3. size - size of the window
    4. data normalization - normolize the data boolean
    5. require labels if the data containes labels. Boolean
    
  Return:
  
    1. List of bathces with the give size
    2. List of coordinates
    3. List of labels
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
  Example:
      frame = rasterio.open("pp_1_sat_modified1.tif")
      points = fiona.open("alltreepoints1.shp", "r")
      collection = extractor(frame,points,size=7)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
  '''
  img = frame.read().T.astype('uint32')
  if normalize:
    img = img/np.max(img)

  assert img.shape[1]>img.shape[2], "channles are in the right place!"+str(img.shape)
  map_tree_speices = {'е':0, 'б':1, 'п':2, "El'":0, "Bereza":1, "Pichta":2, "Sosna":3}
  img_point,locations,labels = [],[],[]

  for p in range(len(points)):

        try:  
              point = np.array(frame.index(*points[p]['geometry']['coordinates']))
              box = img[(point[1]-size):point[1]+size,
                        (point[0]-size):point[0]+size,:].copy()
              if labeling:
                  labels.append(map_tree_speices[ points[p]['properties']['specie'] ]) 
              img_point.append(box)
              locations.append(point)
              del box
        except:
             if verbose:
                print("None coordinate/close to edge")
  return img_point, locations, labels



