# Normal Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#<-----------------------------------------Added imports for 3D graphing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#<-----------------------------------------Added anotation imports
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
#matplotlib inline

#<-----------------------------------------
def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)


# PCA & Standard Scaler
from sklearn.preprocessing import StandardScaler # before PCA you usually need to standardize the data
from sklearn.decomposition import PCA # the PCA library

import warnings
warnings.filterwarnings("ignore")
print('imports done')


# Read in the file
klm = pd.read_csv("klm.csv")

# The date is non numerical (str = object), so first in needs to be changed to a numeric format
klm['Introduction_date'] = pd.to_datetime(klm['Introduction_date'], format = '%d/%m/%Y') # first convert from object to date

# Function to convert from date to numeric
def to_integer(dt_time):
    return 10000*pd.DatetimeIndex(dt_time).year + 100*pd.DatetimeIndex(dt_time).month + pd.DatetimeIndex(dt_time).day

klm['Introduction_date'] = to_integer(klm['Introduction_date'])

# Head of data
klm.head(3)



# STANDARDIZATION

# Separating out the features
x = klm.loc[:, klm.columns != 'Aircraft'].values
# Separating out the target
y = klm['Aircraft'].values

# Standardizing the features
x = StandardScaler().fit_transform(x)


# PCA 

# Creating the pca object
pca = PCA(n_components = 3)

# Fitting it to our X data (the features)
principalComponents = pca.fit_transform(x)

# Creating new dataframe
principalDf = pd.DataFrame(data = principalComponents, 
                           columns = ['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, klm['Aircraft']], axis = 1)

finalDf



# Variability mantained:

print("Variance explained by the 3 PCAs:", pca.explained_variance_ratio_)




plt.matshow(pca.components_, cmap='coolwarm')
plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(klm.columns[1:])),klm.columns[1:],rotation=65,ha='left')
plt.tight_layout()
plt.show()

#For some reason the #D import did not work so I imported something else

# 3D plot - all 3 PCs

fig = plt.figure(figsize = (9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs = finalDf['principal component 1'], ys = finalDf['principal component 2'], zs = finalDf['principal component 3'],
           c='skyblue', s=140)
ax.view_init(30, 185)

for i in range(10):
    ax.text(finalDf['principal component 1'][i],finalDf['principal component 2'][i],finalDf['principal component 3'][i],klm['Aircraft'][i]) 

ax.set_xlabel('Plane Architecture')
ax.set_ylabel('Introduction Year')
ax.set_zlabel('Safety')

#To make the gif
'''
for angle in range(0, 360):
  ax.view_init(30, angle)
  plt.draw()
  plt.pause(.001)
  plt.savefig(str("image_"+ str(angle)))
'''
plt.show()

print("done")
