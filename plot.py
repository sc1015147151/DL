import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
#%matplotlib inline
sns.set()
file_path1='sample_processed.csv'
train_sample = pd.read_csv(file_path1)#
x_src = train_sample.iloc[:, 0:6]
#x_tar = train_sample.iloc[:, 43:49]
y_src=train_sample.iloc[:, 42:43]
#y_tar=train_sample.iloc[:, 85:86]
pca=PCA(n_components=3)
pca.fit(x_src)
x_PCA=pca.transform(x_src)
x_PCA.shape
df_x_PCA = pd.DataFrame({'fakeR':x_PCA[:,0],'fakeG':x_PCA[:,1],'fakeB':x_PCA[:,2]})
df_x_PCA_with_y=pd.concat([df_x_PCA,y_src],axis=1).sample(frac=0.02,replace=True, random_state=6, axis=0)
data=df_x_PCA_with_y
data1=data[data.S_Y==1]
data2=data[data.S_Y==2]
data3=data[data.S_Y==3]
data4=data[data.S_Y==4]
data5=data[data.S_Y==5]
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = np.random.randint(0, 255, size=[40, 40, 40])
plt.figure(figsize=(25,25))
x, y, z = data[0], data[1], data[2]
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
s=5
ax.scatter( data1['fakeG'], data1['fakeB'],data1['fakeR'], c='y',s=s)  # 绘制数据点
ax.scatter( data2['fakeG'], data2['fakeB'],data2['fakeR'], c='b',s=s) 
ax.scatter( data4['fakeG'], data4['fakeB'],data4['fakeR'], c='r',s=s)
ax.scatter( data3['fakeG'], data3['fakeB'],data3['fakeR'], c='g',s=s)
ax.scatter( data5['fakeG'], data5['fakeB'],data5['fakeR'], c='k',s=s)

# ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.view_init(0,90)
plt.show()