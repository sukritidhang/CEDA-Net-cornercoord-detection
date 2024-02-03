import math
import sklearn.metrics
import pandas as pd
#import datashader as ds
#from datashader.mpl_ext import dsshow
import matplotlib.pyplot as plt
#import seaborn as sns

#from sklearn.metrics import mean_absolute_error as mae
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as R2
from scipy.stats import pearsonr

#ground truth four corners co-ordinate
df = pd.read_csv(r"E:/python_programming/phd/billboard_edgedetect/results/all_4cornercordinates.csv")


y_gt = df['average_x_gt'].to_frame() #average_x_gt, average_y_gt
print(y_gt)

#predicted four corners co-ordinate
#y_predicted_adsegcedanet = df['average_x_adceda'].to_frame() #average_x_adceda, average_y_adceda
#y_predicted_unet = df['average_y_unetceda'].to_frame() #average_x_unetceda, average_y_unetceda
#y_predicted_sobel = df['average_y_sobel'].to_frame() #average_x_sobel, average_y_sobel
y_predicted_canny = df['average_x_canny'].to_frame() #average_x_canny, average_y_canny

print(y_predicted_canny)

#mean squared error

mse = sklearn.metrics.mean_squared_error(y_gt, y_predicted_canny)#y_predicted_unet

#root mean square error

rmse = math.sqrt(mse)

#mean absolute error

mae = sklearn.metrics.mean_absolute_error(y_gt, y_predicted_canny)#y_predicted_unet, y_predicted_linknet

mape = sklearn.metrics.mean_absolute_percentage_error(y_gt, y_predicted_canny)#y_predicted_unet, y_predicted_linknet

#print(rmse)

corr_coef = R2(y_gt, y_predicted_canny)
print(corr_coef)

y_gt = y_gt.values.flatten()
y_predicted_canny = y_predicted_canny.values.flatten() 

r = pearsonr(y_gt, y_predicted_canny)
#r= df.corr(method ='pearson')
print(r)


ax2 = df.plot.scatter(x = 'average_x_gt',
                      y = 'average_x_canny',
                      c= df['average_x_canny'],#.map(colors),
                      colormap = 'viridis')
#plt.colorbar()
ax2.plot()
plt.xlabel('x co-ordinate of ground truth')
plt.ylabel('x co-ordinate of canny')
plt.savefig("./results/rmse_plot/x_gt_canny_rmse.jpg")
plt.show()



with open("./results/canny/correl_canny_result.txt", 'a')as f:
    
    f.write('Metrics for gt_canny x co-ordinate \n') 

    '''
    f.write('MSE: %f \n' %mse)
    f.write('RMSE: %f \n' %rmse)
    f.write('MAE: %f \n' %mae)
    f.write('MPE: %f \n' %mape)
    f.write('Correlation Coefficient: %f \n' %corr_coef)
    '''
    f.write('Correlation Coefficient: %f \n' %corr_coef)
    f.write('Correlation Coefficient (statistics): %f \n' %r[0])
    f.write('pvalue: %f \n' %r[1])
    f.write('------------------------- \n')
    f.write('------------------------- \n')
     
   
    
f.close()

