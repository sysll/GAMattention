import numpy as np
import matplotlib.pyplot as plt

# Define example values for MAE, RMSE, MSE, and R-squared for six regression algorithms
# algorithms = ['A: RandomForest', 'B: SCAD-Regression', 'C: Lstm', 'D: GradientBoost', 'E: seq2seq_attention', 'F: seq2seq+SA']

algorithms = ['A','B','C','D','E','F','G','H']
#gam-na
MAE = [0.8591015610379884, 0.9035438491065639, 0.8523229956626892, 0.8728418174802516,0.8705653,0.83570,0.83390, 0.82953]
RMSE = [1.1376566959169114, 1.1500650335156286, 1.1260343790054321, 1.1595702411284006,1.11051415,1.07449,1.07285, 1.06658]
MSE = [1.294262757764584, 1.3226495813153039, 1.2679535150527954, 1.344603144110577,1.233241686,1.15454,1.15100, 1.13760]
R = [0.0888061963542331, 0.10347363709474018, 0.12440804560747419, 0.06508726829306065,0.138873166,0.23535,0.19061, 0.24092]


# Set the positions of the bars on the x-axis
r = np.arange(len(algorithms))

# Create subplots for MAE, RMSE, MSE, and R-squared
fig, axs = plt.subplots(2, 2, figsize=(9, 9))

# Define lighter colors
colors = [(255/255, 208/255, 111/255),(231/255, 98/255, 84/255),
          (55/255, 103/255, 149/255), (114/255, 188/255, 213/255),
          (135/255, 187/255, 164/255),(138/255,176/255,125/255),
          (147/255, 148/255, 231/255), (115/255, 115/230, 255/255)]



# Plot MAE
axs[0,0].bar(r, MAE, color=colors, width=0.8)
axs[0,0].set_ylabel('MAE')
axs[0,0].set_xlabel('(a) MAE')
axs[0,0].set_xticks(r)
axs[0,0].set_xticklabels(algorithms, rotation=0, ha='right')
axs[0,0].set_ylim(0.826, 0.91)
# axs[0,0].set_title('(a) MAE')

# Plot RMSE
axs[0,1].bar(r, RMSE, color=colors, width=0.8)
axs[0,1].set_ylabel('RMSE')
axs[0,1].set_xlabel('(b) RMSE')
axs[0,1].set_xticks(r)
axs[0,1].set_xticklabels(algorithms, rotation=0, ha='right')
axs[0,1].set_ylim(1.06, 1.163)
# axs[0,1].set_title('(b) RMSE')

# Plot MSE
axs[1,0].bar(r, MSE, color=colors, width=0.8)
axs[1,0].set_ylabel('MSE')
axs[1,0].set_xlabel('(c) MSE')
axs[1,0].set_xticks(r)
axs[1,0].set_xticklabels(algorithms, rotation=0, ha='right')
axs[1,0].set_ylim(1.13, 1.35)
# axs[1,0].set_title('(c) MSE')

# Plot R-squared
axs[1,1].bar(r, R, color=colors, width=0.8)
axs[1,1].set_ylabel('R')
axs[1,1].set_xlabel('(d) R')
axs[1,1].set_xticks(r)
axs[1,1].set_xticklabels(algorithms, rotation=0, ha='right')
axs[1,1].set_ylim(0.05, 0.25)
# axs[1,1].set_title('(d) R')
# Add legend outside the plot
# handles = [plt.Rectangle((0, 0), 0, 0, color=color) for color in colors]
# labels = algorithms
# plt.legend(handles, labels, loc='upper left')

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()






