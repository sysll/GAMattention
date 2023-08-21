import numpy as np
import matplotlib.pyplot as plt

# Define example values for MAE, RMSE, MSE, and R-squared for six regression algorithms
# algorithms = ['A: RandomForest', 'B: SCAD-Regression', 'C: Lstm', 'D: GradientBoost', 'E: seq2seq_attention', 'F: seq2seq+SA']

algorithms = ['A','B','C','D','E','F','G','H']

MAE = [0.8793118189960677, 0.8362180092776776, 0.8387797474861145, 0.888212635467835,0.8758203595826558, 0.82409,0.82316, 0.81936]
RMSE = [1.0972590490991134, 1.079678972771948, 1.0712438821792603, 1.1062593734321218,1.0981310637679096,1.02515,1.02333, 1.01659]
MSE = [1.2039774208298906, 1.165706684245889, 1.1475633382797241, 1.2238098013064305,1.2058918332120407, 1.05093,1.04720,1.03346]
R = [0.22448105268029087, 0.24784020687974292, 0.2790097167014518, 0.22139518175254191,0.2973869,  0.32651,0.32883, 0.34071]


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
# axs[0,0].set_title('(a) MAE')
axs[0,0].set_xticklabels(algorithms, rotation=0, ha='right')
axs[0,0].set_ylim(0.81, 0.89)

# Plot RMSE
axs[0,1].bar(r, RMSE, color=colors, width=0.8)
axs[0,1].set_xlabel('(b) RMSE')
axs[0,1].set_ylabel('RMSE')
axs[0,1].set_xticks(r)
# axs[0,1].set_title('(b) RMSE')
axs[0,1].set_xticklabels(algorithms, rotation=0, ha='right')
axs[0,1].set_ylim(1.01, 1.11)

# Plot MSE
axs[1,0].bar(r, MSE, color=colors, width=0.8)
axs[1,0].set_xlabel('(c) MSE')
axs[1,0].set_ylabel('MSE')
axs[1,0].set_xticks(r)
# axs[1,0].set_title('(c) MSE')
axs[1,0].set_xticklabels(algorithms, rotation=0, ha='right')
axs[1,0].set_ylim(1.03, 1.23)

# Plot R-squared
axs[1,1].bar(r, R, color=colors, width=0.8)
axs[1,1].set_ylabel('R')
axs[1,1].set_xlabel('(d) R')
axs[1,1].set_xticks(r)
axs[1,1].set_xticklabels(algorithms, rotation=0, ha='right')
# axs[1,1].set_title('(d) R')
axs[1,1].set_ylim(0.2, 0.35)
# Add legend outside the plot
handles = [plt.Rectangle((0, 0), 0, 0, color=color) for color in colors]
labels = algorithms
# plt.legend(handles, labels, loc='upper left')

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()