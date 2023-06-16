import numpy as np
import matplotlib.pyplot as plt 

fs = 18

def plot_text_list(x, y, s):
    for i in range(len(x)):
        plt.text(x[i] + 0.05, y[i] - 0.1, s[i], fontsize=12)

#plt.xlim([3.5, 8])

fpt_x = np.array([139.97, 147.74, 311.98])
#fpt_x = np.log(fpt_x)
fpt_y = np.array([66.5, 70.0, 72.5])
plt.plot(fpt_x, fpt_y, 'o-', label='FastPointTransformer', markersize=21)
grid_size = ['10cm', '5cm', '2cm']
plt.text(fpt_x[-1] - 0.2, fpt_y[-1] + 0.3, 'FastPointTransformer', weight='bold', fontsize=fs)
plot_text_list(fpt_x, fpt_y, grid_size)

mink_x = np.array([52.9, 73.5, 115.6])
#mink_x = np.log(mink_x)
mink_y = np.array([60.7, 67.0, 72.1])
plt.plot(mink_x, mink_y, 'o-', label='MinkowskiNet', markersize=21)
# grid_size[2] = 'MinkowskiNet-2cm'
plot_text_list(mink_x, mink_y, grid_size)
plt.text(mink_x[0] - 0.2, mink_y[0] - 0.6, 'MinkowskiNet', weight='bold', fontsize=fs)

stf_x = np.array([1690])
#stf_x = np.log(stf_x)
stf_y = np.array([74.3])
plt.plot(stf_x, stf_y, 'o-', label='Stratified Transformer', markersize=10)
# plot_text_list(stf_x, stf_y, ['Stratified Transformer'])
plt.text(stf_x[-1] - 1000, stf_y[-1] + 0.3, 'Stratified Transformer', weight='bold', fontsize=fs)


pcf_x = np.array([29.4, 41.9, 51.0, 59.6, 145.5])
#pcf_x = np.log(pcf_x)
pcf_y = np.array([70.6, 71.4, 73.3, 73.7, 74.5])
plt.plot(pcf_x, pcf_y, 'o-', label='PointConvFormer', markersize=5)
plot_text_list(pcf_x, pcf_y, ['10cm lite', '10cm', '5cm lite', '5cm', '2cm'])
plt.text(pcf_x[-1] - 0.2, pcf_y[-1] + 0.3, 'PointConvFormer', weight='bold', fontsize=fs)


pc_x = np.array([23.4, 36.7, 88.8])
#pc_x = np.log(pc_x)
pc_y = np.array([62.6, 68.5, 70.3])
plt.plot(pc_x, pc_y, 'o-', label='PointConv', markersize=7)
plot_text_list(pc_x, pc_y, grid_size)
plt.text(pc_x[-1], pc_y[-1] + 0.3, 'PointConv', weight='bold', fontsize=fs)


# numbers here for the lite version
#vipc_x = np.array([37.8])
#vipc_x = np.log(vipc_x)
#vipc_y = np.array([71.2])
#plt.plot(vipc_x, vipc_y, 'o-', label='VI-PointConv', markersize=7)
# plot_text_list(vipc_x, vipc_y, ['VI-PointConv'])
#plt.text(vipc_x[-1], vipc_y[-1] + 0.3, 'VI-PointConv', weight='bold', fontsize=fs)



plt.xlabel('running time(log-scale of ms)', fontsize=24)
plt.ylabel('mIoU(%)', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
ax = plt.gca()
ax.semilogx(subs=[10,20,40,80,160,320,640,1280,2560])
# plt.legend()
plt.show()


