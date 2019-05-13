import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import csv

save_dir = '/Users/janghyun/Downloads/Summary_neighbor/imageNet'
index_range = 20

# model_list = [folder for folder in os.listdir(save_dir) if not os.path.isfile(os.path.join(save_dir, folder))]
model_list = ['lls_neigh16_block8', 'lls_neigh0_block8']

final_value = {}
for model in model_list:
    final_value[model] = []

for i in range(index_range):
    plt.figure()

    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title('query-loss')
    ax1.set_ylabel("objective")

    ax2 = plt.subplot(2, 1, 2)
    # ax2.set_xlabel('step-loss')
    ax2.set_xlabel('loss_diff')
    ax2.set_ylabel("objective")

    x_max = 0
    y_min = 0
    for model in model_list:
        summary_dir = os.path.join(save_dir, model)

        with open(os.path.join(summary_dir, 'history_{}.p'.format(i)), 'rb') as f:
            history = pickle.load(f)

        final_value[model].append(history['loss'][-1])

        if len(history['num_queries']) ==1:
            marker = 'o'
        else:
            marker = '-'

        ax1.plot(history['num_queries'], history['loss'], marker, label=model)

        # history_prox = {'step':[], 'loss':[]}
        # prev_step = 0
        # for j in range(len(history['step'])):
        #     cur_step = history['step'][j]
        #     if prev_step != cur_step:
        #         history_prox['step'].append(cur_step)
        #         history_prox['loss'].append(history['loss'][j-1])
        #         prev_step = cur_step
        #
        #     history_prox['step'].append(cur_step)
        #     history_prox['loss'].append(history['loss'][j])
        #
        # ax2.plot(history_prox['step'], history_prox['loss'], marker, label=model)
        ax2.plot(history['loss_diff'], marker, label=model)

        x_max = max(history['num_queries']+[x_max])
        y_min = min(history['loss']+[y_min])

    ax1.set_xlim([0, x_max+500])
    ax1.set_ylim([y_min*1.1-0.001, 0])
    # ax2.set_ylim([y_min*1.1-0.001, 0])
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, 'summary_{}.png'.format(i)))

"""
fig = plt.figure()

ax1 = plt.subplot(2, 1, 1)
ax1.set_title('f(.)-f(x_lls)')
ax1.set_xlabel("index")
ax1.set_ylabel("objective_diff")
sorted_index = np.argsort(np.array(final_value['lls_paral']) - np.array(final_value['lls']))
for model in model_list:
    ax1.plot((np.array(final_value[model]) - np.array(final_value['lls']))[sorted_index], label=model)

ax2 = plt.subplot(2, 1, 2)
ax2.set_title('f(.)')
ax1.set_xlabel("index")
ax2.set_ylabel("objective")
sorted_index = np.argsort(final_value['lls'])
for model in model_list:
    ax2.plot(np.array(final_value[model])[sorted_index], label=model)

fig.tight_layout(rect=[0,0,0.7,1])

ax1.legend(loc=(1.04,0))
ax2.legend(loc=(1.04,0))
ax1.grid()
ax2.grid()
plt.savefig(os.path.join(save_dir, 'sorted.png'))

for model in model_list:
    if model == 'lls':
        continue

    fig = plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_title('{}-lls'.format(model))
    ax1.set_xlabel("index")
    ax1.set_ylabel("objective_diff")
    ax1.set_ylim([-0.5, 0.5])

    sorted_index = np.argsort(np.array(final_value[model]) - np.array(final_value['lls']))
    ax1.plot((np.array(final_value[model]) - np.array(final_value['lls']))[sorted_index])

    ax1.grid()
    plt.savefig(os.path.join(save_dir, 'sorted_{}.png'.format(model)))


# write csv file
csv_file = open(os.path.join(save_dir, 'cifar_admm_lls_diff.csv'), mode='w')
csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

sorted_index = np.argsort(np.array(final_value['admm']) - np.array(final_value['lls']))
diff = (np.array(final_value['admm']) - np.array(final_value['lls']))[sorted_index]

csv_writer.writerow(['index', 'diff'])
for i, val in enumerate(diff):
    csv_writer.writerow([float(i), val])
    print([float(i), val])

csv_file.close()


csv_file = open(os.path.join(save_dir, 'cifar_paral_lls_diff.csv'), mode='w')
csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

sorted_index = np.argsort(np.array(final_value['lls_paral']) - np.array(final_value['lls']))
diff = (np.array(final_value['lls_paral']) - np.array(final_value['lls']))[sorted_index]

csv_writer.writerow(['index', 'diff'])
for i, val in enumerate(diff):
    csv_writer.writerow([float(i), val])

csv_file.close()
"""










