# name_list = ['ICFMDS', 'BLEDS', 'BHPDS']
# num_list = [0.766, 0.791, 0.792]
# num_list1 = [0.494, 0.667, 0.657]
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# from collections import namedtuple
#
# from matplotlib.backends.backend_pdf import PdfPages
#
# # pdf = PdfPages('vsVul_bad.pdf')
# n_groups = 3
#
# means_men = (0.106, 0.791, 0.977)
# # std_men = (2, 3, 4, 1, 2)
#
# means_women = (0.428, 0.700, 0.933)
# # std_women = (3, 5, 2, 3, 3)
#
# fig, ax = plt.subplots()
#
# index = np.arange(n_groups)
# bar_width = 0.35
#
# opacity = 0.4
# error_config = {'ecolor': '0.3'}
#
# rects1 = ax.bar(index, means_men, bar_width,
#                 alpha=opacity, color='b',
#                  error_kw=error_config,
#                 label='VGDetector')
#
# rects2 = ax.bar(index + bar_width, means_women, bar_width,
#                 alpha=opacity, color='r',
#                  error_kw=error_config,
#                 label='Token-based')
#
# ax.set_xlabel('Evaluation Measures')
# ax.set_ylabel('Rates')
# ax.set_title('BLEDS')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('FNR', 'F1-measure', 'ROC-AUC'))
# ax.legend()
#
# fig.tight_layout()
# plt.show()
#
# # pdf.savefig(fig)
# plt.close()
# # pdf.close()
import networkx as nx
from networkx.drawing.nx_pydot import read_dot

dotPath = u"/Users/chengxiao/Desktop/164/pag_final.dot"
g = read_dot(dotPath)

for n, nbrs in g.adjacency():
    for nbr, edict in nbrs.items():
        print()
print()

