from brainrender.atlas_specific import GeneExpressionAPI
import matplotlib.pyplot as plt

gene = "Gpr161"
geapi = GeneExpressionAPI()
expids = geapi.get_gene_experiments(gene)
data = geapi.get_gene_data(gene, expids[1])

grid_size = 4
plt.figure(figsize=(10, 10))
for ii in range(grid_size**2):
	plt.subplot(grid_size, grid_size, ii + 1)
	plt.imshow(data[ii + 10, :, :])
plt.show()

import ipdb; ipdb.set_trace()