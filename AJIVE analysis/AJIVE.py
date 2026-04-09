#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os  
from joblib import dump  
import matplotlib.pyplot as plt 
from jive.AJIVE import AJIVE 
from explore.BlockBlock import BlockBlock  
from explore.Base import Union 
from cbcs_joint.viz_utils import savefig, mpl_noaxis  
from cbcs_joint.Paths import Paths 


# In[ ]:


import pandas as pd 
import numpy as np 


# In[ ]:


expr = pd.read_csv("/data/filtered_gene_expr_top5000_var_t_sorted_tibble.txt",sep="\t",header=0,index_col=0) 
images = pd.read_csv("/data/image_feature_t_sorted_tibble.txt",sep="\t",header=0,index_col=0) 


# In[ ]:


expr_log = np.log2(expr + 1) 
images_log = np.log2(images + 0.01) 


# In[ ]:


from sklearn.preprocessing import StandardScaler  
X_img_scaled = pd.DataFrame( 
    StandardScaler().fit_transform(images_log), 
    index=images.index, 
    columns=images.columns 
) 
X_expr_scaled = pd.DataFrame( 
    StandardScaler().fit_transform(expr_log), 
    index=expr.index, 
    columns=expr.columns 
) 


# In[ ]:


init_signal_ranks = {'images': 10, 'genes': 500}


# In[ ]:


ajive = AJIVE(init_signal_ranks=init_signal_ranks, n_wedin_samples=3000, n_randdir_samples=3000, n_jobs=-1, store_full=False)  
ajive = ajive.fit({'images': X_img_scaled, 'genes': X_expr_scaled}) 


# In[ ]:


dump(ajive, os.path.join("/data", 'fit_ajive')) 


# In[ ]:


plt.figure(figsize=[10, 10])


# In[ ]:


ajive.plot_joint_diagnostic()
savefig(os.path.join("/data/AJIVE_integration", 'ajive_diagnostic.png')) 


# In[ ]:


#gene Joint and Individual
gene_joint_loading = ajive.blocks['genes'].joint.loadings()
gene_joint_loading.to_csv('/data/AJIVE_gene_joint_loading.csv',index=True)
gene_individual_loading = ajive.blocks['genes'].individual.loadings() 
gene_individual_loading.to_csv('/data/AJIVE_gene_individual_loading.csv',index=True)


# In[ ]:


gene_joint_score = ajive.blocks['genes'].joint.scores() 
gene_joint_score.to_csv('/data/AJIVE_gene_joint_score.csv',index=True) 
gene_individual_score = ajive.blocks['genes'].individual.scores() 
gene_individual_score.to_csv('/data/AJIVE_gene_individual_score.csv',index=True) 


# In[ ]:


#image Joint and Individual
image_joint_loading = ajive.blocks['images'].joint.loadings() 
image_joint_loading.to_csv('/data/AJIVE_image_joint_loading.csv',index=True)
image_individual_loading = ajive.blocks['images'].individual.loadings() 
image_individual_loading.to_csv('/data/AJIVE_image_individual_loading.csv',index=True)


# In[ ]:


image_joint_score = ajive.blocks['images'].joint.scores() 
image_joint_score.to_csv('/data/AJIVE_image_joint_score.csv',index=True)
image_individual_score = ajive.blocks['images'].individual.scores() 
image_individual_score.to_csv('/data/AJIVE_image_individual_score.csv',index=True) 

