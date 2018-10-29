# Clustering
We use HDBSCAN algorithm to cluster on features learned using an autoencoder.
Details of the clustering algorithm can be found here:
https://hdbscan.readthedocs.io/en/latest/index.html

The files in the directory are:
1. batch_clustering_AEFeatures.py: This file is used to cluster features learned
   from various Coarse Grain Martini Simulation runs of 3-lipid systems. It follows the
   steps below:
   - Parse command line arguments to get basic parameters for clustering
   - Read the list of directories that have data saved after each training run
   - Read the features from each epoch of individual training runs
   - Compute sparse distance matrix
   - Compute clustering using HDBSCAN
   - Plot and save clustering output
   - Save clustering output in a csv file format that can be read by talass framework

   The script relies on data from different sources:
   - Features saved after training:
     /p/gscratchr/karande1/Benchmarks/Pilot2/P2B1/molecular_AE_results/ (from 09062018)
   - Data that stores the leaflet each molecule belongs to:
     '/p/gscratchr/brainusr/datasets/cancer/pilot2/3k_leaflets/'
   - Data that stores the neighborhood information of each molecule:
     /p/gscratchr/brainusr/datasets/cancer/pilot2/'

2. batch_clustering_HandFeatures.py: This file is used to cluster features designed
   by experts with domain knowledge in Coarse Grain Martini simulations of 3-lilid
   systems. It works similarly to the above script accept that it uses hand engineered
   features of different simulation frames stored in:
   /p/gscratchr/karande1/Benchmarks/clustering/

3. clusteringHelper.py: This files contains several helper functions used by both
   the clustering scripts

4. convert_hdbscan_2_talass.cpp: This cpp file is used the convert the clustering
   output to a format that can be visualizes talass framework. It cannot be run
   as a standalone script from this directory and needs to be placed in the talass
   directory structure in:
   ~/talass/TopologyFileParser/utilities:
