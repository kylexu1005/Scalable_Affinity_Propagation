# Scalable_Affinity_Propagation

This algorithm is used for clustering. It is transferred from the original Affinity Propagation by Frey(2007). The original algorithm is O(N^2). Now I reduce the complexity to O(N) by using nearest neighbors. The work I did is better than the related previous work because I also take exemplars into consideration. 

This algorithm contains two stages, the first stage is affinity propagation with nearest neighbors and exemplars in APNBE.m. The second stage is spectral clustering. The whole algorithm is in SAP.m. 
