# Hyperbolic-Hierarchical-Image-Retrieval

This project builds on the Hierarchical Average Precision Training for Pertinent Image Retrieval method, as described in the ECCV 2022 paper by Elias Ramzi, Nicolas Audebert, Nicolas Thome, Clément Rambour, and Xavier Bitot [1]. Additionally, data was used from the work of Hyun Oh Song, Yu Xiang, Stefanie Jegelka, and Silvio Savarese, as described in their CVPR 2016 paper [2]. During the project, the exponential mapping of Euclidean vectors onto a hyperbolic Poincaré ball was implemented using the Hyperbolic Learning Python Library (HypLL) introduced by van Spengler et al. [3].

**References:**

1. Ramzi, E., Audebert, N., Thome, N., Rambour, C., & Bitot, X. (2022, October). Hierarchical average precision training for pertinent image retrieval. In *European Conference on Computer Vision* (pp. 250-266). Cham: Springer Nature Switzerland.

2. Song, H. O., Xiang, Y., Jegelka, S., & Savarese, S. (2016). Deep metric learning via lifted structured feature embedding. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 4004-4012).

3. van Spengler, M., Wirth, P., & Mettes, P. (2023, October). HypLL: The hyperbolic learning library. In Proceedings of the 31st ACM International Conference on Multimedia (pp. 9676-9679).


This research explores hyperbolic space in deep learning to enhance hierarchical image retrieval. Adapting the HAPPIER framework from Euclidean to hyperbolic space using the Poincaré ball model, it integrates exponential mapping, feature clipping, and prototype scaling. The model incorporates hierarchical and equidistant prototypes to capture complex relationships. Experimental results on the Stanford Online Products dataset show promising improvements in retrieval performance using hierarchical metrics like H-AP, ASI, and NDCG, alongside non-hierarchical metrics like AP and R@1.
