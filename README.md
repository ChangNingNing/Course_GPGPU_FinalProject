# Real-time Collision Culling of a Million Spheres on GPU

## Verify Correctness
![correctness](/figure/correctness.gif)

## Real-time Collision Detection of 0.25M Spheres
|Without Opt.|With Opt.|
|:--------:|:------:|
|![million](/figure/million.gif)|![million](/figure/million-fast.gif)|

## Parallel Sweep and Prune
![GSaP](/figure/GSAP.JPG)

## Optimization
- Workload Balancing
-------------------
![WorkloadBalance](/figure/WorkloadBalance.JPG)
- Choosing the Sweep Direction
-------------------
![BestDir](/figure/BestDir.JPG)
- Workspace Subdivision
-------------------
![WorkspaceSub](/figure/WorkspaceSub.JPG)

## Performance of Sweep and Prune With and Without Optimization
| Number of spheres | 100K | 0.3M | 1M |
|:-----------------:|:----:|:----:|:--:|
| Without Opt.      | ?ms  | ?ms  | ?ms| 
| With Opt          | ?ms  | ?ms  | ?ms| 

## Reference
[Real-time Collision Culling of a Million Bodies on Graphics Processing Units](http://graphics.ewha.ac.kr/gSaP/)
