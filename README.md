#Real-time Collision Culling of a Million Spheres on GPU

## Verify Correctness
(/figure/correctness.gif)

## Real-time Collision Detection of 0.3M Spheres
(/figure/million.gif)

## Parallel Sweep and Prune
(/figure/GSAP.jpg)

## Optimization
- Workload Balancing
-------------------
(/figure/WorkloadBalance.jpg)
- Choosing the Sweep Direction
-------------------
(/figure/BestDir.jpg)
- Workspace Subdivision
-------------------
(/figure/WorkspaceSub.jpg)

## Performance of Sweep and Prune With and Without Optimization
| Number of spheres | 100K | 0.3M | 1M |
| ----------------- | ---- | ---- | -- |
| Without Opt.      | ?ms  | ?ms  | ?ms| 
| With Opt          | ?ms  | ?ms  | ?ms| 

## Reference
[Real-time Collision Culling of a Million Bodies on Graphics Processing Units](http://graphics.ewha.ac.kr/gSaP/)
