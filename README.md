#Real-time Collision Culling of a Million Spheres on GPU

## Verify Correctness
-------------------
https://github.com/ChangNingNing/Course_GPGPU_FinalProject/figure/correctness.gif

## Real-time Collision Detection of 0.3M Spheres
-------------------
https://github.com/ChangNingNing/Course_GPGPU_FinalProject/figure/million.gif

## Parallel Sweep and Prune
-------------------
https://github.com/ChangNingNing/Course_GPGPU_FinalProject/figure/GSAP.jpg

## Optimization
-------------------
- Workload Balancing
-------------------
https://github.com/ChangNingNing/Course_GPGPU_FinalProject/figure/WorkloadBalance.jpg
- Choosing the Sweep Direction
-------------------
https://github.com/ChangNingNing/Course_GPGPU_FinalProject/figure/BestDir.jpg
- Workspace Subdivision
-------------------
https://github.com/ChangNingNing/Course_GPGPU_FinalProject/figure/WorkspaceSub.jpg

## Performance of Sweep and Prune With and Without Optimization
-------------------
| Number of spheres | 100K | 0.3M | 1M |
| ----------------- | ---- | ---- | -- |
| Without Opt.      | ?ms  | ?ms  | ?ms| 
| With Opt          | ?ms  | ?ms  | ?ms| 

## Reference
-------------------
[Real-time Collision Culling of a Million Bodies on Graphics Processing Units](http://graphics.ewha.ac.kr/gSaP/)
