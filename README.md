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

## Experiment Result
![performancd](/figure/performance.png)

|Number of spheres| 100K | 200K | 300K |
|:---------------:|:---:|:------:|:------:|
| SAP    | 66 FPS | 22 FPS | 11 FPS |
| SAP_WB | 119 FPS | 35 FPS | 16 FPS |
| SAP_WS | 120 FPS | 36 FPS | 18 FPS |

## Reference
[Real-time Collision Culling of a Million Bodies on Graphics Processing Units](http://graphics.ewha.ac.kr/gSaP/)
