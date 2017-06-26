# Real-time Collision Culling of a Million Spheres on GPU

## Algotithm

### Sweep and Prune
-------------------
![SaP](/figure/SaP.jpg)
- CPU version
	- Sorting
	- Sweep window
	- Collision detection
- Worst case: O(n^{2})

### Parallel Sweep and Prune
-------------------
![GSaP](/figure/GSAP.JPG)
- GPU version
	- Sorting
	- *Sweep an object per thread*
	- Collision detection
- Worse case: O(n<sup>2</sup>/p)
- Problem
	- Many false positive collided pair
	- Workload imbalance

## Optimization 1 - by Paper

### Choosing the Sweep Direction
-------------------
![BestDir](/figure/BestDir.JPG)
- Variance for x, y, z axis
	- thrust library: transform reduction
- Max variance axis
	- Sweep
- Other axis
	- Prune
- Complexity: O(n)
- Solved problem
	- Reduce false positive collided pair

### Workload Balancing
-------------------
![WorkloadBalance](/figure/WorkloadBalance.JPG)
- Workload estimation
	- Binary search for M<sub>i</sub>
- Workload balance
	- Get required # of threads per object
	- thrust library: inclusive scan (prefix sum)
- Parallel SaP
	- Binary search for object index
	- SaP
- Solved problem
	- Make workload balance

### Workspace Subdivision
-------------------
![WorkspaceSub](/figure/WorkspaceSub.JPG)
- Use prined axis
- Divide to groups
	- Groups for not crossing the line, C<sub>i</sub>
	- Groups for crossing the line, C<sub>ri</sub>
- Duplicate the objects which touch the C<sub>ri</sub>

-------------------
- Choose group per object
- For group C<sub>i</sub>
	- Mark
	- thrust library: inclusive scan (prefix sum)
	- Put into workspace
- For group C<sub>ri</sub>
	- Extension mark
	- Then same as group C<sub>i</sub>
- Problem
	- Twice *global memory access*

## Optimation 2 - by Us

- Less Workspace Duplication
-------------------

- Less Global Memory Access
-------------------

## Expetiment Result

- Verify Correctness
-------------------
![correctness](/figure/correctness.gif)

- Real-time Collision Detection of 0.25M Spheres
-------------------
|Without Opt.|With Opt.|
|:--------:|:------:|
|![million](/figure/million.gif)|![million](/figure/million-fast.gif)|

- Performance of detection lantency
-------------------
![performancd](/figure/performance.png)


- Performance of realtime FPS
-------------------
|Number of spheres| 100K | 200K | 300K |
|:---------------:|:---:|:------:|:------:|
| SAP    | 64 FPS | 20 FPS | 10 FPS |
| SAP_WB | 103 FPS | 29 FPS | 13 FPS |
| SAP_WS | 117 FPS | 48 FPS | 23 FPS |

## Reference
[Real-time Collision Culling of a Million Bodies on Graphics Processing Units](http://graphics.ewha.ac.kr/gSaP/)
