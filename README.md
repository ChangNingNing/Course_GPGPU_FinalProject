# Real-time Collision Culling of a Million Spheres on GPU
## Introduction
-------------------
This is a final project from a course, GPGPU.  We implement a paper of collision detection based on GPU device.
You can see the original website of the paper at Reference.
In this project, we implement the parallel SaP algorithm, and some optimizations which were proposed in the paper.
Moreover, we propose two optimizations focus on <strong>global memory access</strong> by our own.
After all, we can detect collision between the number of 250K spheres in the <strong>real-time</strong>.
And we have 3.2x improvement in the number of 1M spheres.

## Algotithm

### Sweep and Prune
-------------------
![SaP](/figure/SaP.jpg)
- CPU version
	- Sorting by axis
	- Sweep window
	- Collision detection
- Worst case: O(n<sup>2</sup>)

### Parallel Sweep and Prune
-------------------
![GSaP](/figure/GSAP.JPG)
- GPU version
	- Sorting by axis
	- <strong>Sweep an object per thread</strong>
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
- Make each thread having balanced workload.

-------------------
<img src="figure/workload.png" width="500"/>

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
<img src="figure/workspace.png" width="500"/>

- Choose group per object
- For group C<sub>i</sub>
	- Mark
	- thrust library: inclusive scan (prefix sum)
	- Put into workspace
- For group C<sub>ri</sub>
	- Extension mark
	- Then same as group C<sub>i</sub>
- Problem
	- Twice <strong>global memory access</strong>

## Optimation 2 - by Us

### Less Workspace Duplication
-------------------
<img src="figure/lessworkspace.png" width="500"/>

- Choose group per object
- For group C<sub>ri</sub>
	- Extension mark
	- thrust library: inclusive scan (prefix sum)
	- Put into workspace
	- Workload balance and Parallel SaP
- For group C<sub>i</sub>
	- thrust library: stable sort by group index
	- Workload balance and Parallel SaP
- Solved problem
	- <strong>25% less global memory access</strong>

### Less Global Memory Access
-------------------
<img src="figure/lessglobal.png" width="500"/>

- For group C<sub>ri</sub>
	- Duplicate information to workspace pointer.
	- Write bake to the object if collision occur.
- Solved problem
	- <strong>Reduce global memory access</strong>

## Expetiment Result

### Verify Correctness
-------------------
![correctness](/figure/correctness.gif)
- All have the same z position
- Green means no collision.
- Red means collision occuring.

### Real-time Collision Detection of 0.25M Spheres
-------------------
<img src="figure/environment.png" width="800"/>

- Useless comparation because of hardward improvement.

-------------------
- Comparation between our optimazations.

|Without Opt.|With Opt.|
|:--------:|:------:|
|![million](/figure/million.gif)|![million](/figure/million-fast.gif)|

### Performance of detection lantency
-------------------
<img src="figure/performance.png" width="1000"/>


### Performance of realtime FPS
-------------------
|Number of spheres| 100K | 200K | 300K |
|:---------------:|:---:|:------:|:------:|
| SAP    | 64 FPS | 20 FPS | 10 FPS |
| SAP_WB | 103 FPS | 29 FPS | 13 FPS |
| SAP_WS | 117 FPS | 48 FPS | 23 FPS |

## Reference
[Real-time Collision Culling of a Million Bodies on Graphics Processing Units](http://graphics.ewha.ac.kr/gSaP/)
