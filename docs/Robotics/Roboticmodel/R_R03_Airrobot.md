---
typora-copy-images-to: ..\picture
---

# 空中机器人

## 基本知识

定义: 拥有自制和自主能力

### 分类

固定翼
- 机体结构、航电系统、动力系统、起降系统、地面控制站

旋翼

扑翼

气囊

伞翼

### 应用

- 无人机派送
- 消费级无人机



### 导航

定义：将运载体从起始点引导到目的地的技术或方法称为导航(Navigation)

导航三大问题

- 我在那里
- 我要去哪里
- 我如何去

定位的本质是**状态估计问题** 



## 飞行器的动态模型

### 坐标系与姿态转换

常见飞行器可以抽象为一个刚体。描述任意时刻的空间运动需要六个自由度：三个质心运动和三个角运动



**地球中心坐标系**(ECEF)

ECEF坐标系与地球固联，位置在地球质心

**WGS-84坐标系**

当前GPS所用的坐标系（地球是一个椭圆）



!!! note "为什么不用GPS的海拔信息"
	GPS的海拔高度不准<br>



**NED坐标系**





### PID技术

观看视频：[PID讲解](https://www.bilibili.com/video/BV1et4y1i7Gm/?spm_id_from=333.337.search-card.all.click&vd_source=ace17a48ec1787387c4c8d582e6808cb)

$K_P$越大，控制系统的响应速度越大，但产生的震荡也越严重

![image-20240926213044384](../../picture/image-20240926213044384.png)

$K_d$ 减缓震荡，但不能过大，否则会过冲

$\frac{d(e)}{dt} = v$，微分算法可以对无人机的速度发生响应

$K_i$,减小与目标位置之间的误差







## 轨迹规划

### 深度优先（DFS）

队列：先进后出



### 广度优先（BFS）

是**没有权重**的Dijkstra算法

队列：先进先出

[广度优先搜索算法（BFS）](https://blog.csdn.net/aliyonghang/article/details/128724989)

图示：

A为起点，G为终点。一开始我们在起点A上，此时并不知道G在哪里。

![img](../../picture/1bb5820bb9100c7123086a080f2c779b.png)

将可以从A直达的三个顶点B、C、D设为下一步的候补顶点。

![img](../../picture/23eb3167fb9f227f792632cb8ba67ca8.png)

从候补顶点中选出一个顶点。优先选择最早成为候补的那个顶点，如果多个顶点同时成为候补，那么可以随意选择其中一个。

![img](../../picture/01a4f4108a871dc164e6517676120f73.png)

假设选择B点为先进去的，此时的队列[B C D]变为[C D]

![img](../../picture/5e702bd0963309ed1593d3a4d33fe044.png)

移动到选中的顶点B上。此时我们在B上， 所以B变为红色，同时将已经搜索过的顶点变为橙色。

![img](../../picture/7a5220e28bb6c6aa89a6e8829853e97e.png)



将可以从B直达的两个顶点E和F设为候补顶点并加入队列，变为[C D E F]

![img](../../picture/8882353fe5fb1470b8d51b0bf9cb3aad.png)

此时，最早成为候补顶点的是C和D，我们选择了左边的顶点C。

![img](../../picture/a47bcccfe9f7a6bd862819d6d9c39b99.png)

移动到选中的顶点C上。



![img](../../picture/e21810fb145d3f37e82ae7fba1df8fe7.png)

将可以从C直达的顶点H设为候补顶点，并将C移除队列，此时的队列[D E F H]。 

![img](../../picture/25371cf2d3d548018ea64f70f1249edb.png)

重复上述操作直到到达终点，或者所有的顶点都被遍历为止。 



### Dijkstra算法

[B站视频](https://www.bilibili.com/video/BV1zz4y1m7Nq/?spm_id_from=333.337.search-card.all.click&vd_source=ace17a48ec1787387c4c8d582e6808cb)

- 每次从未标记的节点中选取距离出发点最近的节点，标记，收录到最优路径集合中
- 计算刚加入节点A的邻近节点B的距离，若A的距离+A到B的距离小于B的距离，则更新B的距离

例题如下

首先用表格记录该店距离前面点的初始距离，起始的值都为无穷大，前面点都为空

![image-20241026163510021](../../picture/image-20241026163510021.png)

首先节点0到0，距离为0，找到距离最小的值，为0，加入已搜索节点，并标注前面点为0

![image-20241026163726960](../../picture/image-20241026163726960.png)

更新节点0附件的节点1和7的距离

![image-20241026163803202](../../picture/image-20241026163803202.png)

在未被加入已搜索节点的里面找到距离出发点最小的点，是1点，将其加入已搜索点，并跟新1节点周围的点的距离。

![image-20241026163948143](../../picture/image-20241026163948143.png)

依次类推，直到所有点都被搜索完成

**优点**: 

- **准确性**: 总是能找到最短路径。
- **简单性**: 实现相对简单。

**缺点**: 

- **效率较低**: 算法需要遍历图中的大多数节点，可能导致较高的计算成本。
- **实时性差**: 在动态环境中可能不适用，因为它不能快速适应环境的变化





### A*算法

与Dijkstra的区别：是加了猜测H函数的Dijkstra算法

- Dijkstra: G(n)
- A*:F(n)=G(n)+H(n)

[A*讲解](https://blog.csdn.net/Zhouzi_heng/article/details/115035298)

####  搜索区域(The Search Area)

以题目进行解释，我们假设某人要从 A 点移动到 B 点，但是这两点之间被一堵墙隔开。如图 1 ，绿色是 A ，红色是 B ，中间蓝色是墙。

![img](../../picture/0b78c760ac45ec2f8e30745201fecad7.jpeg)

格子的状态分为可走 (walkalbe) 和不可走 (unwalkable)

#### 开始搜索(Starting the Search)

- 从起点 A 开始，并把它就加入到一个由方格组成的 open list( 开放列表 ) 中。 Open list 里的格子是路径可能会是沿途经过的，也有可能不经过。基本上 open list 是一个**待检查**的方格列表。
- 查看与起点 A 相邻的方格 ( 忽略其中unwalk的方格 ) ，把其中可走的 (walkable) 或可到达的 (reachable) 方格也加入到 open list 中。把起点 A 设置为这些方格的父亲 (parent node 或 parent square) 。
- 把 A 从 open list 中移除，加入到 close list( 封闭列表 ) 中， close list 中的每个方格都是现在不需要再关注的。

如下图所示，深绿色的方格为起点，它的外框是亮蓝色，表示该方格被加入到了 close list 。与它相邻的黑色方格是需要被检查的，他们的外框是亮绿色。每个黑方格都有一个灰色的指针指向他们的父节点，这里是起点 A 。

![image002.jpg](../../picture/c232b6a651f1a54116fa38e7b6de142a.jpeg)

#### 路径排序(Path Sorting)

对每个节点，在计算时同时考虑两项**代价**指标：**当前节点与起始点的距离**，以及**当前节点与目标点的距离**：F = G + H

- **欧式距离**：G = 从起点 A 移动到指定方格的移动代价，沿着到达该方格而生成的路径。
  - $G = \sqrt{(x_1 - x_2)^2 +(y_1 - y_2)^2}$ 
- **曼哈顿距离**：H = 从指定的方格移动到终点 B 的估算成本。
  - $H = |x_1 - x_2| + |y_1 - y_2|$
  - 注意，H函数的选取要满足**估算成本小于实际成本**

计算起始点相邻方格的F、G、H的值，分别记录在左上角，左下角和右下角

![image003.jpg](../../picture/908e62d12ad781ced1f767c945c1feda.jpeg)

#### 继续搜索(Continuing the Search)

为了继续搜索，我们从 open list 中选择 F 值最小的 ( 方格 ) 节点，然后对所选择的方格作如下操作：

- 把它从 open list 里取出，放到 close list 中。
- 检查所有与它相邻的方格，忽略其中在 close list 中或是不可走 (unwalkable) 的方格 ( 比如墙，水，或是其他非法地形 ) ，如果方格不在open lsit 中，则把它们加入到 open list 中。把我们选定的方格设置为这些新加入的方格的父亲。
  - 然后计算新加入的方格相对于当前处理方格的F、G、H值(注意G为累加值)
  - 选取其中F值最小的作为下一个待处理的方格。
  - 然后继续上面的操作。
- 如果某个相邻的**所有方格**均已经在 open list 中，则检查所有方格所在的这条路径是否更优，也就是说经由当前方格 ( 我们选中的方格 ) 到达那个方格是否具有更小的 G 值。
  - 如果没有，不做任何操作。
  - 相反，如果 G 值更小，则把那个方格的父亲设为当前方格 ( 我们选中的方格 ) ，然后重新计算那个方格的 F 值和 G 值。



![image004.jpg](../../picture/10b76d7dfb9f62c0621f83409d472e23.jpeg)

1. 对于上图，在我们最初的 9 个方格中，还有 8 个在 open list 中，起点被放入了 close list 中。在这些方格中，起点右边的格子的 **F 值 40 最小**，因此我们选择这个方格作为下一个要处理的方格。它的外框用蓝线打亮。

2. 首先，我们把它从 open list 移到 close list 中  。然后我们检查与它相邻的方格。它右边的方格是墙壁，我们忽略。它左边的方格是起点，在 close list 中，我们也忽略。其他 4 个**相邻的方格均在 open list 中**，因此我们需要检查经由这个方格到达那里的路径是否更好，使用 G 值来判定。让我们看看上面的方格。它现在的 G 值为 14 。如果我们经由当前方格到达那里， G 值将会为 20。显然 20 比 14 大，因此这不是最优的路径。

3. 当把 4 个已经在 open list 中的相邻方格都检查后，**没有发现经由当前方格的更好路径**，因此我们不做任何改变。现在我们已经检查了当前方格的所有相邻的方格，并也对他们作了处理，是时候选择下一个待处理的方格了。

4. 因此再次遍历我们的 open list ，现在它只有 7 个方格了，我们需要选择 F 值最小的那个。有趣的是，这次有两个方格的 F 值都 54 ，选哪个呢？没什么关系。**从速度上考虑，选择最后加入 open list 的方格更快**。

5. 我们选择起点右下方的方格，如下图所示

   ![image005.jpg](../../picture/19efd36ffafcb18f4a5af00078b07849.jpeg)

6. 只有三个方格可以选取，当前处理方格左边的方格，以及新加入的两个方格中。我们检查经由当前方格到达那里是否具有更小的 G 值。没有。因此我们准备从 open list 中选择下一个待处理的方格。

7. 以此类推，找到最短路径

![image007.jpg](../../picture/a2a101733a0277b4346f75b0c3efd89f.jpeg)

### JPS算法






















## 实验课程

```
树莓派密码:
pi
```

!!! attention "主线任务——设置位置控制器，实现无人机悬停"

### 线性控制器

![image-20240926024025533](../../picture/image-20240926024025533.png)

采用串级PID控制，内层控制姿态，外层控制位置。

#### 模型构建

**线性化**

- 平衡悬停态： $(\phi_0 \sim 0, \theta_0 \sim 0, u_{1,0} \sim mg)$

**牛顿方程**：

$$
m\ddot{p} = \begin{bmatrix} 0 \\ 0 \\ -mg \end{bmatrix} + R \begin{bmatrix} 0 \\ 0 \\ F_1 + F_2 + F_3 + F_4 \end{bmatrix} 
$$

$$
其中R = \begin{bmatrix} c\psi c\theta - s\phi s\psi s\theta & -c\theta s\psi & c\psi s\theta + c\theta s\phi s\psi \\ c\theta s\psi + c\psi s\phi s\theta & c\phi c\psi & s\psi s\theta - c\theta c\phi s\phi \\ -c\theta s\theta & s\phi & c\theta c\phi \end{bmatrix}
$$

$$
在小角度近似的情况下，得到\begin{cases}
\ddot{p}_1 = \ddot{x} = g(\theta c\psi + \phi s\psi )\\
\ddot{p}_2 = \ddot{y} = g(\theta s\psi - \phi c\psi) \\
\ddot{p}_3 = \ddot{z} = -g + \frac{u_1}{m}
\end{cases}
$$

**欧拉角微分：**

$$
\begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix} = \begin{bmatrix} c\theta & 0 & -c\phi s\theta \\ 0 & 1 & s\phi \\ s\theta & 0 & c\phi s\theta \end{bmatrix} \begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix}
$$

线性化后

$$
\begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix} = \begin{bmatrix} u_2 \\ u_3 \\ u_4 \end{bmatrix} 
$$

**欧拉方程**：

$$
I\cdot \begin{bmatrix} \dot{\omega}_x \\ \dot{\omega}_y \\ \dot{\omega}_z \end{bmatrix} + \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix} \times I \cdot \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix} = \begin{bmatrix} u_{2x} \\ u_{2y} \\ u_{2z} \end{bmatrix} - \begin{bmatrix} l(F_2 - F_4) \\ l(F_3 - F_1) \\ M_1 - M_2 + M_3 - M_4 \end{bmatrix}
$$

**PID控制**

**位置控制**

$$
\ddot{p}_{i,c} = \ddot{p}_i^{des} + K_{d,i}(\dot{p}_i^{des} - \dot{p}_i) + K_{p,i}(p_i^{des} - p_i) \\
$$

- 由上述方程可以求出$p_{i,c}$
- 再带入牛顿方程的解的到预期的$\phi_c$、$\theta_c$和$u_1$
  - $\phi_c = \frac{1}{g}(\ddot{p}_{1,c}s\psi - \ddot{p}_{2,c}c\psi)$
  - $\theta_c = \frac{1}{g}(\ddot{p}_{1,c}c\psi + \ddot{p}_{2,c}s\psi)$
  - $u_1 = m(g + \ddot{p}_{3,c})$


**姿态控制**

PID控制

$$
\begin{bmatrix} \ddot{\phi}_c \\ \ddot{\theta}_c \\ \ddot{\psi}_c \end{bmatrix} = \begin{bmatrix} K_{p,\phi}(\phi_c - \phi) + K_{d,\phi}(\dot{\phi}_c - \dot{\phi}) \\ K_{p,\theta}(\theta_c - \theta) + K_{d,\theta}(\dot{\theta}_c - \dot{\theta}) \\ K_{p,\psi}(\psi_c - \psi) + K_{d,\psi}(\dot{\psi}_c - \dot{\psi}) 
\end{bmatrix} 
$$


模型

$$
u_2 = I \begin{bmatrix} \ddot{\phi}_c \\ \ddot{\theta}_c \\ \ddot{\psi}_c \end{bmatrix} + \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix} \times I \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}
$$

#### 代码逻辑

##### 位置控制器

!!! note "输入输出"
	输入：$p_des$、$\dot{p}_des$、$\ddot{p}_des$以及当前的yaw角<br>
	输出：当前的位置$P_c$,预期的roll、Pitch角<br>
	返回：当前的yaw与预期的roll、Pitch计算的新的四元素给姿态控制器<br>



```
imu.q              -> Euler
Euler.yaw P_des    -> P_c roll_d pitch_c
yaw pitch_c roll_c -> u.q(输出)
```

