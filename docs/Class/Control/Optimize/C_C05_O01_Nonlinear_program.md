
# 非线性规划

## 基本知识

### 标准模型

目标函数

$$
\min \quad f(x)
$$

约束条件

$$
\text{s.t.} \quad
\begin{cases}
g_j(x) \leq 0, & j = 1,2,\dots,m \\
h_i(x) = 0, & i = 1,2,\dots,p \\
x \in \mathbb{R}^n
\end{cases}
$$

$R^n$代表n维实空间



!!! question "为什么约束不用严格不等式？"
    使得优化问题有最优解，可行域是闭集，可达



**标准模型的集合形式**

$$
\inf_{x \in \chi} f(x)
$$

- 可行域$\chi$：定义域$D$中所有可行（满足所有约束）点的集合
  
$$
\chi = \{x \in D | g_j(x) \leq 0, j = 1, 2, ..., m; h_i(x) = 0, i = 1, 2, ..., p\}
$$

- 定义域$D$：
  
$$
D = \text{dom}f \cap \left( \bigcap_{j=1,m} \text{dom}g_j \right) \cap \left( \bigcap_{i=1,p} \text{dom}h_i \right)
$$



!!! note "函数定义域是优化问题的隐含约束！"



- 函数定义域：
  
$$
\text{dom} f = \{x \in R^n | -\infty < f(x) < +\infty\}
$$



!!! note "函数值有界的区域"



### 相关概念


**上下确界**

- 上界（upper bound）：对于每一个$z \in S$，有$z \leq a$，则$a$称为S的上界。


!!! note 
    所有上界构成一个集合，也可能是空集（例如S=R）。
    上界可能在S中（可达），也可能超出S外（不可达）。
    可达即为函数z最大值

- 上确界（supremum）：上确界是上界集合中最小的一个元素，即最小上界，记为sup(S)。

!!! note "若sup(S)∈S，称上确界可达。"

- 最大值（maximum）：可达的上确界

- 下界（lower bound）：$b \leq z, \forall z \in S$
- 下确界（infimum）：最大下界inf(S)
- 最小值（minimum）：可达的下确界；下确界可达：inf(S) ∈ S。
- 下确界和上确界关系：$\inf(S) = -\sup(-S)$

- 函数$f(x)$的光滑性：$f(x)$无穷阶连续可微。

- 函数$n$阶连续可微：$f(x)$的$n$阶偏导数存在并连续，记为$f(x) \in C^n$。

  

!!! note "实际优化时，算法可根据需要仅关注1阶或2阶偏导数。"


**开集与闭集**


- 开集：对于 $\forall x \in S$，$\exists \varepsilon > 0$，满足 $B(x, \varepsilon) \subseteq S$，则S是开集。球域表示如下
  
$$
B(x, \varepsilon) = \{z \in R^n | \|z - x\| < \varepsilon\}
$$

- 闭集：若S的补集 $R^n \setminus S$ 是开集，则S是闭集。
  

例1：全空间 $R^n$ 和空集 $\emptyset$ 既是开集也闭集。（满足开集定义，且补集是开集）

例2：[1, 2) 既不是开集也不是闭集。（原集和补集都不满足开集定义）

  


性质：S是闭集 $\Leftrightarrow$ S中所有收敛序列的极限点均在S中。

**有界闭集（紧集）**

- 有界集合：$\exists r > 0$，$x \in S$，满足 $S \subseteq B(x, r)$。（在一个球体内）

- 无界闭集
例3：$R^n$ 是无界闭集。



**梯度的几何性质**

- $\nabla f(x)$ 为目标函数$f(x)$等值面在$x$的法向量
- $\nabla f(x)$是目标函数值$f(x)$在$x$点增长最快的方向

!!! note "证明" 
    $f(x + \lambda p) = f(x) + \lambda \nabla f(x)^T p + O(\lambda)$
    - 方向导数（沿$p$方向的导数）：

    $$
    D_p(x) = \lim_{\lambda \to 0} \frac{f(x + \lambda p) - f(x)}{\lambda} = \nabla f(x)^T p = \|\nabla f(x)\|_2 \cos(\theta)
    $$
    
    其中$\Delta x = \lambda p$，$\|p\|_2 = 1$。
    可以得到如下结论：
    - $p$取$f(x)$等值面切方向时，$\nabla f(x)^T p = 0$
    - $p$取$\nabla f(x)$时，方向导数最大。（$\theta = 0$）







### 解的类型

- 最优值 $p*^ \triangleq \inf_{x \in \chi} f(x)$

- 最优解 如果$x^* \in \chi$ 满足$f(x^*) = p^*$，则称$x^*$为最优解。

- 最优解集 $X_{opt} = \{x^* \in \chi | f(x^*) = p^*\}$



**最优值的类型**

 最优值有界                $p^* \in (-\infty, +\infty)$

- 最优值可达 有最优解 存在$x^* \in \chi$ 满足$f(x^*) = p^*$
- 最优值不可达 无最优解 $X_{opt} = \emptyset$


最优值无界

!!! note "无界解不是最优解"

- $p^* = -\infty$                    无界解 $\Leftrightarrow$ 最优值无下界 $\Rightarrow X_{opt} = \emptyset$

  例：$f(x) = \log x \Rightarrow p^* = -\infty$, x=0不是最优解

- $p^* = +\infty$                    无可行解 $\chi = \emptyset \Rightarrow X_{opt} = \emptyset$

反证：只要可行域不是空集，那最优值就不会是$-\infty$

- 广义实值函数：函数值可取正负无穷的函数。$\tilde{f}(x) = 
  \begin{cases} 
  f(x) & x \in \chi \\
  \infty & x \notin \chi 
  \end{cases}$
  现代优化中常引入广义实值函数增强分析的方便性和严格性。

### 最优解的存在性

**定理1---Weierstrass极值定理**

若目标函数$f$连续，且可行域$\chi$为非空紧集，则优化问题存在最优解。

- 紧集：有界闭集。

推论：恰当定义的有约束光滑优化问题有最优解。



!!! note "Weierstrass极值定理是否可分析无约束优化问题？"
		不可以,无约束问题一定是无界的。
		

**定理2**

若无约束优化问题（$\chi = R^n$）的目标函数$f$是连续强制函数，则优化问题存在最小解。

- **强制(coercive)函数**：可行域内趋向边界点的函数值趋于无穷大。
  
  $\{x_k\} \in \text{int dom } f \quad \lim_{k \to \infty} x_k = z \in \text{bd dom } f \implies \lim_{k \to \infty} f(x_k) = +\infty$
  
- **集合内部**：$\text{int } S = \{x \in R^n | B(x, \varepsilon) \subseteq S \text{ for some } \varepsilon > 0\}$
  
  当S可以是开集，此时S = int S
  
- **集合闭包**：$cl\ S = \{z \in R^n | z = \lim_{k \to \infty} x_k, x_k \in S, \forall k \geq 1\}$ ；         $cl\ S = R^n \setminus \text{int } (R^n \setminus S)$
  
  闭包包含了所有收敛序列的极限点，必为闭集。
  
- **集合边界**：$bd\ S = cl\ S \setminus \text{int } S$



!!! note "强制函数判别条件"
    - $f$是强制函数 $\Leftrightarrow$ 所有的$\alpha$下水平集为紧集。
    - $\alpha$下水平集：$C_{\alpha} = \{x \in \text{dom } f | f(x) \leq \alpha\} \quad \alpha \in R$
    ![image-20251119200427999](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202511192004098.png)





## 最优性条件

### 无约束极值问题

!!! note 无约束极小值问题的最优性条件
    - 一阶必要条件：（局部极值、鞍点、拐点）
      $\nabla f(x^*) = 0$
    - 二阶必要条件：（局部极小值）
      $\nabla f(x^*) = 0$ 且 $\nabla^2 f(x^*) \ge 0$
    - 二阶充分条件：（局部极小值）
      $\nabla f(x^*) = 0$ 且 $\exists \varepsilon > 0, \forall x \in B(x^*, \varepsilon), \nabla^2 f(x^*) \ge 0$
      $\nabla f(x^*) = 0$ 且 $\nabla^2 f(x^*) > 0$ （严格局部极小值）



!!! question "无约束问题规划的目标是找哪一种极小点？"
    找全局极小点而非局部极小点



**一阶极值条件（必要条件）**

驻点(stationary point)条件：$\nabla f(x^*) = 0$ 

梯度 $\nabla f(x) \triangleq \begin{bmatrix} \frac{\partial f(x)}{\partial x_1} \\ \frac{\partial f(x)}{\partial x_2} \\ \vdots \\ \frac{\partial f(x)}{\partial x_n} \end{bmatrix} = [Df(x)]^T \quad Df(x) \triangleq \frac{df(x)}{dx}$



!!! question "为什么$\nabla f(x^*) = 0$ 是判断条件" 
    因为如果偏导中有一个分量不是0，必然能够找到一个比当前值更小的解。



**二阶极小值条件（必要条件）**

!!! note "Taylor公式的二阶展开"
    $$
    f(x + \Delta x) = f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T \nabla^2 f(x) \Delta x + O(\|\Delta x\|^2)
    $$

    其中，$\nabla^2 f(x)$是函数$f(x)$的Hessian矩阵，定义为：
    
    $$
    \nabla^2 f(x) = \begin{bmatrix}
    \frac{\partial^2 f(x)}{\partial x_1^2} & \frac{\partial^2 f(x)}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_1 \partial x_n} \\
    \frac{\partial^2 f(x)}{\partial x_2 \partial x_1} & \frac{\partial^2 f(x)}{\partial x_2^2} & \cdots & \frac{\partial^2 f(x)}{\partial x_2 \partial x_n} \\
    \vdots & \vdots & \cdots & \vdots \\
    \frac{\partial^2 f(x)}{\partial x_n \partial x_1} & \frac{\partial^2 f(x)}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_n^2}
    \end{bmatrix}
    $$



函数$f(x + \Delta x)$的二阶泰勒展开式为：

$$
f(x + \Delta x) = f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T \nabla^2 f(x) \Delta x + O(\|\Delta x\|^2)
$$

在驻点$x^*$处

- $\nabla f(x^*) = 0$

- $\nabla^2 f(x^*) \ge 0$

  

!!! attention "局部极小值 $\Rightarrow$ $\nabla^2 f(x^*) \ge 0$（半正定）;局部极大值 $\Rightarrow$ $\nabla^2 f(x^*) \le 0$（半负定）"



!!! question "已有一阶必要条件，为什么还要讨论二阶必要条件？"
    所确定的范围比一阶必要条件小，排除极大值的情况



**二阶极小值条件（充分条件）**



在驻点$x^*$处

- $\nabla f(x^*) = 0$
- $\nabla^2 f(x^*) > 0$

!!! attention "严格局部极小值 $\Rightarrow$ $\nabla^2 f(x^*) > 0$（正定）;严格局部极大值 $\Rightarrow$ $\nabla^2 f(x^*) < 0$（负定）;**“ =”需检验更高阶导数项**"





### 有约束极值问题

#### 一阶条件



##### 等式约束——Lagrange函数法

定义Lagrange函数为：

$$
\min L(x, v) = f(x) + v^T h(x) = f(x) + \sum_{i=1}^{p} v_i h_i(x)
$$

其中，$v = [v_1 \ v_2 \ \cdots \ v_p]^T$

Lagrange函数驻点条件：

1. 对$x$的偏导数为0：
   
$$
\left. \frac{\partial L(x, v)}{\partial x} \right|_{x^*} = 0 \quad \implies \quad \nabla f(x^*) + \sum_{i=1}^{p} v_i \nabla h_i(x^*) = 0
$$

2. 对$v$的偏导数为0：

$$
\left. \frac{\partial L(x, v)}{\partial v} \right|_{v^*} = 0 \quad \implies \quad h_i(x^*) = 0 \quad i = 1, 2, \ldots, p
$$



##### 不等式约束

!!! note "思考过程"
    先考虑全为不等式的约束情况，思考能否转为等式约束问题；
    即：不存在同时满足以下两个条件的方向
    - 满足所有约束的可行方向
    - 目标函数的下降方向



**可行方向**

可行点$x^{(0)}$的可行方向$p$是$x^{(0)}$沿$p$方向移动无限小步后仍在可行域$\chi$内的方向，数学上可表述为：

存在$\lambda_0 > 0$，对于$\forall \lambda \in (0, \lambda_0]$，有：

$$
x^{(0)} + \lambda p \in \chi, \quad \chi = \{x | g_j(x) \leq 0, j = 1, 2, \ldots, m\}
$$

即

$$
g_j(x^{(0)} + \lambda p) \leq 0 \quad j = 1, 2, \ldots, m
$$

判断条件：$\nabla g(x^{(0)})^T p < 0$



!!! note "相关介绍"
    方向是线性化概念！
    $\lambda_0$可趋向于0，故是否是可行方向，取决于方向导数！
    几何含义：与所有起作用约束梯度的夹角大于90°的方向。
    缺点：可行方向的P是直线，对于等式约束来说(比如形成的圆)，存在曲线路径，但无法用直线方向来描述



**下降方向**

定义：令目标函数值$f(x^{(0)})$下降的方向，满足$\exists \lambda_0 > 0$，对于$\forall \lambda \in (0, \lambda_0]$，有$f(x^{(0)} + \lambda p) < f(x^{(0)})$

!!! question "为什么是$\exists \lambda_0 > 0$，对于$\forall \lambda \in (0, \lambda_0]$"
    如果是$\exists \lambda_0 > 0$，有$f(x^{(0)} + \lambda_0 p) < f(x^{(0)})$,可能会存在$\lambda < \lambda_0$,但是$f(x^{(0)} + \lambda_0 p) > f(x^{(0)})$的情况

判别条件：$\nabla f(x^{(0)})^T p < 0$

几何含义：与目标函数梯度方向成钝角


!!! question "可行、下降方向的判别条件是充分条件还是必要条件？"
    都是充分条件非必要



因此，**局部极小值**的判断条件如下

不存在同时满足下面两个不等式

- $\nabla f(x^*)^T p < 0$

- $\nabla g_j(x^*)^T p < 0 \quad j \in J(x^*) \quad J$为起作用约束集合

几何含义：不存在$p$与上述梯度方向均成钝角$\Leftrightarrow$ 上述梯度方向不可能分布在任意超平面的同一侧（Gordan 引理）



=== "Gordan引理"
    设 $\alpha_j$ 为一组已知向量，不存在向量 $p$，使得  

    $$
    \alpha_j^T p < 0,\quad j = 1,2,\dots,m
    $$  
    
    同时成立的充要条件：  
    (正线性相关)存在不全为零的非负实数 $\mu_j \ge 0$，使  
    
    $$
    \sum_{j=1}^m \mu_j \alpha_j = 0
    $$


=== "正线性相关性质"
    性质1：正线性相关的要求比线性相关更强，正线性相关 $\Rightarrow$ 线性相关。

    性质2：正线性相关与负线性相关等价（即存在不全为零的 $\mu_j \geq 0$ 使上式成立，当且仅当存在不全为零的 $\nu_j \leq 0$ 使 $\sum \nu_j \alpha_j = 0$）。
    
    性质3：若部分 $\alpha_j$ 正线性相关，则所有 $\alpha_j$ 正线性相关。



根据上面**局部极小值的判断条件**，可以得到Fritz John条件如下

若 $x^*$ 是局部极小点，则存在不全为零的非负实数  $\mu_j \geq 0,\ j \in J(x^*) \cup \{0\}$，使

$$
\mu_0 \nabla f(x^*) + \sum_{j \in J(x^*)} \mu_j \nabla g_j(x^*) = 0
$$

其中 $J(x^*)$ 为**起作用约束集**。

!!! attention "缺点：需要事先确定起作用约束集"

为了解决需要确定约束集这一难点，对其进行了完善

=== "Fritz John定理"
    若 $x^*$ 是局部极小点，则存在不全为零的非负实数  
    $\mu_j \geq 0,\ j = 0,1,2,\dots,m$，使

    $$
    \mu_0 \nabla f(x^*) + \sum_{j=1}^{m} \mu_j \nabla g_j(x^*) = 0
    \quad \text{(Lagrange 驻点条件)}
    $$
    
    且满足
    
    $$
    \mu_j g_j(x^*) = 0,\quad j = 1,2,\dots,m
    \quad \text{(互补松弛条件)}
    $$
    
    以及
    
    $$
    \sum_{j=0}^{m} \mu_j > 0
    \quad \text{(强非负条件)}
    $$

> 注：上述条件隐含了如下事实  
> 若 $j \notin J(x^*) \iff g_j(x^*) < 0 \implies \mu_j = 0$。

**两类求解情况**


=== "$\mu_0 > 0$"
    可将 Fritz John 条件改写为如下形式：

    1. Lagrange 驻点条件  
    
    $$
    \nabla f(x^*) + \sum_{j=1}^{m} \lambda_j \nabla g_j(x^*) = 0
    $$
    
    2. 互补松弛条件  
      
    $$
    \lambda_j g_j(x^*) = 0,\quad j = 1,2,\dots,m
    $$
    
    3. 非负条件  
      
    $$
    \lambda_j \geq 0,\quad j = 1,2,\dots,m
    $$
    
    （此时 $\lambda_j = \mu_j/\mu_0$，可全为 0。）
    
    > 注：形式上即 **KKT 条件**。  


=== "$\mu_0 = 0$"
    - 从“不全为0” → 起作用约束梯度 $\{\nabla g_j(x)\}_{j\in J(x)}$ 正线性相关  
    - → 不存在使所有 $\nabla g_j(x)^T p < 0\ (j\in J(x))$ 的**内部**可行方向。  

    注：此时仍可能存在**边界**可行方向，即某些 $\nabla g_j(x)^T p = 0$。

!!! question "$\mu_0 = 0$是忽略了目标函数信息，对应的极小点是否值得考虑？"
    仅考虑$\mu_0 = 0$时，可能丢失部分孤立极值点。KKT条件不是必要条件





##### 一般约束

