---
comments: true
---

# 运动控制

???+note "课程资源"


    === "成绩构成"
       平时70%+开卷考试30%<br>
    
    === "资料"
    	[23年复习课智云](https://classroom.zju.edu.cn/livingroom?course_id=57726&sub_id=1007110&tenant_code=112)<br>
    	[SPWM智云](https://interactivemeta.cmc.zju.edu.cn/#/replay?course_id=67855&sub_id=1469287&tenant_code=112)<br>
    	[矢量控制智云](https://interactivemeta.cmc.zju.edu.cn/#/replay?course_id=67855&sub_id=1469290&tenant_code=112)<br>
    	[矢量控制B站(宝藏视频)](https://www.bilibili.com/video/BV1eM4m1U7hi?spm_id_from=333.788.videopod.sections&vd_source=ace17a48ec1787387c4c8d582e6808cb)<br>
    	[直接转矩控制DTC](https://classroom.zju.edu.cn/livingroom?course_id=57726&sub_id=1007102&tenant_code=112)<br>



## 他励直流电动机

### 机械特性

自然机械特性：出厂就固定的

人为机械特性：调压、弱磁、调电阻

电枢两端加额定电压、气隙每极磁通量为额定值、电枢回路不串电阻时的机械特性称为固有机械特性。 即

$$
n =  \frac{U_N}{C_e \Phi_N} - \frac{R_a}{C_e C_T \Phi^2_N} T_{em}
$$

![image-20241229092257805](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412290923902.png)



### 调速方法

**①改变电枢回路外接电阻$R_j$的人为机械特性**

$n = \frac{U_N}{C_e \Phi_N} - \frac{R_a + R_j}{C_e C_T \Phi_N^2} T_{em} = n_0 - \beta T_{em}$

$R_j$越大，斜率也就越大

![image-20241013160046833](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412271445938.png)

**②改变电枢端U的人为机械特性**

$n = \frac{U}{C_e \Phi_N} - \frac{R_a}{C_e C_T \Phi_N^2} T_{em} = n_0' - \beta_N T_{em}$

实际上改变的是空载转速$n_0$

![image-20241013160150636](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412271445946.png)

**③减弱主磁通Φ的人为机械特性(即增大励磁回路外接电阻$r_j$)**

$n = \frac{U_N}{C_e \Phi} - \frac{R_a}{C_e C_T \Phi^2} T_{em} = n_0'' - \beta' T_{em}$

![image-20241013160252925](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412271445953.png)

### 转速控制的要求和调速指标

调速系统转速控制的三个要求：

- 调速——在$n_{max}$，$n_{min}$之间分挡地（有级）或 平滑地（无级）调节$n $
- 稳速——稳定在一定精度内，抗干扰力强
- 加减速——快、稳；频繁起、制动的设备要求加、减速尽量快，以提高生产率；不宜经受剧烈速度变化的机械则要求起，制动尽量平稳

指标有：**调速范围D和静差率 S**

- 调速范围D:额定负载下，**满足静差率指标S**时，电机的最高转速$𝑛_{𝑚𝑎𝑥}$和最低转速$𝑛_{𝑚𝑖𝑛}$

$$
D=\frac{n_{max}}{n_{min}} \\
$$



- 静差率 S：电机负载由理想空载变为额定负载时，电机的转速降低$\Delta{n}$和理想空载转速$𝑛_{0}$之比。
  

$$
s = \frac{\Delta n_N}{n_0}
$$



静差率指标：用及设计指标规定调速范围范围内，最低转速的静差率表示。$𝑛_{0𝑚𝑖𝑛}$ 为最低转速控制电压对应的理想空载转速。


$$
s = \frac{\Delta n_N}{n_0min}
$$


从上面可以得到


$$
D = \frac{n_N s}{\Delta n_N (1-s)}
$$




常用电机的机械特性图，额定转速之前，是降压调速，超过额定转速，为弱磁调速

![image-20241227165747830](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412271657881.png)



闭环系统的方框图如下图所示

![image-20241227170212922](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412271702949.png)



下面是转速负反馈单闭环调速系统的构成，其为有静差系统，否则输入就为0

![image-20241227170225261](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412271702285.png)



开环系统机械特性与闭环系统静特性的比较

![image-20241227201336545](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412272013587.png)



- 静特性是一组机械特性在形成闭环系统以后实现的
- 静特性一定存在控制器


$$
\text{开环机械特性 }n = \frac{K_{p} K_{s} U_{n}^{*}}{C_{e}} - \frac{R I_{d}}{C_{e}} = n_{0,\text{op}} - \Delta n_{\text{op}} \\
\text{闭环机械特性 }n = \frac{K_{p} K_{s} U_{n}^{*}}{C_{e}(1+K)} - \frac{R I_{d}}{C_{e}(1+K)} = n_{0,\text{cl}} - \Delta n_{\text{cl}}
$$


由上面的公式

!!! note "Attention"
	下面条件成立必须在其前提条件下，否则认为这句话是错误的<br>


- 特性硬：速度改变量比开环小


$$
\Delta n_{\text{op}} = \frac{R I_d}{C_e} \\
\Delta n_{\text{cl}} = \frac{R I_d}{C_e (1+K)} \\
\Delta n_{\text{cl}} = \frac{\Delta n_{\text{op}}}{1+K}
$$






- 闭环系统，静差率要小得多（**对同一n0**）

当**理想空载转速相同**，即 $ n_{0,op} = n_{0,cl}$  时


$$
s_{cl} = \frac{s_{op}}{1+K}
$$




- 当**要求S一定时**，闭环系统可以大大提高调速范围


$$
D_{cl} = (1+K)D_{op}
$$




- 当**给定电压相同**时


$$
n_{0,\text{cl}} = \frac{ n_{0,\text{op}}}{1+K}
$$


**结论：闭环系统可以获得比开环系统硬得多的静态特性。在满足一定静差率的要求下，闭环系统大大提高了调速范围。但是闭环系统必须设置检测装置和电压放大器。**



### 无静差调速系统

!!! note "PI控制规律"
	I ——无静差、快速性差；P——有静差、快速性好<br>
	所以， PI联合的优点： 稳态精度提高、动态响应快<br>
	P、I输出；P与I两部分相加<br>
	PI控制器作为校正装置又提高了稳定性，因此是实用的控制器 <br> 




只有积分环节在扰动作用点前的前向通道上，该扰动便不会引起稳态误差

虽然现在$\Delta U_n$ = 0，只要历史上有过 $\Delta U_n$，其积分就有一定数值，足以产生稳态运行所需要的控制电压 Uc。积分控制规律和比例控制规律的根本区别就在于此。 

![image-20241229093742166](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412290937231.png)

**结论：比例调节器的输出只取决于输入偏差量的现状；而积分调节器的输出则包含了输入偏差量的全部历史。**



## 转速、电流双闭环调速系统

### 系统组成

> ASR：转速调节器<br>ACR：电流调节器<br>

电流截止负反馈环节可以起到控制电流的一些作用，但不能很好地控制电流的动态波形。所以，引入电流环，
即电流负反馈，控制电流的动态过程。



![image-20241227204145975](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412272041005.png)

从结构上看:

- 电流调节环:内环
- 转速调节环:外环

特点

- ASR为PI调节器

> 系统无静差<br>起动时ASR的输出饱和，取$U_{im} = \beta * Idm$。

- ACR起电流调节作用

> 对电流（转矩）指令的随动<br>对电流环内的扰动及时调节<br>电流环的本质是转矩环<br>

### 静特性

书本P55

![image-20241227210611014](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412272106047.png)



- 当调节器饱和时：输入和输出之间的联系被暂时隔断，相当于使该调节器所在的闭环成为开环 ，**输出达到限幅值**；  
- 当调节器不饱和时：PI调节器的积分作用使输入偏差电压在稳态时总是等于零 ，输出未达到限幅值，这样的稳态特征是分析双闭环调速系统的关键。

正常运行时，电流调节器ACR不会达到饱和状态，所以在分析静特性时只考虑转速调节器ASR的饱和、不饱和情况

#### 转速调节器不饱和,稳态工作时

此时，两个调节器都不饱和，稳态时（PI）,依靠调节器的调节作用，其输入偏差电压都为零，因此系统具有绝对硬的静特性。


$$
U_n^* = U_n = \alpha n \\
U_i^* = U_i = \beta I_i\\
由式一可得n = \frac{U_n^*}{\alpha} = n_0\\
$$


因此可以得到下面$n_0 - A$段的静特性

![image-20241227212814392](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412272128425.png)



由于转速调节器不饱和， $U_i^* < U_{im}^*$ ，所以 $ I_d < I_{dm}$ 。这表明， $n_0-A$ 段静特性从理想空载状态( $I_d = 0 $)一直延续到电流最大值，而$ I_{dm}$  一般都大于电动机的额定电流  $I_N$。这是系统静特性的正常运行段。是一条水平特性。

#### 转速调节器饱和

稳态时


$$
 U_{im}^* = U_{im} = \beta I_{dm}\\
 I_{dm} = \frac{U_{im}^*}{\beta}
$$

此时的静特性对应上图中的A-B段，呈现很陡的下垂特征



**结论**

- 双闭环调速系统的静特性在负载电流 $ I_d < I_{dm} $ 时表现为转速无静差，这时ASR起主要调节作用。($n_0-A $ 段)
- 当负载电流达到 $ I_{dm}$ 之后，ASR饱和，ACR起主要调节作用，系统表现为电流无静差，实现了过电流的自动保护。( A-B段 )
- 很显然双闭环的静特性显然比带电流截止负反馈的单闭环系统静特性好。



#### 稳态参数的计算

双闭环调速系统在稳态工作中，当两个调节器都不饱和时，系统变量之间存在如下关系：


$$
U_n^* = U_n = \alpha n = \alpha n_0  \\
U_i^* = U_i = \beta I_d = \beta I_{dL} \\
U_{ct} = \frac{U_{s0}}{K_s} = \frac{C_e n + I_d R}{K_s} = \frac{C_e U_n^* / \alpha + I_d R}{K_s}
$$


**结论**

在稳态工作点上，转速$n$是由给定电压$U_n^*$决定的，ASR的输出量$U_i^*$是由负载电流$I_{dl}$决定的，而ACR 的输出量控制电压$U_{ct}$的大小则同时取决于$n$和$I_d$或者说，同时取决于$U_n^*$和$I_{dl}$。



## 典型系统

 一般地，许多控制系统的**开环**传递函数可用下面的公式来表示（即典型系统针对开环）



$$
W（s） = \frac{K(\tau_1 s+1)(\tau_2 s+1)}{s^r(T_1 s+1)(T_2 s+1)}
$$



系统含有r个积分环节。根据r =0，1，2，…等不同数值，分别称为0型、I型、II型、……系统。

### 典型Ⅰ系统

典型一惯性小，只有一个积分环节

开环传递函数


$$
W(s)=\frac{K}{s(Ts+1)}
$$


闭环系统结构图与开环的波特图如下图所示

![image-20241228092018010](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412280920095.png)



该典型系统不仅因为其结构简单，而且对数幅频特性的中频段以-20dB／dec的斜率穿越零分贝线，只要参数的选择能保证足够的中频带宽度，系统就一定是稳定的，且有足够的稳定余量。

需要满足以下条件

- $\omega_c < \frac{1}{T}$
- $\omega_c T< 1$
- $tg^{-1}\omega_c T< 45°$

则相角稳定裕度：$γ = 90°- tg^{-1}\omega_c T$



### 典型Ⅱ系统

开环传递函数


$$
W(s)=\frac{K(\tau s+1)}{s^2(Ts+1)}
$$


相当于PI+典Ⅰ系统

闭环系统结构图和开环对数频率特性如下图所示

![image-20241228092210530](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412280922565.png)



### 动态性能指标

- 跟随性能指标

> 上升时间<br> 超调量<br> 相对稳定性<br>

- 抗扰性能指标

> 动态降落<br> 恢复时间<br>



调速系统的动态指标以抗扰性能为主，随动系统的动态指标则以跟随性能为主

比较分析的结果可以看出，典型I型系统和典型Ⅱ型系统除了在稳态误差上的区别以外，在动态性能中，

- 典型 I 型系统在跟随性能上可以做到超调小，但抗扰性能稍差，
- 典型Ⅱ型系统的超调量相对较大，抗扰性能却比较好。



## 系统设计原则

**先内环后外环**

从内环开始，逐步向外扩展。在这里，首先设计电流调节器，然后把整个电流环看作是转速调节系统中的一个环节，再设计转速调节器。

### 电流调节器的设计

- 电流环结构图的简化
- 电流调节器结构的选择
- 电流调节器的参数计算
- 电流调节器的实现

**电流调节器结构的选择**

选择典型Ⅰ系统的两个原因

- 从稳态要求上看，希望电流无静差，以得到理想的堵转特性，由电流环结构图可知，采用 I 型系统就够了。
- 从动态要求上看，实际系统不允许电枢电流在突加控制作用时有太大的超调，以保证电流在动态过程中不超过允许值，而对电网电压波动的及时抗扰作用只是次要的因素，为此，电流环应以跟随性能为主，应选用典型I型系统。 

![image-20241228100414638](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412281004706.png)



**电流环等效传递函数**

接入转速环内，电流环等效环节的输入量应为$U^*_i(s)$，因此电流环在转速环中应等效为 



这样，原来是**双惯性环节**的电流环控制对象，经闭环控制后，可以近似地**等效成**只有较小时间常数的**一阶惯性环节**。


$$
\frac{I_d(s)}{U_i^*(s)} = \frac{W_{cli}(s)}{\beta} \approx \frac{\frac{1}{\beta}}{\frac{1}{K_1}s + 1}
$$


**物理意义**

​     这就表明，电流的闭环控制改造了控制对象，加快了电流的跟随作用，这是局部闭环（内环）控制的一个重要功能。



### 转速调节器的设计

**转速调节器结构的选择**

设计成典型 Ⅱ 型系统的**两个**原因

为了实现转速无静差，在**负载扰动作用点前面必须有一个积分环节**，它应该包含在转速调节器 ASR 中，现在在扰动作用点后面已经有了一个积分环节，因此转速环开环传递函数应共有两个积分环节，所以应该设计成典型 Ⅱ 型系统，这样的系统同时也能满足动态抗扰性能好的要求。

![image-20241229094933367](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412290949414.png)

**转速环与电流环的关系**

外环的响应比内环慢，这是按上述工程设计方法设计多环控制系统的特点。这样做，虽然不利于快速性，但每个控制环本身都是稳定的，对系统的组成和调试工作非常有利。 

!!! question 例题
    大前提——转速电流双闭环直流无静差调速系统在额定负载下稳定运行<br>
	1、转速调节器ASR和电流调节器ACR的输入分别为多少？分别列出ASR和ACR的输出表达式。<br>
	答：输入：都是0   输出：ASR $U_i^* =\beta I_{dl}$ ACR：$U_c$<br>
	2、若减少转速给定电压Un*而其它参数保持不变，则电动机转速n的变化是增加、减少或是不变<br>
	答：减小<br>
	3、若减少转速反馈系数α而其它参数保持不变，则电动机转速n和转速反馈电压Un的变化是增加、减少或是不变<br>
	答：增加<br>







### 调节器结构的选择

基本思路:  将控制对象校正成为典型系统

![image-20241228104442738](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412281044796.png)

选择规律：

​       几种校正成典型I型系统和典型II型系统的控制对象和相应的调节器传递函数如下表所示，表中还给出了参数配合关系。有时仅靠 P、I、PI、PD及PID几种调节器都不能满足要求，就不得不作一些近似处理，或者采用更复杂的控制规律。



根据控制对象的传递函数确定调节器的传函以及参数的值

![image-20241228104532209](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412281045250.png)



![image-20241228104555407](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412281045441.png)



### 传递函数近似处理

1、高频段小惯性环节的近似处理

​     实际系统中往往有若干个小时间常数的惯性环节，这些小时间常数所对应的频率都处于频率特性的高频段，形成一组小惯性群。例如，系统的开环传递函数为 

![image-20241228110616391](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412281106441.png)

当系统有一组小惯性群时，在一定的条件下，可以将它们近似地看成是一个小惯性环节，其时间常数等于小惯性群中各时间常数之和。 

$$
例如 \quad \frac{1}{(T_2 s + 1)(T_3 s + 1)} \approx \frac{1}{(T_2 + T_3) s + 1} \\
近似条件\quad \omega_c \leq \frac{1}{3\sqrt{T_2 T_3}} 
$$


2、高阶系统的降阶近似处理

  上述小惯性群的近似处理实际上是高阶系统降阶处理的一种特例，它把多阶小惯性环节降为一阶小惯性环节。下面讨论更一般的情况，即如何能忽略特征方程的高次项。以三阶系统为例，设

$$
W(s) = \frac{K}{as^3+bs^2+cs+1} \\
其中a,b,c都是正系数，且bc>a,即系统是稳定的。
$$

降阶处理：若能忽略高次项，可得近似的一阶系统的传递函数为

$$
W(s) = \frac{K}{cs+1} \\
近似条件 \quad \omega_c \le \frac{1}{3}min(\sqrt{\frac{1}{b}},\sqrt{\frac{c}{a}})
$$

3、低频段大惯性环节的近似处理

当系统中存在一个时间常数特别大的惯性环节时，可以近似地将它看成是积分环节，即   

$$
\frac{1}{Ts+1} →\frac{1}{Ts} \\
近似条件 \quad \frac{3}{T} \le \omega_c
$$




## 交流电机

### 机械特性

物理表达式

$$
T_e = C_T \Phi_m I_2^‘ cos\theta_2 \\
$$


- 稳态公式
- Φ与I2’不正交（有互感和漏感）
- Φ与I2’还是中间变量，所以需要考察其他形式的表达式。

即

$$
T_e = \frac{3 n_p}{2 \pi f_1} \cdot \frac{U_1^2 r_2'}{\left(r_1 + \frac{r_2'}{s}\right)^2 + \left(x_{\sigma 1} + x_{\sigma 2}\right)^2} \\
$$

交流电机的力矩与电压的平方成正比



#### 固有机械特性

 三相异步电动机在电压、频率均为额定值不变，定、转子回路不串入任何电路元件条件下的机械特性，称为**固有机械特性**，其T-s曲线(也即T-n曲线)如下图所示．其中曲线1为电源正相序时的，曲线2为负相序时的曲线。

![image-20241228112825114](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412281128157.png)



当 s  很小时

$$
T_e \approx 3 n_p \left(\frac{U_1}{\omega_1}\right)^2 \frac{s \omega_1}{r_2'} \propto s \\
即，机械特性 \quad T_e = f(s) \text{是一段直线。} \\
$$

当 s接近于 1 时，则

$$
T_e \approx 3 n_p \left(\frac{U_1}{\omega_1}\right)^2 \frac{\omega_1 r_2'}{s \left[r_1^2 + \omega_1^2 (L_{\sigma 1} + L_{\sigma 2}')^2\right]} \propto \frac{1}{s}
$$


![image-20241228113608422](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412281136463.png)

#### 人为机械特性

1、降低定子端电压的人为机械特性

仅降低定子电压（由于异步电机的磁路在额定电压下已有点饱和了不宜再升高电压），其它参数都与固有机械特性时相同。

$$
T_{e, \text{max}} = \frac{3 n_p}{4 \pi f_1} \cdot \frac{U_1^2}{(x_{\sigma 1} + x_{\sigma 2}')}\\
 S_{I, \text{max}} = \frac{r_2^2}{(x_{\sigma 1} + x_{\sigma 2}')}
$$


降低定子电压的人为机械特性与固有机械特性相比较，在相同的转差率S下，电磁转矩与$(U_1 / U_N)^2$成正比，即

$$
T_e' = T_e(U_1 / U_N)^2
$$

式中$T_e$为在固有机械特性时的电磁转矩。

![image-20241228141530392](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412281415434.png)

应用：软启动器：防止全压起动时冲击电流过大。



2、转子回路串三相对称电阻时的机械特性

相当于增加了转子绕组每相电阻值。此时，不影响电动机同步转速$n_1$的大小，不改变$T_{e,max}$的大小，其特性都通过同步运行点。但临界转差率$S_{T_emax}$则随转子回路中电阻的增大而成正比地增加。

根据转矩公式$\frac{s'}{s} = \frac{r_2'+R}{r_2'} $

式中,s为固有机械特性上电磁转矩为,$T_e$时的转差率，s'在同一电磁转矩下人为机械特性上的转差率。这表明，若保持电磁转矩不变，则串入附加电阻后电动机的转差率将与转子中的电阻成正比地增加。

应用：绕线电机的转差功率回馈控制

![image-20241228142030803](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412281420845.png)



3、定子回路串三相对称电阻时的机械特性

定子回路串入电阻并不影响同步转速$n_1$，但是最大电磁转矩$T_{em}$、起动转矩$T_{st}$ 和临界转差率 $s_m$都随着定子回路电阻值的增大而减小。

![image-20241229100933795](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291009841.png)



### 恒压频比控制策略的出发点

采用电压型PWM逆变器；

> 假设：①忽略空间和时间谐波<br>
>             ②忽略磁饱和<br>
>            ③忽略铁损<br>



等效电路图和恒压恒频时异步电动机的机械特性如下所示

![image-20241228145218491](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412281452535.png)

$$
T_e = 3 n_p \left(\frac{U_1}{\omega_1}\right)^2 \frac{s \omega_1 r_2'}{(s r_1 + r_2')^2 + s^2 \omega_1^2 (L_{l1} + L_{l2})^2} \quad (6.3-1)\\
$$

当s很小时，机械特性$T_e =f(s)$是一段直线

$$
T_e \approx 3 n_p \left(\frac{U_1}{\omega_1}\right)^2 \frac{s \omega_1}{r_2'} \propto s \quad (6.3-2a)\\
$$

当s接近于1时，则

$$
T_e \approx 3 n_p \left(\frac{U_1}{\omega_1}\right)^2 \frac{\omega_1 r_2'}{s [r_1^2 + \omega_1^2 (L_{l1} + L_{l2})^2]} \propto \frac{1}{s}
$$


在进行电动机调速时，常须考虑的一个重要因素就是：希望保持电动机中每极磁通量$\Phi_m$为**额定值不变**。

- 如果磁通**太弱**，电机也就不能输出最大的转矩；
- 如果**过分增大磁通**，又会使铁心饱和，从而导致过大的励磁电流，严重时会因绕组过热而损坏电机。



三相异步电动机定子每相电动势的有效值是$E_g = 4.44 f_1 W_1 k_{ap1} \Phi_m$

因为 $\Psi_g = W_1 k_{ap1} \Phi_m$，所以 $E_g = 4.44 f_1 \Psi_g$，只要控制好 $E_g$和$ f_1$，便可达到控制有效值 $\Psi_g$ 的目的

#### 基频以下调速

由式 $E_g = 4.44 f_1 \Psi_g$ 知，要保持有效值 $\Psi_g$ 不变，当频率 $f_1$ 从电机的额定频率 $f_{IN}$ 向下调节时必须同时降低 $E_g$，使

$$
\frac{E_g}{f_1} = \text{常值}
$$

即采用**气隙感应电势和定子电压频率之比为恒值**的控制方式。



若忽略定子绕组的漏磁阻抗压降，则定子相电压 $U_1 \approx E_g$

$$
\frac{U_1}{f_1} = \text{常值}
$$


即恒压频比的控制方式。



#### 基频以上调速

电压已经是额定电压，只能 $U_1 = \text{Const.}$  控制。由 $U_1 \approx E_g = 4.44 f_1 \Psi_g$，磁通与频率成反比例降低（弱磁调速）。

按照电力拖动原理，在基频以下，磁通恒定时转矩也恒定，属于“恒转矩调速”性质，而在基频以上，转速升高时转矩降低，基本上属于“恒功率调速”。

![image-20241229101234163](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291012210.png)

!!! note note
	$f_{1N}$一般为定值50Hz<br>



#### 基频以下的电压频-率协调控制时的机械特性

![image-20241228150855525](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412281508568.png)

图中几个新的变量及其物理意义如下：

- $E_g$ 为气隙磁链 $\Psi_g$ 在定子每相绕组中的感应电动势，也就是定子电势 $E_1$ ；
- $E_s$  为定子全磁链（含定子漏磁）$\Psi_s$ 在定子每相绕组中的感应电动势有效值；
- $E_r$ 为转子全磁链（含转子漏磁）$\Psi_r$  在转子绕组中的感应电动势有效值（折合到定子边）。



**1、恒压频比控制**

$U_1/\omega_1 = const$

为了保持$\Phi_m$不变

$$
n_0= \frac{60\omega_1}{2\pi n_p} \\
\Delta n = sn_0 = \frac{60}{2\pi n_p}s\omega_1 \\
$$

由因为在$T_e=f(s)$直线段

$$
T_{e} \approx 3 n_{p} \left(\frac{U_{1}}{\omega_{1}}\right)^{2} \frac{s \omega_{1}}{r_{2}'} \propto s \\
\rightarrow s \omega_1 \approx \frac{r_2' T_{e}}{3 n_{p} \left(\frac{U_1}{\omega_1}\right)^2}
$$

当 $U_1/\omega_1$ 为恒值时，对于同一转矩 $T_e$，$s \omega_1$ 是基本不变的，因而 $\Delta n$ 也是基本不变的。这就是说，在恒压频比的条件下改变频率 $\omega_1$ 时，**机械特性基本上是平行下移**。它们和直流他励电动机变压调速时的情况基本相似。

最大转矩 $T_{e\text{max}}$ 可以通过以下公式计算：

$$
T_{e\text{max}} = \frac{3 n_p}{2} \left(\frac{U_1}{\omega_1}\right)^2 \frac{1}{\frac{r_1}{\omega_1} + \sqrt{\left(\frac{r_1}{\omega_1}\right)^2 + (L_k + L_{lr})^2}}
$$


![image-20241229103809828](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291038899.png)

由于项 $\frac{r_1}{\omega_1}$ 的存在，最大转矩 $T_{\text{max}}$ 是随着 $\omega_1$ 的降低而减小的。频率很低时 $T_{\text{max}}$ 太小，将限制电动机的带载能力。

采用定子压降补偿（算式为 $U_1 = Kf_1 + U_{10}$），适当地提高电压 $U_1$，可以增强带载能力，见图中的红线所示机械特性。



**2、恒$E_g/ \omega_1$控制**

在电压频率协调控制中，如果恰当地提高电压 $U_1$ 的数值来维持 $\frac{E_g}{\omega_1}$ 为恒值，则由 $E_g = 4.44 f_1 \Psi_g$ 可知，无论频率高低，每极磁链有效值 $\Psi_g$ 均为常值。

由等效电路可以看出

$$
I_2' = \frac{E_g}{\sqrt{\left(\frac{r_2'}{s}\right)^2 + \omega_1^2 L_{1r}'^2}}
$$

代入电磁转矩基本关系式，得

$$
T_e = \frac{3 n_p}{\omega_1} \cdot \frac{E_g^2}{\left(\frac{r_2'}{s}\right)^2 + \omega_1^2 L_{1r}'^2} \cdot \frac{r_2'}{s} = 3 n_p \left(\frac{E_g}{\omega_1}\right)^2 \frac{s \omega_1 r_2'}{r_2' + s^2 \omega_1^2 L_{1r}'}
$$

这就是恒 $\frac{E_g}{\omega_1}$ 控制时的机械特性方程式。

将前式对 $s$ 求导，并令 $\frac{dT_e}{ds} = 0$，可得恒 $\frac{E_g}{\omega_1}$ 控制在最大转矩时的转差率 

$$
s_{T_{\text{max}}} = \frac{r_2'}{\omega_1 L_{r2}}
$$

最大转矩 

$$
T_{\text{max}} = \frac{3}{2} n_p \left(\frac{E_g}{\omega_1}\right)^2 \frac{1}{L_{r2}}
$$

值得注意的是，在最大转矩中，当 $\frac{E_g}{\omega_1}$ 为恒值时，**$T_{\text{max}}$ 恒定不变。**可见恒 $\frac{E_g}{\omega_1}$ 控制的稳态性能是优于恒 $\frac{U_1}{\omega_1}$ 控制的，它正是恒 $\frac{U_1}{\omega_1}$ 控制中补偿定子压降所追求的目标。

![image-20241229104201036](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291042104.png)

**3、恒$E_r/ \omega_1$控制**

![image-20241229104406240](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291044302.png)

把电压基波有效值 $U_1$ 再进一步提高，把转子漏抗上的压降也抵消掉，得到恒 $\frac{E_g}{\omega_1}$ 控制。由图可写出 $I_2 = \frac{E_r}{r_2'/s}$，代入电磁转矩基本关系式，得

$$
T_e = \frac{3 n_p}{\omega_1} \cdot \frac{E_r^2}{\left(\frac{r_2'}{s}\right)^2} \cdot \frac{r_2'}{s} = 3 n_p \left(\frac{E_r}{\omega_1}\right)^2 \cdot \frac{s \omega_1}{r_2'}
$$



将上式改写为 

$$
\omega_r = \omega_1 - \frac{r_1'}{3 n_p} \left(\frac{\omega_1}{E_r}\right)^2 T_e = \omega_1 - k T_e
$$

式中，$k = \frac{r_1'}{3 n_p} \left(\frac{\omega_1}{E_r}\right)^2$ 在“恒 $\frac{E_g}{\omega_1}$ 控制”时为常数。



**总结**

![image-20241229104459244](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291044298.png)

给出上述 3 种控制方式下的机械特性曲线。显然，恒 $\frac{E_g}{\omega_1}$ 控制的稳态性能最好，可以获得和直流电动机完全相同的线性机械特性。这正是高性能交流变频调速所要求的特性。



### 转速开环的交—直—交电流源变频调速系统

**开环系统组成**

![image-20241229105510107](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291055171.png)

全数字式系统的实物图(比单闭环没有增加任何硬件)

![image-20241229105537669](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291055723.png)



### 转速闭环、转差频率控制的变频调速系统

任何电力拖动自动控制系统都服从于基本运动方程式 $T_e - T_L = \frac{J}{n_p} \frac{d\omega}{dt}$。

控制转速变化率 $\frac{d\omega}{dt}$ 可以提高调速系统的动态性能；又控制好转矩 $T_e$，就可以控制好 $\frac{d\omega}{dt}$。

!!! question
	对于刚学过的VVVF，问：可否通过控制电压、频率→控制 $T_e$ → 控制 $\frac{d\omega}{dt}$。<br>
	可以<br>
	
重点：一个定义两个规律

定义：令$\omega_s = s\omega$,$\omega_s$定义为转差角频率

**转差频率控制规律**

- 规律一：$T_e \propto \omega_s$


$T_e = K_m \frac{\omega^2 R_2'}{R_2'^2 + (\omega L_{1r})^2}$

就是机械特性 $T_e = f(\omega_s)$。

![image-20241229110808316](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291108373.png)



可以看出：
- 在 $\omega_s$ 较小的稳态运行段上，转矩 $T_e$ 基本上与 $\omega_s$ 成正比，
- 当 $T_e$ 达到其最大值 $T_{e\text{max}}$ 时，$\omega_s$ 达到 $\omega_{s\text{max}}$ 值。



![image-20241229111020311](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291110365.png)



- 规律二

规律一成立的前提假设为：保持 $\Phi_m$ 不变。
如何使 $\Phi_m$ 保持不变？

如图：$\Phi_m \propto I_0$。

由电流平衡方程可以推导出

 $I_1 = I_0 \sqrt{\frac{R_2'^2 + \omega^2 (L_m + L_{1r})^2}{R_2'^2 + \omega^2 L_{1r}^2}}$

当$\Phi_m$或$I_0$不变时，$I_1$与转差频率$ω_s$的关系式如上，绘制成相应的曲线如下

![image-20241229111223400](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291112460.png)

上述关系表明：只要 $I_1$ 与 $\omega_s$ 的关系符合上图的规律，就能保持 $\Phi_m$ 恒定。这样，用转差频率控制代表转矩控制的前提也就解决了。这是转差频率控制的**基本规律之二。**

转差频率控制的规律是：
1. $\omega_s \le \omega_{sm} 时\quad  T_e \propto \omega_s$
2. $I_1 = F(\omega_s)$ 按照图7-47控制，可以保持 $\Phi_m$ 恒定。



转差频率控制的变频调速系统组成

![image-20241229111454819](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291114892.png)



### 异步电动机动态数学模型

定性地，交流异步电动机的数学模型有以下特点：

①异步电动机变压变频调速时需要进行电压（或电流）和频率的协调控制，电压（电流）和频率是两种独立的输入变量。转速和磁通是独立输出变量。异步电动机是一个多变量（多输入多输出）系统，而各个变量之间又互相都有影响，所以是强耦合的多变量系统。

②在异步电动机中，电流乘磁通产生转矩，转速乘磁通得到异步电动势，由于它们都是同时变化的，在数学模型中就含有两个变量的乘积项。因此，即使不考虑磁饱和等因素，数学模型也是非线性的。

③三相异步电动机定子有三个绕组，转子也可等效为三个绕组，每个绕组产生磁通时都有自己的电磁惯性，再算上运动系统的机电惯性，和转速与转角的积分关系，即使不考虑变频装置的滞后因素，也是一个八阶系统。

所以说，异步电动机的动态数学模型是一个**高阶、非线性、强耦合**的**多变量**系统。



在研究异步电动机的多变量非线性数学模型时，作如下假设：

> ①忽略空间谐波，设三相绕组对称，在空间中互差120°  电角度，所产生的磁动势沿气隙周围按正弦规律分布；<br>
> ②忽略磁路饱和，各绕组的自感和互感都是恒定的；<br>
> ③忽略铁心损耗；<br>
> ④不考虑频率变化和温度变化对绕组电阻的影响。<br>



无论电动机转子是绕线型还是笼型的，都等效成三相绕线转子，并且进行了绕组折算（折算后的定子和转子绕组匝数都相等）。这样，电动机绕组就等效成下图所示的三相异步电动机的绕组模型。

![image-20241229120635109](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291206167.png)

讨论异步电动机的数学模型：

-  电压平衡方程
-  磁链平衡方程
-  转矩平衡方程
- 电力拖动系统运动方程



#### dq坐标系上的数学模型

设两相旋转坐标 d 轴与三相定子坐标 A 轴的夹角为 $\theta_s = \omega_{dg} t$，则 $\omega_{sdq} = p\theta_s$ 为 $dq$ 坐标系相对于定子的角转速，$\omega_{rdq} = \omega_{sdq} - \omega_r$ 为 $dq$ 坐标系相对于转子 $a$ 轴的角速度，其中 $\omega_r = \omega_{dgs} - \omega_{dg}$ 为 $n_p = 1$ 条件下电机转子角速度。

三相静止到两相旋转的变换：先 $3/2$ 变换到坐标系 $\alpha\beta$  上，再旋转变换到坐标系 $dq$ 上。

![image-20241229121315904](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291213969.png)



由标轴的旋转速度 $\omega_{sdq}$ 等于定子频率的同步角转速，即$\omega_{sdq} = \omega_1$

可得到另一种很有用的坐标系即两相同步旋转坐标系。坐标轴仍用 $\{d, q\}$ 表示，而转子的转速为 $\omega_r$，因此 $dq$ 轴相对于转子的角转速即转差角频率为$\omega_1 - \omega_r = \omega_s $

将上式代入式带入电压方程



可得同步旋转坐标系上用定、转子电流表示，以及用定子电流和转子磁链表示的电压方程为

$\begin{bmatrix} u_{sdq} \\ u_{rdq} \end{bmatrix} = \begin{bmatrix} R_s I + L_s (pI + \omega_s J) & L_m (pI + \omega_s J) \\ L_m (pI + \omega_s J) & R_r I + L_r (pI + \omega_s J) \end{bmatrix} \begin{bmatrix} i_{sdq} \\ i_{rdq} \end{bmatrix} $

$= \begin{bmatrix} R_s I + \sigma L_s (pI + \omega_s J) & (L_m / L_s) (pI + \omega_s J) \\ -L_m / T_r I & (1 / T_r + p) I + \omega_s J \end{bmatrix} \begin{bmatrix} i_{sdq} \\ \psi_{rdq} \end{bmatrix}$

此时，磁链方程、转矩方程和运动方程均不变。



两相同步旋转坐标系的突出特点是，当三相ABC坐标系中的电压和电流是交流正弦波时，变换到dq坐标系上就表现为直流的形式。

对于异步电动机，$u_{rdq} = 0$。依据上式和转矩方程可绘成动态等效电路图所示。如果把旋转变换做一个调整，即将 $d$ 轴的方向规定为转子磁链的方向，则由于 $\psi_{ra} = 0$，图形将大大被化简。

![image-20241229122913591](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291229640.png)

## 矢量控制

要根据异步机的动态数学模型(方程式)，选择合适的控制策略，使**异步机得到与直流机相类似的控制性能。**

矢量控制与标量控制的主要区别：不仅要控制电流的大小，而且控制电流的相位。

本质上：矢量控制的思想是按产生同样的旋转磁场这一等效原则建立的。

### 基本思路

以产生同样的旋转磁动势为准则，异步电机通过三相-两相变换可以等效成两相静止坐标系上的交流电流 $i_\alpha$ 和 $i_\beta$，再通过同步旋转变换，可以等效成同步旋转坐标系上的直流电流 $i_M$ 和 $i_T$。

站到异步机旋转坐标系上，所看到异步机模型的便是一台与图 7.3.1(a) 结构相同的直流电动机。

![image-20241229123258175](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291232227.png)

**定义：**

既然异步电动机经过坐标变换可以等效成直流电动机，那么，模仿直流电动机的控制策略，就可以控制这个等效的“直流电动机”，从而获得较好的转矩特性，也就能够较好控制异步电动机的电磁转矩了。由于进行坐标变换的是电流（代表磁动势）的空间矢量，所以这样通过坐标变换实现的控制系统就叫作矢量控制系统 (Vector Control System)，简称 VC 系统。由于是以转子磁链的方向作为 M 轴的方向，学术界也将这种控制策略称为转子磁链定向或磁场定向控制 (Field Orientation Control - FOC)。

### 转子磁链定向条件下的电机模型

#### (1)转子磁链定向

选择 $i_{sdq}=\left[\begin{array}{ll}i_{sd}& i_{sq}\end{array}\right]^T, \psi_{rdq}=\left[\begin{array}{ll}\psi_{rd}&\psi_{rq}\end{array}\right]^T$ 作为状态变量，则电压方程进一步可将其改写为式

$$
\left[\begin{array}{c} u_{sdq}\\ 0\end{array}\right]=\left[\begin{array}{cc}\left(R_{s}+\sigma L_{s} p\right) I+\omega_{1}\sigma L_{s} J & \left(L_{m}/ L_{r}\right)\left\{p I+\omega_{1} J\right\}\\ -L_{m}/ T_{r} I & \left(1/ T_{r}+p\right) I+\omega_{s} J\end{array}\right]\left[\begin{array}{c} i_{sdq}\\ \psi_{rdq}\end{array}\right] 
$$

在 动态模型分析中，在进行两相同步旋转坐标变换时只规定了 $d, q$ 两轴的相互垂直关系和与定子频率同步的旋转速度，**并未规定两轴与电机旋转磁场的相对位置**，如果取 **$d$ 轴沿着转子磁链矢量 $\psi_r$ 的方向，**称之为 M (Magnetization) 轴，而 $q$ 轴为逆时针转 90 度，即垂直于矢量 $\psi_r$，称之为 T (Torque) 轴。这样的两相同步旋转坐标系就具体规定为 **MT** 坐标系，即按转子磁链定向 (Field Orientation) 的旋转坐标系。

![image-20241229124057508](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291240560.png)

由上图可知

① 转子磁链 $\psi_r$ 仅由定子电流励磁分量 $i_{M}$ 产生，与转矩分量 $i_{T}$ 无关，从这个意义上看，定子电流的励磁分量与转矩分量是解耦的。此外，$\psi_{M}$ 与 $i_{M}$ 之间的传递函数是一阶惯性环节，其时间常数 $T_r$ 为转子时间常数，当励磁电流分量 $i_{M}$ 突变时，$\psi_{M}$ 的变化要受到励磁惯性的阻挠，这和直流电动机励磁绕组的惯性作用是一致的。

② 电磁转矩 $T_e$ 是变量 $i_{T}$ 和 $\psi_{M}$ 的点积。由于 $T_e$ 同时受到变量 $i_{T}$ 和 $\psi_{M}$ 的影响，仍旧是耦合着的。



**结论**

转子磁链定向使得定子电流的励磁分量于转矩分量解耦，电磁转矩 $T_e$ 与转矩电流分量 $i_{T}$ 变成了线性关系。由此，**对转矩的控制问题就转化为对转矩电流分量的控制问题**。



#### (2)转子磁链定向且磁链赋值为定值

由电机原理知，为了输出最大转矩，也需要电机工作在额定磁链状态。



#### 小结：分析矢量控制系统的一些收获

1. 为了便于理解，将系统的理想实现（假定转子磁链可测）和工程实现分开讨论；
2. **控制的核心仍然是对电磁转矩的控制**。电磁转矩不能直接测量，并与可测量如电流等的关系是非线性、耦合的。
3. 对比它激直流电机的转矩控制，使用“**转子磁链定向并使之幅值一定**”以达到转矩控制的解耦和线性化；
4. 所用到控制理论知识有：坐标变换、状态变量的选取、观测器理论（没有讲）、参数鲁棒性（没有讲）等。

!!! question
	VC控制是如何实现对转矩的解耦的？<br>
	转子磁链定向并使之幅值一定以达到转矩控制的解耦和线性化<br>



## SPWM

### 基本概念

脉宽调制变频的思想：

控制逆变器中的功率开关器件导通或断开，其输出端即获得一系列宽度不等的矩形脉冲波形．而决定开关器件动作顺序和时间分配规律的控制方法即称脉宽调制方法。

!!! note "理论基础"
	冲量相等而形状不同的窄脉冲加在具有惯性的环节上时，其效果基本相同。<br>

![image-20241229133324549](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291333604.png)

如何用一系列等幅不等宽的脉冲来代替一个正弦半波

- 等距分割
- 按照冲量相等的理论，保持中点不变，改变宽度，实现等幅

![image-20241229133538016](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291335073.png)

对于正弦波的负半周，采取同样的方法，得到PWM波形，因此正弦波一个完整周期的等效PWM波为：

![image-20241229134352681](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291343744.png)

根据面积等效原理，正弦波还可等效为下图中的PWM波，而且这种方式在实际应用中更为广泛。

![image-20241229134401362](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291344417.png)



### 对脉宽调制的制约条件

1、开关频率

$N = \frac{f_t}{f_r}$ 是载波比

当 $N$ 增大时，逆变器输出更加接近正弦波。

但 $N$ 不能太大，受功率开关器件允许开关频率的限制。

$N \le \frac{逆变器功率开关器件的允许开关频率}{频段内最高的正弦参考信号频率}$

2、调制度$M = \frac{U_{rm}}{U_{tm}}$  , 其中 $M < 1$ ，$M = 0 \sim 0.99$ 

**SPWM逆变器的同步调制和异步调制**

(一) 同步调制
- $N$ 为常数，$f_t$ 与 $f_r$ 同步变化，当 $f_r$ 减小时 $f_t$ 也小，相邻的两脉冲间距增大，谐波会显著增加。但可以保证 $N$ 为 3 的倍数。

(二) 异步调制
- $f_t$ 为常数，$N$ 不为常数，所以低频时 $N$ 增大，但 3 的倍数无法保证。

(三) 分段同步调制 — 结合上述两者优点
- 在一定频率范围内，用同步调制；当 $f_r$ 减小时 $N$ 增大。

定子下标：1s 转子下标：2 1

**控制模式及其实现**

(一)自然采样法

$$
T_c = t_1 +t_2 +t_3\\
u_r = Msin\omega_1 t
$$

![image-20241229135535042](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291355097.png)

(二)规则采样法

$$
t_2 = \frac{T_c}{2}(1+Msin\omega_1 t)\\
t_1 = t_2 =\frac{1}{2}(T_c -t_2)
$$

![image-20241229135554682](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291355731.png)



## SVPWM

### 概念

 经典的SPWM控制主要着眼于使变压变频器的输出电压尽量接近正弦波，并未顾及输出电流的波形。而电流滞环跟踪控制则直接控制输出电流，使之在正弦波附近变化，这就比只要求正弦电压前进了一步。然而交流电动机需要输入三相正弦电流的最终目的是在电动机空间形成圆形旋转磁场，从而产生恒定的电磁转矩。

  如果对准这一目标，把逆变器和交流电动机视为一体，按照跟踪圆形旋转磁场来控制逆变器的工作，其效果应该更好。这种控制方法称作“磁链跟踪控制”，下面的讨论将表明，磁链的轨迹是交替使用不同的电压空间矢量得到的，所以又称“电压空间矢量PWM（SVPWM，Space Vector PWM）控制”。

#### 电压空间矢量的相互关系

**定子电压空间矢量：**$u_{A0}$、$u_{B0}$、$u_{C0}$ 的方向始终处于各相绕组的轴线上，而大小则随时间按正弦规律脉动，时间相位互相错开的角度也是120°。

**合成空间矢量：**由三相定子电压空间矢量相加合成的空间矢量 $u_s$ 是一个旋转的空间矢量，它的幅值不变，是每相电压值的 $3/2$ 倍。

当电源频率不变时，合成空间矢量 $u_s$ 以电源角频率 $\omega_1$ 为电气角速度作恒速旋转。当某一相电压为最大值时，合成电压矢量 $u_s$ 就落在该相的轴线上。用公式表示，则有

$$
u_s = u_{A0} + u_{B0} + u_{C0} 
$$

与定子电压空间矢量相仿，可以定义定子电流和磁链的空间矢量 $I_s$ 和 $\Psi_s$。



#### 电压与磁链空间矢量的关系

$$
u_s= R_sI_s +\frac{d\Psi_s}{dt} \\
当电阻 R_s 项可以忽略时，u_s \approx \frac{d\Psi_s}{dt}\\
\Psi_s \approx \int u_s dt
$$

交流电动机绕组的电压、电流、磁链等物理量都是随时间变化的，分析时常用时间相量来表示，但如果考虑到它们所在绕组的空间位置，也可以如图所示，定义为空间矢量$U_{A0}$，$U_{B0}$，$U_{C0}$

![image-20241229141812175](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291418271.png)



如图所示，当磁链矢量在空间旋转一周时，电压矢量也连续地按磁链圆的切线方向运动$2 \pi$弧度，其轨迹与磁链圆重合。

这样，电动机旋转磁场的轨迹问题就可转化为电压空间矢量的运动轨迹问题。

![image-20241229141939779](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291419832.png)

**电路原理图**

![image-20241229142237612](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291422673.png)

图中的逆变器采用180°导通型，功率开关器件共有8种工作状态（见附表） ，其中
6 种有效开关状态；2 种无效状态（因为逆变器这时并没有输出电压）。

![image-20241229142320383](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291423440.png)

对于六拍阶梯波的逆变器，在其输出的每个周期中6 种有效的工作状态各出现一次。逆变器每隔 $\pi$/3 时刻就切换一次工作状态（即换相），而在这  $\pi$/3 时刻内则保持不变。 

这样，在一个周期中 6 个电压空间矢量共转过 2$\pi$弧度，形成一个封闭的正六边形，如图所示

![image-20241229142539970](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291425021.png)



### 电压空间矢量的线性组合与SVPWM控制

由上面可知

- 如果交流电动机仅由常规的六拍阶梯波逆变器供电，磁链轨迹便是六边形的旋转磁场，这显然不象在正弦波供电时所产生的圆形旋转磁场那样能使电动机获得匀速运行。
- 如果想获得更多边形或逼近圆形的旋转磁场，就必须在每一个期间内出现多个工作状态，以形成更多的相位不同的电压空间矢量。为此，必须对逆变器的控制模式进行改造。 





**开关顺序原则**

在实际系统中，应该尽量减少开关状态变化时引起的开关损耗，因此不同开关状态的顺序必须遵守下述原则：每次切换开关状态时，只切换一个功率开关器件，以满足最小开关损耗。 

为了使电压波形对称，把每种状态的作用时间都一分为二，因而形成电压空间矢量的作用序列

![image-20241229142827933](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291428993.png)





## 直接转矩控制

### 矢量控制技术

通过旋转坐标变换和转子磁场定向实现转矩控制的解耦。

- 优点：

> 动、静态性能与直流调速系统相当

缺点：

> 采取解耦的办法控制转子磁链和转矩<br>使用电机参数多，控制特性受参数变化影响大<br>结构复杂，运算繁琐，实现成本高

### 原理分析

重新选择状态变量

- 矢量控制选取状态变量 $\left\{i_s, \Psi_r\right\}$

$T_e = n_p \left(\frac{L_m}{L_r}\right) i_{sa\beta}^T J \Psi_{ra\beta}$

其中转子磁链 $\Psi_r$ 不可测可观



因此更换变量

设转子磁链和定子磁链为正弦量电磁转矩 $T_e$ 可以表示为：

$$
T_e = k_{Te} \Psi_{sa\beta}^T J \Psi_{ra\beta}\\
= k_{Te} \left|\Psi_{sa\beta}\right| \left|\Psi_{ra\beta}\right| \sin(\theta_s - \theta_r) \\
= k_{Te} \left|\Psi_{sa\beta}\right| \left|\Psi_{ra\beta}\right| \sin \theta_{sr}
$$

由于转子的机械时间常数很大，所以转子磁链的变化远慢于定子磁链。

如果转子磁链和定子磁链的幅值被控制为一定，则电磁转矩的控制可以通过控制定子磁链的角度 $\theta_s$ 来实现。

按上式控制转矩时需要做两件事：
- 将定子磁链的幅值 $\left|\Psi_s\right|$ 控制为一定。
- 控制定子磁链角度 $\theta_s$ 来控制 $\theta_{sr}$，从而控制电磁转矩 $T_e$。



**控制定子磁链幅值和角度就是控制输出转矩**

定子磁链的控制问题本质上是**空间电压矢量**的控制问题



对于磁链在不同的区域，可以给不同的电压矢量，实现对于磁链幅值和方向的控制，控制表格如下

![image-20241229150846767](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291508834.png)

T是方向，逆时针为正

DTC**总结**

- VC的控制目标是一致的
- 转矩和磁链的控制采用双位式砰-砰控制
- 选择定子磁链作为被控量
- 对磁链和转矩是非解耦控制，结构简单，容易实现
- 没有电流换，不能做电流保护
- 转矩抖动大、逆变器开关周期不恒定



 **直接转矩控制系统和矢量控制系统特点与性能比较**

![image-20241229150925174](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291509237.png)







## 例题

1）**简述调速系统中转速控制的要求和调速指标？**

控制要求：调速、稳速、加减速三个方面

调速指标：调速范围D和静差率S



2）**简述晶闸管相控变流器直流电机调速系统的主要问题。**

- 触发脉冲相位控制
- 电流脉冲的影响及其抑制措施
- 电流波形的连续与断续
- V-M系统的机械特性



3）**典型I型系统和典型II型系统哪一个动态跟踪性能更好？哪一个抗干扰性能更好？**

典Ⅰ跟随性能好，典Ⅱ抗干扰强



4）**在转速、电流双闭环直流调速系统中，系统对于负载扰动和电网扰动，哪一个响应要快一些？为什么？**

见书本P61

5）**简述转速、电流双闭环直流调速系统的起动过程，并画出起动过程中的电流和转速波形。**

见书本P58

6）**直流电机调速有几种控制方式，分别是**

3种，调压、调阻、弱磁

7）**转速电流双闭环直流无静差调速系统在额定负载下稳定运行**

​	①、转速调节器ASR和电流调节器ACR的输入分别为多少？分别列出ASR和ACR的输出表达式。

​	答：输入：都是0   输出：ASR $U_i^* =\beta I_{dl}$ ACR：$U_c$

​	②、若减少转速给定电压Un*而其它参数保持不变，则电动机转速n的变化是增加、减少或是不变

​	答：减小

​	③、若减少转速反馈系数α而其它参数保持不变，则电动机转速n和转速反馈电压Un的变化是增加、减少或是不变

​	答：增加



**8）见右图变频调速时，a、b、c三条曲线分别代表何种情况？**

- a曲线中 $T_e$ 与 $s$ 的关系？
- 一组变频调速机械特性曲线 $T_{e\text{max}}$ 随 $\omega_1$ 的下降而______？

![image-20241229104459244](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291044298.png)

a曲线中 $T_e$ 与 $s$ 的关系

当s很小时，机械特性$T_e =f(s)$是一段直线

$$
T_e \approx 3 n_p \left(\frac{U_1}{\omega_1}\right)^2 \frac{s \omega_1}{r_2'} \propto s \quad (6.3-2a)\\
$$

当s接近于1时，则

$$
T_e \approx 3 n_p \left(\frac{U_1}{\omega_1}\right)^2 \frac{\omega_1 r_2'}{s [r_1^2 + \omega_1^2 (L_{l1} + L_{l2})^2]} \propto \frac{1}{s}
$$

a组变频调速机械特性曲线 $T_{e\text{max}}$ 随 $\omega_1$ 的下降而降低

**9）电压空间矢量 SVPWM 控制技术中，定子磁链的轨迹是__得到的，当逆变器采用 180° 导通型时，其中__个有效工作开关状态，__个无效零电压状态。**

由通过交替使用不同的电压空间矢量  6 2


**10）转差频率控制系统中，**

a) 转差频率 $\omega_s = ?$ 转差频率控制的基本概念是什么？

b) 转差频率控制的主要控制规律是什么，请图示说明？

$\omega_s = s \omega$ 

规律一：$\omega_s \le \omega_{sm} 时\quad  T_e \propto \omega_s$

规律二：$I_1 = F(\omega_s)$按照一定轨迹，可以保持 $\Phi_m$ 恒定。



**11）电压空间矢量编号图见图**

![image-20241229162346239](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412291623323.png)

a. II区的时序图？

b. 试比较下列情况下电压空间矢量与磁链会有什么变化？
   - $f = 50\text{Hz}, f_s = 600\text{Hz}$
   - $f = 100\text{Hz}, f_s = 1200\text{Hz}$
   - $f = 25\text{Hz}, f_s = 600\text{Hz}$

c. 给出相应的磁链轨迹？（以 $f = 50\text{Hz}$ 为例）

```
a: 000-010-110-111-110-010-000
绘制成P141的格式

b. 100Hz时，磁链小于50Hz，电压等于50Hz
	25Hz时，磁链等于50Hz，电压小于50Hz

c. N=$f_s$/ $f$ 即为12变形
```

**12）直接转矩控制**

a、依据下两式如何实现转矩直接控制？

$$
\begin{align*}
T_s &= \frac{3}{2} \rho \left|\Phi_s\right| \left|\Phi_r\right| \sin \gamma \\
u_s &= R_s i_s + \frac{d\Phi_s}{dt}
\end{align*}
$$

b、电压空间矢量调节 $T_e$ 的表，以上面一题分区为标准，作出对 II 区的调节表？

c、写出部分最优开关表？


```
a. 在保持转子磁链幅值不变（负载基本不变）的情况下，通过调节定转子磁链之间的夹角，即可控制转矩。
	又由于定子磁链与电压存在控制关系，因此可以通过电压去控制定子磁链来实现对转矩的控制。
```


结构决定大性能、静差

参数是调指标





