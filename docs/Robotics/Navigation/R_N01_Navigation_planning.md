---
comments: true
---

# 导航规划

!!!note 简介
    包含路径规划、避障规划、轨迹规划三个部分<br>
    最后有一个实践项目<br>

导航规划通常分解为三个问题

1. **路径规划**：根据所给定的地图和目标位置，规划一条使机器人到达目标位置的路径（只考虑工作空间的几何约束，不考虑机器人的运动学模型和约束）。

2. **避障规划**：根据所得到的实时传感器测量信息，调整路径/轨迹以避免发生碰撞。

3. **轨迹生成**：根据机器人的运动学模型和约束，寻找适当的控制命令，将可行路径转化为可行轨迹。

## 路径规划

**全局的，一般只规划一次**



## 避障规划

无完备性、无最优性


## 轨迹规划

**局部的，只处理视觉范围内障碍物造成的轨迹规划**



## 实践项目