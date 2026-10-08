# 华为 AI 机考必刷 35 题：Python 注释解法

题单来源：[华为 AI 机考一周速成题单与攻略](https://my.feishu.cn/wiki/WZLxwSC5qirxfrkvi5McqQmSn8c)。以下按照题单中 🔥 必刷题的原始顺序排列，保留全部 35 个位置。

> **完成情况（2026-10-08）：** 21 道题已提供题意概述、输入输出说明、解题思路、带中文注释的 Python 程序及运行样例；14 道题在截图中显示“会员专享”，按要求跳过，仅保留题名、分类和原题链接。后续读取采用浏览器截图与视觉识别，没有复制题目正文。题意使用重新组织的概述，不是网站题面的逐字转载。

## 如何使用

每道题的代码都是一个独立程序，只复制这一道题的完整代码即可。代码采用 Python 3；标题旁标注 NumPy 的题目需要运行环境支持 NumPy，其他题目只用标准库。原题允许使用的库与评测要求应以网站为准。

常见 Python 写法：

| 写法 | 含义 |
| --- | --- |
| `list(map(int, ...))` | 把输入中的多个值转成整数列表 |
| `a[start:end]` | 切片，包含 start，不包含 end |
| `range(n)` | 依次产生 0 到 n-1 |
| `enumerate(a)` | 同时取得下标和值 |
| `zip(a, b)` | 按位置把两组数据配对 |
| `d.get(key, default)` | 字典里没有 key 时使用默认值 |
| `//`、`**` | 整除、乘方 |
| `f'{x:.2f}'` | 把 x 格式化为两位小数 |
| `@` | NumPy 的矩阵乘法 |
| `axis=0` / `axis=1` | 对二维数组按列 / 按行归约 |
| `if __name__ == '__main__':` | 直接运行该文件时执行入口函数 |

`sys.stdin.buffer.read()` 会一直读到输入结束。在终端手动输入时需要发送结束标记；更方便的是使用编辑器的标准输入功能，或从输入文件重定向。此文档不需要把 21 份程序拼在一起运行。

## 验证与边界说明

21 个程序通过了 44 组本地样例检查，其中 43 组来自页面展示的样例，1 组是页面正文中的卷积演示。另完成前两题合计 800 组穷举对照、连续分段与滑动窗口合计 400 组穷举对照、35 组剪枝枚举对照、量化组合枚举检查及 20 组梯度有限差分检查。未向网站提交代码，不宣称已通过隐藏测试。

少数题面没有说明平票、空簇或无解的处理方式，相关约定已经写在对应章节。DBSCAN 核心点数量条件存在措辞歧义，INT8 题的样例说明也有一处与输出矛盾，均保留了明确提示。

## 题目目录

| 顺序 | 分类 | 题号 | 题单中的题名 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | 动态规划 | P5567 | [激活检查点放置](https://codefun2000.com/new-p/P5567) | 已完成 · 标准库 |
| 2 | 动态规划 | P4625 | [大模型训练显存优化算法](https://codefun2000.com/p/P4625) | 已完成 · 标准库 |
| 3 | 动态规划 | P4274 | [最大能量路径](https://codefun2000.com/p/P4274) | 会员专享，已跳过 |
| 4 | 动态规划 | P3713 | [大模型分词](https://codefun2000.com/p/P3713) | 已完成 · 标准库 |
| 5 | 动态规划 | P4568 | [模型推理量化加速优化问题](https://codefun2000.com/p/P4568) | 已完成 · 标准库 |
| 6 | Kmeans | P3842 | [Yolo检测器中的anchor聚类](https://codefun2000.com/p/P3842) | 会员专享，已跳过 |
| 7 | Kmeans | P4475 | [终端款型聚类识别](https://codefun2000.com/p/P4475) | 已完成 · NumPy |
| 8 | Kmeans | P4571 | [网络流量分析](https://codefun2000.com/p/P4571) | 已完成 · NumPy |
| 9 | Kmeans | P3791 | [无线网络优化中的基站聚类分析](https://codefun2000.com/p/P3791) | 已完成 · NumPy |
| 10 | 逻辑回归 | P3872 | [华为AI方向(留学生)-基于逻辑回归的意图分类器](https://codefun2000.com/p/P3872) | 已完成 · 标准库 |
| 11 | 逻辑回归 | P4344 | [商品购买预测](https://codefun2000.com/p/P4344) | 会员专享，已跳过 |
| 12 | 决策树 | P3492 | [基于决策树预判资源调配优先级](https://codefun2000.com/p/P3492) | 会员专享，已跳过 |
| 13 | 决策树 | P4969 | [华为AI方向-随机森林交易风控算法](https://codefun2000.com/new-p/P4969) | 会员专享，已跳过 |
| 14 | 决策树 | P3480 | [F1值最优的决策树剪枝](https://codefun2000.com/p/P3480) | 已完成 · 标准库 |
| 15 | 卷积 | P4482 | [华为AI方向(留学生)-带Padding的卷积计算](https://codefun2000.com/p/P4482) | 已完成 · 标准库 |
| 16 | 并查集 | P5425 | [语义邻域可达](https://codefun2000.com/new-p/P5425) | 已完成 · 标准库 |
| 17 | 并查集 | P4238 | [利用大规模预训练模型实现智能告警聚类与故障诊断](https://codefun2000.com/p/P4238) | 会员专享，已跳过 |
| 18 | 并查集 | P4343 | [实体匹配结果合并问题](https://codefun2000.com/p/P4343) | 会员专享，已跳过 |
| 19 | 并查集 | P3874 | [数据聚类及噪声点识别](https://codefun2000.com/p/P3874) | 已完成 · 标准库 |
| 20 | 线性回归 | P4532 | [使用线性回归预测手机售价](https://codefun2000.com/p/P4532) | 已完成 · NumPy |
| 21 | 贪心 | P5263 | [最小化流水线并行峰值负载](https://codefun2000.com/new-p/P5263) | 已完成 · 标准库 |
| 22 | 贪心 | P4227 | [动态注意力掩码调度问题](https://codefun2000.com/p/P4227) | 会员专享，已跳过 |
| 23 | 贪心 | P3553 | [大模型训练MOE场景路由优化算法](https://codefun2000.com/p/P3553) | 会员专享，已跳过 |
| 24 | 滑动窗口 | P4547 | [基于样本纯净度指标的大模型训练数据清洗方法](https://codefun2000.com/p/P4547) | 已完成 · 标准库 |
| 25 | 混淆矩阵 | P4538 | [基于混淆矩阵，推导分类模型的核心评估指标](https://codefun2000.com/p/P4538) | 已完成 · 标准库 |
| 26 | Self-Attention | P3712 | [大模型Attention模块开发](https://codefun2000.com/p/P3712) | 会员专享，已跳过 |
| 27 | KNN | P3479 | [标签样本数量](https://codefun2000.com/p/P3479) | 会员专享，已跳过 |
| 28 | KNN | P4626 | [基于KNN的语音数据分类](https://codefun2000.com/p/P4626) | 已完成 · 标准库 |
| 29 | 反向传播 | P4447 | [华为AI方向(留学生)-医疗诊断模型的训练与更新](https://codefun2000.com/p/P4447) | 已完成 · NumPy |
| 30 | DFS | P3657 | [二叉树中序遍历的第k个祖先节点](https://codefun2000.com/p/P3657) | 会员专享，已跳过 |
| 31 | 线性代数 | P4277 | [华为AI方向(留学生)-人脸关键点对齐](https://codefun2000.com/p/P4277) | 会员专享，已跳过 |
| 32 | 结构化剪枝 | P4518 | [基于剪枝的神经网络模型压缩](https://codefun2000.com/p/P4518) | 已完成 · 标准库 |
| 33 | ViT | P4481 | [华为AI方向(留学生)-Vision Transformer中的Patch Embdding层实现](https://codefun2000.com/p/P4481) | 已完成 · 标准库 |
| 34 | INT8量化 | P4464 | [全连接层INT8非对称量化实现](https://codefun2000.com/p/P4464) | 已完成 · 标准库 |
| 35 | RoPE | P5124 | [华为AI方向-动态旋转位置编码](https://codefun2000.com/new-p/P5124) | 会员专享，已跳过 |

## 01. 激活检查点放置（P5567）

分类：动态规划　｜　[原题链接](https://codefun2000.com/new-p/P5567)

#### 题意概述

给定 N 层的前向计算时间和保存激活所需显存，在显存预算内选择中间层检查点，使反向阶段的重计算总代价最小。第 0 层和第 N 层都是免费边界。相邻检查点 c、d 的代价为 sum(forward_time[k-1] × (d-k))，其中 c < k < d。

#### 输入与输出

三行：N MaxMem；N 个前向耗时；N 个显存大小。N≤100，MaxMem≤1000。输出最小总代价整数。

#### 解题思路

先预处理两个前缀和，O(1) 求出任意区间代价。dp[d][w] 表示以第 d 层为最后检查点、使用至多 w 显存时的最小代价；枚举前一个检查点 c 做转移。终点 N 的显存需求为 0。

**复杂度：** O(N² × (MaxMem+1)) 时间；O(N² + N × (MaxMem+1)) 空间。

**细节与约定：** 数组下标从 0 开始，层号从 1 开始；第 N 层的输入耗时与显存不参与中间层重计算或检查点费用。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys


def minimum_cost(n, budget, forward_time, memory):
    # prefix[i]：前 i 层的前向耗时之和。
    # weighted[i]：前 i 层的「层号 × 耗时」之和。
    prefix = [0] * (n + 1)
    weighted = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + forward_time[i - 1]
        weighted[i] = weighted[i - 1] + i * forward_time[i - 1]

    # cost[c][d]：相邻检查点 c、d 之间的重计算代价。
    # 把 sum(f[k-1] * (d-k)) 拆成两项，用前缀和 O(1) 求出。
    cost = [[0] * (n + 1) for _ in range(n + 1)]
    for d in range(1, n + 1):
        for c in range(d):
            time_sum = prefix[d - 1] - prefix[c]
            weighted_sum = weighted[d - 1] - weighted[c]
            cost[c][d] = d * time_sum - weighted_sum

    # dp[d][w]：以 d 为最后一个检查点，使用至多 w 显存的最小代价。
    # 第 0 层是免费起点；无论预算是多少，起点的代价均为 0。
    inf = float('inf')
    dp = [[inf] * (budget + 1) for _ in range(n + 1)]
    dp[0] = [0] * (budget + 1)

    for d in range(1, n + 1):
        # 最后一层是免费终点，不需要为它保存检查点。
        need = 0 if d == n else memory[d - 1]
        if need > budget:
            continue
        current = dp[d]
        for c in range(d):
            previous = dp[c]
            interval_cost = cost[c][d]
            for w in range(need, budget + 1):
                candidate = previous[w - need] + interval_cost
                if candidate < current[w]:
                    current[w] = candidate

    return dp[n][budget]


def main():
    # sys.stdin.buffer.read() 一次读入标准输入；split() 按空白分割。
    # map(int, ...) 把字符串逐个转换成整数；list(...) 保存为列表。
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n, budget = data[0], data[1]
    forward_time = data[2:2 + n]  # 切片左闭右开，恰好取 n 个元素。
    memory = data[2 + n:2 + 2 * n]
    print(minimum_cost(n, budget, forward_time, memory))


if __name__ == '__main__':
    # 直接运行文件时执行 main；被测试程序导入时不会自动读取输入。
    main()
```

#### 运行样例

输入：

```text
5 3
5 10 5 10 20
1 2 1 3 1
```

输出：

```text
15
```

本地验证：本题 1 组样例检查通过；未提交网站评测。

## 02. 大模型训练显存优化算法（P4625）

分类：动态规划　｜　[原题链接](https://codefun2000.com/p/P4625)

#### 题意概述

需要至少释放 m 单位显存。有 n 个张量，每个张量可以不处理、swap 或重计算。后两种操作释放相同空间，代价各不相同；求满足空间要求的最小总代价。

#### 输入与输出

依次输入 m、n、n 个张量大小、n 个 swap 代价、n 个重计算代价。m,n<10000。输出最小代价；总空间不足时输出 error。

#### 解题思路

同一张量只保留 swap 与重计算中较小的代价。用一维 0/1 背包 dp[s] 求解；超过 m 的空间统一合并为状态 m。必须倒序更新，保证每个张量至多使用一次。

**复杂度：** 最坏 O(nm) 时间、O(m) 空间。

**细节与约定：** 不能把条件写成空间恰好等于 m。大规模时最多接近一亿次状态检查，Python 实际耗时取决于时限；没有提交网站评测。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys


def minimum_cost(required, sizes, swap_costs, recompute_costs):
    # 总空间都不够时，无论怎样选择都没有解。
    if sum(sizes) < required:
        return None

    # dp[s]：释放 s 空间的最小代价。
    # s == required 表示「已经释放至少 required 空间」。
    # 超过需求的空间都合并进这个状态，避免数组过大。
    inf = float('inf')
    dp = [inf] * (required + 1)
    dp[0] = 0
    reachable = 0  # 当前已经处理的张量最多能释放多少空间。

    # zip 把三个列表按位置配对，每次循环取出一个张量的信息。
    for size, swap_cost, recompute_cost in zip(sizes, swap_costs, recompute_costs):
        # 两种操作释放相同的空间，只需保留较便宜的一种。
        cost = min(swap_cost, recompute_cost)
        # 倒序遍历是 0/1 背包的关键：每个张量只能选择一次。
        # required 状态已经完成目标，继续加正代价不会更优，所以跳过。
        for space in range(min(reachable, required - 1), -1, -1):
            if dp[space] == inf:
                continue  # 尚不可达的状态不能参与转移。
            next_space = min(required, space + size)
            candidate = dp[space] + cost
            if candidate < dp[next_space]:
                dp[next_space] = candidate
        reachable = min(required, reachable + size)

    return dp[required]


def main():
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    required, n = data[0], data[1]
    sizes = data[2:2 + n]
    swap_costs = data[2 + n:2 + 2 * n]
    recompute_costs = data[2 + 2 * n:2 + 3 * n]
    answer = minimum_cost(required, sizes, swap_costs, recompute_costs)
    print('error' if answer is None else answer)


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
10
5
3 4 5 6 7
1 2 3 5 5
2 3 4 5 6
```

输出：

```text
6
```

本地验证：本题 2 组样例检查通过；未提交网站评测。

## 03. 最大能量路径（P4274）

分类：动态规划　｜　[原题链接](https://codefun2000.com/p/P4274)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 04. 大模型分词（P3713）

分类：动态规划　｜　[原题链接](https://codefun2000.com/p/P3713)

#### 题意概述

把小写字符串完整切分成词典里的词。每个词有基础分，相邻两个词可能有额外转移分；求最高总分。无法完整切分时输出 0。

#### 输入与输出

依次输入 text、词数 n、n 行「词 分数」、转移数 m、m 行「前词 后词 分数」。字符串长度与词数不超过 100。输出一个整数。

#### 解题思路

dp[i] 是以最后一个词为键的字典，保存切完前 i 个字符时的最高分。枚举当前匹配词和上一个词，加上基础分与指定转移分。缺省转移分为 0。

**复杂度：** 词典数 D、字符串长 L 时，状态转移最坏 O(LD²)，另计字符串匹配成本；空间 O(LD)。

**细节与约定：** 有效答案可能为负数，不能用 0 初始化所有状态。只记录位置会丢失影响转移分的上一个词。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys


def main():
    text = sys.stdin.readline().strip()
    n = int(sys.stdin.readline())
    scores = {}
    for _ in range(n):
        word, score = sys.stdin.readline().split()
        scores[word] = int(score)
    m = int(sys.stdin.readline())
    transitions = {}
    for _ in range(m):
        first, second, score = sys.stdin.readline().split()
        transitions[first, second] = int(score)

    # dp[i] 是字典：切完前 i 个字符后，最后一个词 -> 最高分。
    # 只记位置不够，因为下一个转移分数还取决于最后一个词。
    dp = [{} for _ in range(len(text) + 1)]
    dp[0][None] = 0  # None 代表还没有第一个词。
    for end in range(1, len(text) + 1):
        for word, score in scores.items():
            start = end - len(word)
            if start < 0 or text[start:end] != word:
                continue
            for previous_word, previous_score in dp[start].items():
                extra = transitions.get((previous_word, word), 0)
                candidate = previous_score + score + extra
                if word not in dp[end] or candidate > dp[end][word]:
                    dp[end][word] = candidate
    # 不能完整切分才输出 0；有效切分的最优得分可能是负数。
    print(max(dp[-1].values()) if dp[-1] else 0)


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
applepie
2
pen 3
apple 10
2
pen apple 5
pie apple 2
```

输出：

```text
0
```

本地验证：本题 2 组样例检查通过；未提交网站评测。

## 05. 模型推理量化加速优化问题（P4568）

分类：动态规划　｜　[原题链接](https://codefun2000.com/p/P4568)

#### 题意概述

模型每层有若干量化方案，每种方案给出精度损失和内存占用。每层必须选择一个方案，在总损失不超过 T 时，最小化总内存。

#### 输入与输出

第一行 L T。随后每层输入 K，并给出 K 组「位宽字符串 损失 内存」，可跨行或放在同一行。输出最优内存，保留两位小数。

#### 解题思路

分组背包：每处理一层，生成新的损失—内存状态。相同损失只保留最小内存；按损失排序后，删除损失更大且内存也不更小的被支配状态。用 Decimal 精确处理十进制损失。

**复杂度：** 若每层保留 F 个状态、每层最多 K 个方案，约 O(L × (FK + FK log(FK)))；状态数量最坏可能指数增长，题面未给 L 的上界。

**细节与约定：** 页面措辞有「小于 T」，但样例允许损失恰等于 T，因此实现为 ≤T。题面未说明无解时应输出什么；代码显式报错，不虚构输出约定。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys
from decimal import Decimal


def main():
    data = iter(sys.stdin.buffer.read().decode().split())
    layers = int(next(data))
    threshold = Decimal(next(data))
    # 用 Decimal 精确表示输入小数，避免 0.1+0.2 比 0.3 略大的问题。
    # 字典表示「累计精度损失 -> 最小内存」。每层必须恰选一个方案。
    dp = {Decimal(0): Decimal(0)}
    for _ in range(layers):
        count = int(next(data))
        options = []
        for _ in range(count):
            next(data)  # 位宽名称只起说明作用，计算用损失和内存。
            loss = Decimal(next(data))
            memory = Decimal(next(data))
            options.append((loss, memory))
        candidates = {}
        for total_loss, total_memory in dp.items():
            for loss, memory in options:
                new_loss = total_loss + loss
                if new_loss > threshold:
                    continue
                new_memory = total_memory + memory
                if new_loss not in candidates or new_memory < candidates[new_loss]:
                    candidates[new_loss] = new_memory
        # 删除被支配状态：损失更多、内存也不少的状态永远不优。
        dp = {}
        best_memory = Decimal('Infinity')
        for loss in sorted(candidates):
            memory = candidates[loss]
            if memory < best_memory:
                dp[loss] = memory
                best_memory = memory
    if not dp:
        # 页面未规定无解输出；这里报错，避免虚构一个评测答案。
        raise ValueError('题面未说明无解输出；给定数据没有可行量化组合')
    print(f'{min(dp.values()):.2f}')


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
2 0.5
2 8bit 0.2 100.0 16bit 0.1 200.0
2 8bit 0.3 150.0 16bit 0.15 300.0
```

输出：

```text
250.00
```

本地验证：本题 2 组样例检查通过；未提交网站评测。

## 06. Yolo检测器中的anchor聚类（P3842）

分类：Kmeans　｜　[原题链接](https://codefun2000.com/p/P3842)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 07. 终端款型聚类识别（P4475）

分类：Kmeans　｜　[原题链接](https://codefun2000.com/p/P4475)

#### 题意概述

每个终端由 4 个已归一化特征描述。用 K-Means 分成 k 类，初始中心取输入的前 k 个点，迭代至收敛或达到次数上限；输出各类终端数量的升序排列。

#### 输入与输出

第一行 k m n，分别为类别数、终端数和迭代次数上限。接着 m 行，每行 4 个特征。输出 k 个计数，空格分隔。

#### 解题思路

每轮把样本分配给最近中心，再计算各簇均值。比较距离平方即可，省去开平方。所有中心移动距离都小于 1e-8 时停止。使用 NumPy 的广播进行批量距离运算。

**复杂度：** 迭代 I 轮，O(Imk) 时间，距离计算中间数组 O(mk) 空间。

**细节与约定：** 输入已经归一化，不再做二次归一化。相等距离取较小中心编号；空簇保持原中心。这些边界规则题面没有明确说明，已在代码中固定。最终计数使用最后一轮的分配结果。

#### Python 解法（中文注释）

依赖：**NumPy**。

```python
import sys
import numpy as np


def main():
    data = list(map(float, sys.stdin.buffer.read().split()))
    k, m, iterations = map(int, data[:3])
    points = np.array(data[3:], dtype=float).reshape(m, 4)
    centers = points[:k].copy()  # copy 防止修改中心时影响原始样本。
    labels = np.zeros(m, dtype=int)
    for _ in range(iterations):
        # 广播生成 m×k×4 的坐标差；平方后沿特征维求和。
        # 欧氏距离开不开平方不影响谁最近，因此省略 sqrt。
        distance_squared = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distance_squared.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(k):
            members = points[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        shift = np.linalg.norm(updated - centers, axis=1)
        centers = updated
        if np.all(shift < 1e-8):
            break
    # n=0 时未执行循环：按初始化中心分配一次，作为明确的边界处理。
    if iterations == 0:
        labels = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    counts = np.bincount(labels, minlength=k)
    print(*sorted(map(int, counts)))


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
3 20 1000
0.11 0.79 0.68 0.97
1.0 0.8 0.13 0.33
0.27 0.02 0.5 0.46
0.83 0.29 0.23 0.75
0.97 0.08 0.84 0.55
0.29 0.71 0.17 0.83
0.03 0.6 0.88 0.28
0.24 0.26 0.82 0.03
0.96 0.12 0.82 0.36
0.13 0.12 0.86 0.44
0.23 0.7 0.35 0.06
0.42 0.49 0.67 0.84
0.8 0.49 0.47 0.7
0.68 0.03 0.11 0.07
0.77 0.19 0.95 0.44
0.25 0.12 0.98 0.04
0.7 0.11 0.53 0.3
0.73 0.67 0.46 0.96
0.11 0.31 0.91 0.57
0.43 0.61 0.13 0.1
```

输出：

```text
4 6 10
```

本地验证：本题 1 组样例检查通过；未提交网站评测。

## 08. 网络流量分析（P4571）

分类：Kmeans　｜　[原题链接](https://codefun2000.com/p/P4571)

#### 题意概述

给定三维网络流量样本、初始中心和迭代次数，执行 K-Means，并按初始中心的顺序输出更新后的中心坐标。

#### 输入与输出

先输入 k，再输入 k 行三维中心；随后输入迭代数 T、样本数 m 和 m 行三维样本。输出 k 行，每行三个数，保留两位小数。

#### 解题思路

严格执行 T 轮「分配最近中心 → 用簇内样本均值更新中心」。各轮内部保留完整浮点精度，只在最终输出时格式化。

**复杂度：** O(Tmk) 时间，O(mk) 距离数组空间。

**细节与约定：** 截图样例里的「10 05 60」是坐标 (10,5,60)，不是 (10,65,60)。空簇保留原中心，相等距离取较小编号；题面未明确这两个边界约定。

#### Python 解法（中文注释）

依赖：**NumPy**。

```python
import sys
import numpy as np


def main():
    values = iter(sys.stdin.buffer.read().split())
    k = int(next(values))
    centers = np.array([[float(next(values)) for _ in range(3)] for _ in range(k)])
    iterations = int(next(values))
    m = int(next(values))
    points = np.array([[float(next(values)) for _ in range(3)] for _ in range(m)])
    for _ in range(iterations):
        squared = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = squared.argmin(axis=1)  # 距离相同取编号较小的中心。
        updated = centers.copy()
        for cluster in range(k):
            members = points[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        centers = updated
        # 题目要求执行给定次数，不在这里提前停止或四舍五入。
    for center in centers:
        print(' '.join(f'{value:.2f}' for value in center))


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
3
50 25 30
60 15 60
25 75 90
3
9
50 25 30
30 50 30
60 15 60
25 75 90
10 05 60
26 15 30
32 67.5 90
80 7.5 60
20 100 90
```

输出：

```text
35.33 30.00 30.00
50.00 9.17 60.00
25.67 80.83 90.00
```

本地验证：本题 1 组样例检查通过；未提交网站评测。

## 09. 无线网络优化中的基站聚类分析（P3791）

分类：Kmeans　｜　[原题链接](https://codefun2000.com/p/P3791)

#### 题意概述

把二维基站坐标分成 k 簇，计算每个簇内样本的平均轮廓系数。找到平均系数最小的簇，输出其中心坐标。

#### 输入与输出

第一行 n k，接着 n 行整数坐标。n≤500、k≤120。输出 x,y，用逗号分隔，两位小数，采用 HALF_EVEN。

#### 解题思路

初始中心为前 k 个点；最多迭代 100 轮，所有中心移动距离≤1e-6 时停止。轮廓系数用真实欧氏距离：a 为同簇其他点的平均距离，b 为其他非空簇平均距离的最小值，s=(b-a)/max(a,b)。对各簇取平均。

**复杂度：** O(Ink + n² + nk) 时间；O(n²+nk) 空间。

**细节与约定：** 本题规定单点簇的轮廓系数为 1，与常见库默认值不同。k=1 时没有其他簇，代码约定系数为 0；空簇保持原中心，不参与最差簇选择。并列取较小编号，题面未明确这些边界。

#### Python 解法（中文注释）

依赖：**NumPy**。

```python
import sys
import numpy as np
from decimal import Decimal, ROUND_HALF_EVEN


def main():
    values = list(map(int, sys.stdin.buffer.read().split()))
    n, k = values[:2]
    points = np.array(values[2:], dtype=float).reshape(n, 2)
    centers = points[:k].copy()
    for _ in range(100):
        squared = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = squared.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(k):
            members = points[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        converged = np.all(np.linalg.norm(updated - centers, axis=1) <= 1e-6)
        centers = updated
        if converged:
            break
    groups = [np.flatnonzero(labels == c) for c in range(k)]
    # 所有样本两两之间的欧氏距离，用于轮廓系数（此处需要开平方）。
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    cluster_scores = []
    for cluster, members in enumerate(groups):
        if len(members) == 0:
            continue
        silhouettes = []
        for i in members:
            if len(members) == 1:
                silhouettes.append(1.0)  # 本题明确规定单点簇系数为 1。
                continue
            a = distances[i, members].sum() / (len(members) - 1)
            other_means = [distances[i, g].mean() for c, g in enumerate(groups) if c != cluster and len(g)]
            if not other_means:
                silhouettes.append(0.0)  # k=1 的兜底约定，题面未规定。
                continue
            b = min(other_means)
            silhouettes.append((b - a) / max(a, b) if max(a, b) else 0.0)
        cluster_scores.append((sum(silhouettes) / len(silhouettes), cluster))
    _, worst = min(cluster_scores)  # 相等时取较小的簇编号。
    # 由原始整数坐标精确求平均，用十进制实现 HALF_EVEN。
    members = groups[worst]
    output = []
    for axis in range(2):
        total = sum(int(points[i, axis]) for i in members)
        mean = Decimal(total) / Decimal(len(members))
        output.append(str(mean.quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)))
    print(','.join(output))


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
4 2
0 0
0 1
1 0
10 10
```

输出：

```text
0.33,0.33
```

本地验证：本题 2 组样例检查通过；未提交网站评测。

## 10. 华为AI方向(留学生)-基于逻辑回归的意图分类器（P3872）

分类：逻辑回归　｜　[原题链接](https://codefun2000.com/p/P3872)

#### 题意概述

用字母 A—G 的出现情况构造七维特征，训练二分类逻辑回归。权重和偏置初始为 0；学习率 0.1、20 轮、batch=1，按输入顺序更新。测试预测概率大于 0.5 时输出 1，否则输出 0。

#### 输入与输出

第一行 N M；N 行训练数据，每行为字母字符串及 0/1 标签；随后 M 行测试字符串。输出 M 行标签。

#### 解题思路

每个字母对应一个特征，出现记 1，不出现记 0。计算 sigmoid(w·x+b)，交叉熵梯度为 (预测概率-标签)×特征。每读取一个训练样本立即更新参数。

**复杂度：** O(20N×7 + M×7) 时间，O(N×7) 空间。

**细节与约定：** 重复字母只表示出现，不统计次数；不要把 batch=1 改成整批梯度，也不要打乱输入顺序。sigmoid 按正负分支计算避免溢出。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys
import math


def encode(text):
    # 这是「字母是否出现」的多热向量，重复字母不累计次数。
    return [1 if letter in text else 0 for letter in 'ABCDEFG']


def sigmoid(z):
    # 分正负计算，避免对很大的正数调用 exp 发生溢出。
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    value = math.exp(z)
    return value / (1 + value)


def main():
    n, m = map(int, sys.stdin.readline().split())
    training = []
    for _ in range(n):
        text, label = sys.stdin.readline().split()
        training.append((encode(text), int(label)))
    weights = [0.0] * 7
    bias = 0.0
    for _ in range(20):
        # batch=1：每看一个样本立即更新，不能改成整批平均梯度。
        for x, label in training:
            prediction = sigmoid(sum(w * v for w, v in zip(weights, x)) + bias)
            error = prediction - label  # sigmoid + 交叉熵的梯度。
            for i in range(7):
                weights[i] -= 0.1 * error * x[i]
            bias -= 0.1 * error
    for _ in range(m):
        x = encode(sys.stdin.readline().strip())
        probability = sigmoid(sum(w * v for w, v in zip(weights, x)) + bias)
        print(1 if probability > 0.5 else 0)


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
10 2
CBG 0
AFE 0
FGD 1
BFG 0
BBA 0
BDD 0
BEG 1
EGE 0
CAF 0
DGD 1
DBA
DAD
```

输出：

```text
0
0
```

本地验证：本题 2 组样例检查通过；未提交网站评测。

## 11. 商品购买预测（P4344）

分类：逻辑回归　｜　[原题链接](https://codefun2000.com/p/P4344)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 12. 基于决策树预判资源调配优先级（P3492）

分类：决策树　｜　[原题链接](https://codefun2000.com/p/P3492)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 13. 华为AI方向-随机森林交易风控算法（P4969）

分类：决策树　｜　[原题链接](https://codefun2000.com/new-p/P4969)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 14. F1值最优的决策树剪枝（P3480）

分类：决策树　｜　[原题链接](https://codefun2000.com/p/P3480)

#### 题意概述

已给出一棵二分类决策树和验证集。可以把任意子树剪成叶节点，叶标签使用该节点给定 label。求所有合法剪枝中验证集能取得的最大 F1。

#### 输入与输出

第一行 N M K，节点数≤100、验证样本≤300。N 行各为 left right feature threshold label，根为 1；随后 M 行 K 个特征加真实标签。输出六位小数。

#### 解题思路

先把每个样本沿原树路由，记录经过各节点的样本。节点 DP 用「预测正例总数 → 最大 TP」表示可行剪枝：直接剪当前节点，或保留分裂并合并两个子树状态。最终按 F1=2TP/(预测正例数+真实正例数) 取最大值。

**复杂度：** 粗略上界 O(NM² + MN) 时间，O(NM) 状态与样本路由空间。

**细节与约定：** 阈值条件是特征≤threshold 走左子树。不能贪心地逐个剪掉局部 F1 更差的节点，因为 F1 是全局指标。没有预测正例且没有真实正例时，约定 F1=0。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys


def main():
    data = iter(map(int, sys.stdin.buffer.read().split()))
    n, m, k = next(data), next(data), next(data)
    nodes = [None] + [tuple(next(data) for _ in range(5)) for _ in range(n)]
    samples = [tuple(next(data) for _ in range(k + 1)) for _ in range(m)]
    reached = [[] for _ in range(n + 1)]
    for sample in samples:
        node = 1
        while node:
            reached[node].append(sample)
            left, right, feature, threshold, label = nodes[node]
            if left == 0:
                break
            # 特征编号从 1 开始，所以读取 sample[feature-1]。
            node = left if sample[feature - 1] <= threshold else right

    def solve(node):
        left, right, feature, threshold, label = nodes[node]
        count = len(reached[node])
        positives = sum(sample[-1] for sample in reached[node])
        # 字典：预测为正的样本数 -> 能达到的最大真正例 TP。
        # 选择把当前整棵子树剪掉，所有经过此处的样本使用当前 label。
        result = {count: positives} if label == 1 else {0: 0}
        if left:
            a, b = solve(left), solve(right)
            # 或者保留当前分裂，左右子树各自独立选最好的剪枝。
            for predicted_a, tp_a in a.items():
                for predicted_b, tp_b in b.items():
                    predicted = predicted_a + predicted_b
                    tp = tp_a + tp_b
                    result[predicted] = max(result.get(predicted, -1), tp)
        return result

    real_positives = sum(sample[-1] for sample in samples)
    answer = 0.0
    for predicted, tp in solve(1).items():
        denominator = predicted + real_positives
        if denominator:
            # F1 = 2TP / (2TP+FP+FN) = 2TP / (预测正例数+真实正例数)。
            answer = max(answer, 2 * tp / denominator)
    print(f'{answer:.6f}')


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
7 3 2
2 3 1 50 0
4 5 2 50 0
6 7 2 50 1
0 0 0 0 0
0 0 0 0 1
0 0 0 0 0
0 0 0 0 1
30 60 1
30 30 1
60 30 1
```

输出：

```text
0.800000
```

本地验证：本题 1 组样例检查通过；未提交网站评测。

## 15. 华为AI方向(留学生)-带Padding的卷积计算（P4482）

分类：卷积　｜　[原题链接](https://codefun2000.com/p/P4482)

#### 题意概述

给定奇数边长的卷积核和方形整数图像，在图像四周补零，执行不翻转卷积核的二维卷积，保持输出与输入图像大小一致。

#### 输入与输出

第一行 m n，分别为卷积核和图像边长。接着 m 行卷积核、n 行图像。输出 n 行，每行 n 个整数。

#### 解题思路

padding=m//2。逐个输出位置枚举卷积核对应的原图坐标；超出图像边界的值视为 0，其余位置累加 图像值×核值。

**复杂度：** O(n²m²) 时间；输入存储 O(n²+m²)，输出文本另占 O(n²) 空间。

**细节与约定：** 这是深度学习常用的不翻转核的操作；输出可以为负数，也可能超过 255，不做截断。核比图像大时同样按越界为 0 处理。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys


def main():
    data = iter(map(int, sys.stdin.buffer.read().split()))
    m, n = next(data), next(data)
    kernel = [[next(data) for _ in range(m)] for _ in range(m)]
    image = [[next(data) for _ in range(n)] for _ in range(n)]
    padding = m // 2  # 奇数卷积核两侧各补 m//2 个 0。
    output = []
    for row in range(n):
        current = []
        for col in range(n):
            total = 0
            for kr in range(m):
                source_row = row + kr - padding
                if not 0 <= source_row < n:
                    continue  # 图像以外是补的 0，不必真的开新数组。
                for kc in range(m):
                    source_col = col + kc - padding
                    if 0 <= source_col < n:
                        # 深度学习里的卷积不翻转卷积核，不做像素截断。
                        total += image[source_row][source_col] * kernel[kr][kc]
            current.append(total)
        output.append(' '.join(map(str, current)))
    print('\n'.join(output))


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
3 3
1 0 -1
1 0 -1
1 0 -1
1 2 3
4 5 6
7 8 9
```

输出：

```text
-7 -4 7
-15 -6 15
-13 -4 13
```

本地验证：本题 2 组样例检查通过；未提交网站评测。

## 16. 语义邻域可达（P5425）

分类：并查集　｜　[原题链接](https://codefun2000.com/new-p/P5425)

#### 题意概述

二维点集按密度判断是否相连：半径 ε 邻域包含自己，距离≤ε；邻居数至少 MinPts 为核心点。路径只能从核心点扩展，边界点可以作为终点，噪声点不与任何点密度相连。回答 M 组点对。

#### 输入与输出

第一行 N M ε MinPts，N,M≤1000；N 行坐标；M 行 a b，点编号从 0 开始。每组输出 1 或 0。

#### 解题思路

预计算邻域并识别核心点。只合并互为邻居的核心点。每个点记录邻域中核心点所在连通分量的集合；两点的集合有交集即密度相连。

**复杂度：** 预处理 O(N²) 时间与 O(N²) 空间；每次集合交集最坏 O(N)。

**细节与约定：** 边界点不能当作桥梁连接两个核心簇。一个边界点可能属于多个核心分量的邻域，不能只为它保留一个簇编号。噪声点与自身查询也输出 0。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys


def main():
    data = iter(sys.stdin.buffer.read().split())
    n, m = int(next(data)), int(next(data))
    eps, min_points = float(next(data)), int(next(data))
    points = [(float(next(data)), float(next(data))) for _ in range(n)]
    neighbors = [[i] for i in range(n)]  # 本题邻域包含自己。
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if dx * dx + dy * dy <= eps * eps:
                neighbors[i].append(j)
                neighbors[j].append(i)
    core = [len(group) >= min_points for group in neighbors]
    parent = list(range(n))

    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]  # 路径压缩。
            x = parent[x]
        return x

    for i in range(n):
        if core[i]:
            for j in neighbors[i]:
                if core[j]:
                    parent[find(i)] = find(j)
    # 一个边界点可能同时邻接多个互不连通的核心簇。
    # 不能把边界点当桥梁把两个核心簇合并！
    memberships = [{find(j) for j in group if core[j]} for group in neighbors]
    answers = []
    for _ in range(m):
        a, b = int(next(data)), int(next(data))
        # 若存在共同核心簇，就是密度相连；噪声点的集合为空。
        answers.append('1' if memberships[a] & memberships[b] else '0')
    print('\n'.join(answers))


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
6 4 2.0 3
0.0 0.0
1.2 0.0
0.4 1.0
12.0 0.0
12.5 0.4
6.0 6.0
0 2
0 3
3 4
1 5
```

输出：

```text
1
0
0
0
```

本地验证：本题 2 组样例检查通过；未提交网站评测。

## 17. 利用大规模预训练模型实现智能告警聚类与故障诊断（P4238）

分类：并查集　｜　[原题链接](https://codefun2000.com/p/P4238)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 18. 实体匹配结果合并问题（P4343）

分类：并查集　｜　[原题链接](https://codefun2000.com/p/P4343)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 19. 数据聚类及噪声点识别（P3874）

分类：并查集　｜　[原题链接](https://codefun2000.com/p/P3874)

#### 题意概述

对二维或三维数据进行密度聚类，输出核心点连通簇的数量及噪声点数量。该页面半径条件写为距离严格小于 eps。

#### 输入与输出

第一行 eps min_samples x；随后 x 行坐标，每行维数为 2 或 3。输出「簇数 噪声数」。

#### 解题思路

计算邻域，找核心点；只通过核心点—核心点的边建立连通分量。非核心点只要邻接一个核心点就是边界点，否则为噪声。

**复杂度：** O(x²d) 时间、O(x²) 空间，d 为 2 或 3。

**细节与约定：** 重要歧义：页面写核心邻居数量「大于 min_samples」，标准 DBSCAN 通常为「≥ min_samples」。本实现采用标准 ≥、邻域包含自己，两组截图样例均通过，但它们不能区分这两种规则。若评测严格采用「>」，需把核心条件的 >= 改为 >；未验证隐藏测试。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys


def main():
    # 每行一个样本，维数为 2 或 3；保留按行读入以确定维数。
    eps_text, minimum_text, n_text = sys.stdin.readline().split()
    eps, minimum, n = float(eps_text), int(minimum_text), int(n_text)
    points = [list(map(float, sys.stdin.readline().split())) for _ in range(n)]
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            squared = sum((a - b) ** 2 for a, b in zip(points[i], points[j]))
            if squared < eps * eps:  # 本题的半径条件写的是严格小于。
                neighbors[i].append(j)
                if i != j:
                    neighbors[j].append(i)
    # 按标准 DBSCAN：邻域含自身，数量至少 min_samples 时为核心点。
    core = [len(group) >= minimum for group in neighbors]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        if core[i]:
            for j in neighbors[i]:
                if core[j]:
                    parent[find(i)] = find(j)
    clusters = len({find(i) for i in range(n) if core[i]})
    # 非核心点只要在某核心点的邻域里就是边界点，否则是噪声。
    noise = sum(not any(core[j] for j in group) for group in neighbors)
    print(clusters, noise)


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
2 2 10
0 0
3 3
6 6
9 9
12 12
2 6
6 2
9 5
5 9
10 2
```

输出：

```text
0 10
```

本地验证：本题 2 组样例检查通过；未提交网站评测。

## 20. 使用线性回归预测手机售价（P4532）

分类：线性回归　｜　[原题链接](https://codefun2000.com/p/P4532)

#### 题意概述

用手机的三项特征和已知售价拟合含截距的线性回归模型，预测新手机售价，并四舍五入到整数。题目保证最小二乘有唯一解。

#### 输入与输出

依次输入已知手机数 K、K 组「特征1 特征2 特征3 售价」、待预测数 N、N 组三项特征。可按任意空白分隔。输出 N 个整数，空格分隔。

#### 解题思路

为特征矩阵增加一列 1，把截距与三个权重一起求解。用 np.linalg.lstsq 计算最小二乘，比显式求逆更稳定；预测用测试特征矩阵乘权重。

**复杂度：** 特征维数固定为 4，约 O(K+N) 时间与空间。

**细节与约定：** round 默认就近取偶，题面要求四舍五入，所以用 Decimal 的 ROUND_HALF_UP。不能用 int 直接截断；也不要忘记截距。

#### Python 解法（中文注释）

依赖：**NumPy**。

```python
import sys
import numpy as np
from decimal import Decimal, ROUND_HALF_UP


def main():
    data = iter(sys.stdin.buffer.read().split())
    k = int(next(data))
    training = np.array([[float(next(data)) for _ in range(4)] for _ in range(k)])
    n = int(next(data))
    testing = np.array([[float(next(data)) for _ in range(3)] for _ in range(n)])
    # 增加一列 1，把截距 b 也作为一个待学习的参数。
    x = np.column_stack((np.ones(k), training[:, :3]))
    y = training[:, 3]
    # 最小二乘用 lstsq，避免显式求逆放大浮点误差。
    weights, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    test_x = np.column_stack((np.ones(n), testing))
    prediction = test_x @ weights  # @ 是矩阵乘法。
    # 题面要求四舍五入取整，不能直接 int（会向 0 截断）。
    answer = [int(Decimal(str(value)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)) for value in prediction]
    print(*answer)


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
4
30 23 24 1999 55 53 46 2999 68 85 78 3999 113 90 103 4999
1
126 114 143
```

输出：

```text
6009
```

本地验证：本题 1 组样例检查通过；未提交网站评测。

## 21. 最小化流水线并行峰值负载（P5263）

分类：贪心　｜　[原题链接](https://codefun2000.com/new-p/P5263)

#### 题意概述

把按顺序排列的 L 层工作量分成 k 个非空连续段，分别分给计算节点，使各段工作量之和的最大值最小。L≤128，各工作量为正整数。

#### 输入与输出

依次输入 L、L 个工作量、k。输出最小峰值整数。

#### 解题思路

对峰值做二分。给定峰值限制，贪心让每段装尽可能多的连续层，得到所需的最少段数；段数≤k 就可行。正工作量保证可继续拆分成恰好 k 段。

**复杂度：** O(L log(sum(weights))) 时间，O(1) 额外空间，不计输入。

**细节与约定：** 不能打乱工作量顺序；二分下界至少是最大单层工作量，不能只取平均值。k=1 时答案为总和，k=L 时答案为最大值。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys


def minimum_peak(weights, k):
    def feasible(limit):
        groups, current = 1, 0
        for weight in weights:
            if current + weight > limit:
                groups += 1
                current = weight
            else:
                current += weight
        return groups <= k

    # 峰值不能小于最大单层计算量，也不必大于所有层的总和。
    low, high = max(weights), sum(weights)
    while low < high:
        middle = (low + high) // 2  # // 是整除，返回整数。
        if feasible(middle):
            high = middle  # middle 可行，继续尝试更小的峰值。
        else:
            low = middle + 1
    return low


def main():
    data = list(map(int, sys.stdin.buffer.read().split()))
    n = data[0]
    print(minimum_peak(data[1:1 + n], data[1 + n]))


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
1
7
1
```

输出：

```text
7
```

本地验证：本题 4 组样例检查通过；未提交网站评测。

## 22. 动态注意力掩码调度问题（P4227）

分类：贪心　｜　[原题链接](https://codefun2000.com/p/P4227)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 23. 大模型训练MOE场景路由优化算法（P3553）

分类：贪心　｜　[原题链接](https://codefun2000.com/p/P3553)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 24. 基于样本纯净度指标的大模型训练数据清洗方法（P4547）

分类：滑动窗口　｜　[原题链接](https://codefun2000.com/p/P4547)

当前网站标题：**纯净度指标：最长连续非重复片段**。题号与题单链接一致，顺序仍按原题单保留。

#### 题意概述

给定由 ASCII 字母和数字组成、没有空格的字符串，求最长不含重复字符的连续子串长度。长度可能达到 10^7。

#### 输入与输出

输入一行字符串。输出最长长度整数。

#### 解题思路

滑动窗口配合每个字符最近出现的位置。如果字符的旧位置还在窗口中，左边界直接跳到旧位置之后。每个右端点只处理一次。

**复杂度：** O(L) 时间、O(128) 额外空间；输入字符串本身占 O(L)。

**细节与约定：** 子串必须连续，区分大小写。左边界只能向右移动。使用 bytes 读取，迭代时直接得到 ASCII 数值，避免给千万字符创建大量临时字符串。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys


def longest_unique(text):
    # 输入只有 ASCII 字母和数字，固定大小数组比字典更省内存。
    last = [-1] * 128
    left = 0
    answer = 0
    for right, char in enumerate(text):
        # 如果旧位置仍在窗口中，把左边界跳到旧位置的下一位。
        left = max(left, last[char] + 1)
        last[char] = right
        answer = max(answer, right - left + 1)
    return answer


def main():
    # bytes 迭代时直接得到 ASCII 整数，适合处理千万字符的输入。
    text = sys.stdin.buffer.readline().rstrip(b'\r\n')
    print(longest_unique(text))


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
z
```

输出：

```text
1
```

本地验证：本题 4 组样例检查通过；未提交网站评测。

## 25. 基于混淆矩阵，推导分类模型的核心评估指标（P4538）

分类：混淆矩阵　｜　[原题链接](https://codefun2000.com/p/P4538)

当前网站标题：**多分类混淆矩阵的加权指标计算**。题号与题单链接一致，顺序仍按原题单保留。

#### 题意概述

根据每个样本的预测类别、真实类别和各类别权重，分别计算各类 Precision、Recall、F1，再求对应加权和。

#### 输入与输出

共三行：预测类别序列、真实类别序列、类别权重序列。类别编号从 0 开始，权重个数决定类别数。输出加权 P、R、F1，空格分隔，各保留两位小数。

#### 解题思路

累计每类 TP、预测数量与真实数量。P=TP/预测数量，R=TP/真实数量，F1=2PR/(P+R)；分母为零时该指标为 0。最后按给定权重逐类累加。

**复杂度：** 样本数 S、类别数 C 时，O(S+C) 时间，O(C) 额外计数空间。

**细节与约定：** 加权 F1 必须先算各类 F1 再加权；不能把加权 P 和加权 R 再合成一个 F1。不要擅自忽略未出现的类别或重新归一化权重。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys


def main():
    predicted = list(map(int, sys.stdin.readline().split()))
    truth = list(map(int, sys.stdin.readline().split()))
    weights = list(map(float, sys.stdin.readline().split()))
    n = len(weights)
    tp = [0] * n
    predicted_count = [0] * n
    true_count = [0] * n
    for p, t in zip(predicted, truth):
        predicted_count[p] += 1
        true_count[t] += 1
        if p == t:
            tp[p] += 1
    precision = recall = f1 = 0.0
    for i, weight in enumerate(weights):
        p = tp[i] / predicted_count[i] if predicted_count[i] else 0.0
        r = tp[i] / true_count[i] if true_count[i] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        precision += weight * p
        recall += weight * r
        f1 += weight * f
    # 加权 F1 是各类 F1 的加权和，不是加权 P、R 的调和平均。
    print(f'{precision:.2f} {recall:.2f} {f1:.2f}')


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
0
0
1.0
```

输出：

```text
1.00 1.00 1.00
```

本地验证：本题 3 组样例检查通过；未提交网站评测。

## 26. 大模型Attention模块开发（P3712）

分类：Self-Attention　｜　[原题链接](https://codefun2000.com/p/P3712)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 27. 标签样本数量（P3479）

分类：KNN　｜　[原题链接](https://codefun2000.com/p/P3479)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 28. 基于KNN的语音数据分类（P4626）

分类：KNN　｜　[原题链接](https://codefun2000.com/p/P4626)

#### 题意概述

已知若干三维语音特征向量及类别标签，对一个测试向量使用 KNN 分类：找距离最近的 K 个样本，以多数票决定类别。

#### 输入与输出

第一行 N K；随后 N 行三个浮点特征加整数标签；最后一行三个测试特征。输出类别整数。题面示意行号与 N 不一致，按 N 读取。

#### 解题思路

计算距离平方，按距离排序，统计前 K 个邻居的标签频次，再选票数最多的类别。

**复杂度：** O(N log N) 时间，O(N) 空间。

**细节与约定：** 题面未规定相等距离及平票规则。代码明确约定：相等距离保留输入顺序，平票取最小类别编号。样例能验证一般情况，不能验证这些约定；未验证隐藏测试。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys
from collections import Counter


def main():
    data = iter(sys.stdin.buffer.read().split())
    n, k = int(next(data)), int(next(data))
    training = []
    for i in range(n):
        vector = [float(next(data)) for _ in range(3)]
        label = int(next(data))
        training.append((vector, label))
    target = [float(next(data)) for _ in range(3)]
    distances = []
    for index, (vector, label) in enumerate(training):
        squared = sum((a - b) ** 2 for a, b in zip(vector, target))
        distances.append((squared, index, label))
    distances.sort()  # 距离相同时保留输入顺序。
    counts = Counter(label for _, _, label in distances[:k])
    # 题面没给平票规则；这里明确约定票数相同时取最小类别编号。
    winner = min(counts, key=lambda label: (-counts[label], label))
    print(winner)


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
10 3
0.5 0.3 0.4 0
0.6 0.2 0.5 0
0.4 0.3 0.3 0
0.7 0.4 0.6 0
2.1 2.3 2.2 1
2.3 2.2 2.4 1
2.2 2.4 2.3 1
4.5 4.3 4.4 2
4.4 4.5 4.6 2
4.6 4.4 4.5 2
2.2 2.1 2.3
```

输出：

```text
1
```

本地验证：本题 1 组样例检查通过；未提交网站评测。

## 29. 华为AI方向(留学生)-医疗诊断模型的训练与更新（P4447）

分类：反向传播　｜　[原题链接](https://codefun2000.com/p/P4447)

#### 题意概述

序列 X 先通过线性 MLP 矩阵 W_mlp，再通过分类矩阵 W_cls；对序列维取平均得到输出，与真实目标计算 MSE，并对两组权重执行一次 SGD 更新。明确不使用 softmax，也没有 bias 或其他激活。

#### 输入与输出

五行逗号分隔：L,D,K,学习率；K 个目标值；L×D 个 X；D×D 个 W_mlp；D×K 个 W_cls。依次输出预测、MSE、更新后 MLP、更新后分类权重，仍逗号分隔，两位小数。

#### 解题思路

h=mean(XW_mlp)，pred=hW_cls。d_pred=2(pred-target)/K；dW_cls=h 与 d_pred 的外积；dW_mlp=mean(X) 与 d_pred·W_cls.T 的外积。使用旧权重先算完全部梯度，再同时更新。

**复杂度：** O(LD²+DK) 时间，O(LD+D²+DK) 空间。

**细节与约定：** 输出的是更新前的预测和损失、更新后的参数。梯度包含 MSE 对类别数 K 的平均。不要先更新分类权重再用新权重计算另一层梯度。

#### Python 解法（中文注释）

依赖：**NumPy**。

```python
import sys
import numpy as np


def main():
    # 本题输入逗号分隔，用 replace 把逗号变为空格，再分割。
    data = iter(sys.stdin.buffer.read().decode().replace(',', ' ').split())
    length, dimension, classes = int(next(data)), int(next(data)), int(next(data))
    learning_rate = float(next(data))
    target = np.array([float(next(data)) for _ in range(classes)])
    x = np.array([float(next(data)) for _ in range(length * dimension)]).reshape(length, dimension)
    w_mlp = np.array([float(next(data)) for _ in range(dimension * dimension)]).reshape(dimension, dimension)
    w_cls = np.array([float(next(data)) for _ in range(dimension * classes)]).reshape(dimension, classes)
    # 两层都是线性层，无偏置、无激活函数；不要自行添加 softmax。
    hidden = x @ w_mlp
    mean_hidden = hidden.mean(axis=0)
    prediction = mean_hidden @ w_cls
    error = prediction - target
    loss = np.mean(error ** 2)
    grad_prediction = 2 * error / classes  # MSE 对输出的导数。
    grad_cls = np.outer(mean_hidden, grad_prediction)
    # 必须用更新前的分类权重计算另一层梯度，最后同时更新。
    grad_mean_hidden = grad_prediction @ w_cls.T
    grad_mlp = np.outer(x.mean(axis=0), grad_mean_hidden)
    new_mlp = w_mlp - learning_rate * grad_mlp
    new_cls = w_cls - learning_rate * grad_cls

    def show(array):
        print(','.join(f'{value:.2f}' for value in np.asarray(array).reshape(-1)))

    show(prediction)
    show([loss])
    show(new_mlp)
    show(new_cls)


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
2,2,3,0.1
1.0,0.0,0.0
1.0,2.0,3.0,4.0
1.0,1.0,1.0,1.0
1.0,0.0,0.0,0.0,0.0,0.0
```

输出：

```text
5.00,0.00,0.00
5.33
0.47,1.00,0.20,1.00
-0.33,0.00,0.00,-1.33,0.00,0.00
```

本地验证：本题 1 组样例检查通过；未提交网站评测。

## 30. 二叉树中序遍历的第k个祖先节点（P3657）

分类：DFS　｜　[原题链接](https://codefun2000.com/p/P3657)

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 31. 华为AI方向(留学生)-人脸关键点对齐（P4277）

分类：线性代数　｜　[原题链接](https://codefun2000.com/p/P4277)

当前网站标题：**人脸对齐仿射重采样**。题号与题单链接一致，顺序仍按原题单保留。

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。

## 32. 基于剪枝的神经网络模型压缩（P4518）

分类：结构化剪枝　｜　[原题链接](https://codefun2000.com/p/P4518)

当前网站标题：**稀疏模型的行级剪枝与分类预测**。题号与题单链接一致，顺序仍按原题单保留。

#### 题意概述

按权重矩阵各输入特征对应行的 L1 范数进行结构化剪枝，同时删掉输入 X 的相应列；用剪枝后的矩阵做分类，输出各样本的类别。

#### 输入与输出

第一行 n d c；n 行 X 的 d 个浮点数；d 行 W 的 c 个浮点数；最后输入 ratio。n,d,c≤64，ratio∈[0,1]。输出 n 个从 0 开始的类别编号。

#### 解题思路

删除数量 floor(ratio×d)；ratio>0 且结果为 0 时，至少删除 1 行。按 L1 范数和原始下标排序，删除最小的行。剩余特征与权重做线性乘法，再求 argmax。

**复杂度：** O(dc+d log d+ndc) 时间，O(nd+dc) 空间。

**细节与约定：** softmax 不改变 argmax，因此可以省略。范数或输出分数相同时取较小编号。ratio=1 删除全部特征，所有类别分数均为 0，输出类别 0。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys
from decimal import Decimal, ROUND_FLOOR


def main():
    data = iter(sys.stdin.buffer.read().decode().split())
    n, d, c = int(next(data)), int(next(data)), int(next(data))
    x = [[float(next(data)) for _ in range(d)] for _ in range(n)]
    weights = [[float(next(data)) for _ in range(c)] for _ in range(d)]
    ratio = Decimal(next(data))
    remove_count = int((ratio * d).to_integral_value(rounding=ROUND_FLOOR))
    if ratio > 0 and remove_count == 0:
        remove_count = 1
    # 按每行的 L1 范数排序；范数相同取原始行号较小的。
    order = sorted(range(d), key=lambda i: (sum(abs(v) for v in weights[i]), i))
    removed = set(order[:remove_count])
    kept = [i for i in range(d) if i not in removed]
    predictions = []
    for sample in x:
        logits = [sum(sample[i] * weights[i][j] for i in kept) for j in range(c)]
        # softmax 保持大小顺序，求 argmax 不必真的计算 exp。
        # 删除全部特征时 logits 全是 0，取最小编号 0。
        predictions.append(max(range(c), key=lambda j: logits[j]))
    print(*predictions)


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
2 2 2
1 0
0 1
1 -1
-2 3
0
```

输出：

```text
0 1
```

本地验证：本题 3 组样例检查通过；未提交网站评测。

## 33. 华为AI方向(留学生)-Vision Transformer中的Patch Embdding层实现（P4481）

分类：ViT　｜　[原题链接](https://codefun2000.com/p/P4481)

当前网站标题：**计算视觉 Transformer 分块嵌入后的 token 形状**。题号与题单链接一致，顺序仍按原题单保留。

#### 题意概述

只需计算 ViT 的 token 输出形状：方形图像切成不重叠 patch，线性映射到 embedding_dim，再加一个 cls token。无需实现完整 Transformer。

#### 输入与输出

一行四个整数：img_size patch_size channels embedding_dim。输出「token总数 embedding_dim」。

#### 解题思路

每条边可放 img_size//patch_size 个完整 patch；总 patch 数是其平方，再加 1 个 cls token。通道数不改变映射后的第二维。

**复杂度：** O(1) 时间与空间。

**细节与约定：** 不能把不够一个完整 patch 的边缘再算成一个 patch；本题用整除。题单中的旧标题提到 Embedding 层，但目前实际题目只要求计算形状。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys


def main():
    image_size, patch_size, channels, embedding_dim = map(int, sys.stdin.buffer.read().split())
    patches_per_side = image_size // patch_size  # // 取整，不足一块的边缘不计。
    token_count = patches_per_side ** 2 + 1  # **2 表示平方，+1 是 cls token。
    # 通道数影响展平后的输入长度，但不影响线性映射后的形状。
    print(token_count, embedding_dim)


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
1 1 1 8
```

输出：

```text
2 8
```

本地验证：本题 3 组样例检查通过；未提交网站评测。

## 34. 全连接层INT8非对称量化实现（P4464）

分类：INT8量化　｜　[原题链接](https://codefun2000.com/p/P4464)

#### 题意概述

分别对输入向量 x 和整张权重矩阵 W 做 INT8 非对称量化。先输出 x_quant·W_quant.T 的整数结果，再反量化，计算相对原始浮点结果的 MSE×100000，四舍五入成整数。无偏置。

#### 输入与输出

依次输入 n、n 个输入值、m n、m 行各 n 个权重。输出两行：m 个整数计算结果；缩放后的误差整数。

#### 解题思路

scale=(max-min)/255；量化为 clamp(round((v-min)/scale)-128,-128,127)，其中 round 采用就近取偶。常量张量统一量化为 -128，反量化回原值。用 Fraction 精确实现输入十进制和量化，避免半整数舍入误差。

**复杂度：** 算术操作数 O(mn)，存储 O(mn)；Fraction 的实际耗时还受分子分母位数影响。题面未给矩阵大小上界。

**细节与约定：** W 的 min/max 来自整张矩阵，不能逐行量化。整数累加不能用 int8，否则会溢出。量化取偶与最终误差四舍五入是两种不同规则。页面样例3的说明末尾写 MSE 为0，与展示输出矛盾；按公式计算能得到展示输出 812812500，本解法依公式实现。

#### Python 解法（中文注释）

依赖：**Python 标准库**。

```python
import sys
from fractions import Fraction


def quantize(values):
    low, high = min(values), max(values)
    if high == low:
        return [-128] * len(values), [low] * len(values)
    scale = (high - low) / 255
    # Fraction 精确表示十进制输入；round(Fraction) 实现就近取偶。
    # 这可以正确处理恰好为 127.5 等半整数的量化位置。
    quantized = [max(-128, min(127, round((v - low) / scale) - 128)) for v in values]
    restored = [(q + 128) * scale + low for q in quantized]
    return quantized, restored


def main():
    data = iter(sys.stdin.buffer.read().decode().split())
    n = int(next(data))
    x = [Fraction(next(data)) for _ in range(n)]
    m, weight_columns = int(next(data)), int(next(data))
    if weight_columns != n:
        raise ValueError('权重列数必须与输入向量长度一致')
    weights = [[Fraction(next(data)) for _ in range(n)] for _ in range(m)]
    x_quantized, x_restored = quantize(x)
    # 权重必须整张矩阵一起量化，不能逐行单独计算 scale。
    flat_weights = [value for row in weights for value in row]
    w_quantized, w_restored = quantize(flat_weights)
    integer_output = []
    squared_error = Fraction(0)
    for row in range(m):
        begin = row * n
        q_row = w_quantized[begin:begin + n]
        restored_row = w_restored[begin:begin + n]
        # Python 整数不会产生 int8 溢出，累加结果保留完整整数。
        integer_output.append(sum(a * b for a, b in zip(x_quantized, q_row)))
        original_y = sum(a * b for a, b in zip(x, weights[row]))
        restored_y = sum(a * b for a, b in zip(x_restored, restored_row))
        squared_error += (original_y - restored_y) ** 2
    scaled_mse = squared_error / m * 100000
    # 最终 MSE 非负，四舍五入用 floor(value + 1/2)，与量化取偶不同。
    rounded_error = (2 * scaled_mse.numerator + scaled_mse.denominator) // (2 * scaled_mse.denominator)
    print(*integer_output)
    print(rounded_error)


if __name__ == '__main__':
    main()
```

#### 运行样例

输入：

```text
1
3.14
1 1
2.71
```

输出：

```text
16384
0
```

本地验证：本题 4 组样例检查通过；未提交网站评测。

## 35. 华为AI方向-动态旋转位置编码（P5124）

分类：RoPE　｜　[原题链接](https://codefun2000.com/new-p/P5124)

当前网站标题：**长度自适应的动态 RoPE 位置编码**。题号与题单链接一致，顺序仍按原题单保留。

> **已跳过：会员专享。** 本次打开题目页面后，截图显示题面被会员提示遮挡。按要求不读取遮挡内容，不编写未经核对的解法；原题位置与链接保留。
