---
tags: [机器学习, 李航, 监督学习]
category: "章节"
chapter: "Ch08-Boosting"
pian: "第一篇-监督学习"
---
## 第 8 章 Boosting

Boosting是一种常用的机器学习方法，应用广泛且有效。在分类问题中，它通过改变训练样本的权重学习多个分类器，并将这些分类器进行线性组合，提高分类的性能。

本章首先介绍 Boosting 的思路和代表性的 Boosting 算法 AdaBoost；然后通过训练误差分析探讨 AdaBoost 为什么能够提高学习精度，并且从前向分步加法模型的角度解释 AdaBoost；最后叙述 Boosting 更具体的实例——提升树（Boosting tree）。AdaBoost 算法是在 1995 年由 Freund 和 Schapire 提出的，提升树是 2000 年由 Friedman 等人提出的。

## 8.1 AdaBoost 算法

## 8.1.1 Boosting 的基本思路

Boosting 基于这样一种思想：对于一个复杂任务来说，将多个专家的判断进行适当的综合所得出的判断要比其中任何一个专家单独的判断好。实际上，就是“三个臭皮匠顶个诸葛亮”的道理。

历史上，Kearns 和 Valiant 首先提出了“强可学习”（strongly learnable）和“弱可学习”（weakly learnable）的概念。指出：在概率近似正确（probably approximately correct, PAC）学习的框架中，对于一个概念（一个类），如果存在一个多项式的学习算法能够学习它，并且正确率很高，那么就称这个概念是强可学习的；如果存在一个多项式的学习算法能够学习它，学习的正确率仅比随机猜测略好，那么就称这个概念是弱可学习的。非常有趣的是 Schapire 后来证明强可学习与弱可学习是等价的，也就是说，在 PAC 学习的框架下，一个概念是强可学习的充分必要条件是这个概念是弱可学习的。

这样一来，问题便成为，在学习中，如果已经发现了“弱学习算法”，那么能否将它提升（boost）为“强学习算法”。大家知道，发现弱学习算法通常要比发现强学习算法容易得多。那么如何具体实施提升，便成为开发 Boosting 时所要解决的问题。关于 Boosting 的研究很多，有很多算法被提出，最具代表性的是 AdaBoost 算法（AdaBoost algorithm）。

对于分类问题而言, 给定一个训练样本集, 求比较粗糙的分类规则 (弱分类器) 要比求精确的分类规则 (强分类器) 容易得多。Boosting 就是从弱学习算法出发, 反复学习, 得到一系列弱分类器 (又称为基本分类器), 然后组合这些弱分类器, 构成一个强分类器。大多数的 Boosting 都是改变训练数据的概率分布 (训练数据的权值分布), 针对不同的训练数据分布调用弱学习算法学习一系列弱分类器。

这样，对 Boosting 来说，有两个问题需要回答：一是在每一轮如何改变训练数据的权值或概率分布；二是如何将弱分类器组合成一个强分类器。关于第 1 个问题，AdaBoost 的做法是提高那些被前一轮弱分类器错误分类样本的权值，而降低那些被正确分类样本的权值。这样一来，那些没有得到正确分类的数据，由于其权值的加大而受到后一轮的弱分类器的更大关注。于是，分类问题被一系列的弱分类器“分而治之”。至于第 2 个问题，即弱分类器的组合，AdaBoost 采取加权多数表决的方法。具体地，加大分类误差率小的弱分类器的权值，使其在表决中起较大的作用；减小分类误差率大的弱分类器的权值，使其在表决中起较小的作用。

AdaBoost 的巧妙之处就在于它将这些想法自然且有效地实现在一种算法里。

## 8.1.2 AdaBoost 算法

现在叙述 AdaBoost 算法。假设给定一个二类分类的训练数据集

$$
T = \{(x _ {1}, y _ {1}), (x _ {2}, y _ {2}), \dots , (x _ {N}, y _ {N}) \}
$$

其中，每个样本点由实例与标记组成。实例 $x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}$ ，标记 $y_{i} \in \mathcal{Y} = \{-1, +1\}$ ， $\mathcal{X}$ 是实例空间， $\mathcal{Y}$ 是标记集合。AdaBoost 利用以下算法，从训练数据中学习一系列弱分类器或基本分类器，并将这些弱分类器线性组合成为一个强分类器。

算法 8.1 (AdaBoost)

输入: 训练数据集 $T = \{(x_{1}, y_{1}), (x_{2}, y_{2}), \cdots, (x_{N}, y_{N})\}$ , 其中 $x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}$ , $y_{i} \in \mathcal{Y} = \{-1, +1\}$ ; 弱学习算法。

输出：最终分类器 $G(x)$ 。

(1) 初始化训练数据的权值分布

$$
D _ {1} = (w _ {1 1}, \dots , w _ {1 i}, \dots , w _ {1 N}), \quad w _ {1 i} = \frac {1}{N}, \quad i = 1, 2, \dots , N
$$

(2) 对 $m = 1, 2, \cdots, M,$

(a) 使用具有权值分布 $D_{m}$ 的训练数据集学习，得到基本分类器：

$$
G _ {m} (x): \mathcal {X} \rightarrow \{- 1, + 1 \}
$$

(b) 计算 $G_{m}(x)$ 在训练数据集上的分类误差率:

$$
e _ {m} = \sum_ {i = 1} ^ {N} P (G _ {m} (x _ {i}) \neq y _ {i}) = \sum_ {i = 1} ^ {N} w _ {m i} I (G _ {m} (x _ {i}) \neq y _ {i})\tag{8.1}
$$

(c) 计算 $G_{m}(x)$ 的系数:

$$
\alpha_ {m} = \frac {1}{2} \log \frac {1 - e _ {m}}{e _ {m}}\tag{8.2}
$$

这里的对数是自然对数。

(d) 更新训练数据集的权值分布:

$$
D _ {m + 1} = \left(w _ {m + 1, 1}, \dots , w _ {m + 1, i}, \dots , w _ {m + 1, N}\right)\tag{8.3}
$$

$$
w _ {m + 1, i} = \frac {w _ {m i}}{Z _ {m}} \exp (- \alpha_ {m} y _ {i} G _ {m} (x _ {i})), \quad i = 1, 2, \dots , N\tag{8.4}
$$

这里， $Z_{m}$ 是规范化因子，且

$$
Z _ {m} = \sum_ {i = 1} ^ {N} w _ {m i} \exp (- \alpha_ {m} y _ {i} G _ {m} (x _ {i}))\tag{8.5}
$$

它使 $D_{m + 1}$ 成为一个概率分布。

(3) 构建基本分类器的线性组合

$$
f (x) = \sum_ {m = 1} ^ {M} \alpha_ {m} G _ {m} (x)\tag{8.6}
$$

得到最终分类器:

$$
\begin{array}{l} {G (x) = \mathrm{sign} (f (x))} \\ {= \mathrm{sign} \left(\sum_ {m = 1} ^ {M} \alpha_ {m} G _ {m} (x)\right)} \end{array}\tag{8.7}
$$

对 AdaBoost 算法作如下说明:

(1) 步骤 (1) 假设训练数据集具有均匀的权值分布, 即每个训练样本在基本分类器的学习中作用相同, 这一假设保证第 1 步能够在原始数据上学习基本分类器 $G_{1}(x)$ 。

(2) 步骤 (2) 中 AdaBoost 反复学习基本分类器, 在每一轮 $m = 1, 2, \cdots, M$ 顺次地执行下列操作:

(a) 使用当前分布 $D_{m}$ 加权的训练数据集学习基本分类器 $G_{m}(x)$ 。

(b) 计算基本分类器 $G_{m}(x)$ 在加权训练数据集上的分类误差率:

$$
\begin{array}{l} {e _ {m} = \sum_ {i = 1} ^ {N} P (G _ {m} (x _ {i}) \neq y _ {i})} \\ {\qquad = \sum_ {G _ {m} (x _ {i}) \neq y _ {i}} w _ {m i}} \end{array}\tag{8.8}
$$

这里， $w_{mi}$ 表示第 $m$ 轮中第 $i$ 个实例的权值， $\sum_{i=1}^{N} w_{mi} = 1$ 。这表明， $G_m(x)$ 在加权的训练数据集上的分类误差率是被 $G_m(x)$ 误分类样本的权值之和，由此可以看出数据权值分布 $D_m$ 与基本分类器 $G_m(x)$ 的分类误差率的关系。

(c) 计算基本分类器 $G_{m}(x)$ 的系数 $\alpha_{m} \circ \alpha_{m}$ 表示 $G_{m}(x)$ 在最终分类器中的重要性。由式 (8.2) 可知, 当 $e_{m} \leqslant \frac{1}{2}$ 时, $\alpha_{m} \geqslant 0$ , 并且 $\alpha_{m}$ 随着 $e_{m}$ 的减小而增大, 所以分类误差率越小的基本分类器在最终分类器中的作用越大。

(d) 更新训练数据的权值分布为下一轮作准备。式 (8.4) 可以写成

$$
w _ {m + 1, i} = \left\{ \begin{array}{l l} \frac {w _ {m i}}{Z _ {m}} \mathrm{e} ^ {- \alpha_ {m}}, & G _ {m} (x _ {i}) = y _ {i} \\ \frac {w _ {m i}}{Z _ {m}} \mathrm{e} ^ {\alpha_ {m}}, & G _ {m} (x _ {i}) \neq y _ {i} \end{array} \right.
$$

由此可知, 被基本分类器 $G_{m}(x)$ 误分类样本的权值得以扩大, 而被正确分类样本的权值得以缩小。两相比较, 由式 (8.2) 知误分类样本的权值被放大 $\mathrm{e}^{2\alpha_{m}} = \frac{1 - e_{m}}{e_{m}}$ 倍。因此, 误分类样本在下一轮学习中起更大的作用。不改变所给的训练数据, 而不断改变训练数据权值的分布,使得训练数据在基本分类器的学习中起不同的作用, 这是 AdaBoost 的一个特点。

(3) 步骤 (3) 中线性组合 $f(x)$ 实现 $M$ 个基本分类器的加权表决。系数 $\alpha_{m}$ 表示了基本分类器 $G_{m}(x)$ 的重要性, 这里, 所有 $\alpha_{m}$ 之和并不为 1 。 $f(x)$ 的符号决定实例 $x$ 的类, $f(x)$ 的绝对值表示分类的确信度。利用基本分类器的线性组合构建最终分类器是 AdaBoost 的另一特点。

## 8.1.3 AdaBoost 的例子①

例 8.1 给定如表 8.1 所示训练数据。假设弱分类器由 x < v 或 x > v 产生，其阈值 v 使该分类器在训练数据集上分类误差率最低。试用 AdaBoost 算法学习一个强分类器。

表 8.1 训练数据表

| 序号 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| x | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| y | 1 | 1 | 1 | -1 | -1 | -1 | 1 | 1 | 1 | -1 |

解 初始化数据权值分布:

$$
\begin{array}{c} {D _ {1} = (w _ {1 1}, w _ {1 2}, \dots , w _ {1 1 0})} \\ {w _ {1 i} = 0. 1, \quad i = 1, 2, \dots , 1 0} \end{array}
$$

对 $m = 1$

(a) 在权值分布为 $D_{1}$ 的训练数据上, 阈值 $v$ 取 2.5 时分类误差率最低, 故基本分类器为

$$
G _ {1} (x) = \left\{ \begin{array}{l l} 1, & x <   2. 5 \\ - 1, & x > 2. 5 \end{array} \right.
$$

(b) $G_{1}(x)$ 在训练数据集上的误差率 $e_1 = P(G_1(x_i) \neq y_i) = 0.3$ 。

(c) 计算 $G_{1}(x)$ 的系数: $\alpha_{1} = \frac{1}{2} \log \frac{1 - e_{1}}{e_{1}} = 0.4236$ 。

(d) 更新训练数据的权值分布:

$$
\begin{array}{l} {D _ {2} = (w _ {2 1}, \dots , w _ {2 i}, \dots , w _ {2 1 0})} \\ {w _ {2 i} = \frac {w _ {1 i}}{Z _ {1}} \exp (- \alpha_ {1} y _ {i} G _ {1} (x _ {i})), \quad i = 1, 2, \dots , 1 0} \\ {D _ {2} = (0. 0 7 1 4 3, 0. 0 7 1 4 3, 0. 0 7 1 4 3, 0. 0 7 1 4 3, 0. 0 7 1 4 3, 0. 0 7 1 4 3,} \\ {\qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad f _ {1} (x) = 0. 4 2 3 6 G _ {1} (x)} \end{array}
$$

分类器 $\operatorname{sign}(f_1(x))$ 在训练数据集上有3个误分类点。

对 $m = 2$

(a) 在权值分布为 $D_{2}$ 的训练数据上, 阈值 $v$ 是 8.5 时分类误差率最低, 基本分类器为

$$
G _ {2} (x) = \left\{ \begin{array}{l l} 1, & x <   8. 5 \\ - 1, & x > 8. 5 \end{array} \right.
$$

(b) $G_{2}(x)$ 在训练数据集上的误差率 $e_{2}=0.2143$ 。

(c) 计算 $\alpha_{2}=0.6496$ 。

(d) 更新训练数据权值分布:

$$
\begin{array}{r l} D _ {3} & = (0. 0 4 5 5, 0. 0 4 5 5, 0. 0 4 5 5, 0. 1 6 6 7, 0. 1 6 6 7, 0. 1 6 6 7, \\ & \quad 0. 1 0 6 0, 0. 1 0 6 0, 0. 1 0 6 0, 0. 0 4 5 5) \end{array}
$$

$$
f _ {2} (x) = 0. 4 2 3 6 G _ {1} (x) + 0. 6 4 9 6 G _ {2} (x)
$$

分类器 $\mathrm{sign}(f_{2}(x))$ 在训练数据集上有 3 个误分类点。

对 $m = 3$

(a) 在权值分布为 $D_{3}$ 的训练数据上, 阈值 $v$ 是 5.5 时分类误差率最低, 基本分类器为

$$
G _ {3} (x) = \left\{ \begin{array}{l l} 1, & x > 5. 5 \\ - 1, & x <   5. 5 \end{array} \right.
$$

(b) $G_{3}(x)$ 在训练数据集上的误差率 $e_{3}=0.1820$ 。

(c) 计算 $\alpha_{3}=0.7514$ 。

(d) 更新训练数据的权值分布:

$$
D _ {4} = (0. 1 2 5, 0. 1 2 5, 0. 1 2 5, 0. 1 0 2, 0. 1 0 2, 0. 1 0 2, 0. 0 6 5, 0. 0 6 5, 0. 0 6 5, 0. 1 2 5)
$$

于是得到:

$$
f _ {3} (x) = 0. 4 2 3 6 G _ {1} (x) + 0. 6 4 9 6 G _ {2} (x) + 0. 7 5 1 4 G _ {3} (x)
$$

分类器 $\mathrm{sign}(f_{3}(x))$ 在训练数据集上的误分类点个数为 0。

于是最终分类器为

$$
G (x) = \operatorname{sign} \left(f _ {3} (x)\right) = \operatorname{sign} \left(0. 4 2 3 6 G _ {1} (x) + 0. 6 4 9 6 G _ {2} (x) + 0. 7 5 1 4 G _ {3} (x)\right)
$$

## 8.2 AdaBoost 算法的训练误差分析

AdaBoost 最基本的性质是它能在学习过程中不断减少训练误差, 即在训练数据集上的分类误差率。关于这个问题有下面的定理。

定理 8.1 (AdaBoost 的训练误差界) AdaBoost 算法最终分类器的训练误差界为

$$
\frac {1}{N} \sum_ {i = 1} ^ {N} I (G (x _ {i}) \neq y _ {i}) \leqslant \frac {1}{N} \sum_ {i} \exp (- y _ {i} f (x _ {i})) = \prod_ {m} Z _ {m}\tag{8.9}
$$

这里， $G(x)$ , $f(x)$ 和 $Z_{m}$ 分别由式 (8.7)、式 (8.6) 和式 (8.5) 给出。

证明 当 $G(x_{i}) \neq y_{i}$ 时, $y_{i} f(x_{i}) < 0$ , 因而 $\exp (-y_{i} f(x_{i})) \geqslant 1$ 。由此直接推导出前半部分。后半部分的推导要用到 $Z_{m}$ 的定义式 (8.5) 及式 (8.4) 的变形:

$$
w _ {m i} \exp (- \alpha_ {m} y _ {i} G _ {m} (x _ {i})) = Z _ {m} w _ {m + 1, i}
$$

现推导如下:

$$
\begin{array}{l} \frac {1}{N} \sum_ {i} \exp (- y _ {i} f (x _ {i})) = \frac {1}{N} \sum_ {i} \exp \left(- \sum_ {m = 1} ^ {M} \alpha_ {m} y _ {i} G _ {m} (x _ {i})\right) \\ \qquad = \sum_ {i} w _ {1 i} \prod_ {m = 1} ^ {M} \exp (- \alpha_ {m} y _ {i} G _ {m} (x _ {i})) \\ \qquad = Z _ {1} \sum_ {i} w _ {2 i} \prod_ {m = 2} ^ {M} \exp (- \alpha_ {m} y _ {i} G _ {m} (x _ {i})) \\ \qquad = Z _ {1} Z _ {2} \sum_ {i} w _ {3 i} \prod_ {m = 3} ^ {M} \exp (- \alpha_ {m} y _ {i} G _ {m} (x _ {i})) \\ \qquad \vdots \\ \qquad = Z _ {1} Z _ {2} \dots Z _ {M - 1} \sum_ {i} w _ {M i} \exp (- \alpha_ {M} y _ {i} G _ {M} (x _ {i})) \\ \qquad = \prod_ {m = 1} ^ {M} Z _ {m} \end{array}
$$

这一定理说明, 可以在每一轮选取适当的 $G_{m}$ 使得 $Z_{m}$ 最小, 从而使训练误差下降最快。对二类分类问题, 有如下结果。

定理 8.2（二类分类问题 AdaBoost 的训练误差界）

$$
\begin{array}{l} \prod_ {m = 1} ^ {M} Z _ {m} = \prod_ {m = 1} ^ {M} 2 \sqrt {e _ {m} (1 - e _ {m})} \\ = \prod_ {m = 1} ^ {M} \sqrt {1 - 4 \gamma_ {m} ^ {2}} \\ \leqslant \exp \left(- 2 \sum_ {m = 1} ^ {M} \gamma_ {m} ^ {2}\right) \end{array}\tag{8.10}
$$

这里， $\gamma_{m} = \frac{1}{2} -e_{m}$

证明 由 $Z_{m}$ 的定义式(8.5)及式(8.8)得：

$$
\begin{array}{r l} & {Z _ {m} = \sum_ {i = 1} ^ {N} w _ {m i} \exp (- \alpha_ {m} y _ {i} G _ {m} (x _ {i}))} \\ & {\qquad = \sum_ {y _ {i} = G _ {m} (x _ {i})} w _ {m i} \mathrm{e} ^ {- \alpha_ {m}} + \sum_ {y _ {i} \neq G _ {m} (x _ {i})} w _ {m i} \mathrm{e} ^ {\alpha_ {m}}} \\ & {\qquad = (1 - e _ {m}) \mathrm{e} ^ {- \alpha_ {m}} + e _ {m} \mathrm{e} ^ {\alpha_ {m}}} \\ & {\qquad = 2 \sqrt {e _ {m} (1 - e _ {m})}} \\ & {\qquad = \sqrt {1 - 4 \gamma_ {m} ^ {2}}} \end{array}\tag{8.11}
$$

至于不等式

$$
\prod_ {m = 1} ^ {M} \sqrt {1 - 4 \gamma_ {m} ^ {2}} \leqslant \exp \left(- 2 \sum_ {m = 1} ^ {M} \gamma_ {m} ^ {2}\right)
$$

则可先由 $\mathrm{e}^{x}$ 和 $\sqrt{1 - x}$ 在点 $x = 0$ 的泰勒展开式推出不等式 $\sqrt{1 - 4\gamma_{m}^{2}} \leqslant \exp (-2\gamma_{m}^{2})$ ，进而得到。

推论 8.1 如果存在 $\gamma > 0$ ，对所有 m 有 $\gamma_{m} \geqslant \gamma$ ，则

$$
\frac {1}{N} \sum_ {i = 1} ^ {N} I (G (x _ {i}) \neq y _ {i}) \leqslant \exp (- 2 M \gamma^ {2})\tag{8.12}
$$

这表明在此条件下 AdaBoost 的训练误差是以指数速率下降的。这一性质当然是很有吸引力的。

注意，AdaBoost 算法不需要知道下界 $\gamma$ ，这正是 Freund 与 Schapire 设计 Ada-Boost 时所考虑的。与一些早期的 Boosting 不同，AdaBoost 具有适应性，即它能适应弱分类器各自的训练误差率。这也是它的名称（适应的提升）的由来，Ada 是 Adaptive 的简写。

## 8.3 AdaBoost 算法的解释

AdaBoost 算法还有另一个解释, 即可以认为 AdaBoost 算法是模型为加法模型、损失函数为指数函数、学习算法为前向分步算法时的二类分类学习方法。

## 8.3.1 前向分步算法

考虑加法模型（additive model）

$$
f (x) = \sum_ {m = 1} ^ {M} \beta_ {m} b (x; \gamma_ {m})\tag{8.13}
$$

其中， $b(x; \gamma_{m})$ 为基函数， $\gamma_{m}$ 为基函数的参数， $\beta_{m}$ 为基函数的系数。显然，式 (8.6) 是一个加法模型。

在给定训练数据及损失函数 $L(y, f(x))$ 的条件下, 学习加法模型 $f(x)$ 成为经验风险极小化即损失函数极小化问题:

$$
\min _ {\beta_ {m}, \gamma_ {m}} \sum_ {i = 1} ^ {N} L \left(y _ {i}, \sum_ {m = 1} ^ {M} \beta_ {m} b (x _ {i}; \gamma_ {m})\right)\tag{8.14}
$$

通常这是一个复杂的优化问题。前向分步算法（forward stagewise algorithm）求解这一优化问题的想法是：因为学习的是加法模型，如果能够从前向后，每一步只学习一个基函数及其系数，逐步逼近优化目标函数式(8.14)，那么就可以简化优化的复杂度。具体地，每步只需优化如下损失函数：

$$
\min _ {\beta , \gamma} \sum_ {i = 1} ^ {N} L \left(y _ {i}, \beta b (x _ {i}; \gamma)\right)\tag{8.15}
$$

给定训练数据集 $T = \{(x_{1}, y_{1}), (x_{2}, y_{2}), \cdots, (x_{N}, y_{N})\}$ ， $x_{i} \in X \subseteq R^{n}$ ， $y_{i} \in Y = \{-1, +1\}$ ，损失函数 $L(y, f(x))$ 和基函数的集合 $\{b(x; \gamma)\}$ ，学习加法模型 $f(x)$ 的前向分步算法如下。

## 算法 8.2（前向分步算法）

输入: 训练数据集 $T = \{(x_{1}, y_{1}), (x_{2}, y_{2}), \cdots, (x_{N}, y_{N})\}$ , 损失函数 $L(y, f(x))$ , 基函数集 $\{b(x; \gamma)\}$ 。

输出：加法模型 $f(x)$ 。

(1) 初始化 $f_{0}(x)=0;$

(2) 对 $m = 1, 2, \cdots, M,$

(a) 极小化损失函数:

$$
(\beta_ {m}, \gamma_ {m}) = \arg \min _ {\beta , \gamma} \sum_ {i = 1} ^ {N} L (y _ {i}, f _ {m - 1} (x _ {i}) + \beta b (x _ {i}; \gamma))\tag{8.16}
$$

得到参数 $\beta_{m}$ ， $\gamma_{m}$ 。

(b) 更新:

$$
f _ {m} (x) = f _ {m - 1} (x) + \beta_ {m} b (x; \gamma_ {m})\tag{8.17}
$$

(3) 得到加法模型:

$$
f (x) = f _ {M} (x) = \sum_ {m = 1} ^ {M} \beta_ {m} b (x; \gamma_ {m})\tag{8.18}
$$

这样，前向分步算法将同时求解从 $m = 1$ 到 $m = M$ 所有参数 $\beta_{m}, \gamma_{m}$ 的优化问题简化为逐次求解各个 $\beta_{m}, \gamma_{m}$ 的优化问题。

## 8.3.2 前向分步算法与 AdaBoost

由前向分步算法可以推导出 AdaBoost，用定理叙述这一关系。

定理 8.3 AdaBoost 算法是前向分步加法算法的特例。这时，模型是由基本分类器组成的加法模型，损失函数是指数函数。

证明 前向分步算法学习的是加法模型，当基函数为基本分类器时，该加法模型等价于AdaBoost的最终分类器：

$$
f (x) = \sum_ {m = 1} ^ {M} \alpha_ {m} G _ {m} (x)\tag{8.19}
$$

由基本分类器 $G_{m}(x)$ 及其系数 $\alpha_{m}$ 组成， $m=1,2,\cdots,M$ 。前向分步算法逐一学习基函数，这一过程与 AdaBoost 算法逐一学习基本分类器的过程一致。下面证明前向分步算法的损失函数是指数损失函数（exponential loss function）

$$
L (y, f (x)) = \exp (- y f (x))
$$

时，其学习的具体操作等价于 AdaBoost 算法学习的具体操作。

假设经过 m-1 轮迭代前向分步算法已经得到 $f_{m-1}(x)$ :

$$
\begin{array}{r l} & {f _ {m - 1} (x) = f _ {m - 2} (x) + \alpha_ {m - 1} G _ {m - 1} (x)} \\ & {\qquad = \alpha_ {1} G _ {1} (x) + \dots + \alpha_ {m - 1} G _ {m - 1} (x)} \end{array}
$$

在第 m 轮迭代得到 $\alpha_{m}$ ， $G_{m}(x)$ 和 $f_{m}(x)$ 。

$$
f _ {m} (x) = f _ {m - 1} (x) + \alpha_ {m} G _ {m} (x)
$$

目标是使前向分步算法得到的 $\alpha_{m}$ 和 $G_{m}(x)$ 使 $f_{m}(x)$ 在训练数据集 $T$ 上的指数损失最小, 即

$$
(\alpha_ {m}, G _ {m} (x)) = \arg \min _ {\alpha , G} \sum_ {i = 1} ^ {N} \exp [ - y _ {i} (f _ {m - 1} (x _ {i}) + \alpha G (x _ {i})) ]\tag{8.20}
$$

式 $(8.20)$ 可以表示为

$$
(\alpha_ {m}, G _ {m} (x)) = \arg \min _ {\alpha , G} \sum_ {i = 1} ^ {N} \bar {w} _ {m i} \exp (- y _ {i} \alpha G (x _ {i}))\tag{8.21}
$$

其中， $\bar{w}_{mi} = \exp(-y_i f_{m-1}(x_i))$ 。因为 $\bar{w}_{mi}$ 既不依赖 $\alpha$ 也不依赖于 $G$ ，所以与最小化无关。但 $\bar{w}_{mi}$ 依赖于 $f_{m-1}(x)$ ，随着每一轮迭代而发生改变。

现证明使式 (8.21) 达到最小的 $\alpha_{m}^{*}$ 和 $G_{m}^{*}(x)$ 就是 AdaBoost 算法所得到的 $\alpha_{m}$ 和 $G_{m}(x)$ 。求解式 (8.21) 可分两步:

首先，求 $G_{m}^{*}(x)$ 。对任意 $\alpha > 0$ ，使式(8.21)最小的 $G(x)$ 由下式得到：

$$
G _ {m} ^ {*} (x) = \arg \min _ {G} \sum_ {i = 1} ^ {N} \bar {w} _ {m i} I (y _ {i} \neq G (x _ {i}))
$$

其中， $\bar{w}_{mi}=\exp(-y_{i}f_{m-1}(x_{i}))$ 。

此分类器 $G_{m}^{*}(x)$ 即为 AdaBoost 算法的基本分类器 $G_{m}(x)$ , 因为它是使第 $m$ 轮加权训练数据分类误差率最小的基本分类器。

然后，求 $\alpha_{m}^{*}$ 。参照式(8.11)，式(8.21）中

$$
\begin{array}{r l} & {\sum_ {i = 1} ^ {N} \bar {w} _ {m i} \exp (- y _ {i} \alpha G (x _ {i})) = \sum_ {y _ {i} = G _ {m} (x _ {i})} \bar {w} _ {m i} \mathrm{e} ^ {- \alpha} + \sum_ {y _ {i} \neq G _ {m} (x _ {i})} \bar {w} _ {m i} \mathrm{e} ^ {\alpha}} \\ & {\qquad = (\mathrm{e} ^ {\alpha} - \mathrm{e} ^ {- \alpha}) \sum_ {i = 1} ^ {N} \bar {w} _ {m i} I (y _ {i} \neq G (x _ {i})) + \mathrm{e} ^ {- \alpha} \sum_ {i = 1} ^ {N} \bar {w} _ {m i}} \end{array}\tag{8.22}
$$

将已求得的 $G_{m}^{*}(x)$ 代入式 (8.22), 对 $\alpha$ 求导并使导数为 0 , 即得到使式 (8.21) 最小的 $\alpha$ :

$$
\alpha_ {m} ^ {*} = \frac {1}{2} \log \frac {1 - e _ {m}}{e _ {m}}
$$

其中， $e_{m}$ 是分类误差率：

$$
\begin{array}{l} e _ {m} = \frac {\sum_ {i = 1} ^ {N} \bar {w} _ {m i} I (y _ {i} \neq G _ {m} (x _ {i}))}{\sum_ {i = 1} ^ {N} \bar {w} _ {m i}} \\ = \sum_ {i = 1} ^ {N} w _ {m i} I (y _ {i} \neq G _ {m} (x _ {i})) \end{array}\tag{8.23}
$$

这里的 $\alpha_{m}^{*}$ 与 AdaBoost 算法第 2(c) 步的 $\alpha_{m}$ 完全一致。

最后来看每一轮样本权值的更新。由

$$
f _ {m} (x) = f _ {m - 1} (x) + \alpha_ {m} G _ {m} (x)
$$

以及 $\bar{w}_{mi} = \exp(-y_{i}f_{m-1}(x_{i}))$ ，可得：

$$
\bar {w} _ {m + 1, i} = \bar {w} _ {m, i} \exp (- y _ {i} \alpha_ {m} G _ {m} (x))
$$

这与 AdaBoost 算法第 2(d) 步的样本权值的更新只相差规范化因子，因而等价。

## 8.4 提升树

提升树是以分类树或回归树为基本分类器的 Boosting。提升树被认为是机器学习中性能最好的方法之一。

## 8.4.1 提升树模型

Boosting 实际采用加法模型（即基函数的线性组合）与前向分步算法。以决策树为基函数的 Boosting 称为提升树（boosting tree）。对分类问题决策树是二叉分类树，对回归问题决策树是二叉回归树。在例 8.1 中看到的基本分类器 $x < v$ 或 $x > v$ 可以看作是由一个根结点直接连接两个叶结点的简单决策树，即所谓的决策树桩（decision stump）。提升树模型可以表示为决策树的加法模型：

$$
f _ {M} (x) = \sum_ {m = 1} ^ {M} T (x; \Theta_ {m})\tag{8.24}
$$

其中， $T(x;\Theta_{m})$ 表示决策树， $\Theta_{m}$ 为决策树的参数，M 为树的个数。

## 8.4.2 提升树算法

提升树算法采用前向分步算法。首先确定初始提升树 $f_{0}(x) = 0$ ，第 $m$ 步的模型是

$$
f _ {m} (x) = f _ {m - 1} (x) + T (x; \Theta_ {m})\tag{8.25}
$$

其中， $f_{m-1}(x)$ 为当前模型。通过经验风险极小化确定下一棵决策树的参数 $\Theta_{m}$ :

$$
\hat {\Theta} _ {m} = \arg \min _ {\Theta_ {m}} \sum_ {i = 1} ^ {N} L (y _ {i}, f _ {m - 1} (x _ {i}) + T (x _ {i}; \Theta_ {m}))\tag{8.26}
$$

由于树的线性组合可以很好地拟合训练数据，即使数据中的输入与输出之间的关系很复杂也是如此，所以提升树是一个高功能的学习算法。

下面讨论针对不同问题的提升树学习算法，其主要区别在于使用的损失函数不同。包括用平方误差损失函数的回归问题、用指数损失函数的分类问题，以及用一般损失函数的一般决策问题。

对于二类分类问题, 提升树算法只需将 AdaBoost 算法 8.1 中的基本分类器限制为二类分类树即可, 可以说这时的提升树算法是 AdaBoost 算法的特殊情况, 这里不再细述。下面叙述回归问题的提升树。

已知一个训练数据集 $T = \{(x_{1}, y_{1}), (x_{2}, y_{2}), \cdots, (x_{N}, y_{N})\}$ ， $x_{i} \in X \subseteq R^{n}$ ，X 为输入空间， $y_{i} \in Y \subseteq R$ ，Y 为输出空间。在 5.5 节中已经讨论了回归树的问题。如果将输入空间 X 划分为 J 个互不相交的区域 $R_{1}, R_{2}, \cdots, R_{J}$ ，并且在每个区域上确定输出的常量 $c_{j}$ ，那么树可表示为

$$
T (x; \Theta) = \sum_ {j = 1} ^ {J} c _ {j} I (x \in R _ {j})\tag{8.27}
$$

其中，参数 $\Theta=\left\{(R_{1},c_{1}),(R_{2},c_{2}),\cdots,(R_{J},c_{J})\right\}$ 表示树的区域划分和各区域上的常数，J 是回归树的复杂度即叶结点个数。

回归问题提升树使用以下前向分步算法:

$$
\begin{array}{l} {f _ {0} (x) = 0} \\ {f _ {m} (x) = f _ {m - 1} (x) + T (x; \Theta_ {m}), \quad m = 1, 2, \dots , M} \\ {f _ {M} (x) = \sum_ {m = 1} ^ {M} T (x; \Theta_ {m})} \end{array}
$$

在前向分步算法的第 $m$ 步，给定当前模型 $f_{m - 1}(x)$ ，需求解

$$
\hat {\Theta} _ {m} = \arg \min _ {\Theta_ {m}} \sum_ {i = 1} ^ {N} L (y _ {i}, f _ {m - 1} (x _ {i}) + T (x _ {i}; \Theta_ {m}))
$$

得到 $\hat{\Theta}_{m}$ ，即第 m 棵树的参数。

当采用平方误差损失函数时，

$$
L (y, f (x)) = (y - f (x)) ^ {2}
$$

其损失变为

$$
\begin{array}{r l} & L (y, f _ {m - 1} (x) + T (x; \Theta_ {m})) = (y - f _ {m - 1} (x) - T (x; \Theta_ {m})) ^ {2} \\ & \qquad = (r - T (x; \Theta_ {m})) ^ {2} \end{array}
$$

这里，

$$
r = y - f _ {m - 1} (x)\tag{8.28}
$$

是当前模型拟合数据的残差（residual）。所以，对回归问题的提升树算法来说，只需简单地拟合当前模型的残差。这样，算法是相当简单的。现将回归问题的提升树算法叙述如下。

算法 8.3（回归问题的提升树算法）

输入：训练数据集 $T=\left\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{N},y_{N})\right\}$ ， $x_{i}\in\mathcal{X}\subseteq\boldsymbol{R}^{n}$ ， $y_{i}\in\mathcal{Y}\subseteq\boldsymbol{R}$ 。

输出：提升树 $f_{M}(x)$ 。

(1) 初始化 $f_{0}(x)=0$ 。

(2) 对 $m = 1, 2, \cdots, M$ ,

(a) 按式 (8.27) 计算残差:

$$
r _ {m i} = y _ {i} - f _ {m - 1} (x _ {i}), \quad i = 1, 2, \dots , N
$$

(b) 拟合残差 $r_{mi}$ 学习一个回归树，得到 $T(x; \Theta_{m})$ 。

(c) 更新 $f_{m}(x)=f_{m-1}(x)+T(x;\Theta_{m})$ 。

(3) 得到回归问题提升树:

$$
f _ {M} (x) = \sum_ {m = 1} ^ {M} T (x; \Theta_ {m})
$$

例 8.2 已知如表 8.2 所示的训练数据, $x$ 的取值范围为区间 [0.5,10.5], $y$ 的取值范围为区间 [5.0,10.0], 学习这个回归问题的提升树模型, 考虑只用树桩作为基函数。

表 8.2 训练数据表

| $x_i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $y_i$ | 5.56 | 5.70 | 5.91 | 6.40 | 6.80 | 7.05 | 8.90 | 8.70 | 9.00 | 9.05 |

解 按照算法 8.3, 第 1 步求 $f_{1}(x)$ 即回归树 $T_{1}(x)$ 。

首先通过以下优化问题

$$
\min _ {s} \left[ \min _ {c _ {1}} \sum_ {x _ {i} \in R _ {1}} (y _ {i} - c _ {1}) ^ {2} + \min _ {c _ {2}} \sum_ {x _ {i} \in R _ {2}} (y _ {i} - c _ {2}) ^ {2} \right]
$$

求解训练数据的切分点 s:

$$
R _ {1} = \{x | x \leqslant s \}, \quad R _ {2} = \{x | x > s \}
$$

容易求得在 $R_{1}, R_{2}$ 内部使平方损失误差达到最小值的 $c_{1}, c_{2}$ 为

$$
c _ {1} = \frac {1}{N _ {1}} \sum_ {x _ {i} \in R _ {1}} y _ {i}, \quad c _ {2} = \frac {1}{N _ {2}} \sum_ {x _ {i} \in R _ {2}} y _ {i}
$$

这里 $N_{1}$ ， $N_{2}$ 是 $R_{1}$ ， $R_{2}$ 的样本点数。

求训练数据的切分点。根据所给数据，考虑如下切分点：

对各切分点，不难求出相应的 $R_{1}, R_{2}, c_{1}, c_{2}$ 及

$$
m (s) = \min _ {c _ {1}} \sum_ {x _ {i} \in R _ {1}} (y _ {i} - c _ {1}) ^ {2} + \min _ {c _ {2}} \sum_ {x _ {i} \in R _ {2}} (y _ {i} - c _ {2}) ^ {2}
$$

例如，当 s=1.5 时， $R_{1}=\{1\}$ ， $R_{2}=\{2,3,\cdots,10\}$ ， $c_{1}=5.56$ ， $c_{2}=7.50$ ，

$$
m (s) = \min _ {c _ {1}} \sum_ {x _ {i} \in R _ {1}} (y _ {i} - c _ {1}) ^ {2} + \min _ {c _ {2}} \sum_ {x _ {i} \in R _ {2}} (y _ {i} - c _ {2}) ^ {2} = 0 + 1 5. 7 2 = 1 5. 7 2
$$

现将 s 及 $m(s)$ 的计算结果列于表 8.3。

表 8.3 计算数据表

| s | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 | 7.5 | 8.5 | 9.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m(s) | 15.72 | 12.07 | 8.36 | 5.78 | 3.91 | 1.93 | 8.01 | 11.73 | 15.74 |

由表 8.3 可知，当 s = 6.5 时 $m(s)$ 达到最小值，此时 $R_{1} = \{1, 2, \cdots, 6\}$ ， $R_{2} = \{7, 8, 9, 10\}$ ， $c_{1} = 6.24$ ， $c_{2} = 8.91$ ，所以回归树 $T_{1}(x)$ 为

$$
\begin{array}{l} T _ {1} (x) = \left\{ \begin{array}{l l} 6. 2 4, & x <   6. 5 \\ 8. 9 1, & x \geqslant 6. 5 \end{array} \right. \\ f _ {1} (x) = T _ {1} (x) \end{array}
$$

用 $f_{1}(x)$ 拟合训练数据的残差见表 8.4，表中 $r_{2i}=y_{i}-f_{1}(x_{i}), i=1,2,\cdots,10$ 。

表 8.4 残差表

| $x_i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $r_{2i}$ | -0.68 | -0.54 | -0.33 | 0.16 | 0.56 | 0.81 | -0.01 | -0.21 | 0.09 | 0.14 |

用 $f_{1}(x)$ 拟合训练数据的平方损失误差:

$$
L (y, f _ {1} (x)) = \sum_ {i = 1} ^ {1 0} (y _ {i} - f _ {1} (x _ {i})) ^ {2} = 1. 9 3
$$

第2步求 $T_{2}(x)$ 。方法与求 $T_{1}(x)$ 一样，只是拟合的数据是表8.4的残差，可以得到：

$$
T _ {2} (x) = \left\{ \begin{array}{l l} - 0. 5 2, & x <   3. 5 \\ 0. 2 2, & x \geqslant 3. 5 \end{array} \right.
$$

$$
f _ {2} (x) = f _ {1} (x) + T _ {2} (x) = \left\{ \begin{array}{l l} 5. 7 2, & x <   3. 5 \\ 6. 4 6, & 3. 5 \leqslant x <   6. 5 \\ 9. 1 3, & x \geqslant 6. 5 \end{array} \right.
$$

用 $f_{2}(x)$ 拟合训练数据的平方损失误差是

$$
L (y, f _ {2} (x)) = \sum_ {i = 1} ^ {1 0} (y _ {i} - f _ {2} (x _ {i})) ^ {2} = 0. 7 9
$$

继续求得:

$$
T _ {3} (x) = \left\{ \begin{array}{l l} 0. 1 5, & x <   6. 5 \\ - 0. 2 2, & x \geqslant 6. 5 \end{array} , \quad L (y, f _ {3} (x)) = 0. 4 7 \right.
$$

$$
T _ {4} (x) = \left\{ \begin{array}{l l} - 0. 1 6, & x <   4. 5 \\ 0. 1 1, & x \geqslant 4. 5 \end{array} , \quad L (y, f _ {4} (x)) = 0. 3 0 \right.
$$

$$
T _ {5} (x) = \left\{ \begin{array}{l l} 0. 0 7, & x <   6. 5 \\ - 0. 1 1, & x \geqslant 6. 5 \end{array} , \quad L (y, f _ {5} (x)) = 0. 2 3 \right.
$$

$$
\begin{array}{l} T _ {6} (x) = \left\{ \begin{array}{l l} - 0. 1 5, & x <   2. 5 \\ 0. 0 4, & x \geqslant 2. 5 \end{array} \right. \\ f _ {6} (x) = f _ {5} (x) + T _ {6} (x) = T _ {1} (x) + \dots + T _ {5} (x) + T _ {6} (x) \\ = \left\{ \begin{array}{l l} 5. 6 3, & x <   2. 5 \\ 5. 8 2, & 2. 5 \leqslant x <   3. 5 \\ 6. 5 6, & 3. 5 \leqslant x <   4. 5 \\ 6. 8 3, & 4. 5 \leqslant x <   6. 5 \\ 8. 9 5, & x \geqslant 6. 5 \end{array} \right. \end{array}
$$

用 $f_{6}(x)$ 拟合训练数据的平方损失误差是

$$
L (y, f _ {6} (x)) = \sum_ {i = 1} ^ {1 0} (y _ {i} - f _ {6} (x _ {i})) ^ {2} = 0. 1 7
$$

假设此时已满足误差要求，那么 $f(x)=f_{6}(x)$ 即为所求提升树。

## 8.4.3 梯度提升

提升树利用加法模型与前向分步算法实现学习的优化过程。当损失函数是平方损失和指数损失函数时，每一步优化是很简单的。但对一般损失函数而言，往往每一步优化并不那么容易。针对这一问题，Freidman 提出了梯度提升（gradient boosting）算法。这是利用最速下降法的近似方法，其关键是利用损失函数的负梯度在当前模型的值

$$
- \left[ \frac {\partial L (y , f (x _ {i}))}{\partial f (x _ {i})} \right] _ {f (x) = f _ {m - 1} (x)}
$$

作为回归问题提升树算法中的残差的近似值，拟合一个回归树。

算法 8.4（梯度提升算法）

输入: 训练数据集 $T = \{(x_{1}, y_{1}), (x_{2}, y_{2}), \cdots, (x_{N}, y_{N})\}$ , $x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}$ , $y_{i} \in \mathcal{Y} \subseteq \mathbf{R}$ ; 损失函数 $L(y, f(x))$ 。

输出：回归树 $\hat{f}(x)$ 。

(1) 初始化:

$$
f _ {0} (x) = \arg \min _ {c} \sum_ {i = 1} ^ {N} L (y _ {i}, c)
$$

(2) 对 $m = 1, 2, \cdots, M,$

(a) 对 $i = 1, 2, \cdots, N$ ，计算

$$
r _ {m i} = - \left[ \frac {\partial L (y _ {i} , f (x _ {i}))}{\partial f (x _ {i})} \right] _ {f (x) = f _ {m - 1} (x)}
$$

(b) 对 $r_{mi}$ 拟合一个回归树, 得到第 $m$ 棵树的叶结点区域 $R_{mj}, j = 1,2,\dots,J$ 。

(c) 对 $j = 1, 2, \cdots, J$ ，计算

$$
c _ {m j} = \arg \min _ {c} \sum_ {x _ {i} \in R _ {m j}} L (y _ {i}, f _ {m - 1} (x _ {i}) + c)
$$

(d) 更新 $f_{m}(x) = f_{m - 1}(x) + \sum_{j = 1}^{J}c_{mj}I(x\in R_{mj})$

(3) 得到回归树:

$$
\hat {f} (x) = f _ {M} (x) = \sum_ {m = 1} ^ {M} \sum_ {j = 1} ^ {J} c _ {m j} I (x \in R _ {m j})
$$

算法第 1 步初始化, 估计使损失函数极小化的常数值, 它是只有一个根结点的树。第 2(a) 步计算损失函数的负梯度在当前模型的值, 将它作为残差的估计。对于平方损失函数, 它就是通常所说的残差; 对于一般损失函数, 它就是残差的近似值。第 2(b) 步估计回归树叶结点区域, 以拟合残差的近似值。第 2(c) 步利用线性搜索估计叶结点区域的值, 使损失函数极小化。第 2(d) 步更新回归树。第 3 步得到输出的最终模型 $\hat{f}(x)$ 。

## 本章概要

1. Boosting 是将弱学习算法提升为强学习算法的机器学习方法。在分类学习中, Boosting 通过反复修改训练数据的权值分布构建一系列基本分类器 (弱分类器), 并将这些基本分类器线性组合, 构成一个强分类器。代表性的 Boosting 是 AdaBoost 算法。

AdaBoost 模型是弱分类器的线性组合:

$$
f (x) = \sum_ {m = 1} ^ {M} \alpha_ {m} G _ {m} (x)
$$

2. AdaBoost 算法的特点是通过迭代每次学习一个基本分类器。每次迭代中, 提高那些被前一轮分类器错误分类数据的权值, 而降低那些被正确分类的数据的权值。最后, AdaBoost 将基本分类器的线性组合作为强分类器, 其中给分类误差率小的基本分类器以大的权值, 给分类误差率大的基本分类器以小的权值。

3. AdaBoost 的训练误差分析表明, AdaBoost 的每次迭代可以减少它在训练数据集上的分类误差率, 这说明了它作为 Boosting 的有效性。

4. AdaBoost 算法的一个解释是该算法实际是前向分步算法的一个实现。在这个方法里，模型是加法模型，损失函数是指数损失，算法是前向分步算法。

每一步中极小化损失函数

$$
(\beta_ {m}, \gamma_ {m}) = \arg \min _ {\beta , \gamma} \sum_ {i = 1} ^ {N} L (y _ {i}, f _ {m - 1} (x _ {i}) + \beta b (x _ {i}; \gamma))
$$

得到参数 $\beta_{m}$ ， $\gamma_{m}$ 。

5. 提升树是以分类树或回归树为基本分类器的 Boosting，被认为是机器学习中最有效的方法之一。

## 继续阅读

Boosting 的介绍可参见文献 [1] 和文献 [2]。PAC 学习可参见文献 [3]。强可学习与弱可学习的关系可参见文献 [4]。关于 AdaBoost 的最初论文是文献 [5]。关于 AdaBoost 的前向分步加法模型的解释参见文献 [6]，提升树与梯度提升可参见文献 [6] 和文献 [7]。AdaBoost 只是用于二类分类，Schapire 与 Singer 将它扩展到多类分类问题 [8]。AdaBoost 与逻辑斯谛回归的关系也有相关研究 [9]。

## 习题

8.1 某公司招聘职员考查身体、业务能力、发展潜力这 3 项。身体分为合格 1、不合格 0 两级，业务能力和发展潜力分为上 1、中 2、下 3 三级。分类为合格 1、不合格 -1 两类。已知 10 个人的数据，见表 8.5。假设弱分类器为决策树桩，试用 AdaBoost 算法学习一个强分类器。

表 8.5 应聘人员情况数据表

|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 身体 | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| 业务能力 | 1 | 3 | 2 | 1 | 2 | 1 | 1 | 1 | 3 | 2 |
| 发展潜力 | 3 | 1 | 2 | 3 | 3 | 2 | 2 | 1 | 1 | 1 |
| 分类 | -1 | -1 | -1 | -1 | -1 | -1 | 1 | 1 | -1 | -1 |

8.2 比较支持向量机、AdaBoost、逻辑斯谛回归模型的学习策略与算法。

## 参考文献

[1] FREUND Y, SCHAPIRE R E. A short introduction to boosting[J]. Journal of Japanese Society for Artificial Intelligence, 1999, 14(5): 771–780.

[2] HASTIE T, TIBSHIRANI R, FRIEDMAN J. The elements of statistical learning: data mining, inference, and prediction[M]. 范明, 柴玉梅, 畦红英, 等译. Springer, 2001.

[3] VALIANT L G. A theory of the learnable[J]. Communications of the ACM, 1984, 27(11): 1134–1142.

[4] SCHAPIRE R. The strength of weak learnability[J]. Machine Learning, 1990, 5(2): 197–227.

[5] FREUND Y, SCHAPIRE R E. A decision-theoretic generalization of on-line learning and an application to boosting[J]. Lecture Notes in Computer Science, 1995, 904: 23–37.

[6] FRIEDMAN J, HASTIE T, TIBSHIRANI R. Additive logistic regression: a statistical view of boosting (with discussions)[J]. Annals of Statistics, 2000, 28: 337–407.

[7] FRIEDMAN J. Greedy function approximation: a gradient boosting machine[J]. Annals of Statistics, 2001, 29(5): 1189–1232.

[8] SCHAPIRE R E, SINGER Y. Improved boosting algorithms using confidence-rated predictions[J]. Machine Learning, 1999, 37(3): 297–336.

[9] COLLINS M, SCHAPIRE R E, SINGER Y. Logistic regression, AdaBoost and Bregman distances[J]. Machine Learning, 2002, 48(1–3): 253–285.

