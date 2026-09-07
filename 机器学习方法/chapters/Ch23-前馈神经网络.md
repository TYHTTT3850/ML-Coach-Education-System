---
tags: [机器学习, 李航, 深度学习]
category: "章节"
chapter: "Ch23-前馈神经网络"
pian: "第三篇-深度学习"
---
## 第 23 章 前馈神经网络

人工神经网络（artificial neural network）或神经网络（neural network）是受生物神经网络启发而发明的由神经元连接组成的网络状机器学习模型。前馈神经网络（feedforward neural network）或多层感知机（multilayer perceptron，MLP）是最具代表性的神经网络，主要用于监督学习，如分类和回归。

前馈神经网络由多层神经元组成，层间的神经元相互连接，层内的神经元不连接。其信息处理机制是：前一层神经元通过层间连接向后一层神经元传递信号，因为信号是从前往后转递的，所以是“前馈的”信息处理网络。这里，神经元是对多个输入信号（实数向量）进行非线性转换产生一个输出信号（实数值）的函数，整个神经网络是对多个输入信号（实数向量）进行多次非线性转换产生多个输出信号（实数向量）的复合函数。每一个神经元的函数含有参数，神经网络的神经元的参数通过学习得到。当前馈神经网络的层数达到一定数量时（一般大于2），又称为深度神经网络（deep neural network, DNN）。

前馈神经网络学习算法是反向传播（back propagation）算法，是随机梯度下降法的具体实现。学习的损失函数通常在分类时是交叉熵损失，在回归时是平方损失，其最小化等价于极大似然估计。学习的正则化方法包括早停法（early stopping）、暂退法（dropout）。

McCulloch 和 Pitts 于 1943 年提出了最初的人工神经网络模型；Rosenblatt 于 1958 年发明了感知机，可以看作是前馈神经网络的前身；Rumelhart 等于 1986 年重新开发了反向传播算法，用于前馈神经网络学习；Hinton 于 2006 年提出了深度学习的概念，指包括深度神经网络（DNN）等复杂神经网络的机器学习。

本章 23.1 节讲述前馈神经网络的模型, 23.2 节叙述前馈神经网络学习的算法, 23.3 节叙述前馈神经网络学习的正则化方法。

## 23.1 前馈神经网络的模型

神经网络是由神经元连接组成的网络，采用不同类型的神经元以及神经元的不同连接方法可以构建出不同的网络结构，也就是不同的神经网络模型。本节讲述前馈神经网络的基本模型。首先给出前馈神经网络的定义，接着介绍具体例子，最后讨论前馈神经网络的表示能力。

## 23.1.1 前馈神经网络定义

## 1. 神经元

人工神经元（artificial neuron）或者简称神经元（neuron）是神经网络的基本单元。人工神经元是受生物神经元启发而发明的，本质是一个函数。生物神经元一般有多个树突接入，一个轴突接出。输入信号从树突传入，输出信号从轴突传出。当输入信号量达到阈值后，神经元被激活，产生输出信号。与其对应，人工神经元是以实数向量为输入，实数值为输出的非线性函数，表示多个输入信号（实数向量）到一个输出信号（实数值）的非线性转换。

定义 23.1（神经元） 神经元是如下定义的非线性函数:

$$
y = f (x _ {1}, x _ {2}, \dots , x _ {n}) = a \Big (\sum_ {i = 1} ^ {n} w _ {i} x _ {i} + b \Big)\tag{23.1}
$$

或者写作

$$
y = f \left(x _ {1}, x _ {2}, \dots , x _ {n}\right) = a (z), z = \sum_ {i = 1} ^ {n} w _ {i} x _ {i} + b\tag{23.2}
$$

其中， $x_{1}, x_{2}, \cdots, x_{n}$ 是输入，取实数值； $y$ 是输出，取实数值； $z$ 是中间结果，又称作净输入（net input），也取实数值； $w_{1}, w_{2}, \cdots, w_{n}$ 是权重（weight）， $b$ 是偏置（bias），也都取实数值； $z = \sum_{i=1}^{n} w_{i} x_{i} + b$ 是仿射函数； $a(\cdot)$ 是特定的非线性函数，称为激活函数（activation function）。激活函数有多种形式，比如 S 型函数：

$$
a (z) = \frac {1}{1 + \mathrm{e} ^ {- z}}
$$

神经元函数由两部分组成，首先使用仿射函数对输入 $x_{1}, x_{2}, \cdots, x_{n}$ 进行仿射变换，得到净输入 z，然后使用激活函数 $a(z)$ 对净输入 z 进行非线性变换，得到输出 y。权重 $w_{1}, w_{2}, \cdots, w_{n}$ 与偏置 b 是神经元函数的参数，通过学习得到。

图 23.1 是神经元的示意图。图中结点表示变量，有向边表示变量之间的依存关系，有向图整体表示式（23.1）和式（23.2）的神经元函数。结点 $x_{1}, x_{2}, \cdots, x_{n}$ 是神经元的输入变量，结点 y 是神经元的输出变量。通常不显式表示净输入变量 z。习惯上，将权重 $w_{1}, w_{2}, \cdots, w_{n}$ 附在有向边上。为了方便，经常增加一个恒为 +1 的输入，取偏置 b 为其权重，将仿射变换转换成线性变换，得到一个等价的神经元函数，图 23.1 的有向图表示的就是这个形式的函数。

![](../images/e45f32e97c974b96a163be01d0fcef21c494da4ed347bf67521a2e4298c571fb.jpg)  
图23.1 神经元

神经元也可以用向量表示。设向量

$$
\boldsymbol {x} = \left[ \begin{array}{c} x _ {1} \\ x _ {2} \\ \vdots \\ x _ {n} \end{array} \right]
$$

$$
\boldsymbol {w} = \left[ \begin{array}{c c} & w _ {1} \\ & w _ {2} \\ & \vdots \\ & w _ {n} \end{array} \right]
$$

为输入和权重，则神经元为函数

$$
y = f (\boldsymbol {x}) = a \left(\boldsymbol {w} ^ {\mathrm{T}} \boldsymbol {x} + b\right)\tag{23.3}
$$

或者写作

$$
y = f (\boldsymbol {x}) = a (z), \quad z = \boldsymbol {w} ^ {\mathrm{T}} \boldsymbol {x} + b\tag{23.4}
$$

例 23.1 图 23.2 是一个神经元, 画出其函数的三维图形, 其中激活函数是 S 型函数。

![](../images/4dae6f28398a39fe1a4cfbd9013c2ed86c0ef756325e77c175e5511c812ab22d.jpg)  
图23.2 神经元例

解 神经元的仿射函数是 $z = -x_{1} + 2x_{2} + 1$ ，激活函数是 $a(z) = \frac{1}{1 + \mathrm{e}^{-z}}$ ，神经元函数是

$$
y = f (x _ {1}, x _ {2}) = \frac {1}{1 + \mathrm{e} ^ {x _ {1} - 2 x _ {2} - 1}}
$$

图 23.3 是神经元函数的三维图形, 整体是三维空间 $(x_{1}, x_{2}, y)$ 中的一个 S 形曲面。S 形曲面由平行的等高线组成, 每一条等高线对应着二维空间 $(x_{1}, x_{2})$ 中的一条直线 $z = -x_{1} + 2x_{2} + 1$ , 其中 z 是一个定值。当 z 趋于正无穷时, 等高线趋近于平面 y = 1; 当 z 趋于负无穷时, 等高线趋近于平面 y = 0。

## 2. 前馈神经网络

前馈神经网络由多层神经元组成，层间的神经元相互连接，层内的神经元不连接，前一层神经元的输出是后一层神经元的输入。整体表示输入信号（实数向量）到输出信号（实数向量）的多次非线性转换。数学上，前馈神经网络是以实数向量为输入、以实数向量为输出的非线性函数的复合函数（这里，函数都是以向量为输入输出的一般函数的扩展）。前馈神经网络最后的输出也可以是一个实数值，是实数向量的特殊情况。先给出二层前馈神经网络的定义①。

![](../images/c50b7a0e5bc8688814cf809dfeb5b53c758f0bec4d0dc2a557e30ef10e2c2cf9.jpg)  
图 23.3 神经元的三维图形 (见文前彩图)

定义 23.2（二层前馈神经网络） 二层前馈神经网络是如下定义的非线性函数的复合函数。输入是 $x_{i}, i = 1, 2, \cdots, n$ ，输出是 $y_{k}, k = 1, 2, \cdots, l$ 。神经网络有两层。第一层由 m 个神经元组成，其中第 j 个神经元是

$$
h _ {j} ^ {(1)} = a \left(z _ {j} ^ {(1)}\right) = a \left(\sum_ {i = 1} ^ {n} w _ {j i} ^ {(1)} x _ {i} + b _ {j} ^ {(1)}\right), \quad j = 1, 2, \dots , m\tag{23.5}
$$

这里 $x_{i}$ 是输入, $w_{ji}^{(1)}$ 是权重, $b_{j}^{(1)}$ 是偏置, $z_{j}^{(1)}$ 是净输入, $a(\cdot)$ 是激活函数。第二层由 $l$ 个神经元组成, 其中第 $k$ 个神经元是

$$
y _ {k} = g (z _ {k} ^ {(2)}) = g \left(\sum_ {j = 1} ^ {m} w _ {k j} ^ {(2)} h _ {j} ^ {(1)} + b _ {k} ^ {(2)}\right), \quad k = 1, 2, \dots , l\tag{23.6}
$$

这里 $h_{j}^{(1)}$ 是第一层神经元的输出， $w_{kj}^{(2)}$ 是权重，其中 $j=1,2,\cdots,m$ ， $b_{k}^{(2)}$ 是偏置， $z_{k}^{(2)}$ 是净输入， $g(\cdot)$ 是激活函数。神经网络整体是

$$
y _ {k} = g \left[ \sum_ {j = 1} ^ {m} w _ {k j} ^ {(2)} a \left(\sum_ {i = 1} ^ {n} w _ {j i} ^ {(1)} x _ {i} + b _ {j} ^ {(1)}\right) + b _ {k} ^ {(2)} \right], \quad k = 1, 2, \dots , l\tag{23.7}
$$

通常情况第二层只有一个神经元，即 l = 1。

第一层神经元从输入输出的角度不可见，称为隐层。第二层神经元称为输出层。有时把输入也看作是一层，称为输入层。隐层和输出层的激活函数 $a(\cdot)$ 和 $g(\cdot)$ 通常有不同的定义。这里考虑层间的全连接，即前一层的每一个神经元都和后一层的每一个神经元连接。部分连接网络是其特殊情况，相当于未连接边的权重为0。

图23.4是二层前馈神经网络的示意图。图中结点表示变量，有向边表示变量间的依存关系，边上的数值表示权重。结点 $x_{i}$ 是神经网络的输入，结点 $y_{k}$ 是神经网络的输出（也是输出层神经元），结点 $h_j^{(1)}$ 是神经网络的隐层神经元。数值 $w_{ji}^{(1)}$ 是隐层神经元的权重，数值 $w_{kj}^{(2)}$ 是输出层神经元的权重。数值 $b_{j}^{(1)}$ 是隐层神经元的偏置，数值 $b_{k}^{(2)}$ 是输出层神经元的偏置。其中， $i = 1,2,\dots ,n,\quad j = 1,2,\dots ,m,\quad k = 1,2,\dots ,l$ 。

![](../images/903d1ee4b03a2e12950d63b277b4d7835dbf1b5cfc5da98ec23ac6f9afeb7a3a.jpg)  
图 23.4 二层前馈神经网络

二层前馈神经网络也可以用矩阵来表示，简称矩阵表示。

$$
\boldsymbol {h} ^ {(1)} = f ^ {(1)} (\boldsymbol {x}) = a \left(\boldsymbol {z} ^ {(1)}\right) = a \left(\boldsymbol {W} ^ {(1) ^ {\mathrm{T}}} \boldsymbol {x} + \boldsymbol {b} ^ {(1)}\right)\tag{23.8}
$$

$$
\boldsymbol {y} = f ^ {(2)} \left(\boldsymbol {h} ^ {(1)}\right) = g (\boldsymbol {z} ^ {(2)}) = g \left(\boldsymbol {W} ^ {(2) ^ {\mathrm{T}}} \boldsymbol {h} ^ {(1)} + \boldsymbol {b} ^ {(2)}\right)\tag{23.9}
$$

其中，

$$
\boldsymbol {x} = \left[ \begin{array}{c} x _ {1} \\ x _ {2} \\ \vdots \\ x _ {n} \end{array} \right]
$$

$$
\boldsymbol {z} ^ {(1)} = \left[ \begin{array}{c} z _ {1} ^ {(1)} \\ z _ {2} ^ {(1)} \\ \vdots \\ z _ {m} ^ {(1)} \end{array} \right]
$$

$$
\boldsymbol {h} ^ {(1)} = \left[ \begin{array}{c c c} & h _ {1} ^ {(1)} \\ & h _ {2} ^ {(1)} \\ & \vdots \\ & h _ {m} ^ {(1)} \end{array} \right]
$$

$$
\boldsymbol {z} ^ {(2)} = \left[ \begin{array}{c c c} & z _ {1} ^ {(2)} & \\ & z _ {2} ^ {(2)} & \\ & \vdots & \\ & z _ {l} ^ {(2)} & \end{array} \right]
$$

$$
\boldsymbol {W} ^ {(1)} = \left[ \begin{array}{c c c} w _ {1 1} ^ {(1)} & \dots & w _ {1 m} ^ {(1)} \\ \vdots & & \vdots \\ w _ {n 1} ^ {(1)} & \dots & w _ {n m} ^ {(1)} \end{array} \right]
$$

$$
\boldsymbol {W} ^ {(2)} = \left[ \begin{array}{c c c} w _ {1 1} ^ {(2)} & \dots & w _ {1 l} ^ {(2)} \\ \vdots & & \vdots \\ w _ {m 1} ^ {(2)} & \dots & w _ {m l} ^ {(2)} \end{array} \right]
$$

$$
\boldsymbol {b} ^ {(1)} = \left[ \begin{array}{c c} b _ {1} ^ {(1)} & \\ b _ {2} ^ {(1)} & \\ \vdots & \\ b _ {m} ^ {(1)} & \end{array} \right]
$$

$$
\boldsymbol {b} ^ {(2)} = \left[ \begin{array}{c c} b _ {1} ^ {(2)} & \\ b _ {2} ^ {(2)} & \\ \vdots & \\ b _ {l} ^ {(2)} & \end{array} \right]
$$

向量 x 表示输入，向量 y 表示输出，向量 $z^{(1)}$ 表示隐层的净输入，向量 $h^{(1)}$ 表示隐层的输出，向量 $z^{(2)}$ 表示输出层的净输入，矩阵 $W^{(1)}, W^{(2)}$ 表示权重，向量 $b^{(1)}, b^{(2)}$ 表示偏置。整体神经网络由复合函数 $f^{(2)}(f^{(1)}(\boldsymbol{x}))$ 表示。这里 $a(\cdot)$ 和 $g(\cdot)$ ，以及 $f^{(1)}(\cdot)$ 和 $f^{(2)}(\cdot)$ 是一般函数的扩展，作用在向量的每个元素上，得到的仍是一个向量。

下面给出更一般的多层前馈神经网络或深度神经网络的定义。

定义 23.3（多层前馈神经网络） 多层前馈神经网络或前馈神经网络是如下定义的非线性函数的复合函数。输入是 $x_{i}, i = 1, 2, \cdots, n$ ，输出是 $y_{k}, k = 1, 2, \cdots, l$ 。神经网络有 s 层 ( $s \geqslant 2$ )。第一层到第 s-1 层是隐层。假设其中的第 t 层由 m 个神经元组成，第 t-1 层由 n 个神经元组成， $t = 1, 2, \cdots, s-1$ ，第 t 层的第 j 个神经元是

$$
h _ {j} ^ {(t)} = a \left(z _ {j} ^ {(t)}\right) = a \left(\sum_ {i = 1} ^ {n} w _ {j i} ^ {(t)} h _ {i} ^ {(t - 1)} + b _ {j} ^ {(t)}\right), \quad j = 1, 2, \dots , m\tag{23.10}
$$

这里 $h_{i}^{(t-1)}, i = 1, 2, \cdots, n$ ，是第 t-1 层的输出，设 $h_{i}^{(0)} = x_{i}, w_{ji}^{(t)}, i = 1, 2, \cdots, n$ ，是权重, $b_{j}^{(t)}$ 是偏置, $z_{j}^{(t)}$ 是净输入, $a(\cdot)$ 是激活函数。第 $s$ 层是输出层。假设第 $s$ 层由 $l$ 个神经元组成, 第 $s-1$ 层由 $m$ 个神经元组成, 第 $s$ 层的第 $k$ 个神经元是

$$
y _ {k} = g \left(z _ {k} ^ {(s)}\right) = g \left(\sum_ {j = 1} ^ {m} w _ {k j} ^ {(s)} h _ {j} ^ {(s - 1)} + b _ {k} ^ {(s)}\right), \quad k = 1, 2, \dots , l\tag{23.11}
$$

这里 $h_{j}^{(s-1)}, j = 1, 2, \cdots, m$ ，是第 s-1 层的输出， $w_{kj}^{(s)}, j = 1, 2, \cdots, m$ ，是权重， $b_{k}^{(s)}$ 是偏置， $z_{k}^{(s)}$ 是净输入， $g(\cdot)$ 是激活函数。神经网络整体是

$$
y _ {k} = g \left\{\sum_ {j = 1} ^ {m} w _ {k j} ^ {(s)} \dots \left[ a \left(\sum_ {i = 1} ^ {n} w _ {j i} ^ {(1)} x _ {i} + b _ {j} ^ {(1)}\right) \right] \dots + b _ {k} ^ {(s)} \right\}, \quad k = 1, 2, \dots , l\tag{23.12}
$$

层数大于 2 时的前馈神经网络又称为深度神经网络。通常情况是第 s 层只有一个神经元，即 l = 1。

前馈神经网络的矩阵表示如下:

$$
\left\{ \begin{array}{l} \boldsymbol {h} ^ {(1)} = f ^ {(1)} (\boldsymbol {x}) = a (\boldsymbol {z} ^ {(1)}) = a \left(\boldsymbol {W} ^ {(1) ^ {\mathrm{T}}} \boldsymbol {x} + \boldsymbol {b} ^ {(1)}\right) \\ \boldsymbol {h} ^ {(2)} = f ^ {(2)} \left(\boldsymbol {h} ^ {(1)}\right) = a (\boldsymbol {z} ^ {(2)}) = a \left(\boldsymbol {W} ^ {(2) ^ {\mathrm{T}}} \boldsymbol {h} ^ {(1)} + \boldsymbol {b} ^ {(2)}\right) \\ \vdots \\ \boldsymbol {h} ^ {(s - 1)} = f ^ {(s - 1)} \left(\boldsymbol {h} ^ {(s - 2)}\right) = a (\boldsymbol {z} ^ {(s - 1)}) = a \left(\boldsymbol {W} ^ {(s - 1) ^ {\mathrm{T}}} \boldsymbol {h} ^ {(s - 2)} + \boldsymbol {b} ^ {(s - 1)}\right) \\ \boldsymbol {y} = \boldsymbol {h} ^ {(s)} = f ^ {(s)} \left(\boldsymbol {h} ^ {(s - 1)}\right) = g (\boldsymbol {z} ^ {(s)}) = g \left(\boldsymbol {W} ^ {(s) ^ {\mathrm{T}}} \boldsymbol {h} ^ {(s - 1)} + \boldsymbol {b} ^ {(s)}\right) \end{array} \right.\tag{23.13}
$$

其中，向量 x 表示输入，向量 y 表示输出，向量 $z^{(1)}, z^{(2)}, \cdots, z^{(s)}$ 表示第 1 层到第 s 层的净输入，向量 $h^{(1)}, h^{(2)}, \cdots, h^{(s)}$ 表示第 1 层到第 s 层的输出，矩阵 $W^{(1)}, W^{(2)}, \cdots, W^{(s)}$ 表示权重，向量 $b^{(1)}, b^{(2)}, \cdots, b^{(s)}$ 表示偏置。整体神经网络由复合函数 $f^{(s)}(\cdots f^{(2)}(f^{(1)}(\boldsymbol{x}))) \cdots)$ 表示，也写作 $f(\boldsymbol{x}; \boldsymbol{\theta})$ ，其中 $\theta$ 是所有参数组成的向量。

可以看出，前馈神经网络模型是矩阵与向量的乘积的非线性变换的多次重复，其基本结构是非常简单的。因此，多次非线性变换是前馈神经网络的本质。从后面的例子中可以看出，这种变换拥有很强的表示能力，可以进行复杂的信息处理。相比之下，多次线性变换等价于一次线性变换，其表示能力有限。

到目前为止考虑的是一个样本输入到神经网络的情况，这时输入由一个向量表示。也可以是多个样本批量同时输入到神经网络，这时输入样本由一个矩阵表示。可以用式（23.13）的矩阵表示扩展，细节省略，在例 23.3 中介绍。

## 3. 隐层的神经元

隐层神经元函数由两部分组成: 仿射函数和激活函数。这里介绍常用的隐层激活函数, 包括 S 型函数、双曲正切函数、整流线性函数。一个神经网络通常采用一种隐层激活函数。

S 型函数（sigmoid function）又称为逻辑斯谛函数（logistic function），是定义式如下的非线性函数。

$$
a (z) = \sigma (z) = \frac {1}{1 + \mathrm{e} ^ {- z}}\tag{23.14}
$$
其中，z 是自变量或输入， $\sigma(z)$ 是因变量或输出。函数的定义域为 $(-∞,+∞)$ ，值域为 $(0,1)$ 。如图 23.5 所示，S 型函数是将正负实数值映射到 0 和 1 之间实数值的单调递增函数，其特点是输入越接近 $+∞$ ，输出越接近 1；输入越接近 $-∞$ ，输出越接近 0；输入在 0 附近时，输出近似线性递增；输入是 0 时，输出是 0.5。

![](../images/a3da7eb0dc4b288f4eb7e689ce662d1cfadeed2fdc378321e3cd538c776976ef.jpg)  
图23.5 S型函数

S 型函数的导函数是

$$
a ^ {\prime} (z) = a (z) (1 - a (z))\tag{23.15}
$$

双曲正切函数（hyperbolic tangent function）是定义式如下的非线性函数。

$$
a (z) = \tanh (z) = \frac {\mathrm{e} ^ {z} - \mathrm{e} ^ {- z}}{\mathrm{e} ^ {z} + \mathrm{e} ^ {- z}}\tag{23.16}
$$

其中，z 是自变量或输入， $\tanh(z)$ 是因变量或输出。函数定义域为 $(-∞,+∞)$ ，值域为 $(-1,+1)$ 。如图 23.6 所示，双曲正切函数的特点是输入越接近 $+∞$ ，输出越接近 +1；输入越接近 $-∞$ ，输出越接近 -1；输入在 0 附近时，输出近似线性递增；输入是 0 时，输出也是 0。

![](../images/93f8f0247e3f9eb6e36cf8cef26966fa6a9396905cd278d1285fc0cd5cc1f39d.jpg)  
图23.6 双曲正切函数

双曲正切函数的导函数是

$$
a ^ {\prime} (z) = 1 - a (z) ^ {2}\tag{23.17}
$$

双曲正切函数与 S 型函数有以下关系: 直观上双曲正切函数将 S 型函数 “放大” 两倍, 并向下平移 1 个单位。

$$
\tanh (z) = 2 \sigma (2 z) - 1\tag{23.18}
$$

整流线性函数（rectified linear unit, ReLU）是定义式如下的非线性函数。

$$
a (z) = \operatorname{relu} (z) = \max (0, z)\tag{23.19}
$$

其中，z 是自变量或输入， $\mathrm{relu}(z)$ 是因变量或输出。函数定义域为 $(-∞,+∞)$ ，值域为 $(0,+∞)$ 。如图 23.7 所示，整流线性函数的特点是输入是负值时，输出为 0；输入是正值时，输出为正且线性单调递增；输入是 0 时，输出也是 0。

![](../images/ce3ff89dbe707b26aed9fddf8a4a4f3703512f0c846d66bea662c2e30d8f5a4b.jpg)  
图23.7 整流线性函数

整流线性函数的导函数是①

$$
a ^ {\prime} (z) = \left\{ \begin{array}{l l} 1, & z > 0 \\ 0, & \text {其他} \end{array} \right.\tag{23.20}
$$

整流线性函数比 S 型函数和双曲正切函数在计算机上的计算效率更高，其导函数也是如此。整流线性函数在当前深度学习中被广泛使用。

对于激活函数 $a(z)$ , 当其导数满足 $\lim _{z \to -\infty} a'(z) = 0$ 时, 称为左饱和 (left saturating) 函数; 当其导数满足 $\lim _{z \to +\infty} a'(z) = 0$ 时, 称为右饱和 (right saturating) 函数; 同时满足左饱和、右饱和条件时称为 (两边的) 饱和 (saturating) 函数。整流线性函数是左饱和函数, S 型函数和双曲正切函数是饱和函数。

## 4. 模型

前馈神经网络可以作为机器学习模型用于不同任务，有以下几种代表情况。

(1) 用于回归。神经网络的输出层只有一个神经元, 其输出是一个实数值。神经网络表示为 $y = f(x)$ , 其中 $y \in \mathcal{R}$ 。预测时给定输入 $x$ , 计算输出 $y$ 。

(2) 用于二类分类。神经网络的输出层只有一个神经元, 其输出是一个概率值。神经网络表示为 $p = P(y = 1 | \boldsymbol{x}) = f(\boldsymbol{x})$ , 其中 $y \in \{0,1\}$ , 满足条件

$$
0 <   P (y = 1 | \boldsymbol {x}) <   1, \quad P (y = 1 | \boldsymbol {x}) + P (y = 0 | \boldsymbol {x}) = 1
$$

预测时给定输入 x，计算其属于类别 1 的概率。如果概率大于 0.5，则将输入分到类别 1，否则分到类别 0。

(3) 用于多类分类 (multi-class classification)。神经网络的输出层只有一个神经元, 神经元的输出是由 $l$ 个概率值组成的概率向量。神经网络表示为 $\mathbf{p} = [P(y_k = 1|\mathbf{x})] = f(\mathbf{x})$ , 其中 $y_k \in \{0,1\}, k = 1,2,\cdots,l$ , 满足条件

$$
\sum_ {k = 1} ^ {l} y _ {k} = 1, \quad 0 <   P (y _ {k} = 1 | \boldsymbol {x}) <   1, \quad \sum_ {k = 1} ^ {l} P (y _ {k} = 1 | \boldsymbol {x}) = 1, \quad k = 1, 2, \dots , l
$$

也就是说 $[y_1, y_2, \cdots, y_l]$ 是只有一个元素为 1，其他元素为 0 的向量，这样的向量称为独热向量（one-hot vector）， $[P(y_1 = 1|x), P(y_2 = 1|x), \cdots, P(y_l = 1|x)]$ 是定义在独热向量上的概率分布，表示输入 $x$ 属于 $l$ 个类别的概率。预测时给定输入 $x$ ，计算其属于各个类别的概率。将输入分到概率最大的类别，这时输入只可能被分到一个类别。

(4) 用于多标签分类 (multi-label classification)。神经网络的输出层有 $l$ 个神经元, 每个神经元的输出是一个概率值。神经网络表示为 $\mathbf{p} = [P(y_k = 1|\mathbf{x})] = f(\mathbf{x})$ , 其中 $y_k \in \{0,1\}, k = 1,2,\cdots,l$ , 满足条件

$$
0 <   P \left(y _ {k} = 1 | \boldsymbol {x}\right) <   1, \quad P \left(y _ {k} = 1 | \boldsymbol {x}\right) + P \left(y _ {k} = 0 | \boldsymbol {x}\right) = 1, \quad k = 1, 2, \dots , l
$$

$\left[P\left(y_{1}=1|x\right),P\left(y_{2}=1|x\right),\cdots,P\left(y_{l}=1|x\right)\right]$ 表示输入 x 分别属于 1 个类别的概率。预测时给定输入 x，计算其属于各个类别的概率。将输入分到概率大于 0.5 的所有类别，这时输入可以被分到多个类别（赋予多个标签）。

注意，在回归中神经网络的输出和模型的输出是相同的，都是实数值；在分类中神经网络的输出和模型的输出是不同的，前者是概率值，后者是类别。现实中经常对神经网络及其表示的模型不严格区分，这一点其他类型的神经网络也一样。

## 5. 输出层的神经元

输出层神经元函数由两部分组成: 仿射函数和激活函数。输出层激活函数通常使用恒等函数、S型函数、软最大化函数。在回归、二类分类、多类分类、多标签分类中, 激活函数有不同的形式。

回归时，输出层只有一个神经元，其激活函数是恒等函数，神经元函数是

$$
y = g (z) = z, z = \boldsymbol {w} ^ {(s) ^ {\mathrm{T}}} \boldsymbol {h} ^ {(s - 1)} + b ^ {(s)}\tag{23.21}
$$

这里 $\boldsymbol{w}^{(s)}$ 是权重向量， $b^{(s)}$ 是偏置， $g(\cdot)$ 是恒等函数， $\boldsymbol{h}^{(s-1)}$ 是第 s-1 隐层的输出。称这样的输出层为线性输出层。

二类分类时, 输出层只有一个神经元, 其激活函数是 S 型函数, 神经元函数是

$$
P (y = 1 | \pmb {x}) = g (z) = \frac {1}{1 + \mathrm{e} ^ {- z}}, z = \pmb {w} ^ {(s) ^ {\mathrm{T}}} \pmb {h} ^ {(s - 1)} + b ^ {(s)}\tag{23.22}
$$

这里 $\boldsymbol{w}^{(s)}$ 是权重向量， $b^{(s)}$ 是偏置， $g(\cdot)$ 是 S 型函数， $\boldsymbol{h}^{(s-1)}$ 是第 s-1 隐层的输出。

多类分类时, 输出层只有一个神经元, 其激活函数是软最大化函数 (softmax function), 神经元函数是

$$
P (y _ {k} = 1 | \boldsymbol {x}) = g \left(z _ {k}\right) = \frac {\mathrm{e} ^ {z _ {k}}}{l}, z _ {k} = \boldsymbol {w} _ {k} ^ {(s) ^ {\mathrm{T}}} \boldsymbol {h} ^ {(s - 1)} + b _ {k} ^ {(s)}, k = 1, 2, \dots , l   \sum_ {i = 1} ^ {n} \mathrm{e} ^ {z _ {i}}\tag{23.23}
$$

这里 $\boldsymbol{w}_{k}^{(s)}$ 是权重向量， $b_{k}^{(s)}$ 是偏置， $g(\cdot)$ 是软最大化函数， $\boldsymbol{h}^{(s-1)}$ 是第 s-1 隐层的输出。称这样的输出层为软最大化输出层。

软最大化函数的名字来自它是最大化（max）函数的近似这一事实。如果 $z_{k} \gg z_{j}, j \neq k$ ，那么 $p_{k} = P(y_{k} = 1) \approx 1, p_{j} = P(y_{j} = 1) \approx 0$ 。软最大化函数是 $l$ 维实数向量 $z$ 到 $l$ 维概率向量 $p$ 的映射。

$$
\boldsymbol {z} = \left[\begin{array}{c}z _ {1}\\z _ {2}\\\vdots\\z _ {l}\end{array}\right]\rightarrow \boldsymbol {p} = \left[\begin{array}{c}p _ {1}\\p _ {2}\\\vdots\\p _ {l}\end{array}\right]
$$

软最大化函数的偏导数或雅可比矩阵元素是 (推导见附录 F)

$$
\frac {\partial p _ {k}}{\partial z _ {j}} = \left\{ \begin{array}{l l} p _ {k} (1 - p _ {k}), & j = k \\ - p _ {j} p _ {k}, & j \neq k \end{array} , \quad j, k = 1, 2, \dots , l \right.\tag{23.24}
$$

前馈神经网络用于多类分类时, 为了提高效率, 在预测时经常省去激活函数的计算, 选取仿射函数 (净输入) $z_{k}$ 值最大的类别。这样做, 分类结果是等价的, 因为软最大化函数的分母对各个类别是常量, 而分子的指数函数是单调递增函数。实数值 $z_{k}$ 又称为对数几率 (logit)。

多标签分类时, 输出层有 $l$ 个神经元, 每个神经元的激活函数都是 S 型函数, 神经元函数是

$$
P (y _ {k} = 1 | \boldsymbol {x}) = g (z _ {k}) = \frac {1}{1 + \mathrm{e} ^ {- z _ {k}}}, \quad z _ {k} = \boldsymbol {w} _ {k} ^ {(s) ^ {\mathrm{T}}} \boldsymbol {h} ^ {(s - 1)} + b _ {k} ^ {(s)}, \quad k = 1, 2, \dots , l\tag{23.25}
$$

这里 $\boldsymbol{w}_{k}^{(s)}$ 是权重向量， $b_{k}^{(s)}$ 是偏置， $g(\cdot)$ 是 S 型函数， $\boldsymbol{h}^{(s-1)}$ 是第 s-1 隐层的输出。

## 23.1.2 前馈神经网络的例子

例 23.2 图 23.8 是一个二层前馈神经网络, 画出其函数的三维图形, 其中第一层激活函数和第二层激活函数都是 S 型函数。

![](../images/25780f021be3bce38f1624485fbec370c31dc3ee894312c5c91e319413d1343d.jpg)  
图 23.8 二层前馈神经网络例

解 第一层 (隐层) 的第一个神经元与例 23.1 相同, 仿射函数是 $z = -x_{1} + 2x_{2} + 1$ , 激活函数是 $a(z) = \frac{1}{1 + \mathrm{e}^{-z}}$ , 神经元函数是

$$
h _ {1} ^ {(1)} = \frac {1}{1 + \mathrm{e} ^ {x _ {1} - 2 x _ {2} - 1}}
$$

图23.3是该神经元的三维图形。第一层的第二个神经元的仿射函数是 $z = -4x_{1} - x_{2} + 2$ 激活函数是 $a(z) = \frac{1}{1 + \mathrm{e}^{-z}}$ ，神经元函数是

$$
h _ {2} ^ {(1)} = \frac {1}{1 + \mathrm{e} ^ {4 x _ {1} + x _ {2} - 2}}
$$

图 23.9 是该神经元的三维图形。

![](../images/1779fb8483a6075a52cf6d917d7c25186e1a98bf64f4737b4d8c069ef4dd42d6.jpg)  
图 23.9 神经元的三维图形 (见文前彩图)

第二层（输出层）的神经元的仿射函数是 $z = h_{1}^{(1)} + h_{2}^{(1)}$ ，激活函数是 $g(z) = \sigma(z)$ ，神经元函数是 $f(x_{1}, x_{2}) = \sigma(h_{1}^{(1)} + h_{2}^{(1)})$ 。神经网络整体

$$
f (x _ {1}, x _ {2}) = \sigma \left(\frac {1}{1 + \mathrm{e} ^ {x _ {1} - 2 x _ {2} - 1}} + \frac {1}{1 + \mathrm{e} ^ {4 x _ {1} + x _ {2} - 2}}\right)
$$

是一个二类分类模型。图 23.10 是神经网络的三维图形。图形中 “高原” 部分的输出值接近

1, “盆地”部分的输出值接近0。可以看出，整个二层前馈神经网络能够比第一层的两个神经元表示更复杂的非线性关系。

![](../images/9e4cedc2543b67f1612b2ad610c48169374ade9ac9fb8425c30aaaab3b34596d.jpg)  
图 23.10 前馈神经网络例的三维图形 (见文前彩图)

例 23.3 构建一个前馈神经网络实现逻辑表达式 XOR 的功能。

解 采用矩阵表示, 构建一个二层前馈神经网络, 第一层有两个神经元, 其激活函数是整流线性函数, 第二层有一个神经元, 其激活函数是恒等函数, 如图 23.11 所示。

![](../images/73039534699048780b05643ce8ff794e086f9ff4b8915136d0fb49510282ca10.jpg)  
图 23.11 XOR 神经网络例

第一层的权重矩阵和偏置向量是

$$
\boldsymbol {W} ^ {(1)} = \left[ \begin{array}{c c} 1 & 1 \\ 1 & 1 \end{array} \right]
$$

$$
\boldsymbol {b} ^ {(1)} = \left[ \begin{array}{c} 0 \\ - 1 \end{array} \right]
$$

第二层的权重矩阵和偏置向量是

$$
\boldsymbol {W} ^ {(2)} = \left[ \begin{array}{c} 1 \\ - 2 \end{array} \right]
$$

$$
\pmb {b} ^ {(2)} = [ 0 ]
$$

用矩阵表示四种可能的输入:

$$
\boldsymbol {X} = \left[ \begin{array}{l l l l} 0 & 0 & 1 & 1 \\ 0 & 1 & 0 & 1 \end{array} \right]
$$

代表批量处理。代入神经网络，第一层输出是

$$
\begin{array}{l} \boldsymbol {H} ^ {(1)} = \operatorname{relu} \left(\boldsymbol {W} ^ {(1) ^ {\mathrm{T}}} \boldsymbol {X} + \boldsymbol {B} ^ {(1)}\right) \\ = \operatorname{relu} \left(\left[ \begin{array}{l l} 1 & 1 \\ 1 & 1 \end{array} \right] \left[ \begin{array}{l l l l} 0 & 0 & 1 & 1 \\ 0 & 1 & 0 & 1 \end{array} \right] + \left[ \begin{array}{r r r r} 0 & 0 & 0 & 0 \\ - 1 & - 1 & - 1 & - 1 \end{array} \right]\right) \\ = \operatorname{relu} \left(\left[ \begin{array}{r r r r} 0 & 1 & 1 & 2 \\ - 1 & 0 & 0 & 1 \end{array} \right]\right) \\ = \left[ \begin{array}{l l l l} 0 & 1 & 1 & 2 \\ 0 & 0 & 0 & 1 \end{array} \right] \end{array}
$$

其中 relu 计算对矩阵的每一个元素进行。第二层输出是

$$
\begin{array}{r l} & {\pmb {H} ^ {(2)} = \pmb {W} ^ {(2) ^ {\mathrm{T}}} \pmb {H} ^ {(1)} + \pmb {B} ^ {(2)}} \\ & {\qquad = \left[ \begin{array}{l l} 1 & - 2 \end{array} \right] \left[ \begin{array}{l l l l} 0 & 1 & 1 & 2 \\ 0 & 0 & 0 & 1 \end{array} \right] + \left[ \begin{array}{l l l l} 0 & 0 & 0 & 0 \end{array} \right]} \\ & {\qquad = \left[ \begin{array}{l l l l} 0 & 1 & 1 & 0 \end{array} \right]} \end{array}
$$

表示四种可能的输出。可以看出这个二层神经网络实现了 XOR 功能。图 23.12 是 XOR 神经网络函数的三维图形。作为线性模型的感知机不能实现 XOR 是众所周知的事实，而作为非线性模型的前馈神经网络可以实现 XOR，并且以很简单的方式实现。

![](../images/06584e727751160ee6db1616247a9a8ddc619983172943cb4b98af33d26e7943.jpg)  
图 23.12 XOR 神经网络例的三维图形 (见文前彩图)

例 23.4 手写数字识别网络。

MNIST 是一个机器学习标准数据集。每一个样本由一个像素为 $28 \times 28$ 的手写数字灰度图像以及对应的 $0 \sim 9$ 之间的标签组成, 像素取值为 $0 \sim 255$ , 如图 23.13 所示。

![](../images/23f56deb86c3b939d223f70db89a14d52234cd8f8a19aeaca22a3e8789815266.jpg)  
图 23.13 MNIST 手写数字数据例

可以构建图 23.14 所示的前馈神经网络对 MNIST 的手写数字进行识别, 是一个多标签分类模型。输入层是一个 $28 \times 28 = 784$ 维向量, 取自一个图像, 每一维对应一个像素。第一层和第二层是隐层, 各自有 100 个神经元和 50 个神经元, 其激活函数都是 S 型函数。第三层是输出层, 有 10 个神经元, 其激活函数也是 S 型函数。给定一个图像, 神经网络可以计算出其属于 $0 \sim 9$ 类的概率, 将图像赋予概率最大的标签。分类准确率能达到 $90 \%$ 以上。

![](../images/5fe529c319d5abb1a852a8cf56e91ae04974b656d68f6bd0446d44cdac8beafb.jpg)  
图 23.14 手写数字识别网络

从以上的例子可以看出，前馈神经网络拥有很强的语义表示能力，简单的网络就可以表示逻辑关系、抽象数字，进行简单逻辑推理、数字识别等处理。预测时，神经网络的每一层通过非线性变换对输入进行特征转换，多次这样的特征转换产生最终的判断结果。

## 23.1.3 前馈神经网络的表示能力

## 1. 与其他模型的关系

前馈神经网络与逻辑斯谛回归模型、感知机、支持向量机等有密切关系。

对于多类分类的一层神经网络，当其输出层激活函数是软最大化函数时，模型等价于多项逻辑斯谛回归模型。

$$
P \left(y _ {k} = 1 | \boldsymbol {x}\right) = f (\boldsymbol {x}) = \frac {\mathrm{e} ^ {\left(\boldsymbol {w} _ {k} ^ {(1) ^ {\mathrm{T}}} \boldsymbol {x} + b _ {k} ^ {(1)}\right)}}{\sum_ {i = 1} ^ {l} \mathrm{e} ^ {\left(\boldsymbol {w} _ {i} ^ {(1) ^ {\mathrm{T}}} \boldsymbol {x} + b _ {i} ^ {(1)}\right)}}, \quad k = 1, 2, \dots , l\tag{23.26}
$$

所以，前馈神经网络是逻辑斯谛回归模型的扩展。注意：前馈神经网络通常将所有 $l$ 个类别的权重和偏置作为参数使用，而逻辑斯谛回归通常将前 $l - 1$ 个类别的权重和偏置作为自由参数使用（见6.1.4节）。

对于二类分类的一层神经网络，当其输出层激活函数是双曲正切函数时，

$$
f (\boldsymbol {x}) = \tanh (\boldsymbol {w} ^ {(1) ^ {\mathrm{T}}} \boldsymbol {x} + b ^ {(1)})\tag{23.27}
$$

模型可以与感知机对应。感知机模型的定义是

$$
y = \left\{ \begin{array}{l l} {+ 1,} & {\mathrm{sign} (\pmb {w} ^ {\mathrm{T}} \pmb {x} + b) \geqslant 0} \\ {- 1,} & {\text {其他}} \end{array} \right.
$$

所以，可以认为前馈神经网络是感知机的扩展。这也是前馈神经网络又被称为多层感知机的原因。

对于二类分类的多层神经网络，当其输出层激活函数是双曲正切函数时，

$$
\begin{array}{l} f (\boldsymbol {x}) = \tanh (\boldsymbol {w} ^ {(s) ^ {\mathrm{T}}} \boldsymbol {h} ^ {(s - 1)} + b ^ {(s)}) \\ \boldsymbol {h} ^ {(s - 1)} = f ^ {(s - 1)} [ \dots f ^ {(2)} (f ^ {(1)} (\boldsymbol {x})) \dots ] \end{array}\tag{23.28}
$$

模型可以与非线性支持向量机对应。非线性支持向量模型的定义是

$$
y = \left\{ \begin{array}{l l} {+ 1,} & {\mathrm{sign} (\pmb {w} ^ {\mathrm{T}} \phi (\pmb {x}) + b) \geqslant 0} \\ {- 1,} & {\text {其他}} \end{array} \right.
$$

其中, $\phi(x)$ 是从输入空间到特征空间的非线性映射，w 和 b 是模型的参数。前馈神经网络的前 s-1 层函数 $f^{(s-1)}[\cdots f^{(2)}(f^{(1)}(\mathbf{x}))\cdots]$ 与映射函数 $\phi(\mathbf{x})$ 对应。支持向量机学习是凸优化问题，保证可以找到全局最优，而前馈神经网络学习是非凸优化问题（见 23.2 节），不能保证找到全局最优。前馈神经网络比支持向量机有更多的参数可以调节。

## 2. 函数近似能力

前馈神经网络具有强大的函数近似能力。通用近似定理（universal approximation theorem）指出，存在一个二层前馈神经网络，具有一个线性输出层和一个隐层，其中隐层含有充分数量的神经元，激活函数为挤压函数，这个网络可以以任意精度近似任意一个在紧的定义域上的连续函数[7-8]。从这个意义上，前馈神经网络的函数近似能力是通用的。

设有实函数 $G(x)$ : $R \rightarrow [0,1]$ ，如果 $G(x)$ 是非减函数，且满足 $\lim_{x \to -\infty} G(x) = 0$ ， $\lim_{x \to +\infty} G(x) = 1$ ，则称函数 $G(x)$ 为挤压函数 (squashing function)。S 型函数是一种挤压函数。

后续理论研究发现，定理的条件可以放宽，当激活函数是多项式函数以外的其他函数时，或者当被近似函数是波莱尔可测函数时，定理的结论依然成立。波莱尔可测函数包括连续函数、分段连续函数、阶梯函数。

下面的定理是通用近似定理的一个具体形式。

定理 23.1 对任意连续函数 $h: [0,1]^{n} \rightarrow R$ 和任意 $\varepsilon > 0$ ，存在一个二层前馈神经网络：

$$
\begin{array}{l} f (\boldsymbol {x}) = \boldsymbol {\alpha} ^ {\mathrm{T}} \sigma \left(\boldsymbol {W} ^ {\mathrm{T}} \boldsymbol {x} + \boldsymbol {b}\right) \\ = \sum_ {j} \alpha_ {j} \sigma \left(\sum_ {i} w _ {j i} x _ {i} + b _ {j}\right) \end{array}
$$

使得对于任意 $x \in [0,1]^n$ , 有 $|\pmb{h}(x) - f(\pmb{x})| < \varepsilon$ 成立。这里隐层的激活函数是 S 型函数。

下面给出定理 23.1 的直观解释。假设 $h(x)$ 是一个连续函数，定义域是区间 $[0,1]$ ，值域是区间 $[0,1]$ ，可以用二层前馈神经网络 $f(x)$ 以任意精度近似 $h(x)$ 。(为了简单，这里假设 $h(x)$ 取正值，也可以取负值)

![](../images/09a1991d8e3d61a8a90d690f004b6c411d59962e0e68b5016e07a06e5ebc5aa0.jpg)

![](../images/c85de2b047a04c6d841ba074837c11dd098337be44a546792956b0c54df724f4.jpg)  
图23.15 用阶梯函数近似连续函数

如图 23.15(a) 所示, 可以用阶梯函数以任意精度近似函数 $h(x)$ 。如图 23.15(b) 所示, 假设阶梯函数第 $i$ 个分段函数是

$$
s _ {i} \left(x\right) = \left\{ \begin{array}{l l} {\alpha_ {i},} & {x _ {i - 1} <   x \leqslant x _ {i}} \\ {0,} & {\text {其他}} \end{array} \right.
$$

则该分段函数可以由以下二层神经网络近似。

$$
f _ {i} (x) = \alpha_ {i} \cdot \sigma (w \cdot x - x _ {i - 1}) - \alpha_ {i} \cdot \sigma (w \cdot x - x _ {i})
$$

其中隐层有两个神经元，其激活函数是 S 型函数，输出层是线性的。这里参数 $x_{i-1}$ 和 $x_{i}$ 保证与分段函数的区间一致，参数 $\alpha_{i}$ 保证趋近分段函数，参数 w 控制与分段函数的趋近程度。这样，阶梯函数的每一分段函数 $s_{i}(x)$ 都可以用一个二层神经网络 $f_{i}(x)$ 近似，函数 $h(x)$ 整体也可以由所有 $f_{i}(x)$ 相加得到的二层神经网络 $f(x)$ 近似。

通用近似定理叙述的是理论存在性，并不意味着现实可行性。定理 23.1 中近似连续函数的二层前馈神经网络的隐层神经元的个数可能是非常大的，甚至是指数级的，参数个数也是如此，现实中不会有足够多的数据训练这样的网络。经验上，当前馈神经网络的层数增大时，也就是变成深度神经网络时，可以解决这个问题。

## 3. 函数等价性

前馈神经网络 $\boldsymbol{y} = f(\boldsymbol{x}; \boldsymbol{\theta})$ 有大量的等价的函数，即参数 $\theta$ 不同但对相同的输入 x 产生相同的输出 y，而且等价函数的个数是指数级的。

假设某个隐层有 $m$ 个神经元, 其所有参数由向量表示。这一层有 $m!$ 个神经元的排列,每一个排列决定一个参数向量, 因此有 $m!$ 种不同的参数向量。改变神经元的排列, 参数向量发生变化, 但神经网络的输入输出的映射关系不变, 这时隐层有 $m!$ 个等价的参数向量。

假设某个隐层有 $m$ 个神经元, 其所有参数由向量表示, 激活函数是双曲正切函数。双曲正切函数是奇函数, 即满足 $\tanh (-z) = -\tanh (z)$ 。若这一层的某一个神经元的参数以及相连的后一层神经元的参数都反号, 则对相同的输入, 神经网络的输出不变。这时隐层 (与后一层一起) 有 $2^{m}$ 个等价的参数向量。如果同时考虑神经元的不同排列, 这个隐层共有 $m!2^{m}$ 个等价的参数向量。

当神经网络有多个隐层的时候, 整体的等价函数的个数由各层的等价参数向量个数的乘积决定。

## 4. 网络的深度

前馈神经网络的深度指网络的层数，复杂度指神经元的个数。复杂度也代表神经网络的参数个数，因为参数个数与神经元个数成正比。深度神经网络与“浅度神经网络”可以有同等的表示能力，但深度神经网络比浅度神经网络有更低的复杂度。这一点可以由逻辑门电路理论间接论证。

定理 23.2 存在这样的布尔函数, 可以由深度为 $k$ 的多项式复杂度的逻辑门电路表示,等价地深度为 $k - 1$ 的逻辑门电路变为指数复杂度。其中深度指从输入到输出的最长路径的长度, 复杂度指逻辑门的个数。

这里给出定理的直观解释。假设有图 23.16 所示的逻辑门电路，其深度是 3，复杂度是 9。如果根据以下关系，将 AND 的 OR 子电路（可由合取范式表示）转换成等价的 OR 的 AND 子电路（可由析取范式表示）：

$$
\begin{array}{r} (x _ {1} + \bar {x} _ {2}) (\bar {x} _ {3} + x _ {4}) (x _ {5} + x _ {6}) = x _ {1} \bar {x} _ {3} x _ {5} + x _ {1} \bar {x} _ {3} x _ {6} + x _ {1} x _ {4} x _ {5} + x _ {1} x _ {4} x _ {6} + \\ \bar {x} _ {2} \bar {x} _ {3} x _ {5} + \bar {x} _ {2} \bar {x} _ {3} x _ {6} + \bar {x} _ {2} x _ {4} x _ {5} + \bar {x} _ {2} x _ {4} x _ {6} \end{array}
$$

![](../images/be4db2ebabc17cf35ae606699a1c27464ef3317a22adb1d3b1da5a8800ebd48b.jpg)  
图23.16 逻辑门电路例

那么得到图 23.17 所示的等价的逻辑门电路, 其深度是 2 , 复杂度是 17 。深的逻辑门电路复杂度低, 而等价的浅的逻辑门电路复杂度高。

![](../images/79797966178c4ca3d2c3fee477c674a30f9fea888a9b544ab71a81a6916d11c8.jpg)  
图 23.17 等价的逻辑门电路

逻辑门电路可以由前馈神经网络表示。定理 23.2 的推论是存在这样的深度神经网络，其等价的浅度神经网络的复杂度指数级地增加。所以，虽然深度神经网络与浅度神经网络可能有同等表示能力，但复杂度过高的浅度神经网络现实中并不可取。

在同等表示能力下，深度神经网络比浅度神经网络有更少的参数。所以，只需要更少的数据就可以学到，也就是说，深度神经网络有更低的样本复杂度（sample complexity）。这也是深度神经网络更加强大且实用的原因。

## 23.2 前馈神经网络的学习算法

本节讲述前馈神经网络学习算法。首先给出学习问题的定义，之后介绍学习的算法，包括一般的随机梯度下降法和具体的反向传播算法，以及在计算图上的实现，最后介绍学习的技巧。

## 23.2.1 前馈神经网络学习

## 1. 一般形式

前馈神经网络的学习和预测①是如下的监督学习问题。给定训练数据集

$$
\mathcal {T} = \left\{\left(\boldsymbol {x} _ {1}, y _ {1}\right), \left(\boldsymbol {x} _ {2}, y _ {2}\right), \dots , \left(\boldsymbol {x} _ {N}, y _ {N}\right) \right\}
$$

其中， $(x_{i},y_{i}), i=1,2,\cdots,N$ ，表示样本，由输入 $x_{i}$ 与输出 $y_{i}$ 的对组成；N 表示样本容量。学习一个前馈神经网络模型 $f(x;\hat{\theta})$ ，其中 $\hat{\theta}$ 是估计的神经网络的参数向量。用学到的模型对新的输入 $x_{N+1}$ 给出新的输出 $y_{N+1}$ 。

学习时通常假设神经网络的架构已经确定，包括网络的层数、每层的神经元数、神经元激活函数的类型。所以网络的参数已确定，需要从数据中学习或估计的是参数值。

学习问题可以形式化为以下的优化问题:

$$
\hat {\boldsymbol {\theta}} = \underset {\boldsymbol {\theta}} {\operatorname{argmin}} \left[ \sum_ {i = 1} ^ {N} L (f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}), y _ {i}) + \lambda \cdot \Omega (f) \right]\tag{23.29}
$$

其中， $L(\cdot)$ 是损失函数， $\Omega(\cdot)$ 是正则项， $\lambda\geqslant0$ 是系数。当损失函数是对数损失函数、没有正则化时，问题变成极大似然估计。这是前馈神经网络学习的一般形式。

$$
\hat {\pmb {\theta}} = \underset {\pmb {\theta}} {\mathrm{argmin}} \left[ - \sum_ {i = 1} ^ {N} \log P _ {\theta} \left(y _ {i} | \pmb {x} _ {i}\right) \right]\tag{23.30}
$$

这里 $P_{\theta}(y|x)$ 表示输入 x 给定条件下输出 y 的条件概率，由神经网络决定； $\theta$ 是神经网络的参数。

## 2. 具体形式

针对不同的问题，前馈神经网络学习的一般形式可以转化成不同的具体形式。

当问题是回归时, 模型的输入是实数向量 $x$ , 输出是实数值 $y$ 。神经网络 $f(x; \theta)$ 决定输入给定条件下输出的条件概率分布 $P_{\theta}(y|x)$ 。假设条件概率分布 $P_{\theta}(y|x)$ 遵循高斯分布:

$$
P _ {\boldsymbol {\theta}} (y | \boldsymbol {x}) \sim N (f (\boldsymbol {x}; \boldsymbol {\theta}), \sigma^ {2})
$$

其中， $y \in (-\infty, +\infty)$ ， $f(x; \theta)$ 是均值， $\sigma^2$ 是方差。学习问题（极大似然估计）变为优化问题：

$$
\hat {\boldsymbol {\theta}} = \underset {\boldsymbol {\theta}} {\operatorname{argmin}} \left[ \frac {1}{2 \sigma^ {2}} \sum_ {i = 1} ^ {N} (y _ {i} - f (\boldsymbol {x} _ {i}; \boldsymbol {\theta})) ^ {2} + N \log \sigma + \frac {N}{2} \log 2 \pi \right]\tag{23.31}
$$

假设方差 $\sigma^{2}$ 固定不变，有等价的优化问题：

$$
\hat {\boldsymbol {\theta}} = \underset {\boldsymbol {\theta}} {\operatorname{argmin}} \sum_ {i = 1} ^ {N} \frac {1}{2} (y _ {i} - f (\boldsymbol {x} _ {i}; \boldsymbol {\theta})) ^ {2}\tag{23.32}
$$

从另一个角度看，前馈神经网络用于回归时，使用平方损失（square loss）作为损失函数，学习进行的是平方损失的最小化。

当问题是二类分类时, 模型的输入是实数向量 $x$ , 输出是类别 $y \in \{0,1\}$ , 神经网络 $f(x; \theta)$ 决定输入给定条件下类别的条件概率分布:

$$
p = P _ {\boldsymbol {\theta}} (y = 1 | \boldsymbol {x}) = f (\boldsymbol {x}; \boldsymbol {\theta})\tag{23.33}
$$

假设条件概率分布 $P_{\theta}(y = 1|x)$ 遵循贝努利分布, 学习问题 (极大似然估计) 变为优化问题:

$$
\hat {\boldsymbol {\theta}} = \underset {\boldsymbol {\theta}} {\operatorname{argmin}} \left\{- \sum_ {i = 1} ^ {N} \left[ y _ {i} \log f (\boldsymbol {x}; \boldsymbol {\theta}) + (1 - y _ {i}) \log (1 - f (\boldsymbol {x}; \boldsymbol {\theta})) \right] \right\}\tag{23.34}
$$

这时损失函数是交叉熵（cross entropy）损失。离散分布的交叉熵的一般定义是 $-\sum_{k=1}^{l} P_{k} \log Q_{k}$ ，表示经验分布和预测分布的差异，其中 $Q_{k}$ 是预测分布的概率， $P_{k}$ 是经验分布的概率。

当问题是多类分类时, 模型的输入是实数向量 $x$ , 输出是类别 $y_{k} \in \{0,1\}, k = 1,2,\dots ,l,$ $\sum_{k=1}^{l} y_{k} = 1$ , 神经网络 $f(x; \theta)$ 表示输入给定条件下类别的条件概率分布:

$$
\boldsymbol {p} = P _ {\boldsymbol {\theta}} \left(y _ {k} = 1 | \boldsymbol {x}\right) = f (\boldsymbol {x}; \boldsymbol {\theta})\tag{23.35}
$$

假设条件概率分布 $P_{\pmb{\theta}}(y_k = 1|\pmb{x})$ 遵循类别分布（categorical distribution），学习问题变为优化问题：

$$
\hat {\boldsymbol {\theta}} = \underset {\boldsymbol {\theta}} {\operatorname{argmin}} \left\{- \sum_ {i = 1} ^ {N} \left[ \sum_ {k = 1} ^ {l} y _ {i k} \log f (\boldsymbol {x}; \boldsymbol {\theta}) \right] \right\}\tag{23.36}
$$

其中， $y_{ik} \in \{0,1\}$ ， $\sum_{k=1}^{l} y_{ik} = 1, k = 1, 2, \cdots, l, i = 1, 2, \cdots, N$ 。所以，前馈神经网络用于二类和多类分类时以交叉熵为损失函数，进行的是交叉熵的最小化。

## 23.2.2 前馈神经网络学习的优化算法

## 1. 非凸优化问题

前馈神经网络学习变成给定网络架构 $f(\pmb{x};\pmb{\theta})$ 、训练数据集 $\mathcal{T}$ 的条件下, 最小化目标函数 $L(\pmb{\theta})$ , 得到最优参数 $\hat{\pmb{\theta}}$ 的优化问题 (最小化问题)。

$$
\hat {\boldsymbol {\theta}} = \underset {\boldsymbol {\theta}} {\operatorname{argmin}} L (\boldsymbol {\theta}) = \underset {\boldsymbol {\theta}} {\operatorname{argmin}} \frac {1}{N} \sum_ {i = 1} ^ {N} L (f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}), y _ {i})\tag{23.37}
$$

前馈神经网络学习的目标函数一般是非凸函数，优化问题是非凸优化。从前馈神经网络的等价性可以得知，一个神经网路通常有大量等价的参数向量，所以其学习的优化问题有大量等价的局部最优点（最小点）。

![](../images/1fa01d99990276de5302c49cc66172a8a808b8fe25fa11f3af8cb00c42b299d8.jpg)  
图 23.18 非凸优化问题 (见文前彩图)

图 23.18 示意了神经网络的非凸优化问题。参数向量是 $(\theta_{1}, \theta_{2})$ ，目标函数是 $L(\theta_{1}, \theta_{2})$ ，全局最小点是深蓝色，局部最小点是浅蓝色。因为目标函数非凸，有许多局部最小点。

## 2. 梯度下降法和随机梯度下降法

深度学习包括前馈神经网络学习，均使用迭代优化算法，包括梯度下降法（gradient descent）和随机梯度下降法（stochastic gradient descent），后者更为常用（附录 A 给出梯度下降法的一般介绍）。

优化目标函数写作

$$
L (\boldsymbol {\theta}) = \frac {1}{N} \sum_ {i = 1} ^ {N} L _ {i} (\boldsymbol {\theta}) = \frac {1}{N} \sum_ {i = 1} ^ {N} L (f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}), y _ {i})\tag{23.38}
$$

其中， $L_{i}(\theta)$ 是第 i 个样本的损失函数。

梯度下降法首先随机初始化参数向量 $\theta$ ; 之后针对所有样本, 通过以下公式更新参数向量 $\theta$ ; 不断迭代, 直到收敛为止。

$$
\boldsymbol {\theta} \leftarrow \boldsymbol {\theta} - \eta \frac {\partial L (\boldsymbol {\theta})}{\partial \boldsymbol {\theta}}\tag{23.39}
$$

或写作

$$
\boldsymbol {\theta} \leftarrow \boldsymbol {\theta} - \eta \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {\partial L _ {i} (\boldsymbol {\theta})}{\partial \boldsymbol {\theta}}\tag{23.40}
$$

其中， $\eta > 0$ 是学习率， $\frac{\partial L(\theta)}{\partial \theta}$ 是所有样本的损失函数的梯度向量， $\frac{\partial L_i(\theta)}{\partial \theta}$ 是第 $i$ 个样本的损失函数的梯度向量。算法23.1给出梯度下降的具体算法。

梯度下降的基本想法如下。由于负梯度方向 $-\frac{\partial L(\theta)}{\partial \theta}$ 是使函数值下降的方向，所以每一次迭代以负梯度更新参数向量 $\theta$ 的值，从而达到减少函数值 $L(\theta)$ 的目的。函数极小值满足 $\nabla L(\theta) = 0$ 。在迭代过程中，梯度向量趋近 0 向量，参数向量 $\theta$ 也趋近极小点。学习率控制参数更新的幅度。学习率的大小需要适当，学习率过小，参数向量每次更新的幅度会过小，迭代的次数会增加；学习率过大，参数向量每次更新的幅度会过大，产生振荡，迭代的次数也会增加。图 23.18 显示梯度下降的过程。

算法 23.1（梯度下降法）

输入：网络架构 $f(x;\theta)$ ，训练数据集 T。

输出：神经网络参数向量 $\hat{\theta}$ 。

超参数：学习率 $\eta$ 。

1. 随机初始化参数向量 $\theta$ ;

2. Do while( $\theta$ 不收敛) {

针对所有样本，按照以下公式，更新参数向量 $\theta$

$$
\theta \leftarrow \theta - \eta \frac {\partial L (\theta)}{\partial \theta}
$$

3. 返回学习到的参数向量 $\hat{\theta}$ 。

梯度下降用于深度学习（一般的机器学习）有两个缺点。每次迭代需要计算针对所有样本的梯度向量，计算效率不高；得到的解可能是局部最优，不能保证是全局最优。随机梯度下

降能很好地解决这两个问题。

随机梯度下降法首先随机打乱样本顺序，将样本分成 m 个组（小批量），每一组有 n 个样本（假设 $m = \lfloor N/n \rfloor$ ）；接着随机初始化参数向量；之后针对每组样本，通过以下公式更新参数向量，并遍历所有样本组；不断迭代，直到收敛为止。

$$
\theta \leftarrow \theta - \eta \frac {1}{n} \sum_ {j = 1} ^ {n} \frac {\partial L _ {j} (\theta)}{\partial \theta}\tag{23.41}
$$

其中, $\eta > 0$ 是学习率, $\frac{\partial L_{j}(\theta)}{\partial \theta}$ 是一个组中的第 $j$ 个样本的损失函数的梯度向量。算法23.2给出随机梯度下降的具体算法。当 $n$ 是1时, 每次参数更新只使用一个样本, 是一种特殊的随机梯度下降。当 $n$ 是整体样本容量 $N$ 时, 随机梯度下降变为梯度下降 (当前深度学习采用的 Adam 等优化算法, 在随机梯度下降的迭代过程中, 自适应地调整梯度向量。第29章对 Adam 等算法予以介绍)。

算法 23.2（随机梯度下降法）

输入：网络架构 $f(x;\theta)$ ，训练数据集 T。

输出: 神经网络参数向量 $\hat{\theta}$ 。

超参数: 学习率 $\eta$ , 小批量样本容量 $n$ 。

1. 随机打乱样本顺序，将样本分成 m 个组，每一组有 n 个样本；

2. 随机初始化参数向量 $\theta$ ;

3. Do while( $\theta$ 不收敛) {

$$
\text { For } (i = 1, 2, \dots , m) \left\{ \begin{array}{l} \end{array} \right.
$$

针对第 $i$ 个组的 $n$ 个样本，按照以下公式，更新参数向量 $\theta$

$$
\boldsymbol {\theta} \leftarrow \boldsymbol {\theta} - \eta \frac {1}{n} \sum_ {j = 1} ^ {n} \frac {\partial L _ {j} (\boldsymbol {\theta})}{\partial \boldsymbol {\theta}}
$$

4. 返回学习到的参数向量 $\hat{\theta}$ 。

随机梯度下降可以进行分布式并行计算，进一步提高学习效率，特别是当训练数据量大时非常有效。具体地，每组样本分配到不同的工作服务器（worker）上，各台工作服务器基于自己的数据并行更新参数向量，参数服务器（parameter server）再将所有工作服务器的参数更新结果汇总求平均，得到一轮的训练结果。

## 23.2.3 反向传播算法

基于梯度下降或随机梯度下降的学习算法的核心是针对给定样本，计算损失函数对神经网络所有参数的梯度 $\frac{\partial L}{\partial \theta}$ ，更新神经网络的所有参数 $\theta$ 。反向传播（back propagation）算法也称为误差反向传播（error back propagation）算法，提供了一个高效的梯度计算以及参数更新方法。只需要依照网络结构进行一次正向传播（forward propagation）和一次反向传播（backward propagation），就可以完成梯度下降的一次迭代。在梯度下降的每一步，参数已在前一步更新，正向传播旨在基于当前的参数重新计算神经网络所有变量（比如，神经元的输出），反向传播旨在基于当前的变量重新计算损失函数对所有参数的梯度，这样就可以根据梯度下降公式(23.40)和公式(23.41)更新神经网络的所有参数。

考虑一个 $s$ 层神经网络（见图23.19）。其中第 $t$ 层（隐层）的神经元定义如下：

$$
h _ {j} ^ {(t)} = a \left(z _ {j} ^ {(t)}\right), \quad j = 1, 2, \dots , m\tag{23.42}
$$

$$
z _ {j} ^ {(t)} = \sum_ {i = 1} ^ {n} w _ {j i} ^ {(t)} h _ {i} ^ {(t - 1)} + b _ {j} ^ {(t)}\tag{23.43}
$$

第 $t + 1$ 层（隐层）的神经元定义如下：

$$
h _ {k} ^ {(t + 1)} = a \left(z _ {k} ^ {(t + 1)}\right), \quad k = 1, 2, \dots , l\tag{23.44}
$$

$$
z _ {k} ^ {(t + 1)} = \sum_ {j = 1} ^ {m} w _ {k j} ^ {(t + 1)} h _ {j} ^ {(t)} + b _ {k} ^ {(t + 1)}\tag{23.45}
$$

梯度下降需要计算损失函数对所有参数的梯度。损失函数对第 $t$ 层的权重和偏置的梯度分别是 $\frac{\partial L}{\partial w_{ji}^{(t)}}$ 和 $\frac{\partial L}{\partial b_j^{(t)}}$ 。根据链式规则，可以分别展开：

$$
\frac {\partial L}{\partial w _ {j i} ^ {(t)}} = \frac {\partial L}{\partial z _ {j} ^ {(t)}} \frac {\partial z _ {j} ^ {(t)}}{\partial w _ {j i} ^ {(t)}}\tag{23.46}
$$

$$
\frac {\partial L}{\partial b _ {j} ^ {(t)}} = \frac {\partial L}{\partial z _ {j} ^ {(t)}} \frac {\partial z _ {j} ^ {(t)}}{\partial b _ {j} ^ {(t)}}\tag{23.47}
$$

考虑损失函数对第 t 层的净输入 $z_{j}^{(t)}$ 的梯度：

$$
\delta_ {j} ^ {(t)} = \frac {\partial L}{\partial z _ {j} ^ {(t)}}, \quad j = 1, 2, \dots , m\tag{23.48}
$$

称为在第 $t$ 层的“误差”。求解式 (23.46) 和式 (23.47)，并代入式 (24.48)，得到：

$$
\frac {\partial L}{\partial w _ {j i} ^ {(t)}} = \delta_ {j} ^ {(t)} h _ {i} ^ {(t - 1)}\tag{23.49}
$$

$$
\frac {\partial L}{\partial b _ {j} ^ {(t)}} = \delta_ {j} ^ {(t)}\tag{23.50}
$$

其中， $h_{i}^{(t-1)}$ 是 $t-1$ 层的输出，从正向传播得到； $\delta_{j}^{(t)}$ 是第 $t$ 层的误差，从反向传播得到。所以，可以根据式 (23.49) 和式 (23.50) 计算对第 $t$ 层的权重与偏置的梯度。

正向传播是指输入从输入层到输出层的信号传递。给定神经网络参数，根据神经网络的函数计算。反向传播是指“误差”从输出层到输入层的传递。给定神经网络参数，以及正向传

播的结果，通过以下方法计算。

对于第 $t$ 层的误差 $\delta_j^{(t)}$ ，根据链式规则展开：

$$
\delta_ {j} ^ {(t)} = \frac {\partial L}{\partial z _ {j} ^ {(t)}} = \sum_ {k = 1} ^ {l} \frac {\partial L}{\partial z _ {k} ^ {(t + 1)}} \frac {\partial z _ {k} ^ {(t + 1)}}{\partial z _ {j} ^ {(t)}}, \quad j = 1, 2, \dots , m\tag{23.51}
$$

求解得到:

$$
\delta_ {j} ^ {(t)} = \frac {\mathrm{d} a}{\mathrm{d} z _ {j} ^ {(t)}} \sum_ {k = 1} ^ {l} w _ {k j} ^ {(t + 1)} \delta_ {k} ^ {(t + 1)}\tag{23.52}
$$

这里 $\delta_{k}^{(t + 1)}$ 是第 $t + 1$ 层的误差， $w_{kj}^{(t + 1)}$ 是第 $t + 1$ 层的权重， $\frac{\mathrm{d}a}{\mathrm{d}z_j^{(t)}}$ 是第 $t$ 层的激活函数的导数。也就是说可以根据式 (23.52)，从第 $t + 1$ 层的误差 $\delta_{k}^{(t + 1)}$ 计算第 $t$ 层的误差 $\delta_{j}^{(t)}$ 。

第 s 层（输出层）的误差通过以下方法计算。一般形式是

$$
\delta_ {k} ^ {(s)} = \frac {\partial L}{\partial z _ {k} ^ {(s)}}, \quad k = 1, 2, \dots , l\tag{23.53}
$$

求解得到:

$$
\delta_ {k} ^ {(s)} = \frac {\mathrm{d} g}{\mathrm{d} z _ {k} ^ {(s)}} \frac {\partial L}{\partial h _ {k} ^ {(s)}}, \quad k = 1, 2, \dots , l\tag{23.54}
$$

这里 $\frac{\partial L}{\partial h_k^{(s)}}$ 是损失函数对输出的梯度, $\frac{\mathrm{d}g}{\mathrm{d}z_k^{(s)}}$ 是第 $s$ 层的激活函数的导数。注意第 $s$ 层的输出表示为 $h^{(s)}$ 。

回归时, 输入层只有一个神经元, 有一个输出取实数值。损失函数是平方损失 $\frac{1}{2} (h^{(s)} - y)^{2}$ , 激活函数是恒等函数 $g(z) = z$ , 这时误差是

$$
\delta^ {(s)} = h ^ {(s)} - y\tag{23.55}
$$

表示第 s 层的输出 $h^{(s)}$ 与训练样本输出 y 的差。

多类分类（包含二类分类）时，输出层只有一个神经元，有 $l$ 个输出表示 $l$ 个类别的概率。损失函数是交叉熵损失 $-\sum_{k=1}^{l} y_k \log h_k^{(s)}$ ，激活函数是软最大化函数 $g(z) = \frac{\mathrm{e}^z}{\sum_{z'} \mathrm{e}^{z'}}$ ，这时误差是 (推导见附录 F)

$$
\delta_ {k} ^ {(s)} = h _ {k} ^ {(s)} - y _ {k}, \quad k = 1, 2, \dots , l\tag{23.56}
$$

表示第 s 层的输出 $h_{k}^{(s)}$ 与训练样本输出 $y_{k}$ 的差。

可以看出无论是回归还是分类, 由于特殊的激活函数与损失函数的使用, 使得 $\delta^{(s)}$ 表示预测与真实值的差。这也是损失函数 $L$ 对净输入 $z$ 的梯度 $\frac{\partial L}{\partial z}$ 被称为误差的原因。

图 23.19 显示在第 t 层的正向传播和反向传播。首先，第 t-1 层的输出通过网络正向传播到第 t 层，根据式 (23.42) 和式 (23.43) 计算出第 t 层的输出，用于后面各层的正向传播，这时使用第 $t - 1$ 层到第 $t$ 层的权重和偏置。接着，第 $t + 1$ 层的误差通过网络反向传播到第 $t$ 层，根据式(23.52)计算出第 $t$ 层的误差，用于这层和前面各层的反向传播，这时使用第 $t$ 层到第 $t + 1$ 层的权重。然后，根据式(23.49)和式(23.50)计算对第 $t$ 层的权重和偏置的梯度，这时使用第 $t - 1$ 层的输出和第 $t$ 层的误差。最后，根据梯度下降公式更新第 $t$ 层的权重和偏置。正向传播和反向传播都递归地进行，先进行正向传播，然后进行反向传播。正向传播从输入层的计算开始，反向传播从输出层的误差计算开始。

![](../images/73adba780597532ccbc37a62dc6e5e4d3e16a53b8bacc5427328a2cfc7e2df9e.jpg)  
图23.19 在神经元 $h_j^{(t)}$ 的正向传播与反向传播

算法 23.3 是前馈神经网络的反向传播算法的一次迭代算法。不失一般性, 假设是基于一个样本的随机梯度下降 (小批量样本容量为 1)。这里使用神经网络的矩阵表示, 其中 $\odot$ 是向量的逐元素积 (element-wise product) 或阿达玛积 (Hadamard product)。①

算法 23.3 （前馈神经网络的反向传播算法）

输入：神经网络 $f(x;\theta)$ ，参数向量 $\theta$ ，一个样本 $(x,y)$ 。

输出：更新的参数向量 $\theta$ 。

超参数：学习率 $\eta$ 。

{

1. 正向传播，得到各层输出 $\boldsymbol{h}^{(1)}, \boldsymbol{h}^{(2)}, \ldots, \boldsymbol{h}^{(s)}$

$$
\boldsymbol {h} ^ {(0)} = \boldsymbol {x}
$$

For $t = 1, 2, \cdots, s$ , do{

$$
\boldsymbol {z} ^ {(t)} = \boldsymbol {W} ^ {(t)} \boldsymbol {h} ^ {(t - 1)} + \boldsymbol {b} ^ {(t)}
$$

$$
\boldsymbol {h} ^ {(t)} = a (\boldsymbol {z} ^ {(t)})
$$

}

$$
f (\boldsymbol {x}) = \boldsymbol {h} ^ {(s)}
$$

2. 反向传播, 得到各层误差 $\delta^{(s)}, \ldots, \delta^{(2)}, \delta^{(1)}$ , 同时计算各层的梯度, 更新各层的参数。计算输出层的误差

$$
\boldsymbol {\delta} ^ {(s)} = \boldsymbol {h} ^ {(s)} - \boldsymbol {y}
$$

For $t = s, \cdots, 2, 1$ , do{

计算第 $t$ 层的梯度

$$
\nabla_ {\boldsymbol {W} ^ {(t)}} L = \boldsymbol {\delta} ^ {(t)} \cdot \boldsymbol {h} ^ {(t - 1) ^ {\mathrm{T}}}
$$

$$
\nabla_ {\boldsymbol {b} ^ {(t)}} L = \delta^ {(t)}
$$

根据梯度下降公式更新第 t 层的参数

$$
\boldsymbol {W} ^ {(t)} \leftarrow \boldsymbol {W} ^ {(t)} - \eta \nabla_ {\boldsymbol {W} ^ {(t)}} L
$$

$$
\boldsymbol {b} ^ {(t)} \leftarrow \boldsymbol {b} ^ {(t)} - \eta \nabla_ {\boldsymbol {b} ^ {(t)}} L
$$

If $(t > 1)$ {

将第 $t$ 层的误差传到第 $t - 1$ 层

$$
\boldsymbol {\delta} ^ {(t - 1)} = \frac {\partial a}{\partial \boldsymbol {z} ^ {(t - 1)}} \odot \left(\boldsymbol {W} ^ {(t) ^ {\mathrm{T}}} \cdot \boldsymbol {\delta} ^ {(t)}\right)
$$

3. 返回更新的参数向量

## 23.2.4 在计算图上的实现

计算图（computation graph）是表示函数计算过程的有向无环图，其结点表示变量，有向边表示变量（输入变量和输出变量）之间的函数依存关系。每一个非起点的结点对应一个基本函数，如加减乘除运算。图整体对应的是由基本函数组成的复合函数。计算图上的计算有正向传播和反向传播。

正向传播是从起点的输入（数值、向量、矩阵或张量）开始，顺着有向边，依次对结点的基本函数进行计算，直到得到终点的输出（数值、向量、矩阵或张量）的过程。这个过程可以看作是信号在图上的正向传播，在各个结点对信号进行了转换。

反向传播是从终点的梯度（数值、向量、矩阵或张量）开始，逆着有向边，依次对结点的梯度进行运算，直到得到起点的梯度（数值、向量、矩阵或张量）的过程，这里一个结点的梯度是指图整体函数对该结点变量的梯度。梯度计算使用链式规则。这个过程可以看作是梯度在图上的反向传播，在各个结点对梯度进行了展开。

链式法则是复合函数的求导法则, 即一个复合函数的导数可以由构成这个复合函数的各个函数的导数表示。一元复合函数 $y = f(z), z = g(x)$ 的导数是

$$
\frac {\mathrm{d} y}{\mathrm{d} x} = \frac {\mathrm{d} y}{\mathrm{d} z} \frac {\mathrm{d} z}{\mathrm{d} x}
$$

多元复合函数 $y = f(z), z = g(x)$ 的导数是

$$
\frac {\partial y}{\partial x _ {i}} = \sum_ {j} \frac {\partial y}{\partial z _ {j}} \frac {\partial z _ {j}}{\partial x _ {i}}
$$

前馈神经网络学习的随机梯度下降法和梯度下降法可以在计算图上实现。整个图对应的是包含神经网络函数的损失函数。计算图用基本函数（如加减乘除）的组合来表示这个复杂的损失函数，通常是一个很大的图。计算图上的正向传播和反向传播可以实现随机梯度下降和梯度下降，其中正向传播实现的是损失函数的计算，反向传播实现的是损失函数的梯度函数的计算。

下面通过具体例子来说明计算图的原理。图 23.20 是一个含有乘法运算的计算图例，上图显示正向传播，下图显示反向传播。起点 x, w 是输入变量，终点 L 是输出变量，中间结点 u 是中间变量。变量 u 由乘法运算 $u = w \cdot x$ 决定，变量 L 由函数 $L = l(u)$ 决定。计算图整体表示的是复合函数 $L = l(w \cdot x)$ 。在计算图上的正向传播就是计算复合函数 $L = l(w \cdot x)$ 的过程。从起点 x, w 开始，顺着有向边，在结点 u, L 依次进行计算，先后得到函数值 u, L；其中先将 x 和 w 相乘得到 u，然后对 u 计算 $l(u)$ 得到 L。反向传播就是计算复合函数 $L = l(w \cdot x)$ 对变量的梯度的过程。从终点 L 开始，逆着有向边，在结点 u, x, w 依次进行计算，先后得到梯度 $\frac{dL}{du}, \frac{dL}{dx}, \frac{dL}{dw}$ ；其中先根据定义计算 $\frac{dL}{du}$ ，再利用链式规则计算 $\frac{dL}{dx}, \frac{dL}{dw}$ ：

$$
\frac {\mathrm{d} L}{\mathrm{d} w} = \frac {\mathrm{d} L}{\mathrm{d} u} \cdot \frac {\mathrm{d} u}{\mathrm{d} w} = \frac {\mathrm{d} L}{\mathrm{d} u} \cdot x
$$

$$
\frac {\mathrm{d} L}{\mathrm{d} x} = \frac {\mathrm{d} L}{\mathrm{d} u} \cdot \frac {\mathrm{d} u}{\mathrm{d} x} = \frac {\mathrm{d} L}{\mathrm{d} u} \cdot w
$$

梯度在乘法结点 u 的反向传播呈现“翻转”现象，正向传播是 x 时，反向传播是梯度的 w 倍，正向传播是 w 时，反向传播是梯度的 x 倍。

![](../images/1b22009e64b66a24435a9c15a8f72549841766c5f8582a08d29d68c66eeff63d.jpg)

![](../images/737ff15f28ca80f995d4079807cbd0f3e1e8d044dd59f89402c3bb5e8af56ac6.jpg)  
图23.20 计算图例：乘法

图 23.21 是一个含有加法运算的计算图例。起点 u, b 是输入变量，终点 L 是输出变量，中间结点 z 是中间变量。变量 z 由加法运算 $z = u + b$ 决定，变量 L 由函数 $L = l(z)$ 决定。计算图整体表示的是复合函数 $L = l(u + b)$ 。在计算图上的正向传播就是计算复合函数 $L = l(u + b)$ 的过程。从起点 u, b 开始，顺着有向边，在结点 z, L 依次进行计算，先后得到函数值 z, L; 其中先将 u 和 b 相加得到 z, 然后对 z 计算 $l(z)$ 得到 L。反向传播就是计算复合函数 $L = l(u + b)$ 对变量的梯度的过程。从终点 L 开始, 逆着有向边, 在结点 z, u, b 依次进行计算, 先后得到梯度 $\frac{dL}{dz}, \frac{dL}{du}, \frac{dL}{db}$ ; 其中先根据定义计算 $\frac{dL}{dz}$ , 再利用链式规则计算 $\frac{dL}{du}, \frac{dL}{db}$ :

$$
\frac {\mathrm{d} L}{\mathrm{d} u} = \frac {\mathrm{d} L}{\mathrm{d} z} \cdot \frac {\mathrm{d} z}{\mathrm{d} u} = \frac {\mathrm{d} L}{\mathrm{d} z} \cdot 1
$$

$$
\frac {\mathrm{d} L}{\mathrm{d} b} = \frac {\mathrm{d} L}{\mathrm{d} z} \cdot \frac {\mathrm{d} z}{\mathrm{d} b} = \frac {\mathrm{d} L}{\mathrm{d} z} \cdot 1
$$

梯度 $\frac{\mathrm{d}L}{\mathrm{d}z}$ 在加法结点 $z$ 的反向传播保持不变传到输入结点 $u, b$ 。

![](../images/94a2f3acf72e212a06cfb1ec93aae65166237d012a120fb9121fc8bf88be1c1d.jpg)

![](../images/10ad3b431e916101d9ce9f377790ba191dcf237d167764fd0d64d657acd5c4a9.jpg)  
图23.21 计算图例：加法

图 23.22 给出含有 S 型函数的计算图例。起点 z, y 是输入变量，终点 L 是输出变量，中间结点 f 是中间变量。变量 f 由 S 型函数 $f = \sigma(z)$ 决定，变量 L 由损失函数 $L = l(f, y)$ 决定。计算图整体表示的是复合函数 $L = l(\sigma(z), y)$ 。在计算图上进行的正向传播就是计算复合函数 $L = l(\sigma(z), y)$ 的过程。从起点 z, y 开始，顺着有向边，在结点 f, L 依次进行计算，先后得到函数值 $f, L$ ; 其中先对 $z$ 计算 $f(z)$ 得到 $f$ , 然后对 $f$ 和 $y$ 计算 $l(f, y)$ 得到 $L$ 。反向传播就是计算复合函数 $L = l(\sigma(z), y)$ 对变量的梯度的过程。从终止结点 $L$ 出发, 逆着有向边, 在结点 $y, f, z$ 依次进行, 先后得到梯度 $\frac{\mathrm{d}L}{\mathrm{d}y}, \frac{\mathrm{d}L}{\mathrm{d}f}, \frac{\mathrm{d}L}{\mathrm{d}z}$ ; 其中先根据定义计算 $\frac{\mathrm{d}L}{\mathrm{d}y}, \frac{\mathrm{d}L}{\mathrm{d}f}$ , 再利用链式规则计算 $\frac{\mathrm{d}L}{\mathrm{d}z}$ :

![](../images/d2e5f53f9f98cdb0587288066fff1f43af3b9d52f6dde3244379d484c05d1c11.jpg)

![](../images/6937d2b8486f07a8f2c2afaefbf966937dad1b73344cc8e1bc7d49433c38755f.jpg)  
反向传播  
图23.22 计算图例：S型函数

$$
\frac {\mathrm{d} L}{\mathrm{d} z} = \frac {\mathrm{d} L}{\mathrm{d} f} \cdot \frac {\mathrm{d} f}{\mathrm{d} z} = \frac {\mathrm{d} L}{\mathrm{d} f} \cdot f (1 - f)
$$

梯度 $\frac{\mathrm{d}L}{\mathrm{d}f}$ 在结点 $f$ 的反向传播变为梯度的 $f(1 - f)$ 倍，传到输入结点 $z$ 。

考虑一层前馈神经网络的学习，神经网络函数和损失函数分别是

$$
\begin{array}{c} \boldsymbol {f} = \sigma (\boldsymbol {W} \cdot \boldsymbol {x} + \boldsymbol {b}) \\ L = l (\boldsymbol {f}, \boldsymbol {y}) \end{array}
$$

其中，x 和 y 构成一个训练样本，表示输入向量和（真值）输出向量；f 是神经网络的输出向量；W 和 b 是权重矩阵和偏置向量； $l(\cdot,\cdot)$ 是损失函数。图 23.23 的计算图显示针对这个学习的正向传播和反向传播的过程。图上的结点表示神经网络函数和损失函数中的变量。事实上这个计算图是由图 23.20～图 23.22 的计算图组合扩展而得到的，其中变量从一元扩展到多元。注意，在每个结点正向传播和反向传播的向量的维度是相同的，因为正向传播的是结点的变量（向量），反向传播的是整体损失函数对结点变量（向量）的梯度向量。学习的过程中只有参数变量 W 和 b 在更新，训练数据变量 x 和 y 保持不变。可以看出，在计算图上的正向传播和反向传播可以实现神经网络学习的梯度下降法。

反向传播  
![](../images/1eb8ac0646b8b65f7874f2b83e423bbbbe2c3650afebcdfca31db760c5519388.jpg)  
图 23.23 计算图例：一层神经网络的学习  
这里 $\odot$ 表示向量的逐元素积

图 23.24 所示的是同一个前馈神经网络的小批量学习的计算图。前向传播计算针对小批

量训练样本的神经网络的损失，反向传播计算针对小批量训练样本的损失函数的梯度。

$$
\boldsymbol {F} = \sigma (\boldsymbol {W} \cdot \boldsymbol {X} + \boldsymbol {B})
$$

$$
L = l (\boldsymbol {F}, \boldsymbol {Y})
$$

其中, $X$ 和 $Y$ 构成一个小批量训练样本集, 表示多个样本的输入矩阵和 (真值) 输出矩阵; $F$ 是多个样本的神经网络的输出矩阵; $W$ 和 $B$ 是神经网络的权重矩阵和偏置矩阵; $l(\cdot, \cdot)$ 是损失函数。

![](../images/1504349d01c76902146b0409581faeeba9e5c3a8f5784e91836a976e305dacd1.jpg)  
图 23.24 计算图例：一层神经网络的小批量学习  
这里 $\odot$ 表示矩阵的逐元素积

前馈神经网络上的正向和反向传播算法（算法23.3）及计算图上的正向和反向传播算法基于相同的基本原理，但具有不同的形式。

## 23.2.5 算法的实现技巧

深度神经网络学习是一个复杂的非凸优化问题，会产生一些优化上的困难。这里介绍梯度消失和爆炸及其解决方法、内部协变量偏移及其解决方法：批量归一化和层归一化。

## 1. 梯度消失与梯度爆炸

深度神经网络学习中有时会出现梯度消失 (vanishing gradient) 或者梯度爆炸 (exploding gradient) 现象。使用反向传播算法（算法 23.3）时，首先通过正向传播计算神经网络各层的输出，然后通过反向传播计算神经网络各层的误差以及梯度，接着利用梯度下降公式对神经网络各层的参数进行更新。在这个过程中，各层的梯度，特别是前面层的梯度，有时会接近 0（梯度消失）或接近无穷（梯度爆炸）。梯度消失会导致参数更新停止，梯度爆炸会导致参数溢出，都会使学习无法有效地进行。

反向传播中，首先计算误差向量：

$$
\boldsymbol {\delta} ^ {(t - 1)} = \left\{\operatorname{diag} \left(\frac {\partial a}{\partial \boldsymbol {z} ^ {(t - 1)}}\right) \cdot \boldsymbol {W} ^ {(t) ^ {\mathrm{T}}} \right\} \cdot \boldsymbol {\delta} ^ {(t)}\tag{23.57}
$$

之后计算梯度:

$$
\left\{ \begin{array}{c} \nabla_ {\boldsymbol {W} ^ {(t)}} L = \boldsymbol {\delta} ^ {(t)} \cdot \boldsymbol {h} ^ {(t - 1) ^ {\mathrm{T}}} \\ \nabla_ {\boldsymbol {b} ^ {(t)}} L = \boldsymbol {\delta} ^ {(t)} \end{array} \right.\tag{23.58}
$$

造成梯度消失和梯度爆炸的原因有两种。首先，每一层的误差向量实际由矩阵的连乘决定，连乘得到的矩阵的元素可能会接近0，也可能会接近无穷，导致梯度的元素也会接近0或接近无穷，而且越是前面的层这个问题就越严重。考虑一种特殊情况：假设每一层的误差向量都与同一个矩阵 $U$ 相乘。第 $t - 1$ 层有

$$
\boldsymbol {\delta} ^ {(t - 1)} = \boldsymbol {U} \cdot \boldsymbol {\delta} ^ {(t)}
$$

这样第 $t - 1$ 层的误差向量 $\delta^{(t-1)}$ 由矩阵 $U^q$ 决定, 设 $q = s - t$ , $s$ 是网络的层数。假设 $U$ 的特征值分解存在, $U = V \cdot \mathrm{diag}(\lambda) \cdot V^{-1}$ , 其中 $\lambda$ 表示特征值, $V$ 表示特征向量, 则有 $U^q = V \cdot \mathrm{diag}(\lambda)^q \cdot V^{-1}$ 成立, 对角矩阵由特征值的 $q$ 次方组成。对于任意一个特征值 $\lambda_i$ , 如果其绝对值小于 1 , 那么其 $q$ 次方 $\lambda_i^q$ 会接近 0 ; 如果其绝对值大于 1 , 那么其 $q$ 次方 $\lambda_i^q$ 会接近无穷。结果是第 $t - 1$ 层的梯度会消失或爆炸。其次, 得到每一层的误差向量之前每个元素乘以激活函数的导数, 如果激活函数的导数过小, 也容易引起梯度消失, 而且越是前面的层, 这个问题就会越严重。在第 $t - 1$ 层得到误差向量 $\delta^{(t-1)}$ 之前, 各元素乘以 $\frac{\partial a}{\partial z^{(t-1)}}$ , 如果 $\frac{\partial a}{\partial z^{(t-1)}}$ 接近 0 , 就会让 $\delta^{(t-1)}$ 的元素也接近 0 。

有一些防止梯度消失和梯度爆炸的技巧。比如，进行恰当的随机参数初始化，一个经验性的方法是对每个神经元的权重 $\boldsymbol{w} = (w_{1}, w_{2}, \cdots, w_{n})^{\mathrm{T}}$ 根据正态分布 $\mathcal{N}\left(0, \frac{1}{n}\right)$ 随机取值。再比如，使用整流线性函数作为激活函数，而不是 S 型函数或双曲正切函数，也可以一定程度上防止梯度消失，因为整流线性函数只是左饱和函数而不是（两边的）饱和函数。使用特定的网络架构，避免反向传播时只依赖于矩阵连乘，如第 24 章介绍的残差网络（ResNet）和第 25 章介绍的 LSTM 模型，可以更好地避免梯度消失和梯度爆炸问题的发生。

其他神经网络的学习, 如卷积神经网络 (CNN)、循环神经网络 (RNN), 也会出现梯度消失或梯度下降的问题 (详见第 24 章和第 25 章)。

## 2. 批量归一化

批量归一化（batch normalization）是对前馈神经网络的每一层（除输出层外）的净输入或输入在每一个批量的样本上进行归一化，在其基础上训练神经网络的方法[9]。这个方法将特征尺度变换（feature scaling transform）应用到神经网络学习，本质上改变了神经网络的结构。主要作用是防止内部协变量偏移（internal covariate shift），加快学习收敛速度；也可以在一定程度上防止梯度消失和梯度爆炸。

机器学习包括深度学习存在一个普遍现象: 如果将输入向量的每一维的数值进行归一化，使其在一定范围之内，比如0和1之间，那么就可以加快基于梯度下降的学习的收敛速度。其原因是：梯度下降以相同的学习率对每一维进行最小化，如果取值范围差异很大，学习就很难在各个维度上同时收敛；如果将学习率取的很小，可以避免这个问题，但学习效率会降低。归一化可以解决这个问题，称为特征尺度变换。

在深度神经网络的学习过程中, 对于神经网络中间的每一层, 其前面层的参数在学习中会不断改变, 导致其输入也不断改变, 不利于这一层及其后面层的学习, 学习收敛速度会变慢。这种现象在神经网络的各层都会发生, 称作内部协变量偏移。假设第 $t$ 层的神经网络函数是 $\pmb{h}^{(t)} = f^{(t)}(\pmb{h}^{(t-1)})$ , 第 $t-1$ 层的神经网络函数是 $\pmb{h}^{(t-1)} = f^{(t-1)}(\pmb{h}^{(t-2)})$ 。如果要学习第 $t$ 层及其后面层的参数, 输入 $\pmb{h}^{(t-1)}$ 相对比较固定为好, 但 $\pmb{h}^{(t-1)}$ 依赖于第 $t-1$ 层及其前面层的参数, 在学习中会动态变化, 导致第 $t$ 层及其后面层的学习不容易收敛。批量归一化通过在每个批量的样本上的归一化来解决这个问题。

原理上在每一层对输入 x 和净输入 z 都可以进行归一化（两者的关系是 $z = W^{T} \cdot x + b$ ）。现实中对净输入 z 效果略好，也更常用。这里只介绍对净输入的批量归一化。

批量归一化训练时, 针对每一批量数据, 按照以下方法扩展神经网络。假设批量数据在当前层的净输入是 $\{z_{1}, z_{2}, \cdots, z_{n}\}$ , 其中 $z_{j}$ 是第 $j$ 个样本的净输入, $n$ 是批量大小。首先计算当前层的净输入的均值与方差 (无偏估计)。

$$
\boldsymbol {\mu} = \frac {1}{n} \sum_ {j = 1} ^ {n} z _ {j}\tag{23.59}
$$

$$
\sigma^ {2} = \frac {1}{n - 1} \sum_ {j = 1} ^ {n} (z _ {j} - \mu) ^ {2}\tag{23.60}
$$

这里 $\mu$ 和 $\sigma^{2}$ 分别是均值向量和方差向量。然后对每一个样本的净输入进行归一化，得到向量

$$
\bar {z} _ {j} = \frac {z _ {j} - \mu}{\sqrt {\sigma^ {2} + \epsilon}}, \quad j = 1, 2, \dots , n\tag{23.61}
$$

这里 $\epsilon$ 是每一个元素都是 $\epsilon$ 的向量, 保证分母不为 0 , 其中 $\epsilon$ 是一个很小的正数, 向量的除法是元素商。之后再进行仿射变换, 得到向量

$$
\tilde {z} _ {j} = \gamma \odot \bar {z} _ {j} + \beta , \quad j = 1, 2, \dots , n\tag{23.62}
$$

这里 $\gamma$ 和 $\beta$ 是参数向量， $\odot$ 是向量的逐元素积。最后将归一化加仿射变换的结果作为批量数据在这一层的净输入。

图 23.25 显示批量归一化训练网络一层的结构, 是在神经网络一层对每一个批量的样本的归一化。对于每一个批量, 每一层 (除输出层外) 都有同样的结构。注意: 每一个批量在每一层有各自的均值 $\mu$ 和方差 $\sigma^{2}$ , 当输入样本和网络参数确定时, $\mu$ 和 $\sigma^{2}$ 就确定。另一方面, 每一层有各自的参数 $\gamma$ 和 $\beta$ , 归所有批量共有。神经网络原始的参数 $\theta$ 也是所有批量共有。

训练时, 对神经网络每一层的净输入进行归一化, 使净输入的均值是 0、方差是 1 。这样可以提高下一层的学习收敛速度。

![](../images/ed1447ad195e461513329390aeebc01712517326507469edabcd7c5f8cd12600.jpg)  
图 23.25 批量归一化：训练网络结构

当 $\gamma \approx \sigma$ 且 $\beta \approx \mu$ 时，有

$$
\tilde {z} _ {j} \approx z _ {j}, \quad j = 1, 2, \dots , n
$$

归一化加仿射变换接近恒等变换。批量归一化方法保证这种“不做变换”情况也被包括在内。而具体进行怎样的变换通过数据决定。事实上，归一化后的净输入在原点附近时，S型函数的输出也在原点附近，模型的表示能力降低，从学习的角度看未必最优，需要基于数据进行选择。

另外, 每一层的输入到净输入的仿射变换实际根据 $z = W^{\mathrm{T}} \cdot x$ 进行, 省去偏置 $b$ , 因为参数 $\beta$ 能起到相应的作用。

预测时, 通常将一个测试样本输入到推理网络。首先对样本在每一层的净输入进行归一化, 得到向量:

$$
\bar {z} _ {j} = \frac {z _ {j} - E _ {b} (\pmb {\mu})}{\sqrt {E _ {b} (\pmb {\sigma} ^ {2}) + \epsilon}}, \quad j = 1, 2, \dots , n\tag{23.63}
$$

这里 $E_{b}(\mu)$ 和 $E_{b}(\sigma^{2})$ 分别是所有批量的均值和方差的平均。然后再进行仿射变换，得到向量：

$$
\tilde {z} _ {j} = \gamma \odot \bar {z} _ {i} + \beta , \quad j = 1, 2, \dots , n\tag{23.64}
$$

这里 $\gamma$ 和 $\beta$ 是训练过程中得到的参数向量。图23.26显示批量归一化推理网络一层的结构。每一层（除输出层外）都有同样的结构。

![](../images/9880258c87e054cabb02c69aa445e74b35124083f1e937e4e6977a48d0e5f252.jpg)  
图 23.26 批量归一化：推理网络结构

算法 23.4 是批量归一化的算法。首先根据以上批量归一化方法构建训练神经网络，然后使用随机梯度下降法训练神经网络，最后输出推理神经网络对测试样本的预测值。

算法 23.4（批量归一化）

输入：神经网络结构 $f(x;\theta)$ ，训练集，测试样本。

输出：对测试样本的预测值。

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
{
初始化参数  $\theta, \phi$ ，其中  $\phi = \{\gamma^{(t)}, \beta^{(t)}\}_{t=1}^{s-1}$ 
For each (批量 b){
    For  $t = 1, 2, \cdots, s - 1$ 
针对批量 b 计算第 t 层净输入的均值  $\mu^{(t)}$  和方差  $\sigma^{2(t)}$ 
进行第 t 层的批量归一化，得到批量净输入
 $z_{j}^{(t)} \rightarrow \bar{z}_{j}^{(t)} \rightarrow \tilde{z}_{j}^{(t)}, \quad j = 1, 2, \cdots, n$ 
}
构建训练神经网络  $f_{\mathrm{Tr}}(\boldsymbol{x}; \boldsymbol{\theta}, \boldsymbol{\phi})$ 
使用随机梯度下降法训练  $f_{\mathrm{Tr}}(\boldsymbol{x}; \boldsymbol{\theta}, \boldsymbol{\phi})$ ，估计所有参数  $\theta, \phi$ 
For  $t = 1, 2, \cdots, s - 1$ 
针对所有批量计算第 t 层净输入的期待的均值  $E_{b}(\boldsymbol{\mu}^{(t)})$  和方差  $E_{b}(\boldsymbol{\sigma}^{2(t)})$ 
针对测试样本，进行第 t 层的批量归一化，得到净输入
 $z_{j}^{(t)} \rightarrow \bar{z}_{j}^{(t)} \rightarrow \tilde{z}_{j}^{(t)}, \quad j = 1, 2, \cdots, n$ 
}
构建推理神经网络  $f_{\mathrm{Inf}}(\boldsymbol{x}; \boldsymbol{\theta}, \boldsymbol{\phi})$ 
输出  $f_{\mathrm{Inf}}(\boldsymbol{x}; \boldsymbol{\theta}, \boldsymbol{\phi})$  对测试样本的预测值
}
</div>

超参数：批量容量的大小 n。

## 3. 层归一化

层归一化（layer normalization）是另一种防止内部协变量偏移的方法[10]。其基本想法与批量归一化相同，但是是在每一层的神经元上进行归一化，而不是在每一个批量的样本上进行归一化。优点是实现简单，也没有批量大小的超参数需要调节。

层归一化在每一层的神经元的净输入上进行。假设当前层的神经元的净输入是 $z = (z_{1}, z_{2}, \cdots, z_{m})^{\mathrm{T}}$ ，其中 $z_{j}$ 是第 $j$ 个神经元的净输入， $m$ 是神经元个数。训练和预测时，首先计算这一层的神经元的净输入的均值与方差（无偏估计）。

$$
\mu = \frac {1}{m} \sum_ {j = 1} ^ {m} z _ {j}\tag{23.65}
$$

$$
\sigma^ {2} = \frac {1}{m - 1} \sum_ {j = 1} ^ {m} (z _ {j} - \mu) ^ {2}\tag{23.66}
$$

然后对每一个神经元的净输入进行归一化，得到数值：

$$
\bar {z} _ {j} = \frac {z _ {j} - \mu}{\sqrt {\sigma^ {2} + \epsilon}}, \quad j = 1, 2, \dots , m\tag{23.67}
$$

其中, $\epsilon$ 是一个很小的正数。之后再进行仿射变换, 得到数值:

$$
\tilde {z} _ {j} = \gamma \cdot \bar {z} _ {j} + \beta , \quad j = 1, 2, \dots , m\tag{23.68}
$$

其中， $\gamma$ 和 $\beta$ 是参数。最后将归一化加仿射变换的结果作为这一层神经元的实际净输入。在每一层都做同样的处理。神经网络的每一层有两个参数 $\gamma$ 和 $\beta$ 。图23.27显示层归一化的网络结构。

![](../images/ccf456f97dbcabc3b80fa27fd7718df15b2e46c3de73969171867385d099f975.jpg)  
图23.27 层归一化的网络结构

## 23.3 前馈神经网络学习的正则化

正则化（regularization）的目的是提高学习的泛化能力，即不仅使训练误差而且使测试误差达到最小。本节概述深度学习，特别是前馈神经网络学习中的正则化方法，具体介绍常用的早停法和暂退法。

## 23.3.1 深度学习中的正则化

深度学习的正则化有 L1 正则化、L2 正则化或称权重衰减（weight decay）、早停法（early stopping）、暂退法（dropout）等方法。前三种方法是机器学习通用的方法，最后一种方法是深度学习特有的方法。另外，深度学习中不做显式的正则化常常也能达到泛化的效果。具体哪种方法更有效需要在实际的问题和数据上验证。

现实中发现，深度学习中常常不做正则化也不产生过拟合。往往是在大规模训练数据、过度参数化（over-parameterized）神经网络及随机梯度下降训练的情况下发生的，也就是说这种组合能产生泛化能力，这里过度参数化是指神经网络的参数量级大于等于训练数据量级的情况。机器学习理论尚不能很好地分析这种现象，是当前热门的研究课题。普遍的解释是随机梯度下降起到隐式正则化的作用，能保证学到的模型不产生过拟合。

## 23.3.2 早停法

早停法（early stopping）在学习中使用验证集进行评估，判断训练的终止点，进行模型选择，是隐式的正则化方法。

早停法将数据分为训练集、验证集、测试集（比如以 1/2, 1/4, 1/4 的比例）。学习中，持续训练模型，得到训练误差（训练集上的损失），同时用验证集评估，得到验证误差（验证集上的损失）。图 23.28 示意训练过程。横轴表示训练步数，纵轴表示误差。通常训练误差不断减小，逐渐趋近于 0，而验证误差在某个点达到最小，之后逐渐增加。

![](../images/fa5649ebb43615b4562351bd4ba57bd6c372776a741e90c72cd6fe985c360dce.jpg)  
图23.28 训练过程与早停法

如果选择训练误差很小而验证误差已经增大的点的模型，很有可能这个模型的测试误差不是最小的，即产生过拟合。早停法选择验证误差最小的点为训练终止点，将这时的模型作为最终模型输出。有很大概率这个模型也是测试误差最小的模型，即泛化最好的模型，因为用验证集代替测试集进行了模型评估。因为没有等到训练误差降到很小，甚至接近于0时结束训练，所以训练是早停的。早停法的优点是简单有效，缺点是需要将一部分标注数据用于训练评估而不是训练本身。算法23.5给出早停法的具体算法，其中 $l_{\mathrm{dev}}()$ 表示在验证集上的损失，即验证误差。

## 算法 23.5（早停法）

输入：神经网络结构 $f(x;\theta)$ ，训练集，验证集。

输出: 学习得到的神经网络参数向量 $\hat{\theta}$ 。

参数：评估间隔步数 m，持续评估上限 n。

$$
\begin{array}{c} i = 0 \\ j = 0 \\ l _ {\min} = \infty \end{array}
$$

while $(j < n)$ {

用训练集连续训练模型 m 步，得到参数向量 $\theta_{i}$

$$
i \gets i + m
$$

用验证集评估模型的损失

$$
l = l _ {\mathrm{dev}} (\boldsymbol {\theta} _ {i})
$$

$$
\text { if } (l <   l _ {\min}) \left\{ \begin{array}{l} \end{array} \right.
$$

$$
l _ {\mathrm{min}} \gets l
$$

$$
\theta_ {\mathrm{min}} \leftarrow \theta_ {i}
$$

$$
j = 0
$$

$$
j \gets j + 1
$$

$$
\hat {\pmb {\theta}} = \pmb {\theta} _ {\mathrm{min}}
$$

返回参数向量 $\hat{\theta}$ }

## 23.3.3 暂退法

暂退法（dropout）在训练过程中的每一步随机选取一些神经元，让它们不参与（退出）训练，学习结束后，对权重进行调整，然后将整体网络用于预测。暂退法是经验性的方法，在现实中很有效，但目前还没有严格的理论证明。可以认为暂退法是应用于深度学习的一种Bagging 方法[11]。

前馈神经网络训练时, 设输入层和隐层的每一层 (不包括输出层) 都有一个保留概率 $p$ (各层的概率不一定相同), 每层的神经元以概率 $p$ 保留, 以概率 $1 - p$ 退出, 保留概率为 1 时不退出。通常输入层的保留概率设为 0.8 , 隐层的保留概率为 0.5 。在训练的每一步, 针对每一个样本或每一组样本, 在每一层随机判断, 选取保留的神经元和退出的神经元。所有保留的神经元构成了一个退化的神经网络, 也是整体网络的一个子网络。使用随机梯度下降法更新子网络的权重。图 23.29 左边显示一个神经网络, 右边显示一个随机得到的子神经网络。若每层有 $m$ 个神经元, 则每层有 $2^{m}$ 种可能的神经元排列, 所以子网络的种类数是指数级的。(也会以小概率得到一些输入输出层不相连的子网络, 反向传播算法不适用, 对整体学习不产生影响)

![](../images/a82e811aa73a571aec02d84c83af6379dabb06ffec6aabe1ac7ac2de4cbcec98.jpg)  
图 23.29 暂退法：整体网络和子网络

假设某一隐层的输出向量是 h，误差向量是 $\delta$ ，该层神经元保留与退出的结果用随机向量 d 表示，其中 $d \in \{0,1\}^{m}$ 是维度为 m 的 0-1 向量，1 表示对应的神经元保留，0 表示对应的神经元退出。那么，在反向传播算法的每一步，经过保留与退出随机判断后，该层的向量表示变为

$$
\tilde {\boldsymbol {h}} = \boldsymbol {d} \odot \boldsymbol {h}\tag{23.69}
$$

$$
\tilde {\delta} = d \odot \delta\tag{23.70}
$$

这里 $\odot$ 表示逐元素积, 使用 $\tilde{\pmb{h}}$ 进行正向传播和使用 $\tilde{\pmb{\delta}}$ 进行反向传播。注意暂退法中每一步的 $\pmb{d}$ 是随机决定的, 各步之间并不相同。对输入层的处理方法也一样, 细节省略。

预测时，对隐层的输出向量进行调整：

$$
\tilde {\boldsymbol {h}} = \boldsymbol {p} \cdot \boldsymbol {h}\tag{23.71}
$$

其中，p 是这层的保留概率。等价地，可以认为对隐层的神经元输出的权重进行调整，每一个输出权重 w 乘以保留概率 p 作为最终的权重，如图 23.30 所示。其直观解释是学习中神经元以概率 p 参与训练，所以最终使用权重的期待值 $p \cdot w$ 作为真实值。神经元的偏置保持不变。

![](../images/fa07c6187f71ffdfcf10777be473684a3c42a27d89a0386027b5f45addd84eb2.jpg)  
图 23.30 暂退法：权重的调整

为了方便暂退法的实现, 常常采用以下等价的逆暂退法 (inverted dropout)。训练时, 将隐层的输出变量放大 $\frac{1}{p}$ 倍:

$$
\tilde {h} = \frac {1}{p} \cdot d \odot h\tag{23.72}
$$

预测时，隐层的输出权重保持不变。

可以证明在特殊情况下暂退法是一种 Bagging 方法。假设有一个二层的多类分类网络，对其进行基于暂退法的学习。输入层由多个变量组成，保留概率是 1；输出层由一个神经元组成，激活函数是软最大化函数；隐层的输出向量是 h，保留概率是 0.5，暂退法得到的一个随机向量是 $d \in \{0,1\}^{m}$ 。这时，子网络（输出层）可以写作

$$
\begin{array}{r l} P (y _ {k} = 1 | \boldsymbol {h}, \boldsymbol {d}) & = \operatorname{softmax} (\boldsymbol {w} _ {k} ^ {\mathrm{T}} \boldsymbol {d} \odot \boldsymbol {h} + b _ {k}) \\ & = \frac {\exp (\boldsymbol {w} _ {k} ^ {\mathrm{T}} \boldsymbol {d} \odot \boldsymbol {h} + b _ {k})}{\sum_ {k ^ {\prime}} \boldsymbol {w} _ {k ^ {\prime}} ^ {\mathrm{T}} \boldsymbol {d} \odot \boldsymbol {h} + b _ {k ^ {\prime}}} \end{array}\tag{23.73}
$$

隐层共有 $2^{m}$ 个随机向量, 对应 $2^{m}$ 个子模型 (子网络)。所有子模型的权重和偏置是共有的。现实中暂退法训练的是这其中的部分子模型 (因为子模型的个数是指数级的, 现实中一般不可能学到所有的子模型)。暂退法最终的 (经过权重调整的) 模型是

$$
P \left(y _ {k} = 1 | \boldsymbol {h}\right) = \frac {\exp \left(\frac {1}{2} \boldsymbol {w} _ {k} ^ {\mathrm{T}} \boldsymbol {h} + b _ {k}\right)}{\sum_ {k ^ {\prime}} \exp \left(\frac {1}{2} \boldsymbol {w} _ {k ^ {\prime}} ^ {\mathrm{T}} \boldsymbol {h} + b _ {k ^ {\prime}}\right)}\tag{23.74}
$$

式中不包含隐层的随机向量 d。

考虑集成学习，对暂退法中所有子模型的输出概率取几何平均，作为集成模型的输出概率，得到：

$$
\begin{array}{l} P _ {\text {ensemble}} (y _ {k} = 1 | \boldsymbol {h}) = \sqrt [ 2 ^ {m} ]{\prod_ {\boldsymbol {d} \in \{0 , 1 \} ^ {m}} P (\boldsymbol {y} | \boldsymbol {h} , \boldsymbol {d})} \\ = \sqrt [ 2 ^ {m} ]{\prod_ {\boldsymbol {d} \in \{0 , 1 \} ^ {m}} \operatorname{softmax} (\boldsymbol {w} _ {k} ^ {\mathrm{T}} \boldsymbol {d} \odot \boldsymbol {h} + b _ {k})} \\ \propto \sqrt [ 2 ^ {m} ]{\prod_ {\boldsymbol {d} \in \{0 , 1 \} ^ {m}} \exp (\boldsymbol {w} _ {k} ^ {\mathrm{T}} \boldsymbol {d} \odot \boldsymbol {h} + b _ {k})} \\ = \exp \left[ \frac {1}{2 ^ {m}} \sum_ {\boldsymbol {d} \in \{0, 1 \} ^ {m}} (\boldsymbol {w} _ {k} ^ {\mathrm{T}} \boldsymbol {d} \odot \boldsymbol {h} + b _ {k}) \right] \\ = \exp \left(\frac {1}{2} \boldsymbol {w} _ {k} ^ {\mathrm{T}} \boldsymbol {h} + b _ {k}\right) \end{array}\tag{23.75}
$$

中间用到的事实是软最大化函数的分母是对所有类别 $y'$ 的归一化项，属于常量。也就是说，在这种情况下集成模型精确等价于暂退法模型。

暂退法在以下几点对一般的 Bagging 方法进行了改动，以提高神经网络的算法学习和预测的效率。不显式地定义和使用子模型，所有子模型共享整体网络的参数。学习时，每一步将一个随机样本或一组随机样本用于一个子模型的训练。预测时，使用参数调整后的整体网络近似实现子模型的集成（计算子模型预测概率的几何平均）。原理上，可以让子模型拥有不同的参数，或者对子模型进行抽样，然后集成，这些都会降低学习和预测的效率。

## 本章概要

1. 神经元是神经网络的基本单元，本质是一种非线性函数。神经元函数的基本形式是

$$
y = a \left(\sum_ {i = 1} ^ {n} w _ {i} x _ {i} + b\right)
$$

其中， $x_{1},x_{2},\cdots,x_{n}$ 是输入，y是输出， $z=\sum_{i=1}^{n}w_{i}x_{i}+b$ 是净输入， $w_{1},w_{2},\cdots,w_{n}$ 是权重，b是偏置。 $a(\cdot)$ 是激活函数，也可以写成

$$
y = a (\boldsymbol {w} ^ {\mathrm{T}} \boldsymbol {x} + b)
$$

2. 前馈神经网络由多层神经元组成, 层间的神经元相互连接, 层内的神经元不相连。其信息处理机制是前一层神经元通过连接向后一层神经元传递信号。整个神经网络是对多个输入信号 (实数向量) 进行多次非线性转换产生多个输出信号 (实数向量) 的复合函数。前馈神经网络的矩阵表示如下:

$$
\begin{array}{c} \boldsymbol {h} ^ {(1)} = a \left(\boldsymbol {W} ^ {(1) ^ {\mathrm{T}}} \boldsymbol {x} + \boldsymbol {b} ^ {(1)}\right) \\ \boldsymbol {h} ^ {(2)} = a \left(\boldsymbol {W} ^ {(2) ^ {\mathrm{T}}} \boldsymbol {h} ^ {(1)} + \boldsymbol {b} ^ {(2)}\right) \\ \vdots \\ \boldsymbol {h} ^ {(s - 1)} = a \left(\boldsymbol {W} ^ {(s - 1) ^ {\mathrm{T}}} \boldsymbol {h} ^ {(s - 2)} + \boldsymbol {b} ^ {(s - 1)}\right) \\ \boldsymbol {y} = g \left(\boldsymbol {W} ^ {(s) ^ {\mathrm{T}}} \boldsymbol {h} ^ {(s - 1)} + \boldsymbol {b} ^ {(s)}\right) \end{array}
$$

前馈神经网络拥有很强的表示能力，可以进行复杂的信息处理。

3. 激活函数有多种形式。隐层的激活函数主要有 S 型函数:

$$
a (z) = \frac {1}{1 + \mathrm{e} ^ {- z}}
$$

双曲正切函数：

$$
a (z) = \frac {\mathrm{e} ^ {z} - \mathrm{e} ^ {- z}}{\mathrm{e} ^ {z} + \mathrm{e} ^ {- z}}
$$

整流线性函数:

$$
a (z) = \max (0, z)
$$

输出层的激活函数主要有恒等函数：

$$
g (z) = z
$$

软最大化函数:

$$
g \left(z _ {k}\right) = \frac {\mathrm{e} ^ {z _ {k}}}{l}, \quad k = 1, 2, \dots , l   \sum_ {i = 1} ^ {} \mathrm{e} ^ {z _ {i}}
$$

4. 前馈神经网络可以用于不同任务。用于回归时，神经网络表示为 $y = f(x)$ 。用于二类分类时，神经网络表示为 $P(y = 1|x) = f(x)$ 。用于多类分类时，神经网络表示为 $[P(y_k = 1|x)] = f(x)$ ，其中 $y_k \in \{0, 1\}, \sum_{k=1}^{l} y_k = 1, k = 1, 2, \cdots, l$ 。

5. 通用近似定理指出, 对任意连续函数 $h: [0,1]^m \to \mathcal{R}$ 和任意 $\varepsilon > 0$ , 存在一个二层神经网络 $f(x) = \alpha^{\mathrm{T}} \sigma(w^{\mathrm{T}} x + b)$ , 使得对于任意 $x \in [0,1]^m$ , 有 $|h(x) - f(x)| < \varepsilon$ 成立。通用近似定理从理论的角度阐述了深度神经网络的强大表示能力。

6. 深度神经网络与浅度神经网络可以有同等的表示能力, 但深度神经网络比浅度神经网络有更低的样本复杂度。所以, 深度神经网络比起浅度神经网络, 只需要更少的数据就可以学到。

7. 前馈神经网络学习是监督学习，优化目标函数是

$$
L (\boldsymbol {\theta}) = \frac {1}{N} \sum_ {i = 1} ^ {N} L _ {i} (\boldsymbol {\theta}) = \frac {1}{N} \sum_ {i = 1} ^ {N} L (f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}), y _ {i})
$$

分类时损失函数是交叉熵损失，回归时是平方损失。目标函数的优化都等价于极大似然估计。常用的优化算法是随机梯度下降。对于随机梯度下降，每次按以下公式对一个小批量样本进行参数更新，遍历所有样本组，不断迭代，直到收敛为止。

$$
\theta \leftarrow \theta - \eta \frac {1}{n} \sum_ {j = 1} ^ {n} \frac {\partial L _ {j} (\theta)}{\partial \theta}
$$

8. 前馈神经网络学习的具体算法是反向传播算法。只需要依照网络结构进行一次正向传播和一次反向传播，就可以完成梯度下降的一次迭代。主要部分如下。

正向传播，计算各层输出。

第 $t$ 层的输出：

$$
\begin{array}{c} \boldsymbol {z} ^ {(t)} = \boldsymbol {W} ^ {(t)} \boldsymbol {h} ^ {(t - 1)} + \boldsymbol {b} ^ {(t)} \\ \boldsymbol {h} ^ {(t)} = a (\boldsymbol {z} ^ {(t)}) \end{array}
$$

反向传播，计算各层误差 $\delta^{(s)}, \delta^{(s-1)}, \ldots, \delta^{(1)}$ 。

输出层的误差:

$$
\delta^ {(s)} = \boldsymbol {h} ^ {(s)} - \boldsymbol {y}
$$

第 $t - 1$ 层的误差：

$$
\boldsymbol {\delta} ^ {(t - 1)} = \frac {\partial a}{\partial \boldsymbol {z} ^ {(t - 1)}} \odot \left(\boldsymbol {W} ^ {(t) \mathrm{T}} \cdot \boldsymbol {\delta} ^ {(t)}\right)
$$

计算第 $t$ 层的梯度：

$$
\nabla_ {\boldsymbol {W} ^ {(t)}} L = \boldsymbol {\delta} ^ {(t)} \cdot \boldsymbol {h} ^ {(t - 1) ^ {\mathrm{T}}}
$$

$$
\nabla_ {\boldsymbol {b} ^ {(t)}} L = \boldsymbol {\delta} ^ {(t)}
$$

更新第 $t$ 层的参数：

$$
\begin{array}{c} \boldsymbol {W} ^ {(t)} \leftarrow \boldsymbol {W} ^ {(t)} - \eta \nabla_ {\boldsymbol {W} ^ {(t)}} L \\ \boldsymbol {b} ^ {(t)} \leftarrow \boldsymbol {b} ^ {(t)} - \eta \nabla_ {\boldsymbol {b} ^ {(t)}} L \end{array}
$$

9. 计算图是显示函数计算过程的有向无循环图, 其结点表示变量, 有向边表示变量之间的函数依存关系。每一个非起点的结点对应一个基本函数。图整体对应的是由基本函数组成的复合函数。计算图上的计算有正向传播和反向传播。可以将神经网络训练和预测分解为计算图上的矩阵或张量数据计算, 便于在计算机上实现。

正向传播是从起点的输入开始，顺着有向边，依次对结点的基本函数进行计算，直到得到终点的输出为止的过程。反向传播是从终点的梯度开始，逆着有向边，依次对结点的梯度进行运算，直到得到起点的梯度为止的过程，这里一个结点的梯度是指图整体函数对该结点变量的梯度。

10. 深度神经网络学习是一个复杂的非凸优化问题, 会产生一些优化上的困难, 包括梯度消失和梯度爆炸、内部协变量偏移。

梯度消失和梯度爆炸的主要原因是在深度神经网络的学习过程中，每一层的梯度主要由矩阵乘积决定。如果连乘得到的矩阵的元素接近0，那么梯度的元素也会接近0（消失）；如果连乘得到的矩阵的元素接近无穷，那么梯度的元素也会接近无穷（爆炸）。梯度消失会导致参数更新停止，梯度爆炸会导致参数溢出，都会使学习无法有效地进行。

内部协变量偏移的现象是指在深度神经网络的学习过程中，对于网络的每一层，如果其输入相对比较固定，就很容易学习到这一层及其后面层的参数，但现实中这一层前面层的参数在学习中会不断改变，导致其输入也不断改变，不利于这一层及其后面层的学习，学习速度会变缓。

批量归一化和层归一化的主要作用是防止内部协变量偏移。批量归一化是指对网络的每一层（除输出层外）的净输入在批量的样本上进行归一化，然后训练网络的方法。层归一化是指对网络的每一层（除输出层外）的净输入在该层的神经元上进行归一化，然后训练网络的方法。

11. 学习的正则化方法包括早停法、暂退法（dropout）等。现实中发现，深度学习中常常不做正则化也不产生过拟合。往往是在大规模训练数据、过度参数化网络及随机梯度下降训练的情况下发生的，也就是说这种组合能产生泛化能力。

对于暂退法，在前馈神经网络训练时，设输入层和隐层的每一层都有一个保留概率 $p$ ，每层的神经元以概率 $p$ 保留，以概率 $1 - p$ 退出。针对每一个样本或每一组样本，在每一层随机判断，选取保留的神经元和退出的神经元；所有保留的神经元构成了一个退化的子网络，使用随机梯度下降法，更新子网络的权重。预测时，对隐层的神经元输出的权重进行调整，每一个输出权重 $w$ 乘以保留概率 $p$ 作为最终的权重。可以证明在特殊情况下暂退法是一种Bagging方法。

## 继续阅读

进一步学习前馈神经网络和相关的深度学习技术可参见文献 [1]～文献 [5]，学习 Python 和 MXNet 上的实现方法可参阅文献 [3] 和文献 [4]。通用近似定理、模型等价性的介绍可参阅文献 [1] 和文献 [2]，网络深度的讨论可见文献 [6]，暂退法的介绍可见文献 [1] 和文献 [9]，计算图的介绍可参阅文献 [1] 和文献 [3]。通用近似定理的原始论文是文献 [7] 和文献 [8]，批量归一化和层归一化的最初论文是文献 [9] 和文献 [10]，暂退法的论文是文献 [11]。

## 习题

23.1 构造前馈神经网络实现逻辑表达式 XNOR，使用 S 型函数为激活函数。

23.2 写出多标签分类学习中的损失函数以及损失函数对输出变量的导数。

23.3 实现前馈神经网络的反向传播算法，使用 MNIST 数据构建手写数字识别网络。

23.4 写出 S 型函数的正向传播和反向传播的计算图。

23.5 图 23.31 是 3 类分类的正向传播计算图，试写出它的反向传播计算图。这里使用软最大化函数和交叉熵损失。

![](../images/53bd2944ff7667e6161e27bf89b3fe55246ac0ce237929d6c3994b372b474866.jpg)  
图23.31  
23.6 写出批量归一化的反向传播算法。  
23.7 验证逆暂退法和暂退法的等价性。

## 参考文献

[1] GOODFELLOW I, BENGIO Y, COURVILLE A. Deep learning[M]. MIT Press, 2016.

[2] BISHOP C. Pattern recognition and machine learning[M]. Springer, 2006.

[3] 斋藤康毅. 深度学习入门: 基于 Python 的理论与实现 [M]. 陆宇杰, 译. 北京: 人民邮电出版社, 2018.

[4] 阿斯顿·张，李沐，扎卡里·立顿，等. 动手学深度学习 [M]. 北京：人民邮电出版社，2019.

[5] 邱锡鹏. 神经网络与深度学习 [M]. 北京: 机械工业出版社, 2020.

[6] BENGIO Y. Learning deep architectures for AI[M]. Now Publisher, 2002.

[7] CYBENKO G. Approximation by superpositions of a sigmoidal function[J]. Mathematics of Control, Signals and Systems, 1989, 2: 303-314.

[8] HORNIK K, STINCHCOMBE M, WHITE H. Multilayer feedforward networks are universal approximators[J]. Neural Networks, 1989, 2(5): 359-366.

[9] IOFFE S, SZEGEDY C. Batch normalization: accelerating deep network training by reducing internal covariate shift[C]//International Conference on Machine Learning. 2015: 448-456.

[10] BA J L, KIROS J R, HINTON G E. Layer normalization[Z/OL]. arXiv preprint arXiv: 1607.06450, 2016.

[11] SRIVASTAVA N, HINTON G, KRIZHEVSKY A, et al. Dropout: asimple way to prevent neural networks from overfitting[J]. The Journal of Machine Learning Research, 2014, 15(1): 1929-1958.

