---
tags: [机器学习, 李航, 深度学习]
category: "章节"
chapter: "Ch27-预训练语言模型"
pian: "第三篇-深度学习"
---
## 第 27 章 预训练语言模型

在自然语言处理中事先使用大规模语料学习基于 Transformer 等的语言模型，之后用于各种任务的学习和预测，称这种模型为预训练语言模型（pretrained language model）。代表性的模型有 BERT（bidirectional encoder representations from Transformers）和 GPT（generative pre-training）。BERT 的模型是 Transformer 的编码器。首先在预训练中使用大规模语料通过掩码语言模型化的方式估计模型的参数，之后在微调中使用具体任务的标注数据对参数进行进一步调节。前者的过程是无监督学习 $^{①}$ ，后者的过程是监督学习。GPT 的模型是 Transformer 的解码器，预训练通过一般的语言模型化方式进行。

BERT 和 GPT 具有很强的表示自然语言的能力，通过其多层多头自注意力等机制以及在大规模数据上的训练能够有效地表示自然语言的词汇、句法、语义信息，目前已经分别成为语言理解和语言生成的核心技术。Radford 等于 2018 年发表了 GPT，还有之后的改进和增强版 GPT-2 和 GPT-3。Devlin 等于 2019 年发表了 BERT。

本章 27.1 节讲解 GPT 的模型和学习, 27.2 节讲解 BERT 的模型和学习。

## 27.1 GPT 模型

GPT 及其后续版本是有代表性的预训练语言模型，适合于语言生成。本节首先给出预训练语言模型的概述，然后叙述 GPT 模型及其学习算法，最后总结 GPT 模型的特点。

## 27.1.1 预训练语言模型

在实际应用中使用的深度学习主要还是监督学习，如在自然语言处理中的文本分类、文本序列标注。在具体的任务中需要有标注数据，普遍规律是标注数据质量越高和数量越大，学到的模型的准确率就越高。但问题是数据的标注成本通常很高，实际应用中往往很难获取大量的高质量标注数据。另外，不同的任务需要不同的标注数据，标注数据在任务之间很难通用。预训练语言模型是为解决这个问题而开发的用于自然语言处理的深度学习方法。

预训练语言模型的基本想法如下: 基于神经网络, 如 Transformer 的编码器或解码器 (见第 26 章), 实现语言模型, 以计算语言的生成概率。首先使用大规模的语料通过无监督学习的方式学习模型的参数, 称为预训练 (pre-training), 得到的模型可以有效地表示自然语言的特征; 之后将模型用于一个具体任务, 使用少量的标注数据通过监督学习的方式进一步学习模型的参数, 称为微调 (fine tuning), 任务称为下游任务 (downstream task)。预训练使用通用的语料统一进行, 微调使用各个下游任务的标注数据分别进行。微调 (下游任务) 的模型有时在预训练模型的基础上增加新的参数。

Transformer 具有强大的语言表示能力，大规模语料包含丰富的语言表达（这样的无标注数据可以较容易地获取），加之大规模深度学习的训练系统变得越来越高效，所以学习得到的预训练语言模型可以有效地表示语言的词汇、句法和语义特征。这样，当预训练语言模型用于下游任务时，只需要标注少量的数据训练就可以达到很高的准确率。预训练语言模型已成为当前语言理解和语言生成的核心技术。

有代表性的预训练语言模型有 GPT 和 BERT。表 27.1 比较了 GPT 和 BERT 的主要特点，其主要区别在于模型的架构和预训练方式。

表 27.1 GPT 和 BERT 的比较

|  | GPT | BERT |
| --- | --- | --- |
| 语言模型类型 | 单向语言模型 | 双向语言模型 |
| 模型架构 | Transformer 解码器 | Transformer 编码器 |
| 预训练方式 | 语言模型化 | 掩码语言模型化 |
| 预训练原理 | 序列概率估计 | 去噪自动编码器 |
| 下游任务 | 语言理解、语言生成 | 语言理解 |

GPT 是单向语言模型（unidirectional language model），从一个方向对单词序列建模，方向为从左到右或者从右到左，由 Transformer 的解码器实现。假设有单词序列 $x = x_{1}, x_{2}, \cdots, x_{n}$ ，在单词序列的各个位置上，单向语言模型具有以下单词生成的条件概率：

$$
P \left(x _ {i} | x _ {1}, x _ {2}, \dots , x _ {i - 1}\right), \quad i = 1, 2, \dots , n\tag{27.1}
$$

每一个位置的单词依赖于之前位置的单词。可以使用单向语言模型计算单词序列 $x = x_{1}, x_{2}, \cdots, x_{n}$ 的生成概率。

BERT 是双向语言模型（bidirectional language model），从两个方向同时对单词序列建模，由 Transformer 的编码器实现。在单词序列的各个位置上，双向语言模型具有以下单词生成的条件概率：

$$
P \left(x _ {i} | x _ {1}, \dots , x _ {i - 1}, x _ {i + 1}, \dots , x _ {n}\right), \quad i = 1, 2, \dots , n\tag{27.2}
$$

每一个位置的单词依赖于之前位置和之后位置的单词。不可以使用双向语言模型直接计算单词序列 $x = x_{1}, x_{2}, \cdots, x_{n}$ 的生成概率。

GPT 的预训练通过语言模型化（language modeling）的方式进行，基于序列概率估计。对给定的单词序列 $x = x_{1}, x_{2}, \cdots, x_{n}$ ，计算以下负对数似然函数或交叉熵，并通过其最小化估计模型的参数。

$$
- \log P (\boldsymbol {x}) = - \sum_ {i = 1} ^ {n} \log P _ {\boldsymbol {\theta}} (x _ {i} | x _ {1}, x _ {2}, \dots , x _ {i - 1})\tag{27.3}
$$

其中， $\theta$ 表示 GPT 模型的参数。

BERT 的预训练主要通过掩码语言模型化（mask language modeling）的方式进行，可以认为基于后叙去噪自动编码器。假设单词序列 $x = x_{1}, x_{2}, \cdots, x_{n}$ 中有若干个单词被随机掩码，也就是被改为特殊字符 <mask>，得到掩码单词序列 $\tilde{x}$ ，假设被掩码的几个单词是 $\bar{x}$ 。计算以下负对数似然函数，并通过其最小化估计模型的参数。

$$
- \log P (\bar {\boldsymbol {x}} | \tilde {\boldsymbol {x}}) \approx - \sum_ {i = 1} ^ {n} \delta_ {i} \log P _ {\boldsymbol {\theta}} (x _ {i} | \tilde {\boldsymbol {x}})\tag{27.4}
$$

其中， $\theta$ 表示 BERT 模型的参数； $\delta_{i}$ 取值为 1 或 0，表示是否对位置 $i$ 的单词进行掩码处理。

GPT 适合于语言生成, 也可以用于语言理解。BERT 只能用于语言理解。语言理解是指对自然语言进行分析的处理, 如文本分类、文本匹配、文本序列标注。语言生成是指产生自然语言的处理, 可以是无条件的, 也可以是有条件的, 基于语言、图像等输入, 如机器翻译、图像标题生成。

## 27.1.2 模型和学习

## 1. 模型

GPT 是生成式预训练（generative pre-training）的缩写。GPT 的模型基于 Transformer 的解码器①，是单向语言模型。GPT 的预训练就是语言模型化，使用大规模语料基于序列概率估计原理进行模型的参数估计，学习的目标是预测给定单词序列中的每一个单词。学习和预测都是自回归过程（autoregressive process）。

GPT 模型有以下结构。输入是单词序列 $x_{1}, x_{2}, \cdots, x_{n}$ ，可以是一个句子或一段文章。首先经过输入层，产生初始的单词表示向量的序列，记作矩阵 $\boldsymbol{H}^{(0)}$ ：

$$
\pmb {H} ^ {(0)} = \pmb {X} + \pmb {E}\tag{27.5}
$$

其中，矩阵 X 表示单词的词嵌入（表示单词的实数向量）的序列 $\boldsymbol{X} = (\boldsymbol{x}_{1}, \boldsymbol{x}_{2}, \cdots, \boldsymbol{x}_{n})$ ，矩阵 E 表示单词的位置嵌入（表示位置的实数向量）的序列 $\boldsymbol{E} = (\boldsymbol{e}_{1}, \boldsymbol{e}_{2}, \cdots, \boldsymbol{e}_{n})$ 。 $X, E, H^{(0)}$ 是 $d \times n$ 矩阵，设词嵌入和位置嵌入向量的维度是 d。图 27.1 显示的是 GPT 模型输入层的计算。

![](../images/c72bd98c3d9724eb0b8de7a1aaad6c833d1c4a3aa5fa3c48f61d0a3d3953035d.jpg)  
图 27.1 GPT 模型输入层的计算

① 这里的 Transformer 解码器的每一层只包含自注意力子层和前馈网络子层，不包含编码器和解码器之间的注意力子层。

之后经过 $L$ 个解码层, 得到单词表示向量的序列, 记作矩阵 $\pmb{H}^{(L)}$ :

$$
\boldsymbol {H} ^ {(L)} = \text { transformer\_decoder } (\boldsymbol {H} ^ {(0)})\tag{27.6}
$$

具体地，

$$
\boldsymbol {H} ^ {(L)} = \left(\boldsymbol {h} _ {1} ^ {(L)}, \boldsymbol {h} _ {2} ^ {(L)}, \dots , \boldsymbol {h} _ {n} ^ {(L)}\right)
$$

其中, $\pmb{h}_{i}^{(L)}$ 是第 $i$ 个位置的单词表示向量。GPT 模型中, 在每一层, 每一个位置的表示向量是该位置的单词基于之前位置的上下文的表示 (contextualized representation)。注意: 一般的词向量是不依赖于上下文的表示 (见第 25 章)。

GPT 模型的输出是在单词序列各个位置上的条件概率, 第 $i$ 个位置的单词的条件概率 $p_i$ 定义为

$$
P _ {\pmb {\theta}} (x _ {i} | x _ {1}, x _ {2}, \dots , x _ {i - 1}) = \mathrm{softmax} (\pmb {W} _ {x} ^ {\mathrm{T}} \pmb {h} _ {i} ^ {(L)}) = \frac {\exp (\pmb {w} _ {x _ {i}} ^ {\mathrm{T}} \cdot \pmb {h} _ {i} ^ {(L)})}{\sum_ {x _ {i} ^ {\prime}} \exp (\pmb {w} _ {x _ {i} ^ {\prime}} ^ {\mathrm{T}} \cdot \pmb {h} _ {i} ^ {(L)})}\tag{27.7}
$$

其中， $x_{1}, x_{2}, \cdots, x_{i-1}$ 是之前位置的单词序列， $x_{i}$ 是当前位置的单词， $W_{x}$ 表示所有单词的权重矩阵， $\theta$ 表示模型的参数。

图 27.2 显示的是 GPT 模型的架构, 其中输入层进行式 (27.5) 的计算, 解码层整体进行式 (27.6) 的计算, 输出层进行式 (27.7) 的计算。GPT 利用 Transformer 解码器对语言的内容进行层次化的组合式的表示。

![](../images/10256756561c802f000b4c8020826ea33abaa3103b5ac0dc5cb8e4facde3687d.jpg)  
图27.2 GPT模型的架构

GPT 中解码层的多头自注意力都是单向的，也就是各个位置的单词只针对之前所有位置的单词进行自注意力计算。

GPT 模型有三个超参数: 解码层的层数 $L$ 、头的个数 $h$ 、模型的维度 $d$ 。取 $L = 12, h = 12, d = 768$ 。输入单词序列的最大长度一般是 128。

## 2. 预训练

预训练时, 估计模型的参数, 使模型对单词序列数据有准确的预测。损失函数是负对数似然函数或交叉熵 (式 (27.3))。

$$
L _ {\mathrm{PT}} = - \sum_ {i = 1} ^ {n} \log P _ {\boldsymbol {\theta}} \left(x _ {i} | x _ {1}, x _ {2}, \dots , x _ {i - 1}\right)\tag{27.8}
$$

其中， $\theta$ 是模型的参数，通过预训练估计得到，作为下游任务模型的初始值。整个预训练通过Transformer解码器的学习进行，包括强制教学、掩码注意力、反向传播。

## 3. 微调

微调时, 进一步调节参数, 使模型对下游任务有准确的预测。假设下游任务是文本分类, 输入是单词序列 $x' = x_1, x_2, \cdots, x_m$ , 输出是类别 $y$ , 计算条件概率 $P(y|x_1, x_2, \cdots, x_m)$ :

$$
P _ {\pmb {\theta}, \phi} (y | x _ {1}, x _ {2}, \dots , x _ {m}) = \mathrm{softmax} (\pmb {W} _ {y} ^ {\mathrm{T}} \pmb {h} _ {m} ^ {(L)}) = \frac {\exp (\pmb {w} _ {y} ^ {\mathrm{T}} \cdot \pmb {h} _ {m} ^ {(L)})}{\sum_ {y ^ {\prime}} \exp (\pmb {w} _ {y ^ {\prime}} ^ {\mathrm{T}} \cdot \pmb {h} _ {m} ^ {(L)})}\tag{27.9}
$$

其中， $h_{m}^{(L)}$ 是第 $L$ 个解码层最后位置的单词的表示向量， $W_{y}$ 是类别的权重矩阵， $\phi$ 表示分类的参数。

损失函数包括两部分（ $\lambda$ 是系数）：

$$
L _ {\mathrm{FT}} = L _ {\mathrm{CLS}} + \lambda \cdot L _ {\mathrm{LM}}\tag{27.10}
$$

一个是分类的损失函数：

$$
L _ {\mathrm{CLS}} = - \log P _ {\boldsymbol {\theta}, \phi} (y | \boldsymbol {x} ^ {\prime})\tag{27.11}
$$

另一个是语言模型化的损失函数：

$$
L _ {\mathrm{LM}} = - \sum_ {j = 1} ^ {m} \log P _ {\boldsymbol {\theta}} (x _ {j} | x _ {1}, x _ {2}, \dots , x _ {j - 1})\tag{27.12}
$$

前者是微调的主要部分。微调中，预训练模型的参数 $\theta$ 作为初始值，在这个过程中得到进一步学习，同时分类的参数 $\phi$ 也得到学习。

如果下游任务是生成，针对输入单词序列是 $x_{1}, x_{2}, \cdots, x_{m}$ ，进一步调节模型的参数，使得模型对之有准确的预测。损失函数只有语言模型化的部分。

$$
L _ {\mathrm{FT}} = L _ {\mathrm{LM}}
$$

## 4. 模型特点

GPT 的模型是单向语言模型, 而不是双向语言模型。学习 (预训练) 和预测的过程都是自回归的, 保证学习和预测的一致。可用于语言生成, 而用于语言理解时不具备优势。因为语言理解中，输入是一个句子或一段文章，从两个方向同时对语言建模更加合理。BERT 可以解决这个问题。

## 27.2 BERT 模型

BERT 及其扩展版本是常用的预训练语言模型，适合于语言理解任务。BERT 的预训练使用掩码语言模型化，可以认为是一种去噪自动编码器的学习。本节首先介绍自动编码器和去噪自动编码器，之后叙述 BERT 的模型和学习算法，最后总结 BERT 模型的特点。

## 27.2.1 去噪自动编码器

## 1. 自动编码器

自动编码器（auto encoder）是用于数据表示的无监督学习的一种神经网络。自动编码器由编码器网络和解码器网络组成。学习时，编码器将输入向量 x 转换为中间表示向量 z，解码器再将中间表示向量 z 转换为输出向量 y。假设 x 和 y 的维度相同，而 z 的维度远低于 x 和 y 的维度。学习的目标是尽量使输出向量 y 和输入向量 x 保持一致，或者说重建输入向量 x。认为学到的中间表示向量 z 就是数据 x 的表示。图 27.3 显示自动编码器的架构。

![](../images/62520da36ca52e6aefef1b3387b360bca9bde99f8703a219f2e531c19634001b.jpg)  
图 27.3 自动编码器的架构

最基本的情况下，编码器和解码器分别都是一层神经网络，编码器是

$$
\boldsymbol {z} = F (\boldsymbol {x}) = a \left(\boldsymbol {W} _ {\mathrm{E}} \boldsymbol {x} + \boldsymbol {b} _ {\mathrm{E}}\right)\tag{27.13}
$$

其中， $W_{E}$ 是权重矩阵， $b_{E}$ 是偏置向量， $a(\cdot)$ 是激活函数。解码器是

$$
\boldsymbol {y} = G (\boldsymbol {z}) = a \left(\boldsymbol {W} _ {\mathrm{D}} \boldsymbol {z} + \boldsymbol {b} _ {\mathrm{D}}\right)\tag{27.14}
$$

其中， $W_{\mathrm{D}}$ 是权重矩阵， $b_{\mathrm{D}}$ 是偏置向量， $a(\cdot)$ 是激活函数。有时假设 $W_{\mathrm{E}}^{\mathrm{T}} = W_{\mathrm{D}}$ 成立。可见以上的自动编码器是一种特殊的前馈神经网络。

学习时，目标函数是

$$
L = \frac {1}{N} \sum_ {i = 1} ^ {N} L (\boldsymbol {x} _ {i}, \boldsymbol {y} _ {i}) = \frac {1}{N} \sum_ {i = 1} ^ {N} L (\boldsymbol {x} _ {i}, G (F (\boldsymbol {x} _ {i})))\tag{27.15}
$$

其中，N 是样本容量； $L(x_{i},y_{i})$ 是损失函数，比如平方损失：

$$
L (\boldsymbol {x}, \boldsymbol {y}) = | | \boldsymbol {x} - \boldsymbol {y} | | ^ {2}
$$

学习的算法一般是梯度下降。

自动编码器学习实际进行的是对数据的压缩（编码），得到的中间表示能有效地刻画数据的特征。因为通过解压（解码）可以得到原始数据的近似，说明中间表示保留了数据中的主要信息。

预测时，通常用编码器将新的输入向量 $x'$ 转换为中间表示向量 $z'$ 。

$$
\boldsymbol {z} ^ {\prime} = F \left(\boldsymbol {x} ^ {\prime}\right) = a (\boldsymbol {W} _ {\mathrm{E}} \boldsymbol {x} ^ {\prime} + \boldsymbol {b} _ {\mathrm{E}})\tag{27.16}
$$

自动编码器可以用于数据的压缩、聚类等应用。

当编码器和解码器都是线性函数时, 即 $F(x) = W_{\mathrm{E}} x$ , $G(z) = W_{\mathrm{D}} z$ 时, 可以通过主成分分析 (见第 16 章) 学习自动编码器。也就是说主成分分析是自动编码器的一种特殊情况。证明留作习题。

## 2. 去噪自动编码器

去噪自动编码器（denoising autoencoder）是自动编码器的扩展，学习时在输入中加入随机噪声，以学到稳健的自动编码器。去噪自动编码器不仅可以用于数据表示学习，而且可以用于数据去噪。

学习时, 首先根据条件概率分布 $P(\tilde{\boldsymbol{x}} | \boldsymbol{x})$ 对输入向量 $\boldsymbol{x}$ 进行随机变换, 得到有噪声的输入向量 $\tilde{\boldsymbol{x}}$ 。比如随机地选取 $\boldsymbol{x}$ 的一些元素将其置为 0 , 然后以 $\tilde{\boldsymbol{x}}$ 为输入学习自动编码器。编码器将有噪声的输入向量 $\tilde{\boldsymbol{x}}$ 转换为中间表示向量 $\boldsymbol{z}$ , 解码器再将中间表示向量 $\boldsymbol{z}$ 转换为输出向量 $\boldsymbol{y}$ 。学习的目标是尽量使输出向量 $\boldsymbol{y}$ 和原始输入向量 $\boldsymbol{x}$ 保持一致, 或者说重建原始输入向量 $\boldsymbol{x}$ , 比如复原 $\tilde{\boldsymbol{x}}$ 的置为 0 的元素的值。最基本的情况下, 编码器、解码器、目标函数分别是

$$
\boldsymbol {z} = F (\boldsymbol {x}) = a \left(\boldsymbol {W} _ {\mathrm{E}} \tilde {\boldsymbol {x}} + \boldsymbol {b} _ {\mathrm{E}}\right)\tag{27.17}
$$

$$
\boldsymbol {y} = G (\boldsymbol {z}) = a \left(\boldsymbol {W} _ {\mathrm{D}} \boldsymbol {z} + \boldsymbol {b} _ {\mathrm{D}}\right)\tag{27.18}
$$

$$
L = \frac {1}{N} \sum_ {i = 1} ^ {N} L (\boldsymbol {x} _ {i}, \boldsymbol {y} _ {i}) = \frac {1}{N} \sum_ {i = 1} ^ {N} L (\boldsymbol {x} _ {i}, G (F (\tilde {\boldsymbol {x}} _ {i})))\tag{27.19}
$$

因为学习的目标是排除噪声的干扰重建数据，去噪自动编码器能更有效地学到数据的主要特征。

预测时, 用编码器将新的输入向量 $x'$ 转换为中间表示向量 $z'$ , 或者进一步用解码器将中间表示变量 $z'$ 转换为输出向量 $y'$ 。

$$
\boldsymbol {z} ^ {\prime} = F \left(\boldsymbol {x} ^ {\prime}\right) = a (\boldsymbol {W} _ {\mathrm{E}} \boldsymbol {x} ^ {\prime} + \boldsymbol {b} _ {\mathrm{E}})\tag{27.20}
$$

$$
\boldsymbol {y} ^ {\prime} = G \left(\boldsymbol {z} ^ {\prime}\right) = a \left(\boldsymbol {W} _ {\mathrm{D}} \boldsymbol {z} ^ {\prime} + \boldsymbol {b} _ {\mathrm{D}}\right)\tag{27.21}
$$

如果输入向量 $x'$ 是含有噪声的数据，那么 $y'$ 就是去噪后的数据。用去噪自动编码器可以对数据去噪。

## 27.2.2 模型和学习

## 1. 模型

BERT 是双向 Transformer 编码器表示（bidirectional encoder representations from Transformers）的缩写。BERT 的模型基于 Transformer 的编码器，是双向语言模型。BERT 的预训练主要是掩码语言模型化，使用大规模语料基于去噪自动编码器原理进行模型的参数估计，学习的目标是复原给定的掩码单词序列中被掩码的每一个单词。学习和预测都是非自回归过程（non-autoregressive process）。

BERT 模型有以下结构。输入是两个合并的单词序列。

$$
<   \mathrm{cls} >, x _ {1}, x _ {2}, \dots , x _ {m - 1}, <   \mathrm{sep} >, x _ {m + 1}, x _ {m + 2}, \dots , x _ {m + n - 1}, <   \mathrm{sep} >
$$

其中， $x_{1}, x_{2}, \cdots, x_{m-1}$ 是第一个单词序列， $x_{m+1}, x_{m+2}, \cdots, x_{m+n-1}$ 是第二个单词序列，<cls>是表示类别的特殊字符，<sep>是表示序列分割的特殊字符，合并的单词序列共有 $m + n + 1$ 个单词和字符。每一个单词序列是一个句子或一段文章。首先经过输入层，产生初始的单词表示向量的序列，记作矩阵 $\mathbf{H}^{(0)}$ ：

$$
\boldsymbol {H} ^ {(0)} = \boldsymbol {X} + \boldsymbol {S} + \boldsymbol {E}\tag{27.22}
$$

其中，矩阵 X 表示单词的词嵌入的序列 $\boldsymbol{X} = (\boldsymbol{x}_{0}, \boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{m+n})$ ；矩阵 E 表示单词的位置嵌入的序列 $\boldsymbol{E} = (\boldsymbol{e}_{0}, \boldsymbol{e}_{1}, \cdots, \boldsymbol{e}_{m+n})$ ；矩阵 S 是区别前后单词序列的标记序列 $\boldsymbol{S} = (\boldsymbol{a}, \boldsymbol{a}, \cdots, \boldsymbol{a}, \boldsymbol{b}, \boldsymbol{b}, \cdots, \boldsymbol{b})$ ，含有 $m+1$ 个向量 a 和 n 个向量 b。 $\boldsymbol{X}, \boldsymbol{E}, \boldsymbol{S}, \boldsymbol{H}^{(0)}$ 是 $d \times (m+n+1)$ 矩阵，设词嵌入、位置嵌入、标记向量的维度是 d。图 27.4 显示的是 BERT 模型输入层的计算。

![](../images/846e0b0e4ca79032dfdde5c1084e77b922493a5133316a4486d62be82c9e7468.jpg)  
图 27.4 BERT 模型输入层的计算

使用拼接的单词序列（两个单词序列）作为输入是让 BERT 不仅能用于以一个文本为输入的任务，如文本分类，也能用于以两个文本为输入的任务，如文本匹配。

之后经过 $L$ 个编码层, 得到单词的表示向量的序列, 记作 $\pmb{H}^{(L)}$ :

$$
\boldsymbol {H} ^ {(L)} = \text { transformer\_encoder } (\boldsymbol {H} ^ {(0)})\tag{27.23}
$$

具体地，

$$
\boldsymbol {H} ^ {(L)} = \left(\boldsymbol {h} _ {0} ^ {(L)}, \boldsymbol {h} _ {1} ^ {(L)}, \dots , \boldsymbol {h} _ {m + n} ^ {(L)}\right)
$$

其中, $\pmb{h}_i^{(L)}$ 是第 $i$ 个位置的单词的表示向量。BERT 模型中, 在每一层, 每一个位置的表示向量是该位置的单词基于之前位置和之后位置的上下文的表示 (contextualized representation)。

BERT 模型的输出是在合并的单词序列的各个位置上的条件概率, 第 $i$ 个位置的单词 (包括特殊字符) 的条件概率 $p_i$ 定义为

$$
P _ {\pmb {\theta}} (x _ {i} | x _ {0}, \dots , x _ {i - 1}, x _ {i + 1}, \dots , x _ {m + n}) = \mathrm{softmax} (\pmb {W} _ {x} ^ {\mathrm{T}} \pmb {h} _ {i} ^ {(L)}) = \frac {\exp (\pmb {w} _ {x _ {i}} ^ {\mathrm{T}} \cdot \pmb {h} _ {i} ^ {(L)})}{\sum_ {x _ {i} ^ {\prime}} \exp (\pmb {w} _ {x _ {i} ^ {\prime}} ^ {\mathrm{T}} \cdot \pmb {h} _ {i} ^ {(L)})}\tag{27.24}
$$

其中， $x_{0},\cdots,x_{i-1},x_{i+1},\cdots,x_{m+n}$ 是其他位置的单词， $x_{i}$ 是当前位置的单词， $W_{x}$ 表示所有单词的权重矩阵， $\theta$ 表示模型的参数。

图 27.4 显示的是 BERT 模型的架构, 其中输入层进行式 (27.22) 的计算, 编码层整体进行式 (27.23) 的计算, 输出层进行式 (27.24) 的计算。BERT 利用 Transformer 编码器对语言的内容进行层次化的组合式的表示。

![](../images/3efa880b403fded99cb47d0eefeba9251246ba33e7bb51bd1c9de62e91b5bf6c.jpg)  
图 27.5 BERT 模型的架构

BERT 中编码层的多头自注意力都是双向的，也就是各个位置的单词针对其他位置的单词都进行自注意力计算，这一点与 GPT 不同。图 27.6 比较了 BERT 和 GPT 中表示之间关系的差异。BERT 中每一层每一个位置的表示都是由下一层所有位置的表示组合而成，而 GPT 中每一层每一个位置的表示都是由下一层之前所有位置的表示组合而成。

GPT  
![](../images/d711a8bfc65b0101c106b03fc55feecb484e4b9fcba39874a72ef8f8be4b1ade.jpg)  
BERT

![](../images/78e614a9e0baaf48c8b27d77f83e440b245d77f430ed486647ee8b4502639482.jpg)  
图27.6 BERT模型和GPT模型的比较

BERT 模型有三个超参数: 编码层的层数 $L$ 、头的个数 $h$ 、模型的维度 $d$ 。BERT Base 模型取 $L = 12, h = 12, d = 768$ 。输入合并单词序列的最大长度一般是 128。

## 2. 预训练

预训练数据的每个样本由两个单词序列 A 和 B 合并组成, 中间由特殊字符 <sep> 分割。 $50\%$ 的样本中 A 和 B 是同一篇文章中的连续文本, $50\%$ 的样本中 A 和 B 来自不同篇文章。在每一个样本的合并单词序列中, 随机选择 $15\%$ 的位置进行掩码操作。对于掩码操作, 在选择的 $15\%$ 的位置上, 有 $80\%$ 的单词替换为特殊字符 <mask>, 有 $10\%$ 的单词随机替换为其他单词, 剩下 $10\%$ 的单词保持不变。

BERT 模型的预训练由两部分组成, 掩码语言模型化 (mask language modeling) 和下句预测 (next sentence prediction)。掩码语言模型化的目标是复原输入单词序列中被掩码的单词。可以看作是去噪自动编码器学习, 对被掩码的单词独立地进行复原。下句预测的目标是判断输入单词序列是否来自同一篇文章。这里说的下句未必是一个自然句, 也可以是多个自然句。掩码单词序列表示为 $\tilde{\mathbf{x}}$ 。

掩码语言模型化在每一个掩码位置计算条件概率（式(27.4)）：

$$
P _ {\theta} \left(x _ {i} | \tilde {x} _ {0}, \tilde {x} _ {1}, \dots , \tilde {x} _ {m + n}\right) = \mathrm{softmax} \left(\boldsymbol {W} _ {x} ^ {\mathrm{T}} \boldsymbol {h} _ {i} ^ {(L)}\right) = \frac {\exp (\boldsymbol {w} _ {x _ {i}} ^ {\mathrm{T}} \cdot \boldsymbol {h} _ {i} ^ {(L)})}{\sum_ {x _ {i} ^ {\prime}} \exp (\boldsymbol {w} _ {x _ {i} ^ {\prime}} ^ {\mathrm{T}} \cdot \boldsymbol {h} _ {i} ^ {(L)})}\tag{27.25}
$$

假设第 $i$ 个位置是掩码位置, $\pmb{h}_i^{(L)}$ 是在第 $L$ 层第 $i$ 个位置的表示, $x_i$ 是预测的单词, $\pmb{W}_x$ 是单词的权重矩阵。

下句预测计算条件概率:

$$
P _ {\pmb {\theta}} (s | \tilde {x} _ {0}, \tilde {x} _ {1}, \dots , \tilde {x} _ {m + n}) = \sigma (\pmb {w} _ {s} ^ {\mathrm{T}} \cdot \pmb {h} _ {\mathrm{cls}} ^ {(L)}) = \frac {\exp (\pmb {w} _ {s} ^ {\mathrm{T}} \cdot \pmb {h} _ {\mathrm{cls}} ^ {(L)})}{1 + \exp (\pmb {w} _ {s} ^ {\mathrm{T}} \cdot \pmb {h} _ {\mathrm{cls}} ^ {(L)})}\tag{27.26}
$$

其中, $\pmb{h}_{\mathrm{cls}}^{(L)}$ 是在第 $L$ 层的类别特殊字符 $< \mathrm{cls}>$ 的表示向量; $\pmb{w}_s$ 是下句预测的权重向量; $s$ 取值为 1 或 0 , 表示两个单词序列是否来自同一篇文章。

预训练的损失函数为

$$
L _ {\mathrm{PT}} = L _ {\mathrm{MLM}} + \lambda \cdot L _ {\mathrm{NSP}}\tag{27.27}
$$

其中， $L_{MLM}$ 是掩码语言模型化损失， $L_{NSP}$ 是下句预测损失， $\lambda$ 是系数。

$$
L _ {\mathrm{MLM}} = - \sum_ {i = 0} ^ {m + n} \delta_ {i} \log P _ {\theta} (x _ {i} | \tilde {x} _ {0}, \tilde {x} _ {1}, \dots , \tilde {x} _ {m + n})\tag{27.28}
$$

其中， $\delta_{i}$ 取值为 1 或 0，表示第 $i$ 个位置是否被掩码； $\theta$ 是模型的参数。

$$
L _ {\mathrm{NSP}} = - \log P _ {\boldsymbol {\theta}} (s | \tilde {x} _ {0}, \tilde {x} _ {1}, \dots , \tilde {x} _ {m + n})\tag{27.29}
$$

预训练得到的模型参数 $\theta$ 作为下游任务模型的初始值。

掩码语言模型化是预训练的主要部分，下句预测的目标是让 BERT 既能用于以一个单词序列为输入的任务，如文本分类，也能用于以两个单词序列为输入的任务，如文本匹配。后续的研究发现，下句预测未必一定需要。当数据量足够大时，可以只通过掩码语言模型化进行预训练。也就是说，

$$
L _ {\mathrm{PT}} = L _ {\mathrm{MLM}}
$$

改进版 RoBERTa 模型就采用这个方法。

## 3. 微调

微调时, 进一步调节参数, 使模型对下游任务有准确的预测。假设下游任务是文本分类, 输入单词序列是 $x' = x_0, x_1, \cdots, x_l$ , 输出是类别 $y$ , 计算条件概率 $P(y|x_0, x_1, \cdots, x_l)$ :

$$
P _ {\theta , \phi} (y | x _ {0}, x _ {1}, \dots , x _ {l}) = \operatorname{softmax} (\boldsymbol {W} _ {y} ^ {\mathrm{T}} \boldsymbol {h} _ {\mathrm{cls}} ^ {(L)}) = \frac {\exp \boldsymbol {w} _ {y} ^ {\mathrm{T}} \cdot \boldsymbol {h} _ {\mathrm{cls}} ^ {(L)}}{\sum_ {y ^ {\prime}} \exp \boldsymbol {w} _ {y ^ {\prime}} ^ {\mathrm{T}} \cdot \boldsymbol {h} _ {\mathrm{cls}} ^ {(L)}}\tag{27.30}
$$

其中, $\pmb{h}_{\mathrm{cls}}^{(L)}$ 是第 $L$ 层的类别特殊字符 $<\mathrm{cls}>$ 的表示向量, $\pmb{W}_y$ 是类别的权重矩阵, $\phi$ 表示分类的参数。这时单词序列 $x_0, x_1, \cdots, x_l$ 是一个句子或一段文章, 以特殊字符 $<\mathrm{cls}>$ 开始, 以特殊字符 $<\mathrm{sep}>$ 结束。

微调的损失函数为

$$
L _ {\mathrm{FT}} = - \log P _ {\boldsymbol {\theta}, \phi} (y | x ^ {\prime})\tag{27.31}
$$

微调中, 预训练模型的参数 $\theta$ 作为初始值, 在这个过程中进一步得到学习, 以帮助更好地分类; 同时分类的参数 $\phi$ 也得到学习。

如果下游任务是文本匹配, 如判断两句话是否形成一问一答。输入单词序列是 $x_0, x_1, \cdots, x_l$ , 输出是类别 $y$ , 仍然计算条件概率 $P(y|x_0, x_1, \cdots, x_l)$ 。类别有两类, 表示匹配或不匹配。这时单词序列 $x_0, x_1, \cdots, x_l$ 是两个单词序列合并的序列, 如一个问句和一个答句合并而成。以特殊字符 <cls> 开始, 中间以特殊字符 <sep> 间隔, 最后以特殊字符 <sep> 结束。

## 27.2.3 模型特点

BERT 通过其多层多头注意力机制能够有效地表示语言的词汇、语法、语义信息（Transformer 和 GPT 也有类似的特点）。通过自注意力，每一层的每一个位置的单词表示与其他位置的单词表示组合成新的表示，传递到上一层的同一位置。自注意力是多头的，一个头代表一个侧面，因此每一个位置的单词表示由多个不同侧面的表示组合而成。单词表示的内容可以通过自注意力的权重推测。图 27.7 和图 27.8 给出显示 BERT 的权重分布的例子。权重的大小代表了单词表示的组合过程中各个单词表示的作用的大小。

![](../images/df975f9a1904c3ad15d2ddfad56ea765262b09122db64fc2ebde88944135ed2f.jpg)  
图 27.7 BERT 模型的注意力权重分布的例子  
存在于不同层不同头

注意力权重的分布有几种类型。如图 27.7 所示，注意力可能是发散的，可能集中到前一个位置的单词或者后一个位置的单词，可能集中到特殊字符 <sep>，也可能集中到标点符号。这里说的注意力集中是指自注意力计算中只有一个位置的权重很大而其他位置的权重很小的情况。研究发现，有些注意力是冗余的，屏蔽掉它们（权重置为 0），模型预测的结果并没有大的改变，但模型整体的多层多头自注意力机制对语言刻画是有必要的。

BERT 的各层有不同的特点。底层主要表示词汇信息，中层主要表示语法信息，上层主要表示语义信息。从图 27.8 中的例子可以看出，对给定的自然语言输入，不同层不同头可以表示其中的动词-宾语关系、冠词-名词关系、介词-名词关系、代词指代关系等。

对 BERT 和 GPT 的直观解释是: 机器基于大量的语料, 做了大量的词语填空 (BERT) 或词语接龙 (GPT) 练习, 捕捉到了由单词组成句子、再由句子组成文章的各种规律, 并且把它们表示并记忆在模型之中 (注意: 文章不是由单词和句子随机组成的, 而是遵循词汇、语法、语义规则组合而成)。也就是说, BERT 通过无监督学习获取了大量的词汇、语法、语

Head 8-10

Head 8-11

\- Noun modifiers (e.g., determiners) attend to their noun

\- 94.3% accuracy at the det relation

![](../images/b6774346f3a16bbd0de782c939f35397698c399a1aa22175138a634f08cc894b.jpg)

\- Prepositions attend to their objects
- 76.3% accuracy at the pobj relation - Coreferent mentions attend to their antecedents
- 65.1% accuracy at linking the head of a coreferent mention to the head of an antecedent

图 27.8 BERT 模型中的注意力权重分布的例子（见文前彩图）

可以表示词汇、语法、语义关系

义知识。当用于一个下游任务时，只需要很少的标注数据就可以学习到完成该任务所需的知识。

## 本章概要

1. 预训练语言模型是基于具有强大表示能力的神经网络的语言模型。首先在预训练中，使用大规模的语料通过无监督学习的方式学习模型的参数。之后在微调中，将模型用于一个具体任务，使用少量的标注数据通过监督学习的方式进一步调节模型的参数。预训练语言模型通常可以有效地表示语言的词汇、句法和语义特征，用于下游任务。

2. 有代表性的预训练语言模型有 GPT 和 BERT，分别由 Transformer 的解码器和编码器实现。GPT 是单向语言模型，适用于语言生成，也可以用于语言理解。BERT 是双向语言模型，只能用于语言理解。

GPT 的单向语言模型由以下单词的生成条件概率组成:

$$
P \left(x _ {i} | x _ {1}, \dots , x _ {i - 1}\right), \quad i = 1, 2, \dots , n
$$

每一个位置的单词依赖于之前位置的单词。GPT 的预训练通过语言模型化进行，基于序列概率估计原理。

BERT 的双向语言模型由以下单词生成的条件概率组成:

$$
P \left(x _ {i} | x _ {1}, \dots , x _ {i - 1}, x _ {i + 1}, \dots , x _ {n}\right), \quad i = 1, 2, \dots , n
$$

每一个位置的单词依赖于之前位置和之后位置的单词。BERT 的预训练主要通过掩码语言模型化进行，基于去噪自动编码器原理。

3. GPT 模型的输入是单词序列, 可以是一个句子或一段文章。首先经过输入层, 产生初始的单词表示向量的序列。之后经过 $L$ 个 Transformer 解码层, 得到单词表示向量的序列, GPT 模型的输出是在单词序列各个位置上的条件概率。

GPT 预训练时，通过极大似然估计学习模型的参数。

$$
L _ {\mathrm{train}} = - \sum_ {i = 1} ^ {n} \log P _ {\pmb {\theta}} (x _ {i} | x _ {1}, x _ {2}, \dots , x _ {i - 1})
$$

GPT 微调时, 通过优化下游任务的目标函数, 进一步调节模型的参数。

4. 自动编码器是用于数据表示的无监督学习的一种神经网络。自动编码器由编码器网络和解码器网络组成。学习时编码器将输入向量转换为中间表示向量，解码器再将中间表示向量转换为输出向量。编码器和解码器可以是

$$
\begin{array}{l} \boldsymbol {z} = F (\boldsymbol {x}) = a (\boldsymbol {W} _ {\mathrm{E}} \boldsymbol {x} + \boldsymbol {b} _ {\mathrm{E}}) \\ \boldsymbol {y} = G (\boldsymbol {z}) = a (\boldsymbol {W} _ {\mathrm{D}} \boldsymbol {z} + \boldsymbol {b} _ {\mathrm{D}}) \end{array}
$$

学习的目标是尽量使输出向量和输入向量保持一致，或者说重建输入向量。认为学到的中间表示向量就是数据的表示。

$$
L = \frac {1}{N} \sum_ {i = 1} ^ {N} L (\boldsymbol {x} _ {i}, G (F (\boldsymbol {x} _ {i})))
$$

学习的算法一般是梯度下降。自动编码器学习实际进行的是对数据的压缩。

5. 去噪自动编码器是自动编码器的一种扩展, 去噪自动编码器不仅可以用于数据表示学习, 而且可以用于数据去噪。学习时首先根据对输入向量进行的随机变换, 得到有噪声的输入向量。编码器将有噪声的输入向量转换为中间表示向量, 解码器再将中间表示向量转换为输出向量。编码器、解码器、目标函数分别是

$$
\pmb {z} = F (\pmb {x}) = a (\pmb {W} _ {\mathrm{E}} \tilde {\pmb {x}} + \pmb {b} _ {\mathrm{E}})
$$

$$
\boldsymbol {y} = G (\boldsymbol {z}) = a \left(\boldsymbol {W} _ {\mathrm{D}} \boldsymbol {z} + \boldsymbol {b} _ {\mathrm{D}}\right)
$$

$$
L = \frac {1}{N} \sum_ {i = 1} ^ {N} L (\boldsymbol {x} _ {i}, G (F (\tilde {\boldsymbol {x}} _ {i})))
$$

学习的目标是尽量使输出向量和原始输入向量保持一致，或者说重建原始输入向量。因为学习的目标是排除噪声的干扰重建数据，去噪自动编码器能更有效地学到数据的主要特征。

6. BERT 模型的输入是两个合并的单词序列。首先经过输入层，产生初始的单词表示向量的序列。之后经过 $L$ 个Transformer编码层，得到单词的表示向量的序列。BERT 模型的输出是在单词序列的各个位置上的条件概率。

BERT 模型的预训练由掩码语言模型化和下句预测组成。掩码语言模型化的目标是复原输入单词序列中被掩码的单词。下句预测的目标是判断输入单词序列是否来自同一篇文章。预训练以掩码语言模型化为主，其损失函数是

$$
L _ {1} = - \sum_ {i = 0} ^ {m + n} \delta_ {i} \log P _ {\boldsymbol {\theta}} (x _ {i} | \tilde {x} _ {0}, \tilde {x} _ {1}, \dots \tilde {x} _ {m + n})
$$

BERT 微调时, 通过优化下游任务的目标函数, 进一步调节模型的参数。

## 继续阅读

BERT 的原始论文是文献 [1], GPT, GPT-2, GPT-3 的原始论文是文献 [2]～文献 [4]。BERT 的改进工作有 RoBERTa[5], XLNet[6] 等。本章介绍的 BERT 的分析结果见文献 [7]。DAE 的原始论文可见文献 [8]。BERT 和 GPT 之前的预训练语言模型有 ELMo[9]。

## 习题

27.1 设计基于双向 LSTM 的预训练语言模型，假设下游任务是文本分类。

27.2 假设 GPT 微调的下游任务是两个文本的匹配, 写出学习的目标函数。

27.3 设计一个 2 层卷积神经网络编码器和 2 层卷积神经网络解码器组成的自动编码器（使用第 28 章介绍的转置卷积）。

27.4 证明当编码器和解码器都是线性函数时, 主成分分析可以作为自动编码器学习的方法。

27.5 解释为什么 BERT 预训练中的掩码语言模型化是基于去噪自动编码器原理的。

27.6 比较 BERT 与 Transformer 编码器在模型上的异同。

## 参考文献

[1] DEVLIN J, CHANG M W, LEE K, et al. BERT: pre-training of deep bidirectional transformers for language understanding[C]//Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2019: 4171-4186.

[2] RADFORD A, NARASIMHAN K, SALIMANS T, et al. Improving language understanding by generative pre-training[J]. 2018.

[3] RADFORD A, WU J, CHILD R, et al. Language models are unsupervised multitask learners[J]. OpenAI Blog, 2019, 1(8).

[4] BROWN T B, MANN B, RYDER N, et al. Language models are few-shot learners[Z/OL]. arXiv preprint arXiv:2005.14165, 2020.

[5] LIU Y, OTT M, GOYAL N, et al. Roberta: A robustly optimized bert pretraining approach[Z/OL]. arXiv preprint arXiv:1907.11692, 2019.

[6] YANG Z, DAI Z, YANG Y, et al. Xlnet: Generalized autoregressive pretraining for language understanding[J]. Advances in Neural Information Processing Systems, 2019: 5754-5764.

[7] CLARK K, KHANDELWAL U, LEVY O, et al. What does BERT look at? An analysis of BERT's attention[C]//Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP. 2019: 276-286.

[8] VINCENT P, LAROCHELLE H, BENGIO Y, et al. Extracting and composing robust features with denoising autoencoders[C]//Proceedings of the 25th International Conference on Machine Learning. 2018: 1096-1103.

[9] PETERS ME, NEUMANN M, IYYER M, et al. Deep contextualized word representations[C]//Proceedings of NAACL-HLT. 2018: 2227-2237.

