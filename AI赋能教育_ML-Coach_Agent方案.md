# AI 赋能教育 Agent 方案：ML-Coach

# 1. 项目定位

**ML-Coach** 是一个面向机器学习学习场景的监督式自适应教学 Agent。

它的目标不是单纯回答学生问题，而是通过：

- 学习规划
- 知识讲解
- 主动练习
- 学习监督
- 错误诊断
- 阶段验收
- 动态复习

形成完整的学习闭环。

核心目标：

$$
\text{更少时间}
\rightarrow
\text{掌握更多核心知识}
\rightarrow
\text{及时发现薄弱点}
\rightarrow
\text{通过验收后进入下一阶段}
$$

系统重点解决两个问题：

1. **高效**：避免学生重复学习已经掌握的内容，把时间集中在薄弱知识点。
2. **监督**：通过任务计划、学习记录、测试验收和 Mastery Gate，保证学习真正完成。

---

# 2. 整体架构

建议采用：

$$
\text{Supervisor Agent}
+
\text{多个教学子 Agent}
$$

整体结构：

```text
                    ┌─────────────────┐
                    │ Supervisor Agent│
                    │   教学总监督器    │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ↓                  ↓                  ↓
   Planning Agent      Teaching Agent      Evaluation Agent
     学习规划              知识教学              学习验收
          │                  │                  │
          └──────────┬───────┴─────────┬────────┘
                     ↓                 ↓
              Practice Agent     Diagnosis Agent
                 练习系统            薄弱点诊断
                     │                 │
                     └────────┬────────┘
                              ↓
                       Learning Memory
                         学习者模型
```

---

# 3. Supervisor Agent：教学总监督器

Supervisor Agent 是整个系统的核心。

它类似：

> AI 班主任 + 教学主管。

它不主要负责详细知识讲解，而是负责控制整个学习过程。

主要职责：

- 判断当前应该学习什么
- 检查学生是否真正掌握
- 判断是否需要复习
- 判断是否需要回退到前置知识
- 控制是否进入下一知识点
- 监督每日学习任务完成情况

例如学生学习“逻辑回归”时，流程不能只是：

```text
阅读解释
↓
学习结束
```

而应该完成：

```text
前置知识检测
↓
知识学习
↓
概念测试
↓
数学理解
↓
代码练习
↓
错误诊断
↓
综合测试
↓
Mastery Check
```

例如设置掌握标准：

```text
Mastery Score >= 85
```

达到标准后才能进入下一知识点。

否则：

```text
定位薄弱点
↓
重新讲解
↓
针对性训练
↓
再次测试
```

---

# 4. Planning Agent：学习路径规划

Planning Agent 负责根据学生当前能力动态制定学习路径。

机器学习知识存在明显的依赖关系，例如：

```text
线性代数
    ↓
线性回归
    ↓
梯度下降
    ↓
逻辑回归
    ↓
交叉熵
    ↓
神经网络
```

因此可以建立机器学习知识图谱。

示例：

```text
机器学习
│
├── 数学基础
│   ├── 向量
│   ├── 矩阵
│   ├── 概率
│   └── 微积分
│
├── 监督学习
│   ├── Linear Regression
│   ├── Logistic Regression
│   ├── SVM
│   ├── Decision Tree
│   └── Ensemble
│
├── 无监督学习
│   ├── K-Means
│   ├── PCA
│   └── GMM
│
├── 模型训练
│   ├── Loss
│   ├── Gradient Descent
│   ├── Regularization
│   └── Hyperparameter
│
└── 模型评估
    ├── Accuracy
    ├── Precision / Recall
    ├── F1
    ├── ROC-AUC
    └── Cross Validation
```

下一知识点可以由以下信息共同决定：

```text
Next Topic =
学生当前知识状态
+ 历史错误
+ 学习目标
+ 可用时间
+ 前置知识掌握度
```

---

# 5. Teaching Agent：知识教学

Teaching Agent 负责具体知识讲解。

建议统一采用四层教学结构。

## 第一层：直觉理解

先解释这个知识点解决什么问题。

例如逻辑回归：

> 逻辑回归本质上是在线性模型输出之后使用 Sigmoid，将结果映射为概率，用于分类任务。

---

## 第二层：数学定义

例如：

$$
z=w^Tx+b
$$

$$
P(y=1|x)=\sigma(z)
$$

其中：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

---

## 第三层：机制解释

继续解释：

- 为什么需要 Sigmoid
- 为什么输出可以解释为概率
- 为什么常使用交叉熵
- 决策边界为什么仍然是线性的

---

## 第四层：代码实践

例如：

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

继续要求学生理解：

```text
coef_
predict()
predict_proba()
decision_function()
```

最终形成：

```text
直觉
+
数学
+
机制
+
代码
```

四层教学结构。

---

# 6. Practice Agent：主动练习

Practice Agent 的作用是避免学生陷入：

> “听懂了，但不会做。”

因此知识学习完成后必须进入 Active Recall。

例如学习完“过拟合”后，可以立即给出：

```text
训练集准确率：99%
验证集准确率：72%

问题：
1. 模型目前可能存在什么问题？
2. 为什么会出现这种现象？
3. 至少给出三种解决方式。
```

也可以设置：

### 原理题

```text
为什么 L2 Regularization 能缓解过拟合？
```

### 代码题

```python
model = DecisionTreeClassifier(
    max_depth=None
)
```

问题：

> 哪个设置可能导致模型复杂度过高？

### 场景题

给出一个实际机器学习问题，让学生选择模型并说明原因。

---

# 7. Evaluation Agent：学习验收

Evaluation Agent 负责判断学生是否真正掌握一个知识点。

不建议只使用选择题。

可以采用五维评价体系：

| 能力 | 测试方式 |
|---|---|
| 概念理解 | 定义解释 |
| 数学理解 | 公式与推导 |
| 原理理解 | Why Question |
| 工程能力 | Coding |
| 应用能力 | Case Study |

例如 K-Means：

### Level 1

K-Means 的优化目标是什么？

### Level 2

为什么 K-Means 对初始聚类中心敏感？

### Level 3

为什么 K-Means 不适合明显非球形的数据分布？

### Level 4

面对簇大小严重不均衡的数据时，是否推荐 K-Means？

### Level 5

解释下面参数的作用：

```python
KMeans(
    n_clusters=5,
    init="k-means++",
    n_init="auto"
)
```

---

# 8. Diagnosis Agent：错误诊断

Diagnosis Agent 不只是告诉学生：

> 回答错误。

它还需要分析：

> 为什么错误。

例如学生认为：

> Softmax 就是把每个 logit 除以所有 logit 的和。

系统应该识别为：

```text
错误类型：
Softmax 概率归一化机制理解错误
```

然后将相关知识节点标记为薄弱。

例如：

```text
Linear Regression        ██████████ 92
Gradient Descent         ████████░░ 81
Logistic Regression      ███████░░░ 72
Cross Entropy            █████░░░░░ 55
SVM                      未学习
```

系统之后会自动减少已经掌握知识的学习时间，把更多时间分配给：

```text
Cross Entropy
```

---

# 9. Learning Memory：学习者模型

系统需要长期维护学生的知识状态。

例如：

```json
{
  "gradient_descent": {
    "mastery": 0.86,
    "error_count": 2,
    "last_review": "2026-08-28"
  },
  "cross_entropy": {
    "mastery": 0.61,
    "error_count": 7,
    "last_review": "2026-08-30"
  }
}
```

每个知识点可以维护：

```text
Knowledge State =
Mastery
+ Confidence
+ Error History
+ Review History
+ Practice History
```

这样 Agent 才能真正知道：

> 学生会什么，不会什么。

---

# 10. 高效学习机制：Mastery Learning

系统第一次使用时可以进行 Diagnostic Test。

例如机器学习共有 100 个核心知识点。

诊断结果：

```text
已经掌握：42
基本掌握：23
薄弱：20
未学习：15
```

那么系统不需要让学生重新完整学习所有知识。

而是重点学习：

```text
薄弱 20
+
未学习 15
```

即：

```text
35 个重点知识点
```

已经掌握的内容只做周期复习。

这样实现：

```text
减少重复学习
↓
集中解决薄弱点
↓
提高单位时间学习效率
```

---

# 11. 监督机制：Learning Contract

学生可以首先设定学习目标。

例如：

```text
目标：
30 天完成机器学习基础

每天学习时间：
2 小时

每日任务：
3 个知识点
20 道题
1 个 Coding
```

Supervisor Agent 每日生成完成情况：

```text
计划学习：120 min
实际学习：82 min

完成知识点：2 / 3
练习题：16 / 20
Coding：0 / 1

今日完成度：68%
```

然后明确指出未完成任务：

```text
今日未完成：

- SVM Coding
- Cross Validation 剩余 4 道练习

建议：
下一次学习开始前优先补齐欠缺任务。
```

---

# 12. Mastery Gate：阶段解锁机制

系统可以增加课程 Gate。

例如：

```text
Cross Entropy Mastery = 62%

要求：
>= 80%

状态：
未通过
```

那么：

```text
Neural Network
```

暂时不进入正式学习阶段。

只有完成：

```text
补充讲解
↓
专项练习
↓
重新测试
↓
达到掌握标准
```

后才进入下一阶段。

这样可以避免：

> 前面基础没掌握，就不断学习后面的内容。

---

# 13. 复习机制

系统可以结合遗忘规律自动安排复习。

例如：

```text
Day 0   学习
Day 1   Review
Day 3   Review
Day 7   Review
Day 14  Review
Day 30  Review
```

但更推荐动态复习。

复习优先级可以考虑：

```text
Review Priority =
薄弱程度
×
遗忘程度
×
知识重要性
```

例如：

```text
Cross Entropy
Mastery：60%
核心知识
10 天未复习
```

则系统自动提高其 Review Priority。

---

# 14. 每日学习闭环

整个 Agent 每天可以按照以下流程运行：

```text
进入系统
 ↓
Supervisor 查看历史学习状态
 ↓
5 分钟 Retrieval Quiz
 ↓
更新 Knowledge State
 ↓
Planning Agent 生成今日任务
 ↓
Teaching Agent 教学
 ↓
Practice Agent 主动练习
 ↓
Diagnosis Agent 分析错误
 ↓
针对薄弱点补救学习
 ↓
Evaluation Agent 进行验收
 ↓
Mastery Gate 判断是否通过
 ↓
生成今日学习报告
 ↓
安排后续复习任务
```

核心闭环：

```text
Learn
↓
Practice
↓
Test
↓
Diagnose
↓
Review
```

---

# 15. 机器学习课程路径

## 第一阶段：机器学习基础

```text
机器学习基本概念
监督学习 / 无监督学习
Train / Validation / Test
Bias / Variance
Overfitting / Underfitting
```

---

## 第二阶段：基础模型与训练

```text
Linear Regression
Loss Function
Gradient Descent
Regularization
```

---

## 第三阶段：分类

```text
Logistic Regression
Sigmoid
Cross Entropy
Classification Metrics
```

---

## 第四阶段：经典机器学习模型

```text
KNN
Decision Tree
Random Forest
GBDT
XGBoost
```

---

## 第五阶段：SVM

```text
SVM
Margin
Kernel
Soft Margin
```

---

## 第六阶段：无监督学习

```text
K-Means
PCA
Clustering Evaluation
```

---

## 第七阶段：机器学习工程

```text
Cross Validation
Feature Engineering
Hyperparameter Tuning
Data Leakage
Imbalanced Learning
```

---

# 16. 最终综合任务

完成课程后可以设置：

## Machine Learning Final Challenge

提供一个真实数据集，让学生独立完成：

```text
数据分析
↓
数据预处理
↓
模型选择
↓
模型训练
↓
指标评价
↓
错误分析
↓
模型优化
↓
最终汇报
```

最终验证学生是否具备完整的机器学习问题解决能力。

---

# 17. 推荐 MVP

第一版不建议直接开发过多 Agent。

推荐只实现四个核心模块：

```text
Supervisor
+
Tutor
+
Evaluator
+
Student Model
```

重点实现：

1. 学生知识状态追踪
2. 自适应学习路径
3. 知识讲解
4. 自动练习
5. 学习测试
6. Mastery Gate
7. 每日任务监督
8. 学习报告

后续再逐步增加：

```text
Practice Agent
Diagnosis Agent
Coding Agent
Knowledge Graph
题库系统
遗忘曲线
学习数据分析 Dashboard
```

---

# 18. 项目核心价值

ML-Coach 不只是：

> AI 给学生讲机器学习。

而是：

> AI 持续检测学生当前能力，自动规划最值得学习的内容，通过练习和测试验证掌握程度，发现薄弱点后自动补救，并监督学习计划持续执行。

最终形成：

$$
\boxed{
自适应教学
+
主动练习
+
能力诊断
+
学习监督
+
动态复习
}
$$

这五个模块共同构成一个高效、可监督、可持续的机器学习 AI 教育 Agent。
