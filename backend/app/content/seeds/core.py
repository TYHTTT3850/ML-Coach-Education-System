"""Course Content & Assessment MVP, version 1.

All material is explicitly project-authored teaching material, not recovered
textbook content and not a claim of human review. Numeric answers are checked
using a finite catalogue of trusted arithmetic functions; conceptual choices
are cross-checked against the separately authored CHOICE_CATALOGUE. No eval,
student-code execution, model call, database write, or external citation occurs.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


ORIGIN = "instructor_authored"
AUTHOR_LABEL = "项目自编教学示例"


def calculate(kind: str, a: list[Any]) -> Any:
    """Only fixed, trusted calculations used by these seed fixtures."""
    if kind == "linear":
        return a[0] * a[1] + a[2]
    if kind == "sum":
        return sum(a)
    if kind == "length":
        return len(a[0])
    if kind == "at":
        return a[0][a[1]]
    if kind == "count_nonzero":
        return sum(x != 0 for x in a[0])
    if kind == "count_equal":
        return sum(x == a[1] for x in a[0])
    if kind == "count_at_least":
        return sum(x >= a[1] for x in a[0])
    if kind == "max_abs":
        return max(abs(x) for x in a[0])
    if kind == "minimum":
        return min(a[0])
    if kind == "count_outside_unit_interval":
        return sum(not 0 <= x <= 1 for x in a[0])
    if kind == "second_moment":
        return a[0] ** 2 + a[1]
    if kind == "reciprocal_quadratic_derivative":
        return -2 * a[0] / (1 + a[0] ** 2) ** 2
    if kind == "product":
        return math.prod(a)
    if kind == "ratio":
        return a[0] / a[1]
    if kind == "power":
        return a[0] ** a[1]
    if kind == "log":
        return math.log(a[0]) if len(a) == 1 else math.log(a[0], a[1])
    if kind == "exp":
        return math.exp(a[0])
    if kind == "dot":
        return sum(x * y for x, y in zip(a[0], a[1], strict=True))
    if kind == "norm":
        return math.sqrt(sum(x * x for x in a))
    if kind == "vector_add":
        return [x + y for x, y in zip(a[0], a[1], strict=True)]
    if kind == "matrix_product":
        return [[sum(x * y for x, y in zip(row, col, strict=True))
                 for col in zip(*a[1], strict=True)] for row in a[0]]
    if kind == "mean":
        return sum(a) / len(a)
    if kind == "variance":
        mu = sum(a) / len(a)
        return sum((x - mu) ** 2 for x in a) / len(a)
    if kind == "expectation":
        return sum(x * p for x, p in zip(a[0], a[1], strict=True))
    if kind == "discrete_variance":
        mu = calculate("expectation", a)
        return sum((x - mu) ** 2 * p for x, p in zip(a[0], a[1], strict=True))
    if kind == "bernoulli_variance":
        return a[0] * (1 - a[0])
    if kind == "binomial":
        n, k, p = a
        return math.comb(n, k) * p ** k * (1 - p) ** (n - k)
    if kind == "power_derivative":
        coefficient, power, x = a
        return coefficient * power * x ** (power - 1)
    if kind == "polynomial_derivative":
        coefficients, x = a
        return sum(i * c * x ** (i - 1) for i, c in enumerate(coefficients) if i)
    if kind == "chain_power":
        outer_power, inner_slope, inner_offset, x = a
        return outer_power * (inner_slope * x + inner_offset) ** (outer_power - 1) * inner_slope
    if kind == "quadratic_gradient":
        cx, cy, x, y = a
        return [2 * cx * x, 2 * cy * y]
    if kind == "cross_partial":
        x, y = a
        return [2 * x * y + 1, x * x]
    if kind == "sigmoid":
        return 1 / (1 + math.exp(-a[0]))
    if kind == "sigmoid_derivative":
        p = calculate("sigmoid", a)
        return p * (1 - p)
    if kind == "cross_entropy":
        y, p = a
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))
    if kind == "binary_entropy":
        p = a[0]
        return -(p * math.log(p) + (1 - p) * math.log(1 - p))
    if kind == "linear_loss":
        xs, ys, w, b = a
        return sum((w * x + b - y) ** 2 for x, y in zip(xs, ys, strict=True)) / (2 * len(xs))
    if kind == "linear_gradients":
        xs, ys, w, b = a
        residuals = [w * x + b - y for x, y in zip(xs, ys, strict=True)]
        return [sum(r * x for r, x in zip(residuals, xs, strict=True)) / len(xs), sum(residuals) / len(xs)]
    if kind == "gradient_step":
        return a[0] - a[1] * a[2]
    if kind == "finite_difference_square":
        x, h = a
        return ((x + h) ** 2 - x * x) / h
    if kind == "quadratic_value":
        cx, cy, x, y = a
        return cx * x * x + cy * y * y
    if kind == "component":
        operation, arguments, index = a
        return calculate(operation, arguments)[index]
    if kind == "squared_distance":
        return sum((x - y) ** 2 for x, y in zip(a[0], a[1], strict=True))
    if kind == "log_odds":
        return math.log(a[0] / (1 - a[0]))
    if kind == "odds":
        return a[0] / (1 - a[0])
    raise ValueError(f"Unregistered trusted calculation: {kind}")


def N(family: str, stem: str, answer: float, kind: str, args: list[Any],
      solution: str, dimension: str = "calculation", difficulty: str = "medium") -> dict:
    return dict(family=family, stem=stem, answer=answer, kind=kind, args=args,
                solution=solution, dimension=dimension, difficulty=difficulty, type="numeric")


def M(family: str, stem: str, options: list[str], answer: str, explanation: str,
      dimension: str = "concept", difficulty: str = "medium") -> dict:
    return dict(family=family, stem=stem, options=options, answer=answer,
                solution=explanation, dimension=dimension, difficulty=difficulty,
                type="multiple_choice")


def E(name: str, family: str, title: str, problem: str, givens: dict,
      target: str, steps: list[tuple[str, str, str, str, str]],
      answer: str, checks: list[tuple[str, list[Any], Any]], mistakes: list[str]) -> dict:
    return dict(name=name, family=family, title=title, problem=problem,
                givens=givens, target=target, steps=steps, answer=answer,
                checks=checks, mistakes=mistakes)


def T(title: str, objectives: list[str], prerequisites: str, problem: str,
      concepts: list[str], formulas: list[tuple[str, str, str]], summary: str,
      examples: list[dict], practice: list[dict], checkpoint: list[dict],
      prerequisite_test: list[dict], subjective: tuple[str, str, str]) -> dict:
    return dict(title=title, objectives=objectives, prerequisites=prerequisites,
                problem=problem, concepts=concepts, formulas=formulas, summary=summary,
                examples=examples, practice=practice, checkpoint=checkpoint,
                prerequisite_test=prerequisite_test, subjective=subjective)


TOPICS: dict[str, dict] = {
    "la.vectors": T(
        "标量、向量与矩阵：把样本写成有形状的数据",
        ["将两个特征按固定顺序组成向量", "区分样本数、特征数和向量分量", "完成向量加法并检查形状"],
        "只需实数四则运算；特征必须采用一致顺序和单位。",
        "两位学生的学习时长和练习数分别是 (2, 5)、(3, 4)，怎样保存它们并计算合计？",
        ["标量是一个数；向量是按顺序排列的数，分量顺序有含义。", "本课默认样本向量为列向量，两个特征的形状为 2×1。", "数据矩阵按行放样本：n 个样本、d 个特征形成 n×d 的 X。"],
        [("u+v=(u_1+v_1,u_2+v_2)", "u、v 是同维向量；下标标识同一种特征", "只有同维且对应特征单位一致时可相加"),
         ("X\\in\\mathbb R^{n\\times d}", "n 为样本数，d 为特征数", "每个样本具有相同特征定义与顺序")],
        "先写出形状和分量含义，再做逐分量运算；向量长度是分量个数，不是分量之和。",
        [E("feature_addition", "vector.component-addition", "两个特征向量相加", "u=(2,5)，v=(3,4)，求 u+v。", {"u":[2,5],"v":[3,4]}, "合计向量及其形状",
           [("对齐含义","u,v∈R²","时长对应时长，练习数对应练习数","形状均为 2×1","向量相加要求对应分量有一致含义。"),
            ("逐项相加","u+v=(u₁+v₁,u₂+v₂)","(2+3,5+4)","(5,9)","不能把两个向量所有分量合并成一个标量。"),
            ("解释与校验","(u+v)-v=u","(5-3,9-4)","(2,5)","用逆运算恢复原向量检验结果。")],
           "u+v=(5,9)，形状 2×1。", [("vector_add",[[2,5],[3,4]],[5,9])], ["把向量维数误认为分量的和", "交换特征顺序后直接相加"]),
         E("scalar_change", "vector.scalar-scaling", "单位换算是整体缩放", "位移向量 v=(0.5,-1.2) 米，换算成厘米。", {"v_m":[0.5,-1.2],"cm_per_m":100}, "厘米表示的向量",
           [("确定换算因子","1 m=100 cm","两个分量均乘 100","缩放系数为 100","同一单位换算应用到每一个分量。"),
            ("分别计算","100v=(100v₁,100v₂)","(100×0.5,100×(-1.2))","(50,-120)","负号表示方向，换单位不改变方向。"),
            ("检验量纲","v_cm/100=v_m","(50/100,-120/100)","(0.5,-1.2) m","换回米后与原数据一致。")],
           "(50,-120) cm。", [("product",[100,0.5],50),("product",[100,-1.2],-120)], ["只缩放第一个分量", "将负方向误改为正方向"])],
        [N("vector.component-addition","u=(1,4)，v=(3,-2)，u+v 的第二分量是多少？",2,"sum",[4,-2],"对应第二个分量相加：4+(-2)=2。"),
         N("vector.scalar-scaling","v=(2,-3)，3v 的第一分量是多少？",6,"product",[3,2],"标量乘法逐分量进行，3×2=6。"),
         M("vector.feature-order","同一个样本从 (身高,体重) 改为 (体重,身高)，怎样使用原线性模型？",["直接使用，顺序无关","同时调整模型权重的对应顺序","只改变向量长度"],"B","每个权重必须继续对应原来的特征。"),
         M("vector.shape-description","12 个样本，每个有 3 个特征，按行放样本的矩阵形状是？",["12×3","3×12","12×12"],"A","样本数决定行数，特征数决定列数。","application")],
        [N("vector.centroid-as-representation","三个二维点为 (0,0)、(3,6)、(6,3)。用各坐标均值表示中心，中心第一坐标是多少？",3,"mean",[0,3,6],"中心第一坐标为 (0+3+6)/3=3。"),
         N("vector.feature-index-code","固定 Python 列表 x=[8,3,5]，执行 len(x) 得到多少？",3,"length",[[8,3,5]],"len 统计三个元素，而不是计算元素和。","concept"),
         M("vector.semantic-compatibility","向量 (身高厘米,年龄岁) 与 (体重公斤,温度摄氏度) 同为二维，能直接相加作为有意义的样本特征吗？",["能，因为形状相同","不能，特征语义和单位不同","只有分量为正才可以"],"B","同形状只是数学条件；应用中还要保证对应量的语义一致。","application"),
         M("vector.transpose-orientation","列向量 x 的形状是 4×1，xᵀ 的形状是？",["4×4","4×1","1×4"],"C","转置交换行数和列数。"),
         M("vector.zero-representation","二维零向量的正确描述是？",["没有任何维度","有两个分量且每个分量为 0","只有一个分量 0"],"B","零描述分量值，不改变向量所处空间的维数。")],
        [N("vector.coordinate-selection","样本 x=(7,2,9)，第三个特征的值是多少？",9,"at",[[7,2,9],2],"按题目约定的一基序号选择第三个分量 9。"),
         M("vector.same-dimension-rule","u∈R²，v∈R³，通常能直接求 u+v 吗？",["不能，分量无法一一对应","可以，缺少项当 1","可以，只取前两项"],"A","常规定义下向量相加要求同维。","theorem_condition"),
         M("vector.scalar-vector-distinction","下列哪个对象是包含两个特征的向量？",["单个温度 20","有序对 (20,60)","符号 x 不带定义"],"B","有序对保留两个分量的顺序。")],
        ("concept_explanation","说明为什么‘3 个分量的向量’不等于‘长度为 3 的位移’，举一个反例。","区分维数与欧氏范数；例如 (0,0,0) 有 3 个分量但范数为 0。")),
    "la.dot_product": T(
        "向量内积与线性模型分数",
        ["逐项相乘并求和计算内积", "区分内积标量与逐分量乘积", "将内积、偏置和 Sigmoid 串成一次预测"],
        "需要向量分量与固定特征顺序；概率解释是本例的模型约定。",
        "特征 x=(2,1)，权重 w=(0.5,-1)，偏置 b=0.5，模型怎样得到一个预测概率？",
        ["内积将两个同维向量映射成一个标量。", "线性得分 z=wᵀx+b 可以是任意实数，不自动是概率。", "在二分类模型中，Sigmoid 将得分映射到 (0,1)，供指定正类的条件概率解释。"],
        [("w^Tx=\\sum_{j=1}^d w_jx_j", "w、x 形状 d×1，wᵀ 为 1×d，结果 1×1", "两向量同维且特征顺序一致"),
         ("p=1/(1+e^{-z}),\\quad z=w^Tx+b", "b 是标量偏置，p 是正类概率预测", "这里只演示固定模型的前向计算，不表示已校准")],
        "先乘后加得到一个标量；加偏置得到得分，经过链接函数后才得到概率预测。",
        [E("C_dot_sigmoid", "prediction.dot-bias-sigmoid", "例题 C：从向量内积到概率", "x=(2,1)，w=(0.5,-1)，b=0.5，计算 z 和 sigmoid(z)。", {"x":[2,1],"w":[0.5,-1],"b":0.5}, "线性得分与二分类概率",
           [("逐项相乘","wᵀx=w₁x₁+w₂x₂","0.5×2+(-1)×1","1-1=0","同一位置的权重乘对应特征。"),
            ("加上偏置","z=wᵀx+b","0+0.5","z=0.5","偏置在求和之后只加一次。"),
            ("计算非线性输出","σ(z)=1/(1+e⁻ᶻ)","1/(1+e⁻⁰·⁵)","p≈0.622459","指数函数将任意实数得分映射到正概率范围。"),
            ("解释模型输出","p=P(Y=1|x;w,b)","以 0.5 为分类阈值","本例预测正类","这是模型给定特征后的概率预测，不是样本真实标签的证明。")],
           "z=0.5，sigmoid(z)≈0.622459。", [("dot",[[2,1],[0.5,-1]],0),("linear",[1,0,0.5],0.5),("sigmoid",[0.5],0.6224593312018546)], ["遗漏负权重", "将 z=0.5 直接当作概率"]),
         E("orthogonal", "dot.orthogonality-calculation", "通过内积检查直角", "u=(1,2)，v=(2,-1)，用内积判断是否正交。", {"u":[1,2],"v":[2,-1]}, "内积及正交判断",
           [("检查形状","u,v∈R²","两个分量分别对应","可以计算内积","同维是内积定义的前提。"),
            ("求内积","uᵀv=1×2+2×(-1)","2-2","0","正负乘积在求和时相互抵消。"),
            ("应用正交条件","uᵀv=0","u、v 均非零","两向量夹角为 90°","欧氏内积下非零向量内积为零意味着正交。")],
           "uᵀv=0，两非零向量正交。", [("dot",[[1,2],[2,-1]],0)], ["把逐分量乘积 (2,-2) 当最终内积", "忽略欧氏内积和非零向量条件"])],
        [N("dot.orthogonality-calculation","u=(2,-1,3)，v=(1,4,2)，求 uᵀv。",4,"dot",[[2,-1,3],[1,4,2]],"2×1+(-1)×4+3×2=4。"),
         N("prediction.dot-bias-sigmoid","x=(1,2)，w=(2,-0.5)，b=1，求线性得分 z（不做 Sigmoid）。",2,"sum",[2,-1,1],"内积 2×1-0.5×2=1，加偏置后 z=2。","application"),
         M("dot.output-shape","两个 5×1 列向量 u、v，uᵀv 的形状是？",["5×5","1×1","5×1"],"B","(1×5)(5×1) 得到标量。"),
         M("dot.orthogonality-calculation","在通常的欧氏内积下，两个非零向量内积为 0 意味着？",["相互正交","它们相等","一个必为零"],"A","非零向量夹角为直角时余弦为零。")],
        [N("dot.distributive-inference","已知 aᵀb=3，aᵀc=-1，求 aᵀ(2b+c)。",5,"sum",[6,-1],"利用线性性，2(aᵀb)+aᵀc=6-1=5。"),
         N("dot.missing-weight","模型 z=2w₁+3w₂，已知 z=7、w₂=1，求 w₁。",2,"ratio",[4,2],"2w₁+3=7，所以 w₁=(7-3)/2=2。","application"),
         M("dot.joint-permutation","同时用同一个排列重排 x 和 w 的分量，内积如何变化？",["不变","变成相反数","必须重新训练才可计算"],"A","配对乘积不变，仅求和顺序改变。"),
         M("dot.zero-inner-not-equal","aᵀb=0 能推出 a=b 吗？",["总能","不能，例如两个不同坐标轴方向","仅在二维时能"],"B","正交不等于相等。","theorem_condition"),
         M("dot.model-contribution","线性得分中某特征 xⱼ 增大 Δ，其他不变且 wⱼ<0，得分变化方向是？",["增大","不变","减小"],"C","变化为 wⱼΔ，Δ>0 时为负。","application")],
        [N("dot.self-product","u=(3,4)，求 uᵀu。",25,"dot",[[3,4],[3,4]],"平方和为 9+16=25。"),
         M("dot.dimension-condition","x 有 4 个分量，w 有 3 个分量，wᵀx 是否按通常定义成立？",["不成立","成立且等于 0","只舍弃最后一项即可"],"A","特征数必须与权重数相等。","theorem_condition"),
         M("dot.operation-order","求内积应如何操作？",["先求两向量各自和再相乘","对应分量相乘再求和","对应分量相加再求积"],"B","对应乘积求和保留权重与特征的配对。")],
        ("error_correction","某同学把 (1,2)·(3,4) 写成 (1+2)(3+4)=21，请指出错误并更正。","应为 1×3+2×4=11；先求和再相乘多出交叉项。")),
    "math.functions": T(
        "函数与变量：从输入得到唯一输出",
        ["区分自变量、参数和输出", "代入一次函数完成预测", "根据运算限制确定定义域"],
        "使用实数算术；定义域、值域和单位在具体问题中明确。",
        "出租车费用模型 f(x)=3x+10，x 表示公里数；行驶 4 公里时如何计算费用？",
        ["函数把定义域中的每个输入映射到唯一输出，不要求不同输入有不同输出。", "x 是本次输入；3 与 10 是固定参数，换模型时参数才改变。", "分式的分母不能为零；实数平方根内部须非负；应用定义域还可能要求 x≥0。"],
        [("f(x)=ax+b", "x 为输入，a 为每单位变化量，b 为 x=0 时的输出", "a、b 固定，x 在约定定义域内"),
         ("(f\\circ g)(x)=f(g(x))", "先计算内层 g，再把结果作为 f 的输入", "g(x) 必须位于 f 的定义域")],
        "明确哪些量固定、哪些量变化；代入前先检查定义域，再检查输出的单位。",
        [E("fare", "function.affine-evaluation", "计算费用函数", "f(x)=3x+10 元，x≥0，求行驶 4 公里的费用。", {"a":3,"b":10,"x":4}, "费用 f(4)",
           [("确认输入","x=4 km","4≥0","输入在定义域内","实际里程不能为负。"),
            ("代入函数","f(4)=3×4+10","12+10","22 元","先乘法后加固定起步费。"),
            ("解释参数","f(0)=10","3 元/km 与 4 km 相乘","里程费 12 元，起步费 10 元","量纲合并后均为元。")],
           "f(4)=22 元。", [("linear",[3,4,10],22)], ["把起步费乘以里程", "把参数 a 当成输入 x"]),
         E("composition", "function.composition-evaluation", "顺序执行两个函数", "g(x)=x+1，f(t)=t²，求 f(g(2))。", {"x":2,"g":"x+1","f":"t²"}, "复合函数输出",
           [("先算内层","g(2)=2+1","2+1","3","括号表示 g 的结果先产生。"),
            ("再算外层","f(g(2))=f(3)","3²","9","把内层输出整体作为外层输入。"),
            ("对照逆序","g(f(2))=g(4)","4+1","5，与 9 不同","函数复合通常不满足交换律。")],
           "f(g(2))=9；逆序 g(f(2))=5。", [("power",[3,2],9),("linear",[1,4,1],5)], ["把 f(g(x)) 当成 f(x)g(x)", "交换函数执行顺序"])],
        [N("function.affine-evaluation","f(x)=2x-1，求 f(3)。",5,"linear",[2,3,-1],"代入 x=3，2×3-1=5。"),
         N("function.composition-evaluation","g(x)=2x，f(t)=t+3，求 f(g(4))。",11,"linear",[2,4,3],"先 g(4)=8，再 f(8)=11。"),
         M("function.input-parameter","在固定模型 y=wx+b 中预测一个新样本时，通常哪个量是样本输入？",["x","w","b"],"A","w、b 是已确定参数，x 随样本变化。"),
         M("function.reciprocal-domain","实数函数 f(x)=1/(x-2) 必须排除哪个输入？",["0","2","-2"],"B","x=2 会使分母为零。","theorem_condition")],
        [N("function.inverse-input","费用 y=4x+6，已知费用为 30，求里程 x。",6,"ratio",[24,4],"解 4x=30-6=24，得 x=6。","application"),
         N("function.constant-output","常数函数 f(x)=7，求 f(100)-f(-3)。",0,"sum",[7,-7],"两个输入都映射为 7，差为 0。"),
         M("function.uniqueness","哪种关系不符合单值函数定义？",["两个不同输入输出相同","同一输入对应两个不同输出","所有输入输出都为 0"],"B","函数要求每一个合法输入只有一个输出。"),
         M("function.square-root-domain","实数函数 f(x)=√(x-1) 的定义域是？",["x≥1","x≠1","所有实数"],"A","被开方数 x-1 需非负。","theorem_condition"),
         M("function.noninjective","f(x)=x² 在实数上，f(2)=f(-2)，说明什么？",["它不是函数","它的值域是全部实数","函数不一定一一对应"],"C","每个输入仍只有一个输出，只是两个输入共享输出。")],
        [N("function.table-read","函数表给出 f(0)=2、f(1)=5、f(2)=8。只根据表，f(1) 是多少？",5,"sum",[5],"读取输入 1 对应的输出 5，无需猜测表外函数。"),
         M("function.fixed-variable","f(x;a)=ax² 中研究 x 的变化时，‘a 固定’意味着？",["a 在本次讨论中当作参数","a 必须为 0","x 也必须固定"],"A","固定参数不等于参数取零。"),
         M("function.allowed-input","计算一个函数值前最先需要确认什么？",["结果一定为正","输入属于函数定义域","函数一定可逆"],"B","非法输入可能使表达式无意义。","theorem_condition")],
        ("concept_explanation","给出一个不同输入对应相同输出的函数，并解释为什么仍符合函数定义。","例如 f(x)=x²，f(1)=f(-1)=1；每一个输入仍有唯一输出。")),
}


# Independent concept-key catalogue: order is the authored multiple-choice
# order within each usage, excluding numeric items. It is deliberately not
# inferred from question answer_key fields during verification.
CHOICE_CATALOGUE = {
    "la.vectors": {"practice":"BA", "checkpoint":"BCB", "prerequisite_test":"AB"},
    "la.dot_product": {"practice":"BA", "checkpoint":"ABC", "prerequisite_test":"AB"},
    "math.functions": {"practice":"AB", "checkpoint":"BAC", "prerequisite_test":"AB"},
}

from .machine_learning import authored as _machine_learning
from .calculus import authored as _calculus
from .probability import authored as _probability
from .algebra import authored as _algebra

_topics, _choices = _machine_learning(T, N, M, E)
TOPICS.update(_topics)
CHOICE_CATALOGUE.update(_choices)

# Keep similar examples/practice in the same conservative family. These
# aliases are data corrections before the first published v1 seed, not a
# migration of existing student records or a way to label variants independent.
_family_corrections = {
    "optimization.generic-update":"optimization.quadratic-one-step",
    "crossentropy.negative-label-evaluation":"crossentropy.batch-average",
    "partial.quadratic-coordinate":"calculus.quadratic-partial-gradient",
    "gradient.coordinate-extraction":"calculus.quadratic-partial-gradient",
    "gradient.axis-direction":"calculus.quadratic-partial-gradient",
    "moments.population-variance":"moments.affine-transform",
    "bernoulli.label-zero-probability":"bernoulli.single-moments",
}

# Additional modules below are merged before applying the shared aliases.
_topics, _choices = _calculus(T, N, M, E)
TOPICS.update(_topics)
CHOICE_CATALOGUE.update(_choices)
_topics, _choices = _probability(T, N, M, E)
TOPICS.update(_topics)
CHOICE_CATALOGUE.update(_choices)
_topics, _choices = _algebra(T, N, M, E)
TOPICS.update(_topics)
CHOICE_CATALOGUE.update(_choices)

for _spec in TOPICS.values():
    for _question_spec in _spec["practice"]:
        _question_spec["family"] = _family_corrections.get(_question_spec["family"], _question_spec["family"])

from .independent_practice import authored as _independent_practice

_additions, _additional_choices = _independent_practice(N, M)
for _topic_id, _additional_questions in _additions.items():
    TOPICS[_topic_id]["practice"].extend(_additional_questions)
    CHOICE_CATALOGUE[_topic_id]["practice"] += _additional_choices[_topic_id]

from .independent_prerequisites import authored as _independent_prerequisites

_prerequisite_aliases, _prerequisite_additions = _independent_prerequisites(N)
for _spec in TOPICS.values():
    for _question_spec in _spec["prerequisite_test"]:
        _question_spec["family"] = _prerequisite_aliases.get(_question_spec["family"], _question_spec["family"])
for _topic_id, _question_spec in _prerequisite_additions.items():
    TOPICS[_topic_id]["prerequisite_test"].append(_question_spec)


def owner_course(topic_id: str) -> str:
    return {"ml":"machine_learning", "math":"advanced_mathematics",
            "la":"linear_algebra", "prob":"probability_statistics"}[topic_id.split(".")[0]]


def _question(topic_id: str, usage: str, position: int, spec: dict) -> dict:
    question_id = f"bank.{topic_id}.{usage}.{position:02d}"
    dimension = spec["dimension"]
    if topic_id.startswith("ml."):
        dimension = {"calculation":"mathematics", "theorem_condition":"mechanism"}.get(dimension, dimension)
    result = dict(id=question_id, version=1, topic_ids=[topic_id],
                  primary_course=owner_course(topic_id), title=spec["stem"].split("。")[0][:90],
                  question_type=spec["type"], difficulty=spec["difficulty"], stem=spec["stem"],
                  options=None, answer_key={"canonical_answer":str(spec["answer"]),
                    "key_points":[spec["solution"]], "accepted_variants":[]},
                  accepted_variants=[], solution_steps=[part for part in spec["solution"].split("；") if part],
                  rubric=[{"criterion":"答案与指定计算或概念条件一致", "points":100}],
                  tested_competencies=[topic_id+".foundation"], dimensions={dimension:1.0},
                  estimated_minutes=3 if spec["type"] == "numeric" else 2,
                  usage=[usage], family_id=spec["family"], source_refs=[], source_snapshot=[],
                  content_origin=ORIGIN,
                  validation_status="deterministic_checks_passed", publication_status="published")
    if spec["type"] == "numeric":
        result["answer_key"].update(numeric_answer=spec["answer"], tolerance=1e-6)
        result["verification"] = {"kind":"trusted_numeric_fixture", "operation":spec["kind"],
                                  "arguments":deepcopy(spec["args"]), "expected":spec["answer"]}
    else:
        result["options"] = [{"option_id":chr(65+i),"text":s} for i,s in enumerate(spec["options"])]
        result["answer_key"]["correct_option"] = spec["answer"]
        result["verification"] = {"kind":"independent_concept_catalogue", "catalogue_key":question_id,
                                  "expected":spec["answer"]}
    return result


def build_content() -> dict[str, list[dict]]:
    """Return fresh content objects; repeated calls have identical output."""
    lessons, examples, questions = [], [], []
    for topic_id, spec in TOPICS.items():
        example_ids = [f"example.{topic_id}.{e['name']}" for e in spec["examples"]]
        practice_ids = [f"bank.{topic_id}.practice.{i:02d}" for i in range(1, len(spec["practice"])+1)]
        lessons.append(dict(id="lesson."+topic_id, topic_id=topic_id, title=spec["title"],
            learning_objectives=deepcopy(spec["objectives"]), prerequisite_summary=spec["prerequisites"],
            motivating_problem=spec["problem"], key_concepts=deepcopy(spec["concepts"]),
            formulas=[{"expression":f,"symbols":s,"conditions":c} for f,s,c in spec["formulas"]],
            worked_example_ids=example_ids, practice_recommendations=practice_ids,
            summary=spec["summary"], source_refs=[], content_origin=ORIGIN,
            version=1, status="published"))
        for example_id, e in zip(example_ids, spec["examples"], strict=True):
            steps = [{"step_id":f"step-{i}","title":title,"formula":formula,
                      "substitution":substitution,"result":result,"explanation":explanation,
                      "justification":explanation}
                     for i,(title,formula,substitution,result,explanation) in enumerate(e["steps"],1)]
            examples.append(dict(id=example_id, topic_id=topic_id, title=e["title"], difficulty="medium",
                problem=e["problem"], givens=deepcopy(e["givens"]), target=e["target"],
                solution_outline=" → ".join(s["title"] for s in steps), steps=steps,
                final_answer=e["answer"], verification={"kind":"trusted_numeric_fixture",
                    "checks":[{"operation":k,"arguments":deepcopy(a),"expected":v} for k,a,v in e["checks"]]},
                common_mistakes=deepcopy(e["mistakes"]), followup_practice_ids=practice_ids[:2],
                source_refs=[], content_origin=ORIGIN, version=1,
                status="published", family_id=e["family"]))
        for usage in ("practice", "checkpoint", "prerequisite_test"):
            for i,q in enumerate(spec[usage],1):
                questions.append(_question(topic_id,usage,i,q))
        qtype, stem, solution = spec["subjective"]
        questions.append(dict(id=f"bank.{topic_id}.subjective.01",version=1,topic_ids=[topic_id],
            primary_course=owner_course(topic_id),title="待审核开放练习："+spec["title"],
            question_type=qtype,difficulty="medium",stem=stem,options=None,
            answer_key={"canonical_answer":solution,"key_points":[solution],"accepted_variants":[]},
            accepted_variants=[],solution_steps=[solution],rubric=[{"criterion":"概念准确且论证完整", "points":100}],
            tested_competencies=[topic_id+".foundation"],dimensions={"concept":1.0},estimated_minutes=5,
            usage=["practice"],family_id=f"{topic_id}.open-explanation",source_refs=[],source_snapshot=[],
            content_origin=ORIGIN,validation_status="needs_review",
            publication_status="draft",verification={"kind":"requires_content_review"}))
    return {"lessons":lessons,"examples":examples,"questions":questions}


def _close(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list,tuple)) and len(actual)==len(expected) and all(
            _close(x,y) for x,y in zip(actual,expected,strict=True))
    return isinstance(actual,(int,float)) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9)


def verify_content(content: dict | None = None) -> list[str]:
    """Check saved answers against trusted calculations and separate key data."""
    content = build_content() if content is None else content
    errors: list[str] = []
    expected_choices = {}
    for topic_id, spec in TOPICS.items():
        for usage in ("practice","checkpoint","prerequisite_test"):
            answers = iter(CHOICE_CATALOGUE[topic_id][usage])
            for position,q in enumerate(spec[usage],1):
                if q["type"]=="multiple_choice":
                    expected_choices[f"bank.{topic_id}.{usage}.{position:02d}"] = next(answers)
            if next(answers,None) is not None:
                errors.append(f"{topic_id}: unused conceptual catalogue answer")
    all_ids = set()
    for group in content.values():
        for item in group:
            if item["id"] in all_ids:
                errors.append(item["id"]+": duplicate ID")
            all_ids.add(item["id"])
            if item["source_refs"] or item["content_origin"]!=ORIGIN:
                errors.append(item["id"]+": project-authored provenance mismatch")
    for e in content["examples"]:
        for check in e["verification"]["checks"]:
            if not _close(calculate(check["operation"],check["arguments"]),check["expected"]):
                errors.append(e["id"]+": worked calculation mismatch")
        if len(e["steps"])<3 or any(not all(s.values()) for s in e["steps"]):
            errors.append(e["id"]+": missing intermediate teaching steps")
    for q in content["questions"]:
        if q["question_type"]=="numeric":
            v=q["verification"]
            if not _close(calculate(v["operation"],v["arguments"]),q["answer_key"]["numeric_answer"]):
                errors.append(q["id"]+": numeric answer mismatch")
        elif q["question_type"]=="multiple_choice":
            if q["answer_key"]["correct_option"]!=expected_choices.get(q["id"]):
                errors.append(q["id"]+": conceptual key mismatch")
            if len({o["text"] for o in q["options"]}) != len(q["options"]):
                errors.append(q["id"]+": duplicate options")
        elif q["publication_status"]!="draft" or q["validation_status"]!="needs_review":
            errors.append(q["id"]+": subjective content requires review")
        if sum(r["points"] for r in q["rubric"])!=100:
            errors.append(q["id"]+": rubric does not sum to 100")
    return errors
