"""Additional independent practice after studying the worked examples.

The original similar exercises remain available as assisted practice. These
items assess different tasks, not numerical variants of the worked examples,
and allow four unseen practice items before the original Mastery Gate.
"""


def authored(N, M):
    additions = {
        "la.vectors":[
            N("vector.sparse-feature-count","把非零分量称为活跃特征。x=(0,5,0,-2) 有多少个活跃特征？",2,"count_nonzero",[[0,5,0,-2]],"0 不算活跃，只有 5 与 -2 两项非零，因此有 2 个。"),
            N("vector.recover-sample-count","数据矩阵按行放样本，每个样本有 3 个特征，总共存放 15 个标量且没有缺项，共有多少个样本？",5,"ratio",[15,3],"元素总数等于样本数乘特征数，因此样本数为 15/3=5。","application")],
        "la.dot_product":[
            N("dot.normalized-similarity","两个非零向量满足 u·v=6、||u||=2、||v||=3。按 cosθ=(u·v)/(||u||||v||)，求 cosθ。",1,"ratio",[6,6],"分母为 2×3=6，所以归一化内积为 6/6=1。"),
            N("dot.recover-norm-from-self-product","已知 u·u=49。利用 ||u||=√(u·u)，求 u 的欧氏范数。",7,"power",[49,0.5],"范数非负，取平方根 √49=7。"),
            M("dot.real-symmetry","对实数同维向量，u·v 与 v·u 的关系是？",["相等","总是互为相反数","只有 u=v 时相等"],"A","每一项实数乘法满足交换律，求和后内积相同。")],
        "math.functions":[
            N("function.preimage-count","函数表为 f(0)=1、f(1)=1、f(2)=4。在列出的输入中，有多少个输入映射到输出 1？",2,"count_equal",[[1,1,4],1],"输入 0 和 1 都映射到 1，因此有两个；这不违反单值函数定义。"),
            M("function.parameter-intervention","对模型 f(x)=3x+b，固定输入 x，把参数 b 增加 2，输出如何变化？",["增加 6","增加 2","不变"],"B","固定 x 后，输出只增加偏置的变化量 2。","application")],
        "ml.gradient_descent":[
            N("optimization.relative-progress","一次更新使训练损失从 10 变为 8，按 (旧损失-新损失)/旧损失 定义相对降幅，求该比例（用小数）。",0.2,"ratio",[2,10],"绝对降幅为 2，相对降幅为 2/10=0.2，即 20%。","application"),
            M("optimization.nonnegative-loss-certificate","目标是若干实数残差平方的和，某参数处目标达到 0，能确定什么？",["这个目标的全局最小值已达到","所有未来测试误差为 0","所有参数必须为 0"],"A","平方和不可能为负，0 因而是此训练目标的全局下界；不推出泛化结论。","mechanism")],
        "ml.sigmoid":[
            N("sigmoid.probability-reporting","模型已经算出 Sigmoid 概率 p=0.625。用百分比报告时，百分号前的数值是多少？",62.5,"product",[0.625,100],"小数概率乘 100 得百分比数值 62.5；这是 62.5%，不是概率 62.5。","application"),
            M("sigmoid.algebraic-stability","把 1/(1+e^(-z)) 的分子分母同乘 e^z，得到哪个等价式？",["e^z/(1+e^z)","e^z/(1-e^z)","1+e^z"],"A","分母变成 e^z+1，分子为 e^z；当 z 很负时这种形式还可避免计算巨大的 e^(-z)。","mechanism")],
        "ml.logistic_regression":[
            N("logistic.batch-decision-count","三个样本概率依次为 [0.2,0.6,0.9]，固定规则为 p≥0.7 判为正类，这一批有多少个预测正类？",1,"count_at_least",[[0.2,0.6,0.9],0.7],"只有 0.9 达到 0.7，所以预测正类数量为 1。","application"),
            M("logistic.negated-model","若同时将全部权重 w 与偏置 b 变号，原概率为 p，新概率为？",["仍为 p","1-p","-p"],"B","新的线性得分为 -z，利用 σ(-z)=1-σ(z) 得到补概率。","mechanism")],
        "ml.cross_entropy":[
            N("crossentropy.relative-likelihood-gain","真实标签为 1，预测概率从 0.25 提高到 0.5。求旧损失减新损失，使用自然对数。",0.6931471805599453,"log",[2],"差值=-ln0.25+ln0.5=ln(0.5/0.25)=ln2≈0.693147。","mechanism"),
            M("crossentropy.log-base-rescaling","若把所有样本损失中的 ln 换成 log₂，会如何影响两个固定模型的平均损失排序？",["排序反转","必须由模型参数决定","排序不变，只乘相同正系数 1/ln2"],"C","换底公式对每个样本都乘相同正数，因此大小顺序保持。")],
        "math.exponential_logarithmic":[
            N("logarithm.product-combination","不分别求小数近似，计算 log₂8+log₂4。",5,"log",[32,2],"对数和等于乘积的对数，log₂(8×4)=log₂32=5。"),
            M("exponential.base-below-one","对 f(x)=(1/2)^x，当 x 增加 1 时，函数值怎样变化？",["乘以 2","乘以 1/2","加上 1/2"],"B","由指数加法律，f(x+1)=(1/2)f(x)，因此按固定比例减小。","application")],
        "math.derivative_definition":[
            N("derivative.compare-rate-magnitudes","两个测量点的导数分别为 -2 和 5。若只比较变化率的绝对大小，较大的绝对变化率是多少？",5,"max_abs",[[-2,5]],"先取绝对值 2 与 5；负号代表下降方向而不代表变化幅度更大。"),
            M("derivative.output-input-units","函数表示质量 m 随距离 x 变化，m 的单位是 kg，x 的单位是 m，dm/dx 的单位是什么？",["kg/m","kg·m","m/kg"],"A","变化率的单位是输出增量单位除以输入增量单位。","application")],
        "math.derivative_rules":[
            N("derivative.reciprocal-slope","f(x)=1/x，x≠0，求 f'(2)。",-0.25,"ratio",[-1,4],"商法则或负幂法则给出 f'=-1/x²，代入 x=2 得 -1/4。"),
            M("derivative.substitute-before-differentiating","求 f'(2) 时，先把表达式中的 x 全部换成 2 再对 x 求导，为什么通常不对？",["2 不能作为函数输入","代入后得到常数，丢掉了原函数在附近的变化信息","因为所有函数在 2 都不可导"],"B","应先得到导函数再代入指定点；单个函数值无法保留局部变化。","derivation")],
        "math.chain_rule":[
            N("chain.reciprocal-quadratic","f(x)=1/(1+x²)，用内层 u=1+x²、外层 1/u，求 f'(1)。",-0.5,"reciprocal_quadratic_derivative",[1],"外层对 u 导数为 -1/u²，内层导数为 2x；在 x=1 时 u=2，乘得 -2/4=-0.5。"),
            M("chain.units-cancellation","若 dy/du 单位为米/弧度，du/dt 单位为弧度/秒，链式相乘 dy/dt 的单位是？",["米/秒","米·弧度/秒","秒/米"],"A","中间变量 u 的单位在乘积中消去，留下输出对初始输入的单位。","application")],
        "math.partial_derivatives":[
            N("partial.second-mixed-derivative","多项式 f(x,y)=x²y。先对 y 求偏导，再对 x 求偏导，在 x=3 时结果是多少？",6,"power_derivative",[1,2,3],"第一步 ∂yf=x²；第二步对 x 求导为 2x；代入 3 得 6。"),
            M("partial.reordered-parameters","已知按 (x,y) 顺序的梯度是 (2,5)。仅把参数记录顺序改为 (y,x)，梯度分量应记录为？",["(2,5)","(5,2)","(7,7)"],"B","导数值不变，但排列必须继续与对应变量匹配。","application")],
        "math.gradient":[
            N("gradient.unit-descent-component","某点梯度为 (3,4)，其范数为 5。沿最快下降的单位方向，第一分量是多少？",-0.6,"ratio",[-3,5],"单位负梯度方向为 -(3,4)/5=(-0.6,-0.8)，第一分量是 -0.6。"),
            M("gradient.positive-objective-rescaling","把可微目标 f 整体乘以正数 10，在非驻点处，梯度方向怎样变化？",["方向反转","转为垂直方向","方向不变，梯度大小乘 10"],"C","∇(10f)=10∇f，正数缩放只改变长度；负梯度方向也保持。")],
        "prob.basics":[
            M("probability.empty-event","空事件没有任何可能结果，其概率是多少？",["0","1","无法定义"],"A","不可能发生的空事件具有零概率。")],
        "prob.conditional_probability":[
            N("conditional.independent-joint-model","已知 A、B 独立，P(A)=0.2、P(B)=0.3，求 P(A∩B)。",0.06,"product",[0.2,0.3],"独立使 P(A|B)=P(A)，因此联合概率为 0.2×0.3=0.06。"),
            M("conditional.condition-event-itself","P(B)>0，已经知道 B 发生，P(B|B) 是多少？",["P(B)²","0","1"],"C","条件空间中的所有结果都属于 B，所以 P(B∩B)/P(B)=1。")],
        "prob.distributions":[
            N("distribution.transformed-variable-mass","X 分别取 -1、1、2，概率是 0.2、0.3、0.5。定义 Z=X²，求 P(Z=1)。",0.5,"sum",[0.2,0.3],"X=-1 和 X=1 都映射到 Z=1，两个互斥事件概率相加得 0.5。"),
            M("distribution.reorder-mass-pairs","仅重排概率表中的行，同时保留每个取值和它的概率配对，分布是否改变？",["不改变","会改变，第一行必须最小","会改变，概率必须从大到小"],"A","分布由取值及其概率决定，表格行顺序本身没有概率含义。")],
        "prob.expectation_variance":[
            N("moments.standard-deviation","标准差定义为方差的非负平方根。若 Var(X)=9 平方米，标准差的数值是多少米？",3,"power",[9,0.5],"标准差取 √9=3，单位回到米；不取负根。"),
            M("moments.variance-sum-dependence","只知道 Var(X)、Var(Y)，但不了解二者相关性，能否总把 Var(X+Y) 写成两方差之和？",["总能","不能，一般还存在 2Cov(X,Y) 项","只有两者均值为零就总能"],"B","均值为零不意味着不相关；方差之和公式需协方差为零等额外条件。","theorem_condition")],
        "prob.bernoulli":[
            N("bernoulli.empirical-rate-estimate","五次实际点击标签为 [1,0,1,1,0]，用点击次数除以总次数估计伯努利参数 p，估计值是多少？",0.6,"mean",[1,0,1,1,0],"三个 1 表示三次点击，经验点击率为 3/5=0.6；这是数据估计而非事先已知概率。","application"),
            M("bernoulli.degenerate-parameter","当伯努利参数 p=1 时，变量 Y 的正确描述是？",["以概率 1 取 1，方差为 0","在 0 与 1 间等可能","取值变成任意实数"],"A","P(Y=1)=1 且 p(1-p)=0，分布退化为常数。")],
        "la.matrix_multiplication":[
            N("matrix.scalar-multiplication-work","按行列内积直接计算 (2×3) 矩阵乘 (3×4) 矩阵，不复用乘积，每个输出要做 3 次标量乘法。全部输出共做多少次乘法？",24,"product",[2,4,3],"结果含 2×4=8 个元素，每个内积 3 次乘法，总计 8×3=24。","application"),
            M("matrix.identity-action","I_d 是 d×d 单位矩阵，A 为 n×d。AI_d 的结果是？",["A 的转置","A","全零矩阵"],"B","单位矩阵的每列选择 A 对应的列，因此保留 A 不变。")],
    }
    choices = {
        "la.vectors":"", "la.dot_product":"A", "math.functions":"B",
        "ml.gradient_descent":"A", "ml.sigmoid":"A", "ml.logistic_regression":"B", "ml.cross_entropy":"C",
        "math.exponential_logarithmic":"B", "math.derivative_definition":"A", "math.derivative_rules":"B",
        "math.chain_rule":"A", "math.partial_derivatives":"B", "math.gradient":"C",
        "prob.basics":"A", "prob.conditional_probability":"C", "prob.distributions":"A",
        "prob.expectation_variance":"B", "prob.bernoulli":"A", "la.matrix_multiplication":"B",
    }
    return additions,choices
