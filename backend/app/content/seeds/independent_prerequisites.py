"""Small prerequisite checks that remain independent after worked examples.

Conservative family corrections exclude old numerical near-variants from
formal allocation. Replacement tasks test representation, constraints,
inverse reasoning or structural properties rather than the shown solution.
"""


def authored(N):
    corrections={
        "dot.self-product":"dot.orthogonality-calculation",
        "matrix.product-output-rows":"matrix.matrix-vector-product",
        "logarithm.unit-value":"logarithm.inverse-equation",
        "derivative.power-rule-basic":"derivative.polynomial-rules",
        "chain.two-rate-product":"chain.affine-square",
        "partial.fixed-linear-coefficient":"calculus.quadratic-partial-gradient",
        "gradient.simple-direction":"calculus.quadratic-partial-gradient",
        "conditional.simple-positive-fraction":"conditional.ratio-definition",
        "distribution.one-mass-complement":"distribution.discrete-tail",
        "moments.weighted-mean-basic":"moments.two-point-computation",
        "bernoulli.mean-basic":"bernoulli.single-moments",
    }
    additions={
        "la.dot_product":N("dot.bias-shift-sensitivity","线性模型 z=w·x+b，固定 w 和 x，只将偏置 b 增加 0.7。新旧得分之差 z_new-z_old 是多少？",0.7,"sum",[0.7],"两次内积项完全相同，作差时抵消，只剩偏置变化 0.7。","application"),
        "la.matrix_multiplication":N("matrix.entry-index-reading","矩阵第一行为 [5,7,9]，按从 1 开始的行列下标，元素 A₁₂ 是多少？",7,"at",[[5,7,9],1],"下标 1 表示第一行，2 表示该行第二列，因此取 7。"),
        "math.exponential_logarithmic":N("exponential.combine-exponents","利用 e^a·e^b=e^(a+b)，把 e²·e⁻³ 合并为 eᵏ，指数 k 是多少？",-1,"sum",[2,-3],"同底指数相乘时指数相加，k=2+(-3)=-1。"),
        "math.derivative_rules":N("derivative.recover-coefficient","f(x)=c x²，c 是常数。已知 f'(1)=6，求 c。",3,"ratio",[6,2],"幂法则给出 f'(x)=2cx；在 x=1 时 2c=6，因此 c=3。"),
        "math.chain_rule":N("chain.outer-evaluation-point","h(x)=f(g(x))，g(x)=x²。当 x=-3 时，链式法则中的 f' 应在数值几处计算？",9,"power",[-3,2],"外层实际接收 g(-3)=9，所以使用 f'(9)，不是 f'(-3)。"),
        "math.partial_derivatives":N("partial.coordinate-difference-quotient","固定 y=b，已知 f(a,b)=10，f(a+0.2,b)=10.6。求这个 x 增量对应的有限差商（不声称它等于导数极限）。",3,"ratio",[0.6,0.2],"只沿 x 改变，输出增量 0.6 除以输入增量 0.2，有限差商为 3。"),
        "math.gradient":N("gradient.sum-of-objectives","在同一点 ∇f=(2,3)、∇g=(-1,4)。用逐分量求和规则，∇(f+g) 的第二个分量是多少？",7,"sum",[3,4],"第二个分量仍对应第二个变量，两偏导相加为 3+4=7。"),
        "prob.conditional_probability":N("conditional.probability-range-audit","一份报告给出三项条件概率：0.2、-0.1、1.1，其中有多少项违反概率必须在 [0,1] 内的要求？",2,"count_outside_unit_interval",[[0.2,-0.1,1.1]],"-0.1 为负，1.1 大于 1，共两项不合法；条件概率也遵守普通概率范围。"),
        "prob.distributions":N("distribution.read-negative-support","随机变量 X 的可能取值为 -2、1、4，对应概率为 0.2、0.3、0.5。X 最小的可能取值是多少？",-2,"minimum",[[-2,1,4]],"取值和概率是不同列；负取值允许出现，最小取值是 -2。"),
        "prob.expectation_variance":N("moments.recover-second-moment","已知 E[X]=2、Var(X)=3，利用 Var(X)=E[X²]-E[X]² 求 E[X²]。",7,"second_moment",[2,3],"移项后 E[X²]=Var(X)+E[X]²=3+4=7。"),
        "prob.bernoulli":N("bernoulli.observed-centered-outcome","伯努利模型参数 p=0.25，本次实际观察 y=1。按 r=y-p 定义本次标签相对预测均值的偏差，求 r。",0.75,"sum",[1,-0.25],"实际标签是 1，模型均值是 0.25，两者之差为 0.75；不是把本次标签改写为概率。","application"),
    }
    return corrections,additions
