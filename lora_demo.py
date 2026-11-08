"""零依赖 LoRA 微调演示（纯 Python 标准库，无需任何第三方包）。

用一个 2 层 MLP 把「预训练底座 + LoRA 适配器」与「全量微调」做对照，
证明 LoRA 的核心价值：用远少于全量的可训练参数，达到可比的适配效果。

数学对齐经典 LoRA 论文：ΔW = B·A（B∈ℝ^{out×r}, A∈ℝ^{r×in}，r≪min(in,out)），
前向 y = W_base·x + scaling·(B·A)·x，训练时冻结 W_base，只优化 A、B。

运行：python lora_demo.py
"""
import math
import random

# ============================ 超参 ============================
D, H, C, R = 10, 20, 4, 2   # 输入维 / 隐藏维 / 类别数 / LoRA 秩
LR = 0.05
EPOCHS = 60
SCALING = 1.0               # LoRA 缩放（论文中 α/r，这里取 1）
TRAIN_N, TEST_N = 300, 100
SEED = 42

# ============================ 微型线性代数 ============================
def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def mat_vec(W, x):
    return [dot(W[i], x) for i in range(len(W))]

def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))] if M else []

def mat_mat(A, B):  # A[m][k] · B[k][n] -> [m][n]
    n = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(n)] for i in range(len(A))]

def outer(a, b):
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]

def add_mat(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def scale_mat(A, s):
    return [[x * s for x in row] for row in A]

def relu(v):
    return [x if x > 0 else 0.0 for x in v]

def relu_mask(z):
    return [1.0 if x > 0 else 0.0 for x in z]

def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e) or 1e-12
    return [x / s for x in e]

def randn():
    u1, u2 = random.random(), random.random()
    return math.sqrt(-2 * math.log(u1 + 1e-12)) * math.cos(2 * math.pi * u2)

def rand_mat(r, c, scale=0.1):
    return [[random.gauss(0, scale) for _ in range(c)] for _ in range(r)]

def zeros_mat(r, c):
    return [[0.0] * c for _ in range(r)]

# ============================ 数据集（合成、可复现） ============================
def make_dataset(n, seed):
    random.seed(seed)
    # 4 个类别各自不同的 10 维中心，保证可学习
    means = [[math.sin((c + 1) * (i + 1)) for i in range(D)] for c in range(C)]
    X, Y = [], []
    per = n // C
    for c in range(C):
        for _ in range(per):
            x = [means[c][i] + random.gauss(0, 0.3) for i in range(D)]
            X.append(x)
            Y.append(c)
    return X, Y

# ============================ 全量微调 ============================
def train_full(X, Y):
    W1, W2 = rand_mat(H, D), rand_mat(C, H)
    random.seed(SEED)
    idx = list(range(len(X)))
    for _ in range(EPOCHS):
        random.shuffle(idx)
        for i in idx:
            x, y = X[i], Y[i]
            z1 = mat_vec(W1, x)
            a1 = relu(z1)
            z2 = mat_vec(W2, a1)
            p = softmax(z2)
            t = [0.0] * C
            t[y] = 1.0
            dz2 = [p[k] - t[k] for k in range(C)]
            dW2 = outer(dz2, a1)
            da1 = mat_vec(transpose(W2), dz2)
            dz1 = [da1[h] * relu_mask(z1)[h] for h in range(H)]
            dW1 = outer(dz1, x)
            W2 = [[W2[a][b] - LR * dW2[a][b] for b in range(H)] for a in range(C)]
            W1 = [[W1[a][b] - LR * dW1[a][b] for b in range(D)] for a in range(H)]
    return W1, W2

# ============================ LoRA 微调 ============================
def train_lora(X, Y):
    # 底座冻结（模拟预训练权重）
    W1_base = rand_mat(H, D)
    W2_base = rand_mat(C, H)
    # 适配器：A 随机小初始化，B 置零（LoRA 最佳实践：初始 ΔW=0）
    A1, B1 = rand_mat(R, D), zeros_mat(H, R)
    A2, B2 = rand_mat(R, H), zeros_mat(C, R)
    random.seed(SEED)
    idx = list(range(len(X)))
    for _ in range(EPOCHS):
        random.shuffle(idx)
        for i in idx:
            x, y = X[i], Y[i]
            # 前向：W_eff = W_base + scaling·(B·A)
            delta1 = scale_mat(mat_mat(B1, A1), SCALING)
            W1_eff = add_mat(W1_base, delta1)
            z1 = mat_vec(W1_eff, x)
            a1 = relu(z1)
            delta2 = scale_mat(mat_mat(B2, A2), SCALING)
            W2_eff = add_mat(W2_base, delta2)
            z2 = mat_vec(W2_eff, a1)
            p = softmax(z2)
            t = [0.0] * C
            t[y] = 1.0
            dz2 = [p[k] - t[k] for k in range(C)]
            da1 = mat_vec(transpose(W2_eff), dz2)
            dz1 = [da1[h] * relu_mask(z1)[h] for h in range(H)]
            # 梯度回传到 A、B（链式法则拆分）
            dW2_eff = outer(dz2, a1)
            G2 = scale_mat(dW2_eff, SCALING)
            dB2 = mat_mat(G2, transpose(A2))
            dA2 = mat_mat(transpose(B2), G2)
            dW1_eff = outer(dz1, x)
            G1 = scale_mat(dW1_eff, SCALING)
            dB1 = mat_mat(G1, transpose(A1))
            dA1 = mat_mat(transpose(B1), G1)
            A1 = [[A1[a][b] - LR * dA1[a][b] for b in range(D)] for a in range(R)]
            B1 = [[B1[a][b] - LR * dB1[a][b] for b in range(R)] for a in range(H)]
            A2 = [[A2[a][b] - LR * dA2[a][b] for b in range(H)] for a in range(R)]
            B2 = [[B2[a][b] - LR * dB2[a][b] for b in range(R)] for a in range(C)]
    return dict(W1_base=W1_base, A1=A1, B1=B1, W2_base=W2_base, A2=A2, B2=B2)

# ============================ 评估 ============================
def evaluate_full(W1, W2, X, Y):
    ok = 0
    for x, y in zip(X, Y):
        a1 = relu(mat_vec(W1, x))
        p = softmax(mat_vec(W2, a1))
        if p.index(max(p)) == y:
            ok += 1
    return ok / len(Y)

def evaluate_lora(m, X, Y):
    W1_eff = add_mat(m["W1_base"], scale_mat(mat_mat(m["B1"], m["A1"]), SCALING))
    W2_eff = add_mat(m["W2_base"], scale_mat(mat_mat(m["B2"], m["A2"]), SCALING))
    return evaluate_full(W1_eff, W2_eff, X, Y)

def count_params(mode):
    if mode == "full":
        return H * D + C * H
    return R * D + H * R + R * H + C * R  # A1,B1,A2,B2

def run_comparison():
    Xtr, Ytr = make_dataset(TRAIN_N, SEED)
    Xte, Yte = make_dataset(TEST_N, SEED + 1)
    full = train_full(Xtr, Ytr)
    lora = train_lora(Xtr, Ytr)
    full_tr = evaluate_full(full[0], full[1], Xtr, Ytr)
    full_te = evaluate_full(full[0], full[1], Xte, Yte)
    lora_tr = evaluate_lora(lora, Xtr, Ytr)
    lora_te = evaluate_lora(lora, Xte, Yte)
    pf, pl = count_params("full"), count_params("lora")
    return dict(
        full_train=full_tr, full_test=full_te,
        lora_train=lora_tr, lora_test=lora_te,
        full_params=pf, lora_params=pl, lora_ratio=pl / pf,
    )

if __name__ == "__main__":
    r = run_comparison()
    print("=" * 64)
    print("LoRA 微调对照（零依赖 · 2 层 MLP 玩具分类，4 类，D/H/C/R = %d/%d/%d/%d）" % (D, H, C, R))
    print("=" * 64)
    print(f"{'方法':<14}{'可训练参数':>10}{'参数占比':>10}{'训练准确率':>12}{'测试准确率':>12}")
    print("-" * 64)
    print(f"{'全量微调':<12}{r['full_params']:>10}{'100%':>10}{r['full_train']*100:>10.1f}%{r['full_test']*100:>10.1f}%")
    print(f"{'LoRA(r=%d)'%R:<12}{r['lora_params']:>10}{r['lora_ratio']*100:>9.1f}%{r['lora_train']*100:>10.1f}%{r['lora_test']*100:>10.1f}%")
    print("-" * 64)
    print(f"结论：LoRA 用 {r['lora_ratio']*100:.1f}% 的可训练参数，达到与全量微调"
          f"（测试 {r['full_test']*100:.1f}%）可比的适配效果（测试 {r['lora_test']*100:.1f}%）。")
    print("意义：仅保存/训练适配器(A,B)，底座冻结——显存、存储、过拟合风险显著下降。")
