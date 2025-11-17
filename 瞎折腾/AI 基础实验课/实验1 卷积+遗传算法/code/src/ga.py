import random
import numpy as np
import torch
from deap import base, creator, tools, algorithms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，便于在无显示环境保存图像
import matplotlib.pyplot as plt
from pathlib import Path

# 兼容脚本/包两种运行方式
import config as C
from data import load_cifar10
from models import create_model_from_cfg
from train import train_simple_and_save



# 1) 定义适应度与个体
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def build_toolbox():
    tb = base.Toolbox()
    # 2) 基因生成器（7项）
    tb.register("batch_size", lambda: random.choice(C.BATCH_SIZE_CHOICES))
    tb.register("lr", lambda: 10 ** random.uniform(*C.LR_LOG10_RANGE))
    tb.register("dropout_rate", lambda: random.uniform(*C.DROPOUT_RANGE))
    tb.register("use_bn_cnn", lambda: random.choice([0, 1]))
    tb.register("use_dropout_cnn", lambda: random.choice([0, 1]))
    tb.register("use_bn_linear", lambda: random.choice([0, 1]))
    tb.register("use_dropout_linear", lambda: random.choice([0, 1]))

    # 3) 个体与种群
    def create_individual():
        return [
            tb.batch_size(),
            tb.lr(),
            tb.dropout_rate(),
            tb.use_bn_cnn(),
            tb.use_dropout_cnn(),
            tb.use_bn_linear(),
            tb.use_dropout_linear(),
        ]

    tb.register("individual", tools.initIterate, creator.Individual, create_individual)
    tb.register("population", tools.initRepeat, list, tb.individual)
    return tb


def decode(ind):
    cfg = {
        "batch_size": int(ind[0]),
        "lr": float(ind[1]),
        "dropout_rate": float(ind[2]),
        # 固定结构
        "num_cnn_layers": C.FIXED_MODEL["num_cnn_layers"],
        "kernel_size": C.FIXED_MODEL["kernel_size"],
        "pooling_type": C.FIXED_MODEL["pooling_type"],
        "num_linear_layers": C.FIXED_MODEL["num_linear_layers"],
        # 开关
        "use_bn_cnn": bool(int(ind[3])),
        "use_dropout_cnn": bool(int(ind[4])),
        "use_bn_linear": bool(int(ind[5])),
        "use_dropout_linear": bool(int(ind[6])),
    }
    # 若希望不依赖基因、统一用固定开关，则启用覆盖
    if getattr(C, "OVERRIDE_GENE_SWITCHES", False):
        sw = getattr(C, "DEFAULT_SWITCHES", {})
        for k in ("use_bn_cnn", "use_dropout_cnn", "use_bn_linear", "use_dropout_linear"):
            if k in sw:
                cfg[k] = bool(sw[k])
    return cfg


def evaluate(ind):
    cfg = decode(ind)
    # 生成确定性 seed，减小评估方差
    seed = int(abs(hash(tuple(ind))) % (2**32))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(
        "Eval genes: "
        f"bs={int(ind[0])} "
        f"lr={float(ind[1]):.2e} "
        f"dropout={float(ind[2]):.3f} "
        f"BNc={int(ind[3])} DOc={int(ind[4])} BNl={int(ind[5])} DOl={int(ind[6])}",
        flush=True,
    )

    train_loader = load_cifar10(
        is_train=True,
        batch_size=cfg["batch_size"],
        sample_count=C.SAMPLE_COUNT_TRAIN,
        seed=seed,
    )
    test_loader = load_cifar10(
        is_train=False,
        batch_size=cfg["batch_size"],
        sample_count=C.SAMPLE_COUNT_TEST,
        seed=seed,
    )

    model = create_model_from_cfg(cfg)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"]) 

    _, test_loss = train_simple_and_save(
        model, train_loader, test_loader, loss_fn, optimizer,
        num_epochs=C.NUM_EPOCHS_EVAL,
        device=None, save_dir=C.OUTPUT_DIR, id="temp", params=cfg,
    )
    return (float(test_loss),)


def mutate_individual(ind, indpb=0.1):
    """按基因位点概率突变；若被选中突变但无任何位点改变，强制改变1个随机位点。"""
    tb = build_toolbox()  # 简单做法：用现有注册器重采样
    mutated = False
    for i in range(len(ind)):
        if random.random() >= indpb:
            continue
        old = ind[i]
        if i == 0:
            ind[i] = tb.batch_size()
        elif i == 1:
            ind[i] = tb.lr()
        elif i == 2:
            ind[i] = tb.dropout_rate()
        elif i == 3:
            ind[i] = tb.use_bn_cnn()
        elif i == 4:
            ind[i] = tb.use_dropout_cnn()
        elif i == 5:
            ind[i] = tb.use_bn_linear()
        elif i == 6:
            ind[i] = tb.use_dropout_linear()
        mutated = mutated or (ind[i] != old)
    if not mutated:
        # 强制改变1个位点
        i = random.randrange(len(ind))
        if i == 0:
            ind[i] = tb.batch_size()
        elif i == 1:
            ind[i] = tb.lr()
        elif i == 2:
            ind[i] = tb.dropout_rate()
        elif i == 3:
            ind[i] = tb.use_bn_cnn()
        elif i == 4:
            ind[i] = tb.use_dropout_cnn()
        elif i == 5:
            ind[i] = tb.use_bn_linear()
        elif i == 6:
            ind[i] = tb.use_dropout_linear()
    return (ind,)


def run_evolution():
    """运行 GA，带进度条，并在结束后绘制并保存 min/avg 曲线，保存最佳模型。"""
    tb = build_toolbox()
    tb.register("evaluate", evaluate)
    tb.register("mate", tools.cxTwoPoint)
    tb.register("mutate", mutate_individual, indpb=C.MUT_INDPB)
    tb.register("select", tools.selTournament, tournsize=C.TOURN_SIZE)

    out_dir = Path(C.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    pop = tb.population(n=C.POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "min", "max"]

    # 初始评估
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(map(tb.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    min_list = [float(record["min"])]
    avg_list = [float(record["avg"])]

    for gen in range(1, C.NGEN + 1):
        # 选择并产生子代
        offspring = tools.selTournament(pop, len(pop), tournsize=C.TOURN_SIZE)
        offspring = algorithms.varAnd(offspring, tb, cxpb=C.CXPB, mutpb=C.MUTPB)

        # 去重增强：若出现重复个体，对重复项随机重采样1个基因以增加多样性
        seen = set()
        tb_local = build_toolbox()
        for ind in offspring:
            key = tuple(ind)
            if key in seen:
                j = random.randrange(len(ind))
                if j == 0:
                    ind[j] = tb_local.batch_size()
                elif j == 1:
                    ind[j] = tb_local.lr()
                elif j == 2:
                    ind[j] = tb_local.dropout_rate()
                elif j == 3:
                    ind[j] = tb_local.use_bn_cnn()
                elif j == 4:
                    ind[j] = tb_local.use_dropout_cnn()
                elif j == 5:
                    ind[j] = tb_local.use_bn_linear()
                elif j == 6:
                    ind[j] = tb_local.use_dropout_linear()
                # 使其适应度失效，确保会被重新评估
                if hasattr(ind.fitness, 'values'):
                    del ind.fitness.values
            else:
                seen.add(key)

        # 评估无效个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(tb.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 选择下一代种群
        pop = tb.select(offspring, len(pop))
        hof.update(pop)

        # 统计与记录
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        min_list.append(float(record["min"]))
        avg_list.append(float(record["avg"]))
        
        # 打印进度
        print(
            f"\nGen {gen}/{C.NGEN}  evals={len(invalid_ind)}  "
            f"min={record['min']:.4f}  avg={record['avg']:.4f}  max={record['max']:.4f}"
        )   

    # 最优个体复评并保存模型
    best_ind = hof[0]
    best_cfg = decode(best_ind)
    best_seed = int(abs(hash(tuple(best_ind))) % (2**32))
    # 与 evaluate() 保持一致的确定性随机种子设置，尽量对齐结果
    random.seed(best_seed)
    np.random.seed(best_seed)
    torch.manual_seed(best_seed)
    train_loader = load_cifar10(True, best_cfg["batch_size"], C.SAMPLE_COUNT_TRAIN, best_seed)
    test_loader = load_cifar10(False, best_cfg["batch_size"], C.SAMPLE_COUNT_TEST, best_seed)
    model = create_model_from_cfg(best_cfg)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_cfg["lr"]) 
    _, reval = train_simple_and_save(
        model, train_loader, test_loader, loss_fn, optimizer,
        num_epochs=C.NUM_EPOCHS_EVAL,
        device=None, save_dir=C.OUTPUT_DIR, id="hof_reval", params=best_cfg,
    )

    # 保存最佳模型权重与元信息
    best_model_path = out_dir / "best_model.pth"
    torch.save(model.state_dict(), best_model_path)
    meta = {
        "best_ind": list(map(float, best_ind)),
        "best_cfg": best_cfg,
        # 进化阶段选择该 best_ind 时的适应度（来源于个体上一次评估的缓存）
        "fitness_at_selection": float(best_ind.fitness.values[0]) if hasattr(best_ind, "fitness") else None,
        # 复评（重新训练）后的测试损失
        "best_test_loss": float(reval),
        "ga_params": {
            "POP_SIZE": C.POP_SIZE, "NGEN": C.NGEN, "CXPB": C.CXPB,
            "MUTPB": C.MUTPB, "TOURN_SIZE": C.TOURN_SIZE,
        },
    }
    with open(out_dir / "best_model_meta.json", "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 绘制并保存曲线图
    gens = list(range(0, C.NGEN + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(gens, min_list, label="min (test_loss)")
    plt.plot(gens, avg_list, label="avg (test_loss)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (test_loss, lower is better)")
    # 图题包含关键参数与最佳配置（简版）
    # 简洁标题：不展示参数详情
    plt.title("GA fitness curve (test_loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path = out_dir / "fitness_curve.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return pop, logbook, hof, float(reval)
