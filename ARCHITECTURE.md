# 架构说明

这份文档是给"改代码的人（或 AI）"看的上下文地图。改动前先在这里定位，
再打开对应的 1-2 个文件，不需要通读全项目。

## 依赖方向

```
        ┌──────────────────────────────────────────────┐
        │  main.py  只负责组装：日志 / 中间件 / 路由注册 │
        └───────────────────┬──────────────────────────┘
                            │
        ┌───────────────────▼──────────────────────────┐
        │  api/        HTTP 形状：schemas, deps, routers│
        └───────────────────┬──────────────────────────┘
                            │
        ┌───────────────────▼──────────────────────────┐
        │  agent/      问答编排：意图路由 / 规则 / LLM   │
        └───────────────────┬──────────────────────────┘
                            │
        ┌───────────────────▼──────────────────────────┐
        │  services/   用例编排：筛选 / 详情 / 组合      │
        └──────┬───────────────────────┬───────────────┘
               │                       │
   ┌───────────▼──────────┐  ┌─────────▼──────────────┐
   │  sources/  数据出入口 │  │  analysis/  纯计算      │
   │  （唯一 import akshare）│ │ （无 I/O，可单测）      │
   └───────────┬──────────┘  └─────────┬──────────────┘
               │                       │
        ┌──────▼───────────────────────▼───────────────┐
        │  core/（缓存 / 限流 / 熔断 / 异常）  config.py │
        └──────────────────────────────────────────────┘
```

只允许上层依赖下层。`analysis` 不知道有 HTTP，`sources` 不知道有指标计算，
`api` 里不写业务判断。`container.py` 是唯一的组合根，把下层实例装配起来。

## 要改 X，看哪个文件

| 你想做的事 | 改这里 | 备注 |
|---|---|---|
| 加/改配置项、缓存时长、费率假设 | `config.py` | 唯一读 env 的地方，别在其它模块 `os.getenv` |
| akshare 换了列名 / 接口报错 | `sources/normalize.py`、`sources/akshare_client.py` | 列名漂移只在这层吸收 |
| 加一个新数据源（比如天天基金直连） | `sources/akshare_client.py` 同层新增，`repository` 编排 | 上层无需改动 |
| 改风险指标、加新指标 | `analysis/metrics.py` | 纯函数，顺手在 `tests/test_quant.py` 补断言 |
| 改滤波器（P/V/A、残差尺度、置信度） | `analysis/abg.py` | 公式来自 `docs/reference/abg_exit_design.md`，偏离要注明 |
| 改仓位规则、天平地板 | `analysis/signals.py` 的 `gate_factors` | **唯一定义**，标量与向量路径共用，别在别处再写一遍公式 |
| 改止盈止损 / 出场规则 | `analysis/risk_rules.py` | 任何异常必须返回"不动作"，不得因状态算错而误赎 |
| 改策略模拟管线 | `analysis/simulation.py` | 敞口只能由 t-1 及以前的状态决定（`test_simulation.py::test_no_lookahead` 守着） |
| 改参数搜索 / 进化 | `analysis/search.py` | 新参数要同时进 `PARAM_SPACE` 和某个 `GENE_BLOCKS` 块 |
| 改组合优化 / 仓位规则 | `analysis/portfolio.py` | 改权重必须同时过 `leverage_guard` |
| 改回测逻辑（再平衡、成本） | `analysis/backtest.py` | 注意 monthly 是**真**月末再平衡 |
| 改因子权重、加打分因子 | `analysis/factors.py` + `config.Settings.factor_weights` | 新因子要进 `OPTIONAL_FACTORS` |
| 费用假设、负期望判定 | `analysis/expectancy.py`、`config.DEFAULT_FEE_PROFILES` | |
| 样本外检验、Deflated Sharpe | `analysis/validation.py` | |
| 问答新增一类问题 | `agent/intent.py` 加关键词 + `agent/rules.py` 加 handler | 不用读懂其它分支 |
| 改系统提示词 | `agent/prompts.py` | |
| 换 LLM 服务商 | `agent/llm.py`（或只改 `.env` 的 `LLM_API_BASE`） | OpenAI 兼容即可 |
| 加一个接口 | `api/schemas.py` + `api/routers/*.py`，逻辑放 `services/` | 路由里不写计算 |
| 前端旧字段/布局 | `static/index.html` | 后端响应只**追加**字段，不改已有字段名 |
| 前端新功能（体检/模拟面板） | `static/js/sim-panel.js`、`static/js/paper-panel.js` | 新逻辑一律加在这里，不再往 index.html 堆 |
| 改模拟盘撮合规则（T+1、阶梯赎回费、批次） | `paper/account.py` | 纯逻辑，改一处要同步 `tests/test_paper.py` |
| 改模拟盘的自动下单逻辑 | `paper/engine.py` | 决策只能读 t-1 及以前的已公布净值 |
| 改冠军/挑战者档案与晋升纪律 | `analysis/archive.py` | 冻结日语义不能动，否则样本外统计失真 |
| 改自训练闭环、复盘报告 | `agent/training.py`、`train.py` | LLM 只写叙述，不参与打分与晋升 |
| 模拟盘/自训练的接口 | `api/routers/paper.py`、`api/routers/training.py` | 档案与账户实例必须从 `AppState` 注入 |
| 改配色/暗色主题/组件质感 | `static/css/theme.css` | 覆盖层，只写覆盖规则，别去改内联 `<style>` |
| 主题切换/状态胶囊/表格着色/问答 LLM 开关 | `static/js/ui-tweaks.js` | 暴露 `window.UITweaks`，内联 onclick 直接调用 |

## 硬约束（改代码时不要破坏）

1. **akshare 只出现在 `sources/`**。其它层要数据，找 `FundRepository`。
2. **`analysis/` 无 I/O、无全局可变状态**。输入 ndarray/Series，输出 dict。
   这样任何函数都能单独喂给模型、单独测试。
3. **运行期实例只有一个**：`container.AppState`。新增状态请挂在它上面，
   不要重新写模块级全局字典（这正是重构前 `_cache / _nav_cache / LLM_STATE` 的问题）。
4. **同步阻塞调用不能直接写在 `async def` 路由里**：akshare 是同步的，
   统一用 `core.resilience.run_blocking` 挪到线程池，否则整个事件循环被卡住。
5. **单个文件不超过 ~300 行**。超了就说明该拆了。
6. 外部接口失败**必须有降级路径**（旧缓存 → 明确错误码），不能 500 糊上去。
7. **熔断的粒度是接口，不是应用**：`AkshareClient.breakers` 是按接口名分配的
   `BreakerRegistry`。数据级失败（基金代码不存在、返回结构变化）由
   `core.resilience.is_infrastructure_error` 识别为"这条数据没有"，
   **不重试也不计入熔断** —— 否则查 3 只垃圾基金就会把正常查询封在门外
   （这是重构过程中真实踩到过的坑）。
8. **环境层参数不进搜索空间、不进请求体**。`Environment`（杠杆上限、long-only、
   全局熔断、费用下限、执行滞后）既不在 `analysis/search.PARAM_SPACE` 里，
   也不在 `services/simulation.build_genes` 的白名单里。客户端传
   `{"max_leverage": 5}` 会被记进 `ignored_params` 并原样拒绝。
   理由：任何"分数更高"的杠杆解都是过拟合，不该有被搜到的可能。
9. **同一套公式只能有一份实现**。仓位天平的标量版（问答解释）和向量版
   （回测/进化）都调用 `analysis.signals.gate_factors`；
   复制公式会造成"页面上解释的仓位和回测用的仓位不一致"这种查不动的 bug。

## 已知偏差

- `static/index.html` 仍是 ~1470 行的单页，超出第 5 条。已经拆出去的是：
  新功能 `static/js/sim-panel.js`、交互层 `static/js/ui-tweaks.js`、
  主题 `static/css/theme.css`（加载在内联样式之后，只做覆盖）。
  存量仪表盘/问答/对比的 HTML 与 JS 仍在单页里，待拆。
- ECharts 走 CDN，并带一份本地副本 `static/js/vendor/echarts.min.js` 兜底
  （`window.echarts || document.write(...)`），断网也能出图。
- `docs/reference/kline_free_plan.rar` 是 RAR 压缩包，本机没有解压工具，内容尚未阅读。
- 第三、四、五期的**转写稿没拿到**（YouTube 字幕接口要求 POT 令牌），
  对应模块的设计依据是：视频标题 + 完整简介 + 时间戳，以及作者公开在简介里的
  两份设计文档（GA 进化系统、ABG 平仓原理）。凡属推断的地方都在模块
  docstring 里标注了。

## 数据流：一次 `/api/portfolio/analyze`

```
routers/portfolio.py   解析 PortfolioRequest
  └─ services/portfolio.analyze
       ├─ sources/repository.require_aligned   （限流→重试→熔断→缓存降级）
       ├─ analysis/metrics.risk_metrics        （每只基金的风险指标）
       ├─ analysis/correlation                 （相关矩阵、高相关对、有效下注数）
       ├─ analysis/factors.multi_factor_score  （综合打分）
       └─ services/portfolio.risk_contributions（谁在真正决定组合风险）
```

`run_blocking` 在路由层把整条链丢进线程池，所以阻塞只发生在 worker 线程。

## 期望值约定（这个项目的世界观）

来自"残酷数学真相"的那部分逻辑被固化在代码里，而不是写在文档里靠人自觉：

| 结构性负期望来源 | 代码位置 | 如何拦截 |
|---|---|---|
| 做空 | `analysis/portfolio.leverage_guard` | 闭式解出负权重 → 裁剪为 long-only，并在响应里返回 `guard.notes` |
| 杠杆 | 同上 + `config.AnalysisConfig.max_gross_exposure=1.0` | 总敞口 >100% 直接标记不达标；目标波动率缩放**只减仓不加杠杆**，多余留现金 |
| 交易/持有费用 | `analysis/expectancy.fee_drag` | 每个详情与批量接口都返回 `net_annual_return_pct`、`net_sharpe` |
| 惩罚性赎回费（<7 天） | `config.FeeConfig.penalty_*` | `fee_drag(penalty_applies=True)`，问答会明确警告 |
| 把噪声当规律 | `analysis/expectancy.luck_concentration` | 前 5 天贡献超 50% 时触发 `把噪声当规律` 检查 |
| 排名无预测力 | `analysis/validation.walk_forward` | 样本外 Rank IC≈0 时直接说"没有可依赖的预测力" |
| 搜索出来的漂亮回测 | `analysis/validation.deflated_sharpe_ratio` | 按尝试次数 `n_trials` 抬高夏普门槛 |
| 进化出来的针尖解 | `analysis/search.plateau_check` | 邻域扰动 10% 后掉分超阈值 → 判为"在针尖上"，不是平原 |
| 悬崖式清仓（规则本身的风险） | `analysis/signals.gate_factors` | 波动用逆比率缩放而非 sigmoid；每个天平带地板，单一状态恶化不会清零仓位 |
| 前视偏差 | `analysis/simulation.simulate` + `Environment.execution_lag_days` | t 日敞口只读 t-1 及以前的状态；`test_no_lookahead` 守住 |
| 出场系统自身崩溃 | `analysis/risk_rules.evaluate_exit` | 任何异常路径一律返回 hold，绝不因状态算错而误赎 |
| 频繁调仓的摩擦 | `signals.min_trade_delta`（不交易带）+ `search.fitness` 摩擦惩罚 | 偏差不到阈值不动手；进化时按双边费用直接扣分 |

结论性的一句话：**毛收益为正不代表可以继续持有，费后为正 + 样本外有效 + 参数在平原上，才算。**

## 模拟盘与自训练的纪律（不可绕过）

"自动模拟炒股 + 自己训练自己"合在一起最大的风险不是亏模拟币，而是
**自我确认**：在同一段历史上进化一百次，第一百次的成绩依然是样本内。
所以这套闭环建立在四条硬规矩上：

| 规矩 | 落在哪 | 破坏它会怎样 |
|---|---|---|
| 进化时把最近 N 天**整段藏起来**，挑战者只准用更早数据进化，留出段是它第一次见 | `agent/training.run_round` 的 `_split` | 留出段泄漏进训练，"样本外"变成自欺 |
| 晋升那一刻才记 `frozen_at`，只有其后的净值算样本外证据 | `analysis/archive.promote`、`PaperAccount.out_of_sample_stats` | 把样本内成绩说成"验证通过" |
| 挑战者不自动晋升，闸门（参数平原 / DSR / walk-forward）任一不过直接 `rejected` | `StrategyArchive.add` / `promote` | 进化器总会找到针尖解，并把它叫做最优 |
| LLM 只把代码算好的事实写成话，不参与打分、筛选、晋升 | `agent/training.narrate_review` | 模型一幻觉，整条链的数字就不可信 |

基金专属的撮合细节也都在代码里，不在文档里：T 日提交按 T+1 未知净值确认、
申购按金额而赎回按份额、份额按批次记账（否则算不清阶梯赎回费）、
持有不足 7 天惩罚性 1.5%、现金不足直接拒单不透支。

命令行入口（可挂 Windows 计划任务定时跑）：

```bash
python train.py --codes 002910,110022,161725 --rounds 1     # 进化一轮并入库
python train.py --open --codes 002910,110022                # 开户并等权建仓（记冻结日）
python train.py --auto                                      # 所有账户按冠军规则走一天
python train.py --review pa-xxxx                            # 复盘（含样本外天数）
python train.py --promote st-xxxx                           # 人工晋升
```

## API 一览

| 路径 | 方法 | 用途 |
|---|---|---|
| `/api/funds` | GET | 排名数据（前 200 条） |
| `/api/screen` | POST | 条件筛选，返回 `applied_filters` / `ignored_filters` |
| `/api/fund/{code}` | GET | 详情：净值 + 指标 + 费后期望 |
| `/api/export` | GET | CSV/JSON 导出 |
| `/api/chat` | POST | 问答；`use_llm=false` 可强制走规则引擎省 token |
| `/api/intent` | GET | 调试：这句话被识别成什么意图 |
| `/api/news`、`/api/knowledge` | GET | 资讯 / 知识库 |
| `/api/portfolio/analyze` | POST | 指标 + 相关性 + 打分 + 有效下注数 + 风险贡献 |
| `/api/portfolio/optimize` | POST | 三种优化 + 有效前沿 + 无杠杆护栏 + 目标波动率仓位 |
| `/api/portfolio/backtest` | POST | 回测，支持 `cost_pct` 计入换手成本 |
| `/api/portfolio/robustness` | POST | walk-forward + Deflated Sharpe 反过拟合体检 |
| `/api/portfolio/expectancy` | POST | 一组基金的费后期望对照 |
| `/api/paper/accounts` | GET/POST | 模拟账户列表 / 开户（开户即记冻结日） |
| `/api/paper/accounts/{id}` | GET/DELETE | 账户明细（持仓批次、订单、样本外状态）/ 删除 |
| `/api/paper/accounts/{id}/orders` | POST | 申购（按金额）/ 赎回（按份额），T+1 确认 |
| `/api/paper/accounts/{id}/settle` | POST | 用最新公布净值结算待确认订单并盯市 |
| `/api/paper/accounts/{id}/auto` | POST | 按冠军规则自动走一天（结算 → 判状态 → 下单） |
| `/api/paper/accounts/{id}/decision` | GET | 只算不动：今天规则会怎么决定仓位 |
| `/api/train/round` | POST | 跑一轮自训练（进化 → 三重闸门 → 留出段首考） |
| `/api/train/archive`、`/api/train/promote` | GET/POST | 策略档案 / 人工晋升（记录冻结日） |
| `/api/train/review` | GET | 复盘报告（数字由代码算，叙述可选 LLM） |
| `/api/portfolio/policy` | GET | 每只基金当前风险状态与建议敞口档位 |
| `/api/portfolio/simulate` | POST | 策略模拟：状态→仓位→出场→成本（含定投基准对比） |
| `/api/portfolio/evolve` | POST | 遗传参数进化 + 参数平原 + 蒙特卡洛终审 + DSR |
| `/api/llm/{models,model,usage,budget}` | GET/POST | 模型切换与 token 预算护栏 |
| `/api/health`、`/api/config`、`/api/cache/invalidate` | — | 运维 |

## 配置项（`.env`）

除原有 `LLM_API_KEY / LLM_API_BASE / LLM_MODEL / ALLOWED_ORIGINS` 外，新增可选：

```
CACHE_RANKING_TTL / CACHE_DETAIL_TTL / CACHE_NAV_TTL / CACHE_NEWS_TTL
SOURCE_MIN_INTERVAL / SOURCE_MAX_RETRIES / SOURCE_BREAKER_THRESHOLD / SOURCE_BREAKER_RESET
ANNUAL_RISK_FREE / TARGET_VOLATILITY
LLM_TIMEOUT / LLM_MAX_TOKENS / LLM_BUDGET_TOKENS
```

不填即用 `config.py` 里的默认值。

## 兼容层与迁移状态

- `quant.py` → 已变成转发 `analysis/*` 的薄壳，import 时会发 `DeprecationWarning`。
  新代码用 `from analysis import ...`。确认无人引用后可删除。
- `main.py` → 底部有「向后兼容导出」块，让旧的 `from main import ...` 继续可用。
  新代码按层导入；这块可以删。
- `tests/test_main.py` 仍从 `main` 取符号（证明兼容层有效）；
  `tests/test_api.py`、`test_screening.py` 等按新层次导入。

## 启动与验证

```bash
pip install -r requirements.txt
python main.py                 # http://localhost:8000
python -m pytest tests/ -q     # 209 项，全部离线，无需 API Key
```

## 设计来源对照（可乐AI实验室 量化系列）

这一节说明"哪些是作者文档里明确写的，哪些是我根据标题推断的"，
便于后续判断哪些改动需要再校准。作者资料在 `docs/reference/`。

| 期数 | 借了什么 | 落在哪 | 依据强度 |
|---|---|---|---|
| 一 残酷数学真相 | 做空/杠杆期望为负；波动率决定仓位 | `portfolio.leverage_guard`、`signals.gate_factors`、`expectancy` | 标题+简介+章节+部分字幕 |
| 二 遗传算法 | GA 目的不是最高收益；环境/季节/基因三层；10-40-50 配比；精英保留；块级正交交叉；突变斜坡；挑战者不自动晋升；**蒙特卡洛不进 fitness**；MinTradeThreshold | `analysis/search.py`、`signals.Environment/SEASON_MULTIPLIERS`、`risk_rules.no_trade_band` | **作者公开设计文档全文**（`docs/reference/ga_evolution_design.md`） |
| 二 附带彩蛋 | AI 幻觉的低成本对策 → 用分层+可单测的纯函数压缩模型自由发挥的空间 | 本文件通篇 | 标题+简介 |
| 三 删掉 K 线 | 指标堆叠与 if-else 长尾导致状态过载；最悲观原则；"8 年策略跑不赢定投" | `signals.py`（只用 4 个状态量）、`simulation.dca_baseline`（定投作为唯一基准）、各处"信息不足则不动作" | 标题+简介+时间戳（文档为 RAR，未能解压阅读） |
| 四 降次求解 | 泰勒展开、P/V/A 三阶、把绝对价格降成 Boolean、Sigmoid 约束高次 | `analysis/abg.py`（完整公式）、`signals._gate`、`abg.boolean_regime` | **作者公开设计文档全文**（ABG 原理即第四/六期共用） |
| 五 噪声处理 | 噪声分类、降噪、t 统计量要多大才够、参数取值范围、大周期为何"看起来"准 | `validation.information_coefficient`（t_stat）、`validation.walk_forward`、`search.PLATEAU`、`PARAM_SPACE` 边界 | 标题+简介+时间戳 |
| 六 止盈止损 | 出场比进场重要；平仓系统崩盘的巨坑；ABG 决策顺序（止损→止盈/抛物底→持有）；导函数与原函数对齐 | `analysis/risk_rules.py`、`abg.exit_decision`、`FUND_EXIT_REALITY` | **作者公开设计文档全文**（`docs/reference/abg_exit_design.md`） |

刻意没有照搬的三处，以及理由：

1. **ABG 的默认参数**（α=0.60/β=0.20/γ=0.05/H=30/k_stop=5）是给 1 分钟 K 线定的，
   本项目输入是日频公募净值，量纲完全不同。已重设为日频起点值，
   并在 `abg.py` 顶部注明"这不是作者验证过的参数，必须经样本外检验"。
2. **限价挂单退出**在公募基金上不存在（只能按未知净值赎回），
   所以 `risk_rules.FUND_EXIT_REALITY` 把它降级为"择时赎回 + 1 日滑点"，
   并用 `execution_lag_days` 强制执行。
3. **微观 1 分钟动量开火**整层没搬。基金申赎按日结算，搬过来只会制造假精确。
