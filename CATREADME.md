AlphaGPT 仓库速读

当前仓库是“纯 A 股多因子量化研究与模拟盘”系统，核心思路沿用原版：用模型自动生成可解释的因子公式，通过回测评分训练公式生成器，再把高分公式接入模拟撮合与组合管理。原加密（Solana meme）链路已归档到 `_obsolete_crypto/`。

代码组织（按功能划分）
- ashare_data/：数据层。AkShare 拉取交易日历/股票列表/指数成分/日线，DuckDB + Parquet 本地存储，清洗复权、股票池过滤。
- ashare_model/：策略挖掘。把行情转成因子（factors），定义算子语言（ops）与公式词表（vocab），StackVM 解释执行，Transformer（LoopedTransformer + MTPHead）生成公式 token 序列，REINFORCE + value baseline 训练，回测评分。
- ashare_trading/：模拟盘。券商撮合（涨跌停/停牌/T+1/整手/费用）、组合管理、风控过滤、日频运行器。
- dashboard/：Streamlit 看板，展示回测净值/基准、选股快照、模拟盘状态与数据状态。
- lord/：独立研究实验（低秩衰减/grokking），不参与主流程。
- _obsolete_crypto/：原加密链路与 times.py 遗留脚本归档。

主流程（从数据到模拟盘）
1) ashare_data.sync 拉取并入库（日历/股票/成分/日线，Parquet 缓存）
2) ashare_model.train 训练生成最优公式（best_ashare_strategy.json + ashare_model.pt）
3) ashare_model.backtest 用公式回测，输出净值/基准/持仓快照（backtest_result.json）
4) ashare_trading.run_sim 按日生成订单，撮合成交，更新组合状态
5) dashboard 展示回测与模拟盘

核心思想
- 不是直接预测价格，而是“生成公式 → 解释执行 → 回测评分 → 优化生成器”。
- 公式 = token 序列；token 由“特征 + 算子”组成，StackVM 执行成因子信号。
- 交易层只消费最终信号分数，负责现实规则（涨跌停、停牌、T+1、整手、费用）与风控。

当前因子与算子一览
- 特征（50 个，分代追加，v1 token id 永不偏移）：v1（34 个）动量/反转/波动（RET_1/5/10/20、VOL_20/60、MOMENTUM_20/60、REVERSAL_5、SKEW_20、KURT_20）、量价（TURNOVER、TURNOVER_CHG、VOLUME_RATIO、VOLUME_IMPACT、AMPLITUDE、CLOSE_POSITION）、事件（LIMIT_UP/DOWN_EVENT）、基本面（PE/PB/PS/ROE/ROA/毛利率/净利率/营收与利润增速/负债率/市值/股息率，暂无数据源，全中性占位）、暂中性占位（NORTHBOUND_CHG、MARGIN_BALANCE_CHG、INDUSTRY_MOMENTUM）；v2（16 个，全部本地计算，无新接口依赖）日内分解（OVERNIGHT_RET、INTRADAY_RET）、流动性（ILLIQ_20 阿米胡德、AMOUNT_SHARE 成交额占比）、彩票/锚定（MAX_20、HIGH_52W 52 周新高距离）、风险回归（BETA_60、IVOL_60、RSQ_60，对全市场等权组合做滚动 CAPM）、技术（BIAS_20 均线距离、RSI_14、ATR_14、MACD_DIF/DEA）、微观结构（SUSPEND_DAYS_60、LIST_AGE）。因子以元数据注册表（家族/所需列/预热期）驱动，新增因子=加一条注册+测试。
- 算子（18 个）：ADD/SUB/MUL/DIV/NEG/ABS/SIGN/GATE/JUMP/DECAY/DELAY1/MAX3/DELTA5/MA20/STD20/TS_RANK20/CORR20/DOWNVOL20，全部无未来泄漏（窗口算子用可用历史，除零有保护，非有限结果归零）。

现状与依赖（实话版）
- 只需 Python + AkShare 网络访问，无数据库服务依赖；配置在 config/ashare_config.yaml，密钥可选 config/.env。
- 已知局限：成分股为当前快照（幸存者偏差）；北向/两融/行业因子中性占位；基本面 12 特征全中性（旧快照接口已失效并移除，point-in-time 回填在路线图中）；离线日历用工作日近似。
- 词表版本化：训练产物带 feature_names/feature_version，加载时按名称重映射；无元数据的旧公式对照 v1 首发词表（34 特征/16 算子）重映射，语义永不漂移。
- best_ashare_strategy.json 需先训练或提供，仓库默认不带。

Takeaway（可对外复述）
这不是一套“预测模型”，而是一个“自动写因子的系统”：Transformer 生成公式，回测奖励（含交易成本）训练生成器，StackVM 保证公式可解释、可执行；研究层与模拟交易层分离，交易层严格实现 A 股现实规则。
