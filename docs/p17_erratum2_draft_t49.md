# P17 勘误草案 #2（t49：industry_membership_pit 三级展开映射 + source_row_hash 粒度）——待用户批准，未生效

- 性质：**勘误草案**（第二次）。只起草、不修改 p17 正文；批准后由获授权任务
  按 p13 t46/t57 / p17 t44 追认式模式回写（一次原子 commit）。
- 起草：governance-auditor（AgentTeams tushare-pit-migration t49），2026-09-04。
- 触发链：t9 step5-membership（index_member_all 回填）被 p17 §15.6 拦截——
  p17 §5.2 假设源行携带单一 `index_code`，探针与实测证明源行为 l1/l2/l3
  三级同行结构、**无 index_code 列**。按 §15.6（"来源字段无法确认 → 停下
  修订本文，禁止猜测"）走勘误，禁止 D2/D3 静默换映射。
- 证据权威与事实来源：
  1. **探针报告**（`D:\minequant\tushare_probe\probe_report_d1.txt`，
     2026-09-04T10:15:39，23/23 OK）——`[index_member_all] status: ok /
     rows: 126 / columns (11): l1_code, l1_name, l2_code, l2_name,
     l3_code, l3_name, ts_code, name, in_date, out_date, is_new /
     range in_date: 19930430 .. 20260701`（逐字）。
  2. **展开映射提案**：data-engineer（t9 step5 被 §15.6 拦截后提出，
     captain 转达为草案主体）。
  3. **t47 实现的批内预去重机制**：按 `source_row_hash` keep-first
     （D2 实现事实，captain 转达；t45/D2 复审复核）——构成勘误项 2 的
     坍缩风险依据。
- 勘察基线：分支 codex/g0-tushare-pit-contract @ `5d05987`（p17 已含 t44
  勘误：附录 B 四行 + 附录 C）。

## 勘误项 1：industry_membership_pit 源行三级展开映射

- **p17 原文**（§5.2 表 8，逐字）：

```sql
-- 8. 申万行业成员 PIT（index_member_all；区间 [in_date, out_date)，
--    out_date 为空 = 仍在任；约束见 §14）
CREATE TABLE IF NOT EXISTS industry_membership_pit (
    index_code VARCHAR NOT NULL, ts_code VARCHAR NOT NULL,
    in_date VARCHAR NOT NULL,
    out_date VARCHAR, is_new VARCHAR NOT NULL,
    industry_name VARCHAR NOT NULL, available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (index_code, ts_code, in_date)
);
```

- **探针实证**：源行 11 列 = l1_code, l1_name, l2_code, l2_name, l3_code,
  l3_name, ts_code, name, in_date, out_date, is_new——**无 index_code**；
  行业三级（L1/L2/L3）的 code+name 同行携带。
- **展开映射规则（钉死，data-engineer 提案 + 本草案细化）**：一行源数据
  展开为**最多 3 行契约行**：
  1. `level=1`：index_code = l1_code，industry_name = l1_name；
     `level=2`：index_code = l2_code，industry_name = l2_name；
     `level=3`：index_code = l3_code，industry_name = l3_name；
  2. 某级 code 与 name 任一为空 → **跳过该级**（不产出空行业行）；
  3. in_date / out_date / is_new **各级沿用源值**（L1 原则，同 t44 项 2
     对 is_new 的处理：真实源字段留在成员表）；
  4. available_date 按 §6.1 由 in_date 推导（in_date 当日 23:59 → 下一
     开市会话）；同一源行展开的 2–3 行 available_date 相同；
  5. 展开行沿用 t44 后的写入语义（KEYED：`ON CONFLICT DO NOTHING` +
     hash 冲突整批失败）。
- **DDL 连带修正（批准后执行）**：industry_membership_pit **增列
  `level INTEGER NOT NULL`**——展开判别符，非源列；本勘误明确授权的
  唯一派生列（理由：它承载展开本身的信息，不入列则 L2/L3 行不可区分、
  D5 的层级过滤无从落地）；`name`（源列，股票名称）为 §5.0 additive
  可选追加，由 D2 deliberate extension 决定。
- **核对 ①：PK (index_code, ts_code, in_date) 展开后仍成立**——同一源行
  展开的 2–3 行 index_code 互异（SW 分类不同层级不同码），PK 无自冲突；
  跨源行同键 = KEYED `DO NOTHING` + hash 冲突整批失败（修订走人工裁决，
  禁止覆盖）。防御性说明：若源出现同码跨级病态行，KEYED 冲突 fail-closed
  拦截、不静默合并；PK 不扩大（维持 t44 后文本，避免身份语义二次变更）。
- **核对 ②：§14.3 同股同层同日至多一个有效行业**——展开后"同层"即
  同 level；§14.3 校验按 level 分组执行，规则文本零修改；发现重叠 →
  红线 §15.3 停止不变。§6.3 的 as-of 规则（in_date ≤ s < out_date）
  亦零修改（"同一层级"经展开后即 level 分组）。

## 勘误项 2（硬性）：source_row_hash 粒度 = 展开后行，不是源行

- **p17 原文**（§4.3）：source_row_hash = 对行 canonical 序列化（键排序、
  UTF-8、值原样、不含 token/账户信息）后 SHA-256；append-only 幂等与
  修订检测的唯一依据。
- **坍缩风险（实测机制，必须在本勘误钉死）**：t47 实现的批内预去重按
  `source_row_hash` **keep-first**——同一源行的 3 个展开行若以源行 11 列
  为哈希输入，则三行哈希相同 → keep-first 坍缩为 1 行 → **L2/L3 成员数据
  静默丢失**（隐性数据丢失，违反 §4.3 append-only 完整性与 §1 原则 1）。
- **钉死规则**：source_row_hash = SHA-256(canonical(**展开后契约行**全部
  列))；canonical 输入**显式含 level 判别符**（同源行的 2–3 行哈希必异，
  即使病态同码场景也可区分）；**禁止以源行 11 列为哈希输入**。展开行哈希
  粒度 = 展开后行，不是源行。
- **落点**：§4.3 补一句（展开映射场景的哈希输入定义）；§5.2 表 8 注释
  补一行；附录 B 对应行备注。

## 范围外顺带核对 ④：SW 双代际（SW2021/SW）在 taxonomy 侧的取值确认

- 可验证现状：index_classify 源含 `src` 列（t44 后 DDL 已登记，探针 7 列
  之一）；但探针报告仅留档列名与行数（[index_classify] rows: 31），**未含
  样本值**——`src` 的取值分布（SW2014/SW2021 的标记形态）及其与 index_code
  代际前缀的对应关系，当前**无留档证据可确认**。
- 预注册确认步骤（不猜测，§15.6 同纪律）：D3 首批真实拉取时，对
  index_classify 全量行落档 (index_code, level, src, is_pub, parent_code)
  分布到测量日志；taxonomy 代际版本记录 = (index_code 代际 + src) 组合，
  衔接 §14.3"2021 版分类切换显式记录 taxonomy 代际版本"的既有要求。
- 结论：本勘误**不钉死 src 取值**（无证据不猜测），只预注册确认步骤；
  §14.3 的代际记录要求不变。

## 范围外顺带核对 ⑤：与附录 B 既有四行的衔接

- 附录 B 现有四行均为 t44（proc / rqchl / suspension / is_new），本勘误
  新增行以 **t49** 标识追加，不触碰 t44 行；表结构（勘误/表/源列/契约列/
  备注）不变；t49 行为展开映射形态（一行源列组 → 三个契约列组合，备注
  注明 level=1/2/3 与空级跳过），与 t44 的单列改名行并存无冲突。
- 附录 C 勘误记录追加 t49 段（引用本草案 + 用户批准原文 + 表决范围）。

## 修正位置清单（批准后由获授权任务执行；本草案不预写结果）

1. §5.2 表 8 DDL：增 `level INTEGER NOT NULL` 列 + 注释（三级展开映射、
   空级跳过）；
2. §4.3：哈希输入定义补句（展开映射场景 = 展开后行 canonical 含 level，
   禁止以源行为输入）；
3. §6.1 industry 行 / §6.3 / §13 index_member_all 行：措辞核对本（§6.1/
   §6.3 规则本体零修改；§13 行可补"源行展开 ≤3 契约行"备注）；
4. 附录 B：追加 t49 展开映射行；
5. 附录 C 勘误记录：追加 t49 段（批准原文逐字 + 表决范围）。

## 表决范围建议与未生效声明

- **建议项 1 + 项 2 捆绑表决**（拆开会产生中间态错误：有展开无哈希钉死
  = 数据丢失；有哈希钉死无展开 = 无意义），范围外 ④/⑤ 为核对记录随批
  落盘；执行任务批准后另派（p17 文件单 commit，CONTRACTS.md 不动）。
- **未生效声明**：本草案未修改 p17 任何正文；批准前 p17 现行文本继续
  有效，t9 step5-membership 继续 §15.6 停止等待，禁止静默换映射。
