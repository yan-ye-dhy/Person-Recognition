# 国产多模态大模型影视角色识别 Benchmark（vcrb）

评测 **DeepSeek / Kimi（月之暗面）/ GLM（智谱）/ Qwen（阿里百炼）** 云端视觉模型对**影视角色姓名**（而非演员真名）的识别准确率，为车载影视问答系统选型。

```
截图采集 → 清洗去重 → 人工标注(ground truth) → 匿名化导出
   → 实验 A(纯视觉) / 实验 B(提供作品名) → 自动评分(七类判定+双口径+95%CI)
   → 可视化报告(图表/Markdown/HTML错误浏览器)
```

- **四部固定测试作品**：《狂飙》《庆余年第二季》《功夫女足》（近期电影）《剑来》（国漫），每部目标 40 张、合计 160 张。优先保证身份可靠、角色与场景覆盖；五组各 8 张是参考目标（主角/重要配角/普通配角/复杂画面/多人同框），不足如实报告。
- **红线**：无真实数据不生成虚构结论；付费 API 调用一律 dry-run 优先、`--yes` + 人工授权才执行；素材不足如实报告缺口

2026-11-13 实测已完成：四部各40张，7个模型分别执行A/B两组，共2240个成功接口响应。GitHub可查看 [测试报告](reports/2026-11-13/README.md)、[模型对比CSV](reports/2026-11-13/model_comparison.csv) 和 [160张标注清单](data/final/dataset.json)。完整本机汇总仍在 `results/benchmark_20261113/`，图片审核入口为 `data/selection_reviews/review.html`。接口成功不等于角色识别正确，格式错误、无法识别也计入准确率分母。免费 `glm-4.6v-flash` 因反复返回“模型当前访问量过大”停止后续全量，114项部分测试单列为可用性记录，不参加准确率排名。

同步范围为代码、`reports/2026-11-13/` 和 `data/final/` 的160张图片及原始标注JSON。`.env`、本地视频、客户端缓存、SQLite库和原始API日志保留本地。标注中的 `source_url` 保留原视频文件与时间点以供追溯；运行测试只需随仓库提供的相对 `image_path`，不会尝试访问这些原视频路径。复制项目到新机器后，按下方步骤安装依赖并自行配置API Key即可使用这些样本。

---

## 1. 环境准备

```bash
cd Person-Recognition
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # 含测试、Streamlit 标注页、视频抽帧
cp .env.example .env           # 填入四家 API Key（.env 不会进入 git）
```

- Python ≥ 3.11（开发环境 3.12）
- 视频抽帧可选装 ffmpeg（否则用 `imageio-ffmpeg` 自带二进制）
- McNemar 精确检验可选 `pip install scipy`（不装则自动正态近似并在报告标注方法）

## 2. 模型清单（2026-11 官方文档核实）

| 厂商 | 首批模型（enabled） | base_url | Key 环境变量 |
|---|---|---|---|
| DeepSeek | `deepseek-flash`（唯一视觉模型） | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| Kimi | `kimi-k3.1-pro` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` |
| GLM | `glm-5.3-flash`、`glm-4.6v` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` |
| Qwen | `qwen3.8-max`、`qwen3.7-plus`、`qwen3.8-flash` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |

- **不支持视觉（仅登记，不参与评测）**：`deepseek-v4-pro`、`glm-5.3`、`glm-4.6` 等纯文本模型
- 可选对照模型（`enabled: false`）见 `config/models.yaml`：GLM 免费档 `glm-4.1v-thinking-flash`/`glm-4v-flash`、高速版 `glm-5.3-flashx`、Qwen `qwen3-vl-plus`/`qwen3.8-omni-flash` 等，改 yaml 即可启用
- 每个模型记录：`official_series`（厂商官方定位原文）、`tier`（项目分组标签）、`supports_vision`；`vcrb models verify` 会把"文档确认"升级为"实测确认"（写入 `results/runs.db`）
- 2026-11-13 实测本账号的 `kimi-k2.5`、`kimi-k2.6` 返回404，两旧型号本轮禁用并保留记录。免费 `glm-4.6v-flash` 在部分测试后按实用性偏好排除；当前共7个启用模型，全量A/B基础调用量为160×7×2=2240。
- 本轮可关闭思考的型号统一关闭，K3.1与GLM5.3-Flash使用不可关闭的low思考；K3.1固定温度1，其余接受者用0.1。统一输出总预算4096，思考与正文分开保存，推理文字不参与答案评分。模式差异随run记录，不能视为完全相同的推理配置。

```bash
vcrb models list          # 首批模型
vcrb models list --all    # 含可选与纯文本登记
```

## 3. 完整使用流程

### 3.1 素材采集（每部目标 40 张正式样本）

**优先从已确认作品的正片视频抽帧**。建议每部先准备约 100–150 张候选，分散到多集、不同剧情段和角色；避开片头片尾、演员表和直接给出目标姓名的字幕，再清洗、审核、标注，最后保留 40 张可靠且有代表性的图片。候选池采集上限仍为 150，与正式样本目标分开配置。

获取途径按优先级：可读取的本地正片文件 → 官方公开正片/剧情片段 → 人工确认的正片截图。公开网页可访问不代表视频流一定可读取；付费、登录、地域或加密限制需按实际访问情况处理。官方剧情片段也要排除预告标题卡、采访、花絮和海报，并留意角色覆盖不足。网络图片搜索用于补缺，查询词和可信域不能证明图片来自正片。

```bash
# ① 网络采集（Bing 图片搜索；查询词在 config/works.yaml 维护）
vcrb collect web --work kuangbiao
vcrb collect web --work qingyunian2 --limit 60

# ② 本地视频抽帧（scene=场景切换检测，避免连续重复帧）
vcrb collect video --work qingyunian2 --video /path/to/episode01.mp4 --strategy scene
# 已裁好正片时间段时，也可按间隔抽帧；分多集/片段执行以覆盖角色与场景
vcrb collect video --work qingyunian2 --video /path/to/episode02_body.mp4 --strategy interval --interval 30 --max-frames 40

# ②b 腾讯客户端下载的 HLS 缓存：指向含 .m3u8 的那一集 .hls 目录，非 .url 快捷方式
# 已实测一份 H.266/VVC 缓存可分片抽关键帧；其他格式需先试采，不保证均可读取
vcrb collect cache --work jianlai --cache-dir /path/to/episode.hls --title "剑来 第二季 第04话" --max-frames 10 --out-dir data/cache_probe/jianlai_s2e04
# 试采合格后可去掉 --out-dir，采集到 data/raw/{work}，再 clean / annotate

# ③ 手动投放（网络受阻时把图片放进 data/manual_import/{slug}/ 再登记）
vcrb collect import --work gongfunvzu
```

- 每张图在 `data/raw/{slug}/manifest.jsonl` 记录来源 URL、**来源页 URL 与页面标题**（作品归属核验证据）、素材类型（剧照/截图/抽帧/手动）、获取时间与许可备注
- 网络搜索的素材类型目前由查询词推断，仅作检索标签；`official_still` / `episode_screenshot` 不代表内容已经人工核实。审核时仍须确认是正片画面，不能只看页面标题或来源域。
- `collect cache` 只读取本地分片，不请求索引中的网络地址。默认跳过片头 180 秒、片尾 120 秒，按时间分散抽帧；不缩放图片，并记录来源季集标题、时间点与解码警告。无法解码/空白画面计入失败，不将报错当作成功；生成图片仍需目视检查。不会输出完整 MP4，也不保证读取加密缓存。
- **证据预检默认开启**：仅下载页面标题/URL 含作品标识（如「庆余年第二季」「qingyunian2」）的命中，无证据的通用搜索结果直接跳过（`--no-evidence-check` 可关闭）——防止关键词搜索混入其他作品/无关素材
- 补充通道：`data/url_lists/{slug}.txt`（每行一个 URL，支持 JSON 行带 source_type；人工清单视为已核，不做预检）

### 3.2 清洗

```bash
vcrb clean                 # 全部作品
vcrb clean --work jianlai
```

流水线：文件有效性 → 精确去重(md5) → 感知哈希近重复去重 → 质量筛选（最小尺寸、纯色剔除）→ `data/candidates/{slug}/`（候选以内容 md5 为身份登记进标注库，重跑 clean 重新编号不会丢失已有标注）。

### 3.2b 作品归属核验与素材批量审核（数据质量红线）

关键词图片搜索的候选池会混入其他作品海报、宣传拼贴、漫画/二创等污染素材。正式导出前必须过归属核验：

```bash
vcrb screen run            # 证据规则 + 视觉启发式自动筛查（离线，不调模型 API）
vcrb screen backfill       # 重放搜索词回补「来源页标题/URL」证据后自动重筛（免费搜索）
vcrb screen report         # 六项数字：候选/自动排除/归属核实/待人工/可入评测/缺口
```

判定规则（`vcrb/screening/`，全部离线启发式，不依赖任何大模型）：

| 状态 | 自动判定条件 |
|---|---|
| `verified` | 可信域（works.yaml `trusted_domains`，如 iqiyi/新浪/百科/腾讯视频/豆瓣）**且**页面标题或 URL 含作品标识——只有来源证据能给出该状态，纯视觉信号永远不能 |
| `rejected` | 二创平台拉黑域（pixiv/civitai/lofter）；动画作品中的黑白图（漫画截图嫌疑）。被排除素材**保留原文件与记录，仅隔离**，不再进入正式集，也不会被重新采集复活 |
| `needs_review` | 有作品标识但域不可信 / 可信域无标识 / 竖版海报嫌疑 / 异常画幅 / 插画留白嫌疑 / 疑似拼贴（启发式实测有误报——水墨云雾、贯穿字幕条也会触发，故不自动排除，交人工） |

已知局限（如实声明）：Cosplay、同画风二创、AI 生成图无法离线可靠识别，依赖来源证据与人工批量审核兜底；拼贴检测只对「单方向 ≥2 条等距贯穿分割线」的 3 联以上组图有效。

**Streamlit「素材批量审核」页**（`vcrb annotate` 后第 1 个页面）：
1. 左侧选作品 → 点「▶ 自动筛查本作品」
2. 4 列缩略图网格浏览（每张显示归属徽章 + 首条理由，展开可见页面标题/URL/视觉标记）
3. 勾选图片 → 填操作原因 → 「✅ 确认归属 / 🗑 排除并隔离 / ❓ 设为待人工复核」→ 预览即将更改的缩略图 → 确认执行
4. 全选/清空本页、按状态筛选、分页（24 张/页）批量处理
5. 每次变更（自动或人工）都写入 `attribution_log` 审计表（actor + 原因 + 前后状态），自动筛查不会覆盖人工决定（`--force` 才会）

> 导入门禁：只有 `标注 verified` **且** `归属 verified` 的样本才会进入正式测试集；被排除素材不计入、可随时在审核页翻查。

### 3.3 人工标注（Ground Truth）

```bash
vcrb annotate              # 启动 Streamlit（默认 http://localhost:8501）
```

标注页五个页面：
0. **素材批量审核**：见 3.2b（归属核验，先于标注）
1. **标注**：浏览候选 → 红框框选目标人物（canvas 框选，归一化坐标自动换算）→ 选择角色 → 三维度独立标注（人物重要度/视觉难度/多人同框）→ 状态改为「已确认」保存（默认隐藏已排除素材）
2. **角色名单**：维护别名/演员/重要度（评分时演员名命中 → `答演员名`，不算正确；不确定条目保留 ⚠️待确认）
3. **进度总览**：四作品 × 状态计数 + 归属筛查计数 + 五组配额达成表
4. **人工复核**：评分后对模糊匹配项改判

> 只有 `annotation_status=verified` 且归属核验 `verified` 的样本才会进入正式测试集；有争议的保持「待确认」。

### 3.4 导出与校验

```bash
vcrb export-dataset        # 仅 verified → data/final/{slug}/{slug}_0001.jpg（匿名化+清EXIF）+ dataset.json
vcrb validate-dataset      # schema/图片存在性/配额达成/文件名防泄露检查
```

正式集数量读取 `config/settings.yaml` 的 `export.target_per_work: 40`、`target_per_group: 8`。导出先覆盖五组参考目标，再用其他已通过双门禁的样本补足总量，每部最多 40 张；不足时记录真实数量和缺口。精简筛选会将未入选图及其原标注归档，当前候选池仅保留选定样本，避免重导出时换图。

2026-11-13 最终集为四部各40张、共160张，均由assistant按用户授权代标。独立对照原片连续场景、带姓名角色海报后更正了17项原角色标签；另2项无法确认身份，归档并用2张已核实的本地正片帧替换。158张原输入图像与目标框未变，沿用已有真实响应；仅替换帧补测，原标签和全部来源保留审计。当前审核入口为 `data/selection_reviews/review.html`，最终编号与标注库PK对应见 `data/selection_reviews/final_review_20261113/`。本次付费输入快照已冻结，不要用常规export重排编号覆盖；当前候选/原图另保留2项uncertain供溯源，不计入正式40张。狂飙覆盖两集素材中的8个角色，并非全剧覆盖；旧版300张审核及初选记录均保留为历史版本。

### 3.5 执行评测（分级验证，每级需你的明确授权）

```bash
# ① 连通性检查：每模型 1 张小图（确认 kimi-k3.1-pro 中国站可调用等）
vcrb models verify --yes

# ② 小规模预检：每作品 2 张（8 张/模型/实验）
vcrb run --small  --experiment both --yes

# ③ 正式预检：每作品 10 张（40 张 × 模型 × A/B）
vcrb run --pilot  --experiment both --yes

# ④ 完整测试：全部样本
vcrb run --experiment both --yes
```

- **默认 dry-run**：不加 `--yes` 只打印执行计划与预计调用量，不产生费用
- 中断（Ctrl-C）后使用原命令加 `--run-id 原runID` 续跑；已有成功记录不重复调用。必须保留原筛选参数，样本哈希不一致会拒绝续跑；small/pilot/full各自保存独立数据集快照，评分只计算该次选中的样本。
- `settings.yaml` 的 `runner.concurrency` 控制有界并发，`per_model_concurrency` 可覆盖单型号上限；Kimi充值后实测使用4并发、无新增限流，免费GLM留有单并发设置但默认禁用。请求返回后由主线程立即保存，Ctrl-C停止新请求并收回已发请求的结果。
- 常用过滤：`--models qwen3.8-max,glm-5.3-flash`、`--vendors Kimi`、`--works kuangbiao`、`--experiment A`、`--retry-failed`
- 公平性：所有模型同一 dataset 快照、同一红框画法、同一提示词（B 组仅多一句作品名）；请求不带 tools/联网搜索；run 元数据记录各模型 `response.model` 实际版本与参数差异

### 3.6 评分、复核与报告

```bash
vcrb score                       # 评分（只读原始响应，可反复重评，零费用）
vcrb review list                 # 查看待复核（相似但未精确命中）
vcrb review set --run-id run_xxx --model m1 --experiment A \
     --sample kuangbiao_0007 --outcome correct --note "确认是别名"   # 改判后重跑 score 生效
vcrb report                      # 图表×6 + Markdown + HTML 错误案例浏览器
```

结果位置：`results/scores/{run_id}/`（metrics.csv / detail.csv / summary.json）、`results/reports/{run_id}/`（report.md、6 张 PNG、errors.html）。本次最终ID为 `run_20261113_final_practical`，组合记录逐条保留源请求时间、原响应和来源运行；组合过程不新增API调用。

评分七类：`correct / wrong_name（错误识别）/ actor_name（答演员名）/ uncertain（无法识别）/ refused（拒答）/ parse_error（格式错误）/ api_failure（API失败，与拒答严格区分）`。

双口径指标（网络故障不扭曲模型评价）：
- **核心选型指标**：有效响应识别准确率 = correct / (计划样本 − API失败 − 未执行)
- 并列展示：端到端正确率 = correct / 计划样本；API 失败率单列
- 附 95% Wilson 置信区间、A/B 配对 McNemar 检验、按角色统计 + 去重角色宏平均

## 4. 命令速查

| 命令 | 作用 |
|---|---|
| `vcrb status` | 四作品数据进度 + 归属筛查状态 + 最近运行 + 连通性验证结果 |
| `vcrb models list [--all]` / `models verify --yes` | 模型注册表 / 实测视觉能力 |
| `vcrb collect web\|video\|import` | 三通道素材采集（web 默认证据预检） |
| `vcrb roster show --work …` / `roster build` | 角色名单查看 / 生成模板 |
| `vcrb clean [--work]` | 清洗去重 |
| `vcrb screen run\|backfill\|report` | 归属核验：自动筛查 / 证据回填 / 六项数字报告 |
| `vcrb annotate` | Streamlit 标注页（含素材批量审核） |
| `vcrb export-dataset` / `validate-dataset` | 匿名化导出（归属+标注双门禁）/ 校验 |
| `vcrb run …[--yes]` | 执行评测（幂等断点续跑） |
| `vcrb score` / `report` / `review` | 评分 / 报告 / 复核 |

## 5. 运行测试

```bash
pytest                 # 40+ 离线用例：配置/模型注册/Mock调用/图片校验/去重/数据集/
                       # Prompt/JSON解析/别名匹配/评分统计/断点恢复/导出/报告
```

## 6. 目录说明

```
config/          models.yaml(模型注册) works.yaml(四作品) settings.yaml rosters/(角色名单)
vcrb/            主包：providers/ collectors/ dataset/ annotation/ runner/ evaluation/ reports/
data/            final/最终160图及标注可同步；raw、candidates、annotations.db保留本地
results/         运行时(gitignored)：runs.db raw/(每次请求JSONL) scores/ reports/
reports/         可移植测试报告、统计图、CSV及轻量修标摘要
tests/           pytest 离线测试
```

## 7. 已知限制与说明

- 网络采集受来源反爬影响时，用 `url_lists`/手动投放补足；**不会**用其他作品替代或伪造素材
- 《功夫女足》本次覆盖6个已独立核对的角色；各部样本覆盖有限，不能将这160张的成绩直接视为全剧或所有电影的准确率。
- K3.1固定温度1且不能关闭思考，GLM-5.3-Flash也不能关闭思考；两者均采用官方支持的low力度。这些差异如实记录在run元数据与报告中，不虚构等价。
- 图表中文字体依赖系统字体（Noto Sans CJK/文泉驿等）；缺失时中文可能显示为方块，不影响数据
