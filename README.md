# 国产多模态大模型影视角色识别 Benchmark（vcrb）

评测 **DeepSeek / Kimi（月之暗面）/ GLM（智谱）/ Qwen（阿里百炼）** 云端视觉模型对**影视角色姓名**（而非演员真名）的识别准确率，为车载影视问答系统选型。

```
截图采集 → 清洗去重 → 人工标注(ground truth) → 匿名化导出
   → 实验 A(纯视觉) / 实验 B(提供作品名) → 自动评分(七类判定+双口径+95%CI)
   → 可视化报告(图表/Markdown/HTML错误浏览器)
```

- **四部固定测试作品**：《狂飙》《琅琊榜》《功夫女足》（近期电影）《剑来》（国漫），每部目标 100 张、五组采样配额各 20 张（主角/重要配角/普通配角/复杂画面/多人同框）
- **红线**：无真实数据不生成虚构结论；付费 API 调用一律 dry-run 优先、`--yes` + 人工授权才执行；素材不足如实报告缺口

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
| Kimi | `kimi-k3.1-pro`、`kimi-k2.6`、`kimi-k2.5` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` |
| GLM | `glm-5.3-flash`、`glm-4.6v`、`glm-4.6v-flash`(免费) | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` |
| Qwen | `qwen3.8-max`、`qwen3.7-plus`、`qwen3.8-flash` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |

- **不支持视觉（仅登记，不参与评测）**：`deepseek-v4-pro`、`glm-5.3`、`glm-4.6` 等纯文本模型
- 可选对照模型（`enabled: false`）见 `config/models.yaml`：GLM 免费档 `glm-4.1v-thinking-flash`/`glm-4v-flash`、高速版 `glm-5.3-flashx`、Qwen `qwen3-vl-plus`/`qwen3.8-omni-flash` 等，改 yaml 即可启用
- 每个模型记录：`official_series`（厂商官方定位原文）、`tier`（项目分组标签）、`supports_vision`；`vcrb models verify` 会把"文档确认"升级为"实测确认"（写入 `results/runs.db`）

```bash
vcrb models list          # 首批模型
vcrb models list --all    # 含可选与纯文本登记
```

## 3. 完整使用流程

### 3.1 素材采集（每部目标 ≥100 张候选）

```bash
# ① 网络采集（Bing 图片搜索；查询词在 config/works.yaml 维护）
vcrb collect web --work kuangbiao
vcrb collect web --work langyabang --limit 60

# ② 本地视频抽帧（scene=场景切换检测，避免连续重复帧）
vcrb collect video --work langyabang --video /path/to/episode01.mp4 --strategy scene

# ③ 手动投放（网络受阻时把图片放进 data/manual_import/{slug}/ 再登记）
vcrb collect import --work gongfunvzu
```

- 每张图在 `data/raw/{slug}/manifest.jsonl` 记录来源 URL、**来源页 URL 与页面标题**（作品归属核验证据）、素材类型（剧照/截图/抽帧/手动）、获取时间与许可备注
- **证据预检默认开启**：仅下载页面标题/URL 含作品标识（如「琅琊榜」「langyabang」）的命中，无证据的通用搜索结果直接跳过（`--no-evidence-check` 可关闭）——防止关键词搜索混入其他作品/无关素材
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
- 中断（Ctrl-C）后**重跑同一命令自动续传**（`--run-id` 可指定续跑）；已成功记录绝不重复付费调用
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

结果位置：`results/scores/{run_id}/`（metrics.csv / detail.csv / summary.json）、`results/reports/{run_id}/`（report.md、6 张 PNG、errors.html）。

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
data/            运行时(gitignored)：raw→candidates→final + annotations.db
results/         运行时(gitignored)：runs.db raw/(每次请求JSONL) scores/ reports/
tests/           pytest 离线测试
```

## 7. 已知限制与说明

- 网络采集受来源反爬影响时，用 `url_lists`/手动投放补足；**不会**用其他作品替代或伪造素材
- 《功夫女足》角色名单资料暂缺（`config/rosters/gongfunvzu.yaml` 待调研补充），补齐并人工确认前该作品不导出正式样本
- Kimi k2.5/k2.6 官方不接受显式采样参数、GLM-5.3-Flash 无法关闭 thinking —— 这些差异如实记录在 run 元数据与报告中，不虚构等价
- 图表中文字体依赖系统字体（Noto Sans CJK/文泉驿等）；缺失时中文可能显示为方块，不影响数据
