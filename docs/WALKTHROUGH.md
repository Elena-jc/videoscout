# VideoScout 讲解：从代码学会 Agent 的关键知识点

这份文档按「概念 → 在本项目哪一行 → 面试会怎么问」的顺序组织。建议读的时候把对应文件打开对照着看。

---

## 0. 一句话讲清这个项目

> 长视频问答如果把几十帧均匀抽样一股脑塞给 VLM，很容易漏掉关键瞬间，而且又贵又慢。VideoScout 让 LLM 像侦探一样工作：先用**混合检索**定位可疑片段，用**开放词汇检测**按文字描述找任意物体，再用**视觉子 agent** 放大去看，必要时用 **SQL 查目标轨迹**做计数和时序推理；答案交给一个**独立的验证器**核对证据，全过程在**代码强制的预算**内完成。

数据流：

```
离线建索引（一次）                        在线回答（每个问题）
video ─┬─ 切成 10s 片段 (chunk)            planner (LLM) ──工具调用──► tools 节点（预算在代码里强制）
       ├─ SigLIP 2 关键帧向量 (稠密索引)      ▲                               │
       ├─ YOLO26 + ByteTrack → SQLite       └──── tool_result + 预算 ◄──────┘
       └─ 字幕 / 物体标签 / 标题 → BM25      │ submit_answer          （find_objects：YOLOE-26 按文字现场检测）
                                            ▼
                                          verifier（全新上下文）── 驳回+反馈 ──► planner
                                            │ 通过
                                            ▼
                                          答案 + 校准后的置信度
```

## 1. 推荐阅读顺序

1. [react_minimal.py](../videoscout/agent/react_minimal.py)：50 行的裸 agent 循环。**先把它读懂**，后面的内容都是在它上面加东西。
2. [graph.py](../videoscout/agent/graph.py)：同一个循环用 LangGraph 重写，加上预算、验证器、兜底和追踪。
3. [tools/](../videoscout/tools)：三个工具，以及它们怎么变成 MCP 服务。
4. [retrieval/hybrid.py](../videoscout/retrieval/hybrid.py)、[index/](../videoscout/index)：视频 RAG 的底座。
5. [llm.py](../videoscout/llm.py)：生产级 LLM 调用的细节。
6. [eval/](../videoscout/eval)：怎么证明它比 baseline 好。
7. [tests/test_graph.py](../tests/test_graph.py)：用假模型测 agent，也是学 tool-use 协议最直观的地方。

---

## 2. Agent 循环的本质（ReAct / Tool Calling）

**概念**：agent = 一个循环。模型读完整的对话记录 → 决定调用哪些工具 → 你的代码执行工具 → 把结果作为新消息追加 → 再让模型读。直到模型调用一个「终止工具」（这里是 `submit_answer`）。

**代码**：[react_minimal.py:25](../videoscout/agent/react_minimal.py#L25)

必须记住的几条协议细节，面试常考，也是新手最常踩的坑：

| 规则 | 为什么 | 代码位置 |
|---|---|---|
| assistant 的 `response.content` **原样**追加回历史，包括 thinking block | thinking block 带签名，只在产生它的那段历史里有效；改动它会导致缓存失效，甚至 400 错误 | [llm.py:32](../videoscout/llm.py#L32)（`to_dict()` 只保留 API 实际返回的字段） |
| 每个 `tool_use` 在下一条 user 消息里都必须有对应的 `tool_result`（按 id 匹配） | API 强制要求 | [conftest.py](../tests/conftest.py) 里的 `check_protocol` 在每次假调用时都会检查 |
| 一轮里的多个并行调用，结果放进**同一条** user 消息 | 拆成多条消息，会让模型逐渐学会不做并行调用 | [graph.py:234](../videoscout/agent/graph.py#L234) |
| 工具出错时返回 `is_error: true` 和错误说明，不要抛异常 | 让模型看到错误并自己纠正，例如改参数重试 | [registry.py](../videoscout/tools/registry.py) 的 `Tool.run` |
| 先检查 `stop_reason`：`refusal` 或 `max_tokens` 时不执行工具 | 被截断的 tool_use 也可能解析成「合法但残缺」的 JSON | [llm.py:162](../videoscout/llm.py#L162) |

**面试题**：「Function calling 和 agent 有什么区别？」
答：function calling 是单次能力，模型输出一个结构化的调用请求。agent 是把它放进循环，**由模型决定下一步做什么、什么时候停**。我奇瑞实习做的是前者加上多步查询；这个项目做的是后者，外加预算、验证和评测。

---

## 3. 为什么用 LangGraph：Workflow vs Agent

**概念**：Anthropic 的《Building effective agents》区分了两类系统：
- **Workflow**：控制流由代码预先写死，例如固定的「检测 → 提案 → VLM 验证」，就是你中科院实习做的 pipeline。
- **Agent**：控制流由模型动态决定。

这个项目是两者混合：**「去哪看、看几次」交给模型决定；预算、验证、兜底这些硬约束由代码保证**。LangGraph 适合的正是这种场景：把循环拆成节点和条件边，每条出口都可以单独测试、记录和做消融。

**代码**：[graph.py](../videoscout/agent/graph.py) 顶部的状态图；节点定义见 [graph.py:124](../videoscout/agent/graph.py#L124)（agent）、[:171](../videoscout/agent/graph.py#L171)（tools）、[:273](../videoscout/agent/graph.py#L273)（verify）、[:344](../videoscout/agent/graph.py#L344)（force_answer）。

关键知识点：
- **State + Reducer**：[state.py:17](../videoscout/agent/state.py#L17) 里 `messages: Annotated[list, operator.add]` 表示节点返回的消息会被**追加**，而不是覆盖。这样从结构上保证了对话记录只增不改（append-only）。这件事同时关系到 prompt caching（缓存按前缀匹配）和 thinking block 的有效性。
- **条件边**：`route_after_agent` 根据最后一条消息决定下一步：去执行工具、去验证、提醒模型（nudge），还是强制作答。
- **为什么不用 SDK 自带的 Tool Runner**：Tool Runner 适合「工具随便调，模型说停就停」的场景。我们需要在代码里强制预算、在另一个上下文里做验证、为失败设计兜底出口，这些都需要自己掌控循环。
- **Checkpointer**：`build_graph(deps, checkpointer=...)` 预留了接口。接上 `InMemorySaver` 就能支持中断恢复、人工介入（human-in-the-loop）、回放调试。可以作为扩展方向来讲。

**面试题**：「为什么不直接写个 while 循环？」
答：while 循环能跑，但控制流有多个出口（验证通过、强制作答、提醒），还有代码强制的预算。拆成图之后，每个节点能单独测试（见 test_graph.py），也能单独关掉做消融（`--set agent.verify=false`）。

---

## 4. 工具设计（Tool Design）

**概念**：工具就是 agent 的「API」。模型只能通过描述和 schema 理解工具，所以**工具描述就是 prompt**。

**代码**：[registry.py](../videoscout/tools/registry.py)、[search.py](../videoscout/tools/search.py)、[inspect.py](../videoscout/tools/inspect.py)、[tracks_sql.py](../videoscout/tools/tracks_sql.py)

知识点：
1. **单一事实源**：每个工具的参数是一个 pydantic 模型。同一个模型既生成发给 API 的 JSON Schema（[schema.py:52](../videoscout/schema.py#L52)），又负责校验模型传回来的参数。
2. **Strict tool use**：`strict: true` 保证参数符合 schema。但 strict 模式只支持 JSON Schema 的一个子集，比如不支持 `minimum`/`maximum`，所以 schema.py 会去掉这些约束，数值范围改在代码里做截断（例如 `num_frames` 最多 8 帧）。**Claude Opus 5.5 不支持强制 tool_choice**（`any`/`tool` 会返回 400），所以只能靠 prompt 引导加 strict 来保证模型最终调用 `submit_answer`，没调用时由 nudge 节点提醒。
3. **描述怎么写**：说明「什么时候用」「返回什么」「有什么坑」。例如 `query_tracks` 的描述直接告诉模型：track ID 会切换，所以 `COUNT(DISTINCT track_id)` 会多算；低置信度的短轨迹往往是重复框。
4. **工具输出是数据，不是指令**：字幕和标题来自视频本身，可能被人为注入「忽略之前的指令……」之类的文字。system prompt 和工具描述里都明确说明了这一点。这是 **prompt injection** 防护，面试加分点。
5. **为每个工具设计清晰的错误**：`inspect_clip` 窗口超出视频长度、`track_id` 不存在时，都返回可以据此行动的错误说明（「用 query_tracks 查它什么时候出现」）。

---

## 5. 视频 RAG：切分、稠密检索、稀疏检索、融合、去冗余

这一节完整对应帖子里的「文档处理、切分、向量检索、混合检索、Rerank」，只是把「文档」换成了视频。

| 文本 RAG | 本项目 | 代码 |
|---|---|---|
| chunk | 10 秒片段 | [build.py:21](../videoscout/index/build.py#L21) |
| embedding | SigLIP 2 关键帧向量（每 2 秒一帧） | [embedder.py](../videoscout/index/embedder.py) |
| 稀疏检索 | 自己实现的 BM25，文本来源是字幕、物体标签、标题 | [bm25.py](../videoscout/retrieval/bm25.py) |
| 混合检索融合 | RRF | [hybrid.py:38](../videoscout/retrieval/hybrid.py#L38) |
| 去冗余 / rerank | 以时间为相似度的 MMR | [hybrid.py:47](../videoscout/retrieval/hybrid.py#L47) |
| query 改写 | 模型分别写 `query`（关键词）和 `visual_query`（画面描述） | [search.py](../videoscout/tools/search.py) |

要能讲清楚的细节：
- **Chunk 大小的权衡**：片段太长，一个向量会把几个事件糊在一起；太短，上下文又不够。事件跨越片段边界的问题靠两点缓解：`inspect_clip` 的时间窗口可以任意指定，MMR 会看到相邻片段。
- **MaxSim（late interaction）**：一个片段的得分取它**最匹配的那一帧**，而不是所有帧向量的平均值，因为取平均会把短暂出现的目标稀释掉。见 [hybrid.py:88](../videoscout/retrieval/hybrid.py#L88)。
- **SigLIP 2 的一个坑**：文本要转成小写，并且 `padding="max_length", max_length=64`，和训练时一致，否则检索质量会悄悄下降（见 [embedder.py](../videoscout/index/embedder.py) 的 `embed_text`）。面试时可以顺带讲 sigmoid loss 和 CLIP 的 softmax loss 的区别，以及 SigLIP 2 相比第一代加了什么（多语言、带定位和密集特征的训练目标）。实测：用中文「写着字的白色牌子」搜索，第一名就是标牌所在的场景。
- **BM25 公式**写在 [bm25.py](../videoscout/retrieval/bm25.py) 的 docstring 里。k1 控制词频饱和（出现第 5 次几乎不再加分），b 控制长度归一化。建议能手写出来。
- **RRF 为什么好**：余弦相似度和 BM25 分数不在同一个量纲，没法直接相加。RRF 只用排名，`score = Σ 1/(k + rank)`，不需要做分数校准。
- **RRF 的代价，这是真实观察到的现象**：在演示视频上搜「标牌」，第 1 条结果是对的（80–90 秒，视觉和字幕两路都命中），但第 2、3 条被 MMR 推到了其他场景，而不是同样拍到标牌的 90–120 秒。原因是 RRF 丢掉了分数的大小，视觉第 1 到第 5 名融合后只差约 7%，相关性差距太小，时间冗余惩罚就占了上风。λ（`retrieval.mmr_lambda`）和 τ（`mmr_tau_seconds`）就是调这个取舍的旋钮，适合写进消融实验。面试时讲出「我观察到 X，原因是 Y，用 Z 调」比只说「我用了 RRF」有说服力得多。

---

## 6. 结构化记忆 + Text-to-SQL 护栏

**概念**：长期记忆不一定是向量库。对「几个人」「谁停留最久」「谁先出现」这类问题，结构化表加 SQL 远比向量检索准确。

**代码**：[tracker.py:82](../videoscout/index/tracker.py#L82) 为每条轨迹计算时长、路径长度、净位移、平均面积。徘徊检测的思路就是「时长长、净位移小」，正好接上你中科院实习做的 loitering 检测。

**五层护栏**（[tracks_sql.py](../videoscout/tools/tracks_sql.py)，和奇瑞实习的「只读 API + 参数校验」一脉相承）：
1. 文件以 `mode=ro` 只读方式打开（[store.py:156](../videoscout/index/store.py#L156)）：操作系统层面就写不了；
2. SQLite authorizer 只放行 SELECT / READ / FUNCTION（[tracks_sql.py:26](../videoscout/tools/tracks_sql.py#L26)），PRAGMA、ATTACH、DROP 在编译阶段就被拒绝；
3. 每次只执行一条语句，`SELECT 1; DELETE ...` 这种拼接会直接报错；
4. progress handler 超时中断（[tracks_sql.py:47](../videoscout/tools/tracks_sql.py#L47)），防止递归 CTE 死循环；
5. 结果最多返回 50 行，防止一次查询把上下文撑爆。

每一层都有对应测试：[test_sql_guard.py](../tests/test_sql_guard.py)。

**真实发现（面试时很好用的故事）**：最早用 YOLO11n 时，集成测试断言「办公室场景最多同时 2 个人」，结果失败了，查出来是 3。排查发现 track 24 只在 71–79.5 秒出现（镜头拉得最近时），置信度 0.46，框面积占画面的 46%。它是检测器对同一个人给出的第二个框。换成 YOLO26s（端到端，不需要 NMS）之后，这个重复框消失了，不过滤也是 2 人；但在场景切换处仍然多出一条只有 2 帧的「bus」轨迹（track 12）。这说明：
- 检测和跟踪得到的统计量是**线索，不是证据**，换更好的模型能减少错误，但不能消除。所以工具描述里让模型过滤 `mean_conf >= 0.5`，并用 `inspect_clip` 做视觉复核；
- 这也正是需要独立 verifier 的原因（见第 8 节）。

---

## 6.5 开放词汇检测（find_objects）与模型选型

**问题**：离线索引用的 YOLO26 只认识 COCO 的 80 个类别，可题目里会问「红色夹克」「叉车」「价签」。闭集检测器根本找不到这些东西。

**做法**：[grounding.py](../videoscout/tools/grounding.py) 里的 `find_objects` 工具在提问时现场运行 **YOLOE-26**。它把类别名当作文字提示，用 MobileCLIP2 编码成向量，替换掉检测头的类别向量，所以任意名词都能检测，而且是 YOLO 级别的速度，在本地跑、不花 API 钱。返回的框可以直接交给 `inspect_clip(box=...)` 放大细看。

实测：在演示视频的门口场景搜「red square」（COCO 里没有这个类），6 帧全部检出，框随方块从左往右移动；手画的「sign board」没检出。这再次说明检测结果只是候选，最终要靠视觉模型确认。

工程细节：Ultralytics 默认每次换提示词都会重新加载文本编码器（254MB），而且下载到当前目录。代码里改成只加载一次、存到项目的 `weights/` 下面；`set_classes` 会修改模型状态，所以调用时加了锁。

**按 2026 年的标准，为什么选这些模型**（不是因为你以前用过）：

| 角色 | 选择 | 理由 | 备选 |
|---|---|---|---|
| 离线检测 + 跟踪 | YOLO26-s + ByteTrack | Ultralytics 最新一代，端到端、不需要 NMS，笔记本 GPU 能实时跑 | RF-DETR（Apache-2.0 协议，没有 AGPL 的问题）、带 ReID 的 BoT-SORT（减少 ID 切换） |
| 开放词汇检测 | YOLOE-26-s | 支持文字提示，速度和 YOLO 一样 | Grounding DINO、OWLv2（更慢）；SAM 3（概念分割 + 视频跟踪，更强但更重，权重要申请） |
| 文本 → 画面检索 | SigLIP 2 so400m@384 | 检索比第一代强，支持多语言 | 只有 CPU 时用 SigLIP 2 base；Perception Encoder |
| 规划、验证、视觉 | Claude Opus 5.5 / Gemini | 多步工具调用和看图能力强 | 本地 Qwen3-VL（通过 Ollama） |

面试时要强调：**这个项目的核心是 agent 的设计**，所有感知模型都是挂在工具后面、可以替换的组件。换模型只需要改配置，评测框架能直接告诉你换完之后是变好还是变差。

---

## 7. Sub-agent 与 Context Engineering

**概念**：上下文是 agent 最稀缺的资源。一个问题可能要看几十帧，每帧约 450 个 token。如果全部塞进 planner 的上下文，成本会暴涨，注意力也会被稀释。

**做法**：
- **视觉子 agent**：`inspect_clip` 在一次独立调用里把帧交给视觉模型，只把不超过 120 词的文字观察返回给 planner（[vision.py](../videoscout/vision.py)）。planner 从头到尾看不到像素，这就是 sub-agent 或上下文隔离模式。
- **证据账本（evidence ledger）**：每次工具调用都在 state 里记录（工具、参数、结果），verifier 和兜底作答只读这份账本，不读整段对话（[graph.py](../videoscout/agent/graph.py) 的 `render_ledger`）。
- **预算行**：每批工具结果后面追加一行 `[budget] tool calls used 3/12, frames used 14/48`，让模型对剩余资源有感知。
- **稳定前缀**：system prompt 里不放时间戳，工具列表顺序固定（[registry.py:72](../videoscout/tools/registry.py#L72)），这样 prompt cache 能持续命中。

---

## 8. Verifier：自我反思、LLM-as-judge、校准置信度

**概念**：模型自己报的置信度往往偏高。用一个**独立上下文**的验证器，只看问题、答案和工具观察记录，判断「证据是否真的支持这个答案」，而不是判断「答案看起来对不对」。

**代码**：[graph.py:273](../videoscout/agent/graph.py#L273)。几个设计点：
- 验证器看不到 planner 的推理过程，只看证据账本，从而避免「自己说服自己」；
- 输出走结构化输出（`output_config.format` + pydantic 二次校验）：`supported / confidence / issues / next_step`；
- 被驳回时，把 `issues` 和 `next_step` 作为 `submit_answer` 的 tool_result 反馈给 planner，让它补证据（最多 `max_verify_rounds` 轮）。这就是帖子里说的「自我反思」；
- 最终报告的置信度用 verifier 给出的值，同时保留 `self_confidence`，可以对比两者的校准情况。

**选择性回答（selective answering）**：`accepted=False` 的答案仍然会输出，但会被标记为低置信度。评测时看 **coverage**（有多少题被信任）和 **selective accuracy**（被信任的那部分准确率多高）。这和你 SigLIP-KAN 项目里的「10% review rate 下的错误捕获」是同一种思路，面试时可以串起来讲。

---

## 9. 预算与兜底：让 agent 可控

| 机制 | 触发条件 | 代码 |
|---|---|---|
| 工具次数和帧数预算 | 在代码里检查并截断参数，超出预算的调用不执行，直接返回错误 | [graph.py:176](../videoscout/agent/graph.py#L176) |
| 失败的视觉检查不扣帧数 | inspect 出错时退回预扣的帧数 | tools_node |
| nudge | 模型结束了回合却没调用工具 | `nudge_node` |
| 强制作答 | 连续两轮所有调用都因超预算被拒，或达到回合上限 | [graph.py:165](../videoscout/agent/graph.py#L165)、[:344](../videoscout/agent/graph.py#L344) |

**重点**：预算必须**由代码强制**，不能只在 prompt 里「请求」模型遵守。模型可以不听，代码不会。

---

## 10. MCP（Model Context Protocol）

**概念**：MCP 是把工具标准化暴露给任意 LLM 客户端的协议，相当于工具界的 USB-C。写一次 server，Claude Code、Claude Desktop 和你自己的 agent 都能用。

**代码**：
- Server：[mcp_server.py:36](../videoscout/tools/mcp_server.py#L36)。用的是 MCP SDK 2.x 的 low-level `Server`，把 handler 作为构造参数传入。没用装饰器风格的 MCPServer，是为了让 schema 直接来自同一批 pydantic 模型。
- Client：[mcp_client.py:23](../videoscout/tools/mcp_client.py#L23)。MCP SDK 是异步的，stdio 连接必须在同一个 task 里打开和关闭，所以用一个后台事件循环持有连接，同步代码往这个循环里提交调用。
- 同一个 agent 可以一键切换两种模式：`--set agent.tools=mcp`（通过子进程和 stdio 调用）或 `inprocess`（进程内直接调函数）。

**面试题**：「MCP 和 function calling 是什么关系？」
答：function calling 是模型层面的能力（输出结构化调用）；MCP 是工具层面的协议，负责工具如何被发现（`tools/list`）和调用（`tools/call`）。MCP 的工具定义和 Claude 的工具定义可以一一映射，见 mcp_client.py 里的 `input_schema` 转换。

**注意**：MCP 模式下视觉模型在 server 进程里调用，所以客户端统计不到那部分费用。评测成本请用 inprocess 模式。

---

## 11. 生产级 LLM 调用细节

全部集中在 [llm.py](../videoscout/llm.py)：

| 点 | 说明 |
|---|---|
| 按角色选模型和 effort | planner / verifier / vision / captioner 各自配置（[configs/default.yaml](../configs/default.yaml)）。Opus 5.5 的 thinking 不能关闭，用 `output_config.effort` 控制思考深度 |
| Prompt caching | planner 循环开启自动缓存（[llm.py:192](../videoscout/llm.py#L192)）。越来越长的对话记录每轮按约 0.1 倍输入价读取；一次性的视觉调用不开缓存，因为写缓存要付 1.25 倍的价格 |
| Refusal fallback | beta `server-side-fallback-2026-07-01` + `fallbacks="default"`（[llm.py:196](../videoscout/llm.py#L196)）：请求被安全策略拒绝时，服务端自动换模型重跑 |
| stop_reason 检查 | 遇到 refusal 或 max_tokens 直接抛异常，不执行被截断的工具调用 |
| Batches API | 离线给片段生成标题时走批处理，价格减半（[captioner.py](../videoscout/index/captioner.py)）；计费器按 0.5 倍计价 |
| 成本计量 | `UsageMeter` 按实际服务的模型价格累计，包括缓存读写。每道题一个独立的计量器，评测结果里有 `cost_usd` |
| 重试 | SDK 自带：408/409/429/5xx 和连接错误会指数退避重试，客户端设了 `max_retries=4` |

### 11.1 模型无关：一个适配器接所有 OpenAI 兼容接口

[llm_openai.py](../videoscout/llm_openai.py) 让 Gemini（免费额度）、Ollama（本地）、OpenRouter、Groq 等都能直接用，agent、工具和测试一行不用改。做法：

- **内部统一一种消息格式**（Anthropic 风格的 content blocks），只在出入口做翻译：一个带 `tool_result` 的 user 回合会拆成若干条 `role: tool` 消息，后面再跟一条 user 消息放剩余的文字；图片转成 `data:` URL；工具 schema 简化为各家都能接受的子集（展开 `$ref`、把可空的 `anyOf` 折叠、去掉 `additionalProperties`）。
- **原样回传 provider 的 assistant 消息**：和 Claude「response.content 原样追加」是同一个原则。比如 Gemini 会在工具调用上附带 thought signature，必须原样送回去，否则会报错。
- **免费额度的限流**：429 时按 `Retry-After` 退避重试，`min_request_interval` 控制两次请求之间的最小间隔。
- `make_llm()`（[llm.py](../videoscout/llm.py)）根据 `models.provider` 选择后端，两个后端实现同一个 `LLMClient` 接口。

面试可以这样讲：「核心逻辑和模型厂商解耦。同一套评测可以横向比较不同模型的准确率和成本，换模型只需要改一份配置。」

---

## 12. 评测体系（面试的重中之重）

**代码**：[eval/run.py](../videoscout/eval/run.py)、[eval/report.py](../videoscout/eval/report.py)

1. **Baseline**：均匀抽 N 帧，一次性问 VLM（[baselines.py](../videoscout/eval/baselines.py)），这是大多数长视频 VLM 报告的做法。
2. **消融**：通过 `--set` 开关：去掉 verifier、去掉 BM25、去掉稠密检索、去掉 SQL 工具、改预算大小。
3. **指标**：
   - accuracy；
   - selective accuracy 和 coverage：被信任的答案有多准、有多少答案被信任；
   - **ECE**（期望校准误差，[report.py:23](../videoscout/eval/report.py#L23)）：模型说 70% 把握时，是否真的约有 70% 答对；
   - false accept rate：答错了但被 verifier 放行的比例，这是最危险的一类错误；
   - 平均工具调用次数、帧数、延迟、每题成本。
4. **组件级评测（grounding）**：如果数据带 `gt_windows`（答案所在的时间段），就计算 search_hit 和 inspect_hit（[run.py:41](../videoscout/eval/run.py#L41)），用来区分「没找到」和「找到了但看错了」。
5. **失败分类**（[report.py:41](../videoscout/eval/report.py#L41)）：retrieval_miss / looked_in_wrong_place / no_visual_check / perception_or_reasoning / forced_answer。report 会导出 failures.csv，带一列空的 notes 给你人工标注。
6. **可恢复的评测**：results.jsonl 逐条追加写入，重跑同一条命令会跳过已完成的题目。这和你 Pathfinder 项目里的「幂等写入」是同一个思路。

**面试题**：「你怎么知道 agent 真的有用？」
答：同样的帧预算下对比均匀抽帧 baseline；用消融量化每个组件的贡献；用 grounding 指标把检索错误和感知错误拆开；报告每题成本，因为 agent 的多轮调用必须用准确率提升来证明值得。

---

## 13. 怎么跑

所有命令都在项目根目录（`F:\VideoScout`）下运行。环境、模型权重和缓存都在这个文件夹里，不占 C 盘。

```bash
.venv\Scripts\activate          # 激活环境（CUDA 版 torch，会用上 3070）
pytest                          # 49 个单元测试，不需要 API key
pytest -m integration           # 5 个集成测试：用真实的 YOLO26 / YOLOE / SigLIP 2 建索引和检测

# 把 key 粘贴进项目根目录的 .env 文件（这个文件已被 git 忽略），然后：
python -m videoscout ask --config configs/gemini.yaml --index indexes/demo "What does the sign at the entrance say?" -o "A. GATE A OPEN" -o "B. GATE B CLOSED"
python -m videoscout ask --config configs/gemini.yaml --index indexes/demo --minimal "..."   # 50 行的裸循环版本，对比学习

# 换一个视频
python -m videoscout index 你的视频.mp4 --out indexes/my_video [--srt 字幕.srt]
```

费用粗估（Opus 5.5）：agent 每题大约 $0.1–0.3，取决于调用轮数，以 `summary.json` 的实测为准。5 道演示题大约 $1；Video-MME long 抽 30 个视频（90 题）大约 $10–30，另外还要加上建索引时可选的标题生成费用。

---

## 13.5 v0.3 升级：对齐 2025–2026 长视频 agent 的前沿做法

升级前的问题：YOLO 和 SigLIP 只是把同一家族换成最新版，没有回答「这个位置现在该用什么」。v0.3 的改动是按业界与论文的做法重新设计的：

| 模块 | 之前 | 现在 | 为什么（面试这样讲） |
|---|---|---|---|
| 记忆结构 | 一层：固定 10 秒片段 | 四层：故事线 L0 → 事件 L1 → 片段 L2 → 帧 L3（[memory.py](../videoscout/index/memory.py)） | Deep Video Discovery、VideoSeek 等 LVBench 头部方法都是「多粒度 + 由粗到细」。VideoSeek 在 LVBench 上用约 27 帧就超过了用 8000 帧的方法：**省帧数本身就是核心指标** |
| 事件切分 | 无 | 相邻片段向量的余弦相似度低于「均值 − z·标准差」处切开，并限制最短/最长时长 | 阈值是自适应的：讲座和动作片的相似度基线完全不同，固定阈值不可用 |
| 文本记忆 | 只有字幕 | 本地 Qwen3-VL-2B 给每个片段写描述、给每个事件写摘要，再写全片故事线 | 免费、离线；agent 用 `browse_timeline` 浏览全片只花几百个 token，**不花帧预算** |
| 稠密检索 | SigLIP 2 单帧 MaxSim | Qwen3-VL-Embedding：一个片段（几帧 + 字幕 + 描述）编码成一个向量 | 单帧编码器看不到动作和顺序，片段级多模态向量可以；SigLIP 2 保留为消融基线 |
| 精排 | 无 | Qwen3-VL-Reranker 交叉编码器对融合后的前 20 个重排（[hybrid.py](../videoscout/retrieval/hybrid.py)） | 双塔（bi-encoder）快但粗，交叉编码器（cross-encoder）把 query 和片段放在一起读，准但慢，所以只用于 top-N：这就是工业界标准的「召回 + 精排」两阶段 |
| 开放词汇 | YOLOE-26 逐帧检测 | SAM 3 概念分割 + 跟踪（[concepts.py](../videoscout/tools/concepts.py)），没有权重时自动退回 YOLOE | SAM 3 跨帧保持身份，所以「有几个不同的人」「出现了多久」是跟踪出来的，不是逐帧猜的 |
| YOLO26 的定位 | 主力 | 便宜的常驻第一级（级联设计） | 面试官问「为什么还用 YOLO」：全片跑便宜模型，问题相关的窗口才跑重模型，这是工业里常见的成本级联 |

**8GB 显存怎么同时跑 4 个模型**（[gpu.py](../videoscout/gpu.py)）：同一时刻只有一个大模型在 GPU 上，其余的停在内存里（换入换出约 3 秒）。实测发现每次搜索要换两次模型（约 10 秒），于是把短的文本查询放到 CPU 上编码（fp32，0.9 秒），GPU 留给 reranker：**搜索从约 10 秒降到 1.5 秒**。这是一个很好的「先测量、再优化」的例子。

**评测怎么做才有说服力**：
1. **检索单独评测（免费）**：LVBench 每道题标了答案所在的时间段。直接看 `search_segments` 的前 k 个结果有没有覆盖这个时间段，算 Recall@1/@5 和 MRR（[eval/retrieval.py](../videoscout/eval/retrieval.py)）。完全本地运行、不调 API，可以覆盖所有下载了的视频的所有题。对比 `bm25 / keyframe / clip / clip+bm25 / clip+bm25+rerank`，每一级改进都有数字支撑。
2. **端到端问答**：agent 对比三个基线：均匀抽 32 帧、Gemini 原生「整段看视频」（static）、Gemini 原生 agentic 模式（Gemini 自己在时间轴上跳着看）。面试官一定会问「为什么不直接把视频丢给 Gemini？」，这张表就是回答：准确率、token、延迟、能否给出证据时间段。

## 14. 做实验的建议顺序

1. 跑通演示视频（5 题），读 `runs/agent/traces/*.json`，看 agent 实际怎么决策；
2. 下载 Video-MME，用 long 分割抽 20–30 个视频（`eval.data --max-videos`），先跑 uniform-32 baseline；
3. 跑完整 agent 和 3–4 个消融，用 `eval.report` 生成对比表，填进 README；
4. 看 failures.csv，人工标 20 个错例，找出最大的失败类别，针对性改进一处（例如调 MMR、改 prompt、加镜头切分），再跑一次。**这一轮「发现 → 改进 → 验证」就是面试里最有价值的故事**；
5. 如果时间允许，做扩展（第 16 节）。

---

## 15. 简历条目（数字必须来自你自己的 runs/）

**Agentic Long-Video QA System (VideoScout)**, Oct 2026 – Present
- Built a LangGraph agent that answers questions over hour-long videos with a multi-granular memory (storyline → events → clips → frames) and MCP-served tools: text-memory browsing, two-stage retrieval (Qwen3-VL clip embeddings + BM25 with RRF, Qwen3-VL cross-encoder reranking, temporal MMR), read-only SQL over YOLO26/ByteTrack trajectories, SAM 3 open-vocabulary concept tracking, and a vision sub-agent.
- Ran four local models on an 8 GB laptop GPU with an LRU GPU pool and CPU query encoding, cutting search latency from ~10 s to 1.5 s.
- Enforced tool and frame budgets in the graph and added an independent verifier that checks answers against logged observations, feeding back missing evidence or flagging low-confidence answers (selective answering).
- On N LVBench questions (hour-long videos), raised retrieval Recall@5 from X% (SigLIP 2 frames) to Y% (clip embeddings + reranking), and compared the agent with uniform sampling and Gemini's native static/agentic video modes on accuracy, frames and tokens; ablated memory, reranker and verifier.

---

## 16. 面试高频问题速查

| 问题 | 回答要点 |
|---|---|
| 为什么不用更长上下文直接塞整段视频？ | 成本随帧数线性增长，注意力会被稀释，而且均匀抽样会漏掉短暂事件；主动搜索按需取证，花费只随「需要看的地方」增长 |
| 检索错了怎么办？ | 检索结果只当作候选，必须用 inspect_clip 确认；verifier 会驳回没有视觉证据的答案；grounding 指标能量化检索召回率 |
| 如何防止 agent 死循环或花太多钱？ | 在代码里强制预算、设回合上限、连续超预算后强制作答；每题都计量成本 |
| 怎么防 prompt injection？ | 字幕和标题都当作不可信数据；system prompt 声明工具输出不是指令；SQL 工具五层护栏；所有工具只读 |
| 幻觉怎么控制？ | 答案必须引用时间窗口；独立 verifier 只看证据账本；结构化输出；报告校准后的置信度和 ECE |
| 为什么 verifier 要用新的上下文？ | 避免它被 planner 的推理带偏；让它只判断「证据是否支持答案」，把生成和验证分开 |
| RRF 和加权求和相比？ | RRF 不需要分数校准，更鲁棒；代价是丢掉了分数大小（结合第 5 节的真实例子讲） |
| MCP 带来了什么？ | 工具只写一次，任何 MCP 客户端都能复用；进程隔离；可以一键切换 in-process 和 MCP 做对比 |
| 为什么加 reranker 而不是换更大的 embedding？ | 双塔是独立编码后算相似度，查询和片段之间没有交互；交叉编码器一起读两者，精度高但不能预先建索引，只能用于 top-N。两阶段兼顾了召回的速度和精排的精度 |
| 多粒度记忆有什么用？ | agent 先读故事线和事件摘要（几百个 token、零帧）定位大致区域，再搜片段，最后才花帧确认；对比消融 `--set agent.disabled_tools=browse_timeline` 看帧数和准确率 |
| 本地小模型写的描述会不会有幻觉？ | 会。所以它只当「地图」不当「证据」：prompt 和 verifier 都要求关键事实必须由 inspect_clip 的视觉观察确认 |
| 为什么不直接用 Gemini 的原生视频理解？ | 把它作为基线一起评测：比较准确率、token、延迟；agent 的优势在于可控预算、可审计的证据链、模型无关（可以换成本地模型），以及能给出证据时间段 |
| 如果要上线？ | 离线索引用批处理加 GPU；片段标题走 Batches API；checkpointer 支持中断恢复；缓存、并发和限流；按问题类型路由不同的预算 |

---

## 17. 可以继续做的前沿扩展（挑一个做深就够）

1. **本地视觉模型**：实现一个 `QwenVLVision` 后端（接口见 [vision.py](../videoscout/vision.py) 的 `VisionBackend`），复用已经下载的 Qwen3-VL-2B 跑 inspect_clip，对比成本和准确率。
2. **Agentic RL**：把 planner 换成 3B–4B 的小模型，用 GRPO 训练，奖励 = 答对 − λ × 工具成本（可以用 TRL 或 verl）。这是 2025–2026 年的热点方向，需要 GPU。
3. **ASR**：没有字幕的视频用 faster-whisper 生成字幕，补上 BM25 的文本来源。
4. **跨视频长期记忆**：把多个视频的索引合并，支持「哪个监控视频里出现过穿红衣服的人」这类问题。
