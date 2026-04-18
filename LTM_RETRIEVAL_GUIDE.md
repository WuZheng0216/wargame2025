# LTM Retrieval Guide

## 主链路

1. `BaseCommander` 在 RED slow 决策前构建 `memory_packet`
2. `LongTermMemoryRetriever.build_query_for_context(...)` 生成检索 query
3. `LongTermMemoryRetriever.retrieve_for_context(...)` 从 lessons 中做 hybrid 检索
4. `format_lessons_block(...)` 将 Top-K lesson 格式化为 prompt 文本
5. `graph.py` 中的 `ltm_lessons` 注入 `commander/allocator` prompt

## 关键文件

- `ltm_retriever.py`
- `base_commander.py`
- `graph.py`
- `reflection_agent.py`

## 当前检索策略

- 关键词检索：`keyword_score`
- 向量检索：`vector_score`
- 混合得分：`hybrid_score = alpha * keyword + (1 - alpha) * vector`

默认配置：

- `RED_LTM_TOKENIZER=hybrid`
- `RED_LTM_VECTOR_BACKEND=tfidf`
- `RED_LTM_HYBRID_ALPHA=0.6`

## 分词策略

### `RED_LTM_TOKENIZER=hybrid`

- 英文/标识符：正则切词
- 中文：
  - 若已安装 `jieba`，使用 `jieba.cut(...)`
  - 同时补充 CJK n-gram
- 若未安装 `jieba`，退化为：
  - 单字
  - CJK n-gram

### 可选值

- `hybrid`
- `jieba`
- `char`

## 向量后端

### `RED_LTM_VECTOR_BACKEND=tfidf`

- 使用 `sklearn.feature_extraction.text.TfidfVectorizer`
- 对中文分词结果和英文 token 做统一 TF-IDF
- 这是当前默认且最稳的本地方案

### `RED_LTM_VECTOR_BACKEND=tei`

- 调用外部 embedding 服务
- 默认读取：
  - `RED_LTM_EMBEDDING_ENDPOINT`
  - 若未设置则回退 `TEI_MODEL_ENDPOINT`
- 需要服务提供 `/embed` 接口

## lessons 来源

- 优先：`*_lessons_structured.jsonl`
- 回退：`*_reflections.jsonl`

批量实验时，当前 run 会使用自己的：

- `run_xxxx/knowledge/red_lessons_runtime.jsonl`

## 推荐实验方式

### 先做本地稳态对比

1. `RED_LTM_TOKENIZER=hybrid`
2. `RED_LTM_VECTOR_BACKEND=tfidf`
3. 实验解释器固定为 `C:\Users\Tk\.conda\envs\scene\python.exe`
4. 当前推荐依赖记录见 `requirements-experiment.txt`

### 再做增强对比

1. 安装 `jieba`
2. 保持 `RED_LTM_TOKENIZER=hybrid`
3. 观察召回质量和分数变化

### 最后再尝试外部 embedding

1. 启动 TEI / embedding 服务
2. `RED_LTM_VECTOR_BACKEND=tei`
3. 设置 `RED_LTM_EMBEDDING_ENDPOINT`

## 当前已知边界

- `scene` 环境现已安装 `jieba` 和 `scikit-learn`
- TEI 后端依赖外部服务可用
- lessons 结构仍偏轻量，后续仍建议继续增强 schema
