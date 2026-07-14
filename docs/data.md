# 数据接入

本仓库不分发 Provo、ZuCo、逐参与者数据、转换后的 fixation 表、模型 checkpoint 或 cache。使用者必须从下列官方来源自行获取数据，并分别遵守语料许可、引用和参与者数据条款；仓库的 MIT 许可不覆盖第三方数据。`data/raw/` 与 `data/processed/` 默认仅供本地使用并由根 `.gitignore` 排除；版本化 manuscript release 可单独提供经精确 allowlist 审核的 aggregate-results attachment，其中不含参与者级记录、fixation 表、checkpoint 或 cache。

## 规范化 fixation CSV

核心代码不绑定某个语料，输入文件必须包含：

```text
subject_id,text_id,word_index,fixation_order,duration_ms
```

可选列：

```text
line_id
```

约束：

- `word_index` 使用从 0 开始的文本内词位置。
- `fixation_order` 在每个 `(subject_id, text_id)` 内唯一，但不要求连续。
- 每行表示一次 fixation；连续落在同一词的 fixation 不应预先合并。
- 只有同一被试、同一文本内相邻 fixation 才构成转移。
- `line_id` 存在时，跨到后续行的转移被标记为 `line_return`，并优先于按全局词索引判断的前向或回视。

## Provo Corpus

官方来源：<https://osf.io/sjefs/>
许可：CC BY 4.0
正式引用：Luke & Christianson (2018), <https://doi.org/10.3758/s13428-017-0908-4>

| 文件 | 官方下载 | SHA-256 |
|---|---|---|
| 词级眼动汇总 | <https://osf.io/download/a32be/> | `38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7` |
| Predictability norms | <https://osf.io/download/e4a2m/> | `965fb72eab55f51e08fc1b5622638b85b1085976ff513e2a7bee4adbbd4e6489` |
| 逐 fixation report | <https://osf.io/download/z3eh6/> | `0d961a6508ed6caafdb4bc1025c067ecc97a0be07b13d3de0acafb5ef6c4fb7e` |

逐 fixation 文件的候选字段映射：

```text
subject_id       <- RECORDING_SESSION_LABEL
fixation_order   <- CURRENT_FIX_INDEX
duration_ms      <- CURRENT_FIX_DURATION
AOI/word label   <- CURRENT_FIX_INTEREST_AREA_LABEL
AOI index        <- CURRENT_FIX_INTEREST_AREA_INDEX
```

不能不经核验直接完成的映射：

- 逐 fixation 文件中的 `trial` 和 `page` 尚无作者提供的可靠说明，不能直接假定等于论文数据中的 `Text_ID`。
- `IA_ID` 不保证等于 `Word_Number`。论文明确提到 typo 和 text parsing error 会造成错位。
- 词级汇总文件的 `IA_FIRST_FIXATION_INDEX` 只表示第一次进入某词，不能恢复完整 scanpath。

因此，在建立经过人工/程序核验的 trial-to-text 和 AOI-to-word crosswalk 前，不应自动生成 Provo 规范化 CSV。转换后应记录原始文件哈希、下载日期、排除规则和所有修正。

当前转换器 `scripts/convert_provo.py` 已执行全量 crosswalk 审计：共同 trial 中 `page == Text_ID` 无冲突，主表的 `(Text_ID, IA_ID) -> Word_Number` 无歧义。转换器只使用通过这些约束的记录，并输出 JSON 排除报告；不映射合并多个 norms token 的 AOI。

## Python API

```python
from eyetrack2llm import (
    aggregate_transitions,
    extract_events,
    read_fixation_csv,
    split_half_reliability,
)

fixations = read_fixation_csv("fixations.csv")
events = extract_events(fixations)
matrices = aggregate_transitions(
    events,
    n_words={"text-1": 120},
    event_types=("forward", "regression", "refixation", "line_return"),
)
reliability = split_half_reliability(
    events,
    text_id="text-1",
    event_type="forward",
    n_words=120,
    repeats=100,
)
```

矩阵的 `mask` 表示某个 source word 是否有可靠的出边分布。有效 source 的整行 destination 都进入比较，而不是只比较观察到的非零边。
