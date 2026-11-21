---
title: "搜索匹配重构：从最小距离到首次命中"
description: "记录文件与应用搜索中 Damerau-Levenshtein 容错匹配的简化，以及 find_map 提前结束的取舍。"
date: 2026-10-24
order: 1
authors:
  - maokaihe
tags:
  - Rust
  - Search
  - Refactoring
draft: true
---

昨天改了一段文件和应用搜索的代码。原本只是觉得其中一段迭代器写得有点别扭，改到后面发现，真正需要调整的其实是匹配策略。

## 原来的匹配流程

我有一个函数叫 `score_name`：

```rust
fn score_name(
    name: &str, // 原始名字
    lower: &str, // name 小写
    query: &str, // 用户输入
    pattern: &Pattern,
    matcher: &mut Matcher,
    buffer: &mut Vec<char>,
) -> Option<u32>
```

它包含四个步骤：

```rust
// 精确匹配
if lower == query {
    return Some(EXACT_SCORE);
}

// 前缀匹配
if lower.starts_with(query) {
    return Some(200_000 - lower.chars().count().min(10_000) as u32);
}

// nucleo 模糊匹配
// buffer 是传进来用来复用的，减少重复内存分配
if let Some(score) = pattern.score(Utf32Str::new(name, buffer), matcher) {
    return Some(100_000 + score.min(65_000));
}

// Damerau-Levenshtein 容错匹配
...
```

这个搜索其实主要使用 nucleo 做模糊匹配。比如文件名是 `docker-compose-cli`，输入 `doker`，由于少了一个 `c`，nucleo 可能还能匹配；但如果把某个字符输错了，导致查询不再是文件名的子序列，就可能匹配失败。

所以我在 nucleo 后面加了一层 Damerau-Levenshtein（DL）容错匹配。它支持插入、删除、替换和相邻字符换位。查询长度小于等于 3 或大于等于 32 时不执行，其他情况允许 1～2 次编辑。

## 将两次距离计算合并为一次

最开始的代码大概是这样的：

```rust
let distance = std::iter::once(lower)
    .chain(
        lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| *word != lower && !word.is_empty()),
    )
    .map(|word| {
        // Rust 的字符串以 UTF-8 存储，字节数不等于字符数。中文字符通常占多个字节，
        // 直接用字符个数当字节下标可能截在字符中间，因此需要用 char_indices
        let end = word
            .char_indices()
            .nth(length)
            .map_or(word.len(), |(index, _)| index);

        let prefix = &word[..end];
        let prefix_distance = strsim::damerau_levenshtein(query, prefix);

        if prefix != word && word.chars().count().abs_diff(length) <= tolerance {
            prefix_distance.min(strsim::damerau_levenshtein(query, word))
        } else {
            prefix_distance
        }
    })
    .min()?;
```

这里先把文件名拆成多个候选词。比如 `docker-compose-cli` 会得到完整文件名，以及 `docker`、`compose`、`cli` 三个部分。

随后对每个候选词计算编辑距离。

我当时考虑了两种情况。一种是只比较与 query 等长的前缀，另一种是当候选词长度与 query 接近时，再比较一次完整词，最后取更小的距离。

逻辑没什么大问题，但看着还是有点绕。同一个候选词最多会调用两次 DL，而 DL 本身需要动态规划，没必要在这里反复计算。

于是我试着简化成一次。

最开始想的是，把截取长度从 `length` 改为 `length + tolerance`。既然允许一两次编辑，那前缀稍微长一点似乎也合理。

但很快发现，这会引入另一个问题。

比如查询 `docler`，候选词是 `docker-compose`。原本只需要将 `l` 替换成 `k`，编辑距离就是 1。但如果截取时把后面的 `-` 也带进来，就需要额外删除一个字符，距离变成 2。

我想要的其实是搜索容错，不是严格计算某个固定字符串的编辑距离。只要用户输入的内容接近文件名的开头，就应该尽量搜到。

后面又改成根据候选词长度决定是否截取：

```rust
let target = if word.chars().count() > length + tolerance {
    let end = word
        .char_indices()
        .nth(length)
        .map_or(word.len(), |(i, _)| i);

    &word[..end]
} else {
    word
};

strsim::damerau_levenshtein(query, target)
```

这样每个候选词只计算一次 DL。

如果候选词比较短，就使用完整词；如果明显比 query 长，就只比较前 `length` 个字符。

## 固定前缀的边界

不过这个方案也不是完全准确。

例如查询 `xabc`，候选词是 `abcdefgh`。实际上只需要删除查询词开头的 `x`，就能匹配候选词的前缀 `abc`。但由于截取的是 `abcd`，计算出来的编辑距离可能是 2。

要准确解决这个问题，需要计算查询词到候选词所有可能前缀的最小编辑距离，或者实现专门的 Prefix DL。

我暂时没有这样做。

一方面，DL 已经是整个搜索流程的最后一层。前面还有完全匹配、前缀匹配以及 nucleo 模糊匹配，大部分正常搜索不会进入这里。

另一方面，我更在意搜索时的响应速度。为了覆盖一些相对少见的输入错误，增加大量编辑距离计算不太划算。

## 从最优距离改为首次命中

后来又注意到一个细节。

原来的代码使用 `.map().min()`，也就是把每个候选词的距离都计算出来，再选最小的那个。

但对于这个搜索工具，我并不需要知道哪个候选词最相似。

只要有一个候选词的编辑距离在容错范围内，就可以把这个文件加入结果。至于它是距离 1 还是距离 2，对我的排序影响不大，因为 DL 匹配本来就排在 nucleo 匹配后面。

所以我把 `.map().min()` 换成了 `.find_map()`。

```rust
let tolerance = if length >= 7 { 2 } else { 1 };

std::iter::once(lower)
    .chain(
        lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty() && *word != lower),
    )
    .find_map(|word| {
        if word.starts_with(query) {
            return Some(10_000);
        }

        let target = if word.chars().count() > length + tolerance {
            let end = word
                .char_indices()
                .nth(length)
                .map_or(word.len(), |(i, _)| i);

            &word[..end]
        } else {
            word
        };

        let distance = strsim::damerau_levenshtein(query, target);

        (distance <= tolerance).then(|| 10_000 - distance as u32 * 1_000)
    })
```

`find_map` 在这里刚好符合需求。

每次迭代返回 `None`，就继续检查下一个候选词；一旦返回 `Some(score)`，就立即停止，不再计算后面的候选。

比如搜索 `docke`，第一个候选词 `docker-compose-cli` 就可能满足容错条件。这时候直接返回即可，不需要继续检查 `docker`、`compose` 和 `cli`。

## 评分结果与取舍

这里还有一个很容易写错的地方。

原来的 `.min()` 返回的是最小编辑距离，因此后面还需要判断 `distance <= tolerance`，再转换成分数。

现在 `.find_map()` 内部已经完成了判断和评分，返回的就是 `Option<u32>`。如果还在外面把结果当作 distance 判断，就把分数和编辑距离混在一起了。

改完之后，DL 阶段每个实际检查的候选词最多只调用一次编辑距离计算，并且一旦命中就会提前结束。

当然，提前结束也意味着可能错过后面距离更小的候选词。例如第一个候选距离为 2，后一个候选距离为 1，现在只会使用第一个候选的分数。

不过目前的评分体系里，DL 命中的结果本来就属于低优先级。我先接受了这个误差。

回头看，这次最主要的改动不是把两次 DL 缩减到一次，而是从“计算所有候选词中最小的编辑距离”，改成了“找到第一个满足容错条件的候选词”。

原来那段 `.map().min()` 其实一直在计算一个后续逻辑并不真正需要的最优值。
