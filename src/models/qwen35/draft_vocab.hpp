#pragma once

#include <vector>

namespace gufo_npu::qwen35 {

// The token ids the MTP draft's LM head scores, ascending: every special
// token, then the most frequent tokens of a multilingual sample (Wikipedia
// in 20 languages, chat answers, math and code), ranked by
// tools/draft_vocab/rank_tokens.py. Scoring these rows instead of the whole
// vocabulary makes a draft several times cheaper; the main model checks every
// draft over the full vocabulary, so the output does not change. Throws if
// an id is not below `vocab`.
std::vector<int> DraftVocab(int vocab);

}  // namespace gufo_npu::qwen35
