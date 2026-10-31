# Changelog

## [0.2.0](https://github.com/vantreeseba/gufo-npu/compare/v0.1.0...v0.2.0) (2026-10-31)


### Features

* **bench:** add NPU bandwidth and server benchmarks with FLM baselines ([#1](https://github.com/vantreeseba/gufo-npu/issues/1)) ([680757b](https://github.com/vantreeseba/gufo-npu/commit/680757bc42c658d2ba4cfa4a7397c9d9f80a2020))
* **npu:** run Qwen3.5-2B decode on the NPU with q4nx GEMVs ([#3](https://github.com/vantreeseba/gufo-npu/issues/3)) ([2ca2cc4](https://github.com/vantreeseba/gufo-npu/commit/2ca2cc47522c684b68503d8925c785df6df11847))
* **q4nx:** per-model fit, search for Qwen3.6-35B-A3B and Qwen3.5-4B (KL -24%, -22%) ([#15](https://github.com/vantreeseba/gufo-npu/issues/15)) ([7686d55](https://github.com/vantreeseba/gufo-npu/commit/7686d55a801dc8e3c18b3c2579fd8ce13dea35c4))
* **qwen35_4b:** run Qwen3.5-4B on its own NPU programs with MTP and prefill (27-31 tok/s) ([#10](https://github.com/vantreeseba/gufo-npu/issues/10)) ([c45593f](https://github.com/vantreeseba/gufo-npu/commit/c45593fe75bc81d8a5856a2c0957ec6ae592878a))
* **qwen35:** add CPU reference, tokenizer and greedy generate matching HF ([#2](https://github.com/vantreeseba/gufo-npu/issues/2)) ([9345628](https://github.com/vantreeseba/gufo-npu/commit/9345628dd3f35d3d699f6fe1cd8d3b6e4e6acb93))
* **qwen35:** add Qwen3.6-35B-A3B CPU reference, llama.cpp oracle and 23 tok/s CPU decoder ([#6](https://github.com/vantreeseba/gufo-npu/issues/6)) ([c2d8d9c](https://github.com/vantreeseba/gufo-npu/commit/c2d8d9cbef682deb347e3e916e6be5832f9867bf))
* **qwen35:** decode Qwen3.6-35B-A3B on the NPU with expert gathers (20 tok/s) ([#7](https://github.com/vantreeseba/gufo-npu/issues/7)) ([e36aae5](https://github.com/vantreeseba/gufo-npu/commit/e36aae5fd8a3f50e7decd6c69233356ef3e14758))
* **qwen35:** MTP speculation for Qwen3.6-35B-A3B on the NPU (26-29 tok/s) ([#9](https://github.com/vantreeseba/gufo-npu/issues/9)) ([9492042](https://github.com/vantreeseba/gufo-npu/commit/9492042d4472381451a42dffdc88fefa41256258))
* **qwen35:** process prompts on the NPU (0.11 s TTFT at 35 tokens, 1.2 s at 4k) ([#5](https://github.com/vantreeseba/gufo-npu/issues/5)) ([d47fe15](https://github.com/vantreeseba/gufo-npu/commit/d47fe158044a816dca97a07ae81f6ee53bca3e69))
* **qwen36_35b_a3b:** process prompts on the NPU (5.6 s TTFT at 1k tokens, FLM 8.75 s) ([#11](https://github.com/vantreeseba/gufo-npu/issues/11)) ([828ea92](https://github.com/vantreeseba/gufo-npu/commit/828ea925bd77d100facd30252eca93c573c699db))
* **qwen3vl_embed_2b:** CPU reference for image inputs and the NPU plan for the vision path ([#22](https://github.com/vantreeseba/gufo-npu/issues/22)) ([2e67e06](https://github.com/vantreeseba/gufo-npu/commit/2e67e06f34fdc5a4be9778df333f051243e544ef))
* **qwen3vl_embed_2b:** Qwen3-VL-Embedding-2B text embeddings on the NPU, embed CLI and /v1/embeddings ([#21](https://github.com/vantreeseba/gufo-npu/issues/21)) ([6b00e68](https://github.com/vantreeseba/gufo-npu/commit/6b00e688d67113a97ee99db51216ab9970ce6ad6))
* scaffold gufo-npu with pinned NPU toolchain, CI and dispatch benchmark ([76f9de0](https://github.com/vantreeseba/gufo-npu/commit/76f9de03b11fcf6316e0d41189cc2a934c85291d))
* **serve:** gufo-style request, generation and model-load logs ([#20](https://github.com/vantreeseba/gufo-npu/issues/20)) ([944a738](https://github.com/vantreeseba/gufo-npu/commit/944a738aa23226c4156151eb9cad16c64ce8c56d))
* **serve:** OpenAI-compatible server with streaming, tools, sampling and a multi-turn prompt cache ([#17](https://github.com/vantreeseba/gufo-npu/issues/17)) ([bed7a16](https://github.com/vantreeseba/gufo-npu/commit/bed7a16c76684117eb35acc9fa587ca05ce321fe))
* **serve:** structured outputs, forced tool calls, logprobs, completions and several models per server ([#19](https://github.com/vantreeseba/gufo-npu/issues/19)) ([0ec8d0d](https://github.com/vantreeseba/gufo-npu/commit/0ec8d0d14dbcbfc0ca84fc6ee6c31bda1e7f04ca))


### Bug Fixes

* **core:** SpinPool workers no longer miss a Run issued before they start ([#12](https://github.com/vantreeseba/gufo-npu/issues/12)) ([e37f4ee](https://github.com/vantreeseba/gufo-npu/commit/e37f4eef335af5ba2e42de6b7b4f6c071871e643))


### Performance

* **qwen35:** first token from prefill on the 2B and 4B (-20 ms / -47 ms TTFT) ([#14](https://github.com/vantreeseba/gufo-npu/issues/14)) ([0b89091](https://github.com/vantreeseba/gufo-npu/commit/0b89091209ee13a0fe3392c4667305d906e37c2e))
* **qwen36_35b_a3b:** prefill experts on the CPU (1.25 s TTFT at 1k tokens, FLM 8.75 s) ([#13](https://github.com/vantreeseba/gufo-npu/issues/13)) ([190d2bd](https://github.com/vantreeseba/gufo-npu/commit/190d2bdc45eab5d41cea84033b66382badd5f836))


### Documentation

* diagrams of the request path, the NPU GEMV array and each model's decode and prompt passes ([#18](https://github.com/vantreeseba/gufo-npu/issues/18)) ([e22d84a](https://github.com/vantreeseba/gufo-npu/commit/e22d84a2440ed08df4d537144f244e53bab4e293))
* prompt speeds and the NPU/CPU share of the work in the benchmarks, against FastFlowLM ([#16](https://github.com/vantreeseba/gufo-npu/issues/16)) ([b884191](https://github.com/vantreeseba/gufo-npu/commit/b88419145941e0c5b5fcabe4dad5f0a44b59f45e))


### Code Refactoring

* per-model kernel registration and test folders, README benchmarks table ([#8](https://github.com/vantreeseba/gufo-npu/issues/8)) ([0d6cd4b](https://github.com/vantreeseba/gufo-npu/commit/0d6cd4b23f2d824a149b7c9661d163ea961a0969))
