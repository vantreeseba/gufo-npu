# Changelog

## [0.4.0](https://github.com/vantreeseba/gufo-npu/compare/v0.3.0...v0.4.0) (2026-11-07)


### Features

* **npu:** time the submit alone, a polled round trip and a one-run runlist in bench dispatch ([16aaa42](https://github.com/vantreeseba/gufo-npu/commit/16aaa42a12be5e0e5fe2eb70c1576a8cc684ff1a))
* **qwen35_2b:** split a decode step's host work and each submission's submit time in the timeline ([e45a58d](https://github.com/vantreeseba/gufo-npu/commit/e45a58da2204f90ffb0ef5ae87735f6ec9d661b4))


### Performance

* **qwen35_2b:** run the DeltaNet decode step as one pool dispatch per layer, bit-identical and 1.5-2% faster decode ([7a84117](https://github.com/vantreeseba/gufo-npu/commit/7a84117ee613295b694c1cb4b44f448b859b34eb))
* **qwen35_2b:** run the DeltaNet decode step as one pool dispatch per layer, bit-identical to the Mixer's ([c462681](https://github.com/vantreeseba/gufo-npu/commit/c462681434d98e7a09e5537464aec4f175c97954))
* **qwen35_2b:** time the host part of each decode step and the NPU submit path, which is not where the time goes ([c36dee5](https://github.com/vantreeseba/gufo-npu/commit/c36dee566f79c4028544aefdee05450763e883bf))
* **qwen35:** fuse Q4_K, Q5_K and F32 rows into the fp32 CPU decoder's dot ([e6c27cf](https://github.com/vantreeseba/gufo-npu/commit/e6c27cfac82fee779a067688791b1521c5a35b1b))
* **qwen35:** fuse Q4_K, Q5_K and F32 rows into the fp32 CPU decoder's dot, 6-10% faster 35B CPU decode ([d188903](https://github.com/vantreeseba/gufo-npu/commit/d188903ae0c8bc395dd3a9e637851cadb7ada60f))
* **qwen35:** fuse Q6_K rows into the fp32 CPU decoder's dot, 2x CPU decode on a Q4_K_M 35B ([c9ee0f6](https://github.com/vantreeseba/gufo-npu/commit/c9ee0f68f41a34b589ae02e8febf2b0f5d453621))
* **qwen35:** fuse Q6_K rows into the fp32 CPU decoder's dot, 2x CPU decode on a Q4_K_M 35B ([f72e175](https://github.com/vantreeseba/gufo-npu/commit/f72e175166cc38afa3067319982d66515b8c2ec8))


### Documentation

* **qwen35_2b:** put the decode step's measured times on the data-flow diagrams, with a pie of where a step goes ([73e3e53](https://github.com/vantreeseba/gufo-npu/commit/73e3e5333689859d139a4d63db73b8adc4e00ec7))
* **qwen35_2b:** record the fused DeltaNet decode step, 1.5-2% faster decode with the same replies ([f36812d](https://github.com/vantreeseba/gufo-npu/commit/f36812dea56303425738478f62338c45c6371253))
* **qwen35_2b:** record the Gated DeltaNet decode step on the NPU as rejected for now, 2-4% for a second core program mode ([022fc85](https://github.com/vantreeseba/gufo-npu/commit/022fc851d71e5eb461e14fb59d00600b197d1d64))
* **qwen35_2b:** record the Gated DeltaNet decode step on the NPU as rejected for now, 2-4% for a second core program mode ([fd3898a](https://github.com/vantreeseba/gufo-npu/commit/fd3898adc0bbed25310d7be62557077b081a796e))
* **qwen35_2b:** record the host's part of a decode step, the submit path is not where the time goes ([e6af6df](https://github.com/vantreeseba/gufo-npu/commit/e6af6df36028726e0e7c00316e04344f7b805b27))
* **qwen36_35b_a3b:** note the UD GGUF's near-tie routing differences, same as main ([f810af4](https://github.com/vantreeseba/gufo-npu/commit/f810af42af6c9341cd9a7a06b4b92c7e266211f8))
* **qwen36_35b_a3b:** record prefill experts v2 as rejected, slower than the CPU expert step at every prompt length ([f3573c8](https://github.com/vantreeseba/gufo-npu/commit/f3573c813d30280e10631da0584e7235e56bd32b))
* **qwen36_35b_a3b:** record prefill experts v2 as rejected, slower than the CPU expert step at every prompt length ([9f9bf4c](https://github.com/vantreeseba/gufo-npu/commit/9f9bf4c8597a028afc6f313523e95ca032ab30a4))
* **qwen36_35b_a3b:** record the fused Q4_K, Q5_K and F32 CPU dots ([980a14b](https://github.com/vantreeseba/gufo-npu/commit/980a14b5bb7811ff168aec6e6fc9251d641b1295))


### Code Refactoring

* **qwen35:** expose the Mixer's worker pool for a model's own CPU steps ([0a57630](https://github.com/vantreeseba/gufo-npu/commit/0a5763050b627e88db3076a7b6a8ca055a7db9c0))

## [0.3.0](https://github.com/vantreeseba/gufo-npu/compare/v0.2.0...v0.3.0) (2026-11-04)


### Features

* **bench:** time chained GEMV calls (bench gemv --chain) and a per-submission decode timeline for Qwen3.5-2B ([cacebec](https://github.com/vantreeseba/gufo-npu/commit/cacebec4583291bed6289bdeaa670b350c8c741a))
* **npu:** patch the expert gathers inside a chain sequence (Q4nxGatherChain), sharing Q4nxGather's patch discovery ([525e0ac](https://github.com/vantreeseba/gufo-npu/commit/525e0acf779a0ff5c823b72042c2d8c0bcb690ff))
* **qwen36_35b_a3b:** add a CPU q4nx GEMV shaped like the NPU core's, for the mixer output projection ([f9fd5ef](https://github.com/vantreeseba/gufo-npu/commit/f9fd5ef9690d9e6f6f853542def07a6aa5774d82))


### Bug Fixes

* **npu:** initialize every Buffer::Impl field, silencing -Wmissing-field-initializers ([2241722](https://github.com/vantreeseba/gufo-npu/commit/224172258632d9ef5d1288be2e1e0158afb25ca6))
* **npu:** make every sub-buffer from its root buffer, as syncing a sub-buffer of a sub-buffer left stale data ([5892a65](https://github.com/vantreeseba/gufo-npu/commit/5892a656133b9ec1a25ace02d321caa7cbb163bb))


### Performance

* **qwen35_2b:** run each layer's chained GEMVs as one instruction sequence ([6044314](https://github.com/vantreeseba/gufo-npu/commit/6044314ef3c8d577b536905155fa2781698c3f9c))
* **qwen35_2b:** run each layer's out, gate + up, down and next projection as one chain sequence (one call per layer) ([0d35013](https://github.com/vantreeseba/gufo-npu/commit/0d350130982d09afb8dd423ee8b9df62d1c8ef80))
* **qwen36_35b_a3b:** build one chain sequence per MoE layer (gate + up gather, down gather, next call) with RTP words written only when they change ([5fe929b](https://github.com/vantreeseba/gufo-npu/commit/5fe929b48df52e6e65b2a24deed0dabfc0a8b255))
* **qwen36_35b_a3b:** run each MoE layer's gathers and next projection as one chain sequence on one activation buffer, with a decode timeline ([36aa325](https://github.com/vantreeseba/gufo-npu/commit/36aa3257f01d4db1fcdf64efcbcd59b6aaf97fa4))
* **qwen36_35b_a3b:** run each MoE layer's gathers and next projection as one instruction sequence, 7% faster decode ([5aa2d5d](https://github.com/vantreeseba/gufo-npu/commit/5aa2d5d4aea80bdd168f90782c54d3dc4e57fd1c))
* **qwen36_35b_a3b:** run the MoE layers' mixer output projection on the CPU, 8-10% faster decode ([f2b0c51](https://github.com/vantreeseba/gufo-npu/commit/f2b0c5189ceb17b659b5c2fbeebbdbc211b5c76b))
* **qwen36_35b_a3b:** run the MoE layers' mixer output projection on the CPU, one NPU submission per layer ([19cb688](https://github.com/vantreeseba/gufo-npu/commit/19cb6889a87368f7c4d30b35b2e09511f7744f7c))
* **qwen36_35b_a3b:** write each gate + up gather round's router scales just before its weights, 13% faster gather ([1f133dc](https://github.com/vantreeseba/gufo-npu/commit/1f133dcb6114dc39a60817d30258312e373e6da6))


### Documentation

* point AGENTS.md, NOTICE and comments at the per-model decode and prefill programs ([4c964c9](https://github.com/vantreeseba/gufo-npu/commit/4c964c98c5f99b791e2f61b528cb8fa5f38e5e59))
* **qwen35_2b:** record the chain sequences as retained, 3-4% faster decode with the same output ([ae08249](https://github.com/vantreeseba/gufo-npu/commit/ae08249e3fc75e3ab39de9b76379e203997e0867))
* **qwen35_2b:** record the chain sequences as unmeasured and how to measure them on the box ([09ff1fa](https://github.com/vantreeseba/gufo-npu/commit/09ff1fad67ae24eaa760273b553c0bbc05433604))
* **qwen35_2b:** say where Qwen3.5-2B's NPU programs and tests now live ([3f16e56](https://github.com/vantreeseba/gufo-npu/commit/3f16e568e78e5b1d5e3b027c78ef7dd1df9a249e))
* **qwen36_35b_a3b:** point to the 35B's own decode program and decoder ([a4d7042](https://github.com/vantreeseba/gufo-npu/commit/a4d7042683e0ad0ab9e822049a6f18922d49dca0))
* **qwen36_35b_a3b:** record the chain sequences and per-round gather scales, 7% faster decode with the same replies ([c054238](https://github.com/vantreeseba/gufo-npu/commit/c054238cac95c36434be15d9fe9b9b12aba098df))
* **qwen36_35b_a3b:** record the CPU mixer output projection, 8-10% faster decode ([6f1c45d](https://github.com/vantreeseba/gufo-npu/commit/6f1c45df85519548060cb1e518539a027359df26))


### Code Refactoring

* **npu:** drop the q4nx_gemv defaults from the shared GEMV runtime; the 35B passes its own design ([cd461b3](https://github.com/vantreeseba/gufo-npu/commit/cd461b35aed54d824fea8bfdc5a5697a6ae35de2))
* **npu:** run Qwen3.5-2B and Qwen3.6-35B-A3B on their own decode programs, decoders and prefills ([#25](https://github.com/vantreeseba/gufo-npu/issues/25)) ([53b44a2](https://github.com/vantreeseba/gufo-npu/commit/53b44a25ad89734bd5507ae30cfb96a64e8bad83))
* **qwen35_2b:** give Qwen3.5-2B its own decode GEMV program (qwen35_2b_decode) ([a5b4eba](https://github.com/vantreeseba/gufo-npu/commit/a5b4ebae72adfadb1a70ddbd39fe777b378c5241))
* **qwen35_2b:** give Qwen3.5-2B its own prefill GEMM program (qwen35_2b_prefill) ([27006cc](https://github.com/vantreeseba/gufo-npu/commit/27006cc90000e2dcaa7dc6012ecef6fc1094dca2))
* **qwen35_2b:** move Qwen3.5-2B's NPU prefill into its package ([6e1ae27](https://github.com/vantreeseba/gufo-npu/commit/6e1ae275d531c2f317ca3e3dc4d7f0e52a8c40bd))
* **qwen35_2b:** run Qwen3.5-2B on its own NPU decoder ([3c7d19b](https://github.com/vantreeseba/gufo-npu/commit/3c7d19bbcdadb7e495c2d73559279d4079b75483))
* **qwen35_2b:** run Qwen3.5-2B's GEMV op tests and bench gemv on its own decode program ([166c84e](https://github.com/vantreeseba/gufo-npu/commit/166c84e7a1dc9e92df10d34b36da258c7ff8c1da))
* **qwen36_35b_a3b:** build the decode GEMV as qwen36_35b_a3b_decode with only the 35B's shapes ([0c18f41](https://github.com/vantreeseba/gufo-npu/commit/0c18f41ea7f913b99552ff984968d6930a317d51))
* **qwen36_35b_a3b:** move the q4nx GEMV design and the NPU decoder into the 35B's package ([4262546](https://github.com/vantreeseba/gufo-npu/commit/426254686d19d4170fe721475292c2bd7dcedff6))
* **qwen36_35b_a3b:** share the CPU's q4nx block layout and code permutations in q4nx_tile.hpp ([57a9986](https://github.com/vantreeseba/gufo-npu/commit/57a9986b020b2cdb835532b1fef0a5433ad15eed))
* **qwen36_35b_a3b:** spell the shared NPU types gufo_npu::npu:: where the package's own npu namespace hides them ([8b17d32](https://github.com/vantreeseba/gufo-npu/commit/8b17d3269e914c360094a1bbf443995467596c96))
* **qwen36_35b_a3b:** the CLI, the server and the MoE spike run the 35B on its own decoder and program ([7fc24c2](https://github.com/vantreeseba/gufo-npu/commit/7fc24c25b852bde1cd6cf3db7840a5e23b9c4c63))
* **qwen36_35b_a3b:** the NPU decoder lives in the 35B's package and loads qwen36_35b_a3b_decode ([28d71c3](https://github.com/vantreeseba/gufo-npu/commit/28d71c3d0eaa3485df5f25b481954bf3026b3420))

## [0.2.0](https://github.com/vantreeseba/gufo-npu/compare/v0.1.0...v0.2.0) (2026-11-02)


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
