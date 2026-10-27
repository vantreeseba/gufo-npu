# Qwen3.5-4B's own NPU programs (kernels/build.py compiles them; no NPU
# needed). The shapes are in the pure-data *_variants.py next to each design.

# Decode GEMV: a copy of the 2B's q4nx_gemv built with the 4B's geometry.
gufo_npu_add_kernel(NAME qwen35_4b_decode DESIGN decode_gemv.py
                    SOURCES aie/decode_gemv.cc decode_gemv_variants.py
                    MODEL qwen35_4b)

# Prefill GEMM: the 2B's bfp16_gemm core program with the 4B's shapes.
gufo_npu_add_kernel(NAME qwen35_4b_prefill DESIGN prefill_gemm.py
                    SOURCES prefill_gemm_variants.py
                    MODEL qwen35_4b)
