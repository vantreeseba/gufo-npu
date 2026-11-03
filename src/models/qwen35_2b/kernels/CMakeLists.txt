# Qwen3.5-2B's own NPU programs (kernels/build.py compiles them; no NPU
# needed). The shapes are in the pure-data *_variants.py next to each design.

# Decode GEMV: kernels/designs/q4nx_gemv.py's core program with only the
# 2B's shapes.
gufo_npu_add_kernel(NAME qwen35_2b_decode DESIGN decode_gemv.py
                    SOURCES aie/decode_gemv.cc decode_gemv_variants.py
                    MODEL qwen35_2b)
