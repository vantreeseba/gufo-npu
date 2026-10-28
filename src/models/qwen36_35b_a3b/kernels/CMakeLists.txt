# Qwen3.6-35B-A3B's own NPU programs for prompt processing (kernels/build.py
# compiles them; no NPU needed). The shapes are in the pure-data
# *_variants.py next to each design. Its decode GEMVs are still the shared
# kernels/designs/q4nx_gemv.py.

# q4nx (the decoder's weights, read in place) to bfp16 in the GEMM's B order.
gufo_npu_add_kernel(NAME qwen36_35b_a3b_dequant DESIGN prefill_dequant.py
                    SOURCES aie/prefill_dequant.cc
                    MODEL qwen36_35b_a3b)

# Prefill GEMM: the 2B's bfp16_gemm core program with the 35B's shapes,
# reading B from the dequant's output.
gufo_npu_add_kernel(NAME qwen36_35b_a3b_prefill DESIGN prefill_gemm.py
                    SOURCES prefill_dequant_variants.py
                    MODEL qwen36_35b_a3b)
