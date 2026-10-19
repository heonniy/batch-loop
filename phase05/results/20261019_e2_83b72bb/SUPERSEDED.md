# Superseded E2 attempt (kept for provenance)

Source `83b72bb`. Bug: when a cell raised CUDA OOM, the local `a` from the previous successful cell
(the largest KV cache built so far) stayed referenced, so every later cell of the same model ran out of
memory spuriously (ouro14: B=1 L=1024 "OOM" after the true OOM at B=128 L=512). ouro26 and huginn were
stopped manually before completion. Fixed in the next commit (cache references cleared + gc + empty_cache
in `finally`, free memory recorded for every failed cell). Do not use these rows for E2/E3.
