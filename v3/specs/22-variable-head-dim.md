# Variable head dimension

Attention launchers accept head dimensions 128, 256, and 512. The per-head
scale is `1 / sqrt(head_dim)`. Total query width is
`num_attention_heads * head_dim`; total K/V width is
`num_key_value_heads * head_dim`. A “global” attention layer changes context
masking, not these widths or the scale formula.

Every backend validates the requested dimension and its own shared-memory/
artifact limits. A backend unable to execute a supported public dimension must
select a tested explicit route or fail. Tests cover each dimension, query/KV
head ratio, sliding/global masks, partial pages, and eager/replay reference
parity.
