// The cross-provider suite covers all three hosted providers as a
// unit (model-switching, function-calling round-trips, snapshot
// fidelity). The llama-gguf variant is additive — gated on its own
// feature inside the module tree.
#![cfg(all(feature = "openai", feature = "google", feature = "anthropic-vertex"))]

mod cross_provider;
