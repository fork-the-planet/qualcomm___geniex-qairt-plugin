#pragma once

// Umbrella header for the continuous-batching demo library.
//
// Includes the full public API in one shot — scheduler, KV cache manager,
// CBStepContext / CBInputProvider interface, greedy token extractor, and
// CBLLMModel. Models under examples/continuous_batching/<model>/ include
// this header and implement their own CBInputProviders.
//
// See examples/continuous_batching/README.md for the full integration guide.

#include "cb/cb_llm_model.h"
#include "cb/input_provider.h"
#include "cb/kv_cache_manager.h"
#include "cb/scheduler.h"
#include "cb/session.h"
#include "cb/token_sampler.h"
