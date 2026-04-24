#pragma once

// Umbrella header for the continuous-batching demo library. Models under
// examples/continuous_batching/<model>/ include this and supply their own
// CBInputProvider implementations. See README.md for the integration guide.

#include "cb/cb_llm_model.h"
#include "cb/input_provider.h"
#include "cb/kv_cache_manager.h"
#include "cb/scheduler.h"
#include "cb/session.h"
#include "cb/token_sampler.h"
