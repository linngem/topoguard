"""Empirical validation: topology → real error propagation among LLM agents."""
from .analysis import (AdoptionFit, Comparison, compare, fit_adoption, fit_all, observed,
                       simulate)
from .backends import AnthropicBackend, OpenAICompatBackend
from .cases import CASES, ERROR_ROLE, Candidate, Case
from .harness import (Condition, conditions, error_seeds, load, n_calls, pair_seeds, run_grid,
                      run_trial, seeds)
from .scenario import (ERROR_KEY, ERRORS, TRACKED, TRUTH_KEY, VOCAB, LLMPolicy, Scenario,
                       SimulatedPolicy, parse, render_user, system_prompt)
