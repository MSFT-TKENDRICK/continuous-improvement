import os

# Importing order_support.agent calls assert_ai.auto_trace.enable(); keep it from
# instrumenting third-party libraries during unit tests.
os.environ.setdefault("PHOENIX_DISABLE_AUTO_INSTRUMENT", "1")
