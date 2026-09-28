"""LangChain LLM wrapper."""
try:
    from langchain.llms.base import LLM
except ImportError:
    LLM = object


class PanguLangChain(LLM):
    """Wrap a panguforge engine as a LangChain LLM."""

    engine = None

    @property
    def _llm_type(self):
        return "panguforge"

    def _call(self, prompt, stop=None, **kwargs):
        out = self.engine.generate(prompt, max_tokens=64)
        if isinstance(out, list):
            return "".join(chr(t % 128) for t in out)
        return str(out)
