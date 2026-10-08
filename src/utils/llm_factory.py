"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from langchain_core.embeddings import Embeddings

_gemini_rate_limiter = None


def _get_gemini_rate_limiter():
    """
    Rate limiter dùng chung cho MỌI Gemini LLM trong process (quota free tier tính
    theo API key, nên RAGAS chạy song song cũng phải xếp hàng chung 1 limiter).
    """
    global _gemini_rate_limiter
    if _gemini_rate_limiter is None:
        from langchain_core.rate_limiters import InMemoryRateLimiter
        _gemini_rate_limiter = InMemoryRateLimiter(
            requests_per_second=config.GEMINI_RPM / 60,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    return _gemini_rate_limiter


class _RetryingEmbeddings(Embeddings):
    """
    Bọc 1 Embeddings: chia lô nhỏ và tự chờ + thử lại khi gặp 429 (quota/phút).

    Gemini free tier giới hạn 100 text/phút cho embeddings, trong khi knowledge base
    có > 100 chunks → embed 1 lần là bị từ chối nếu không chia lô.
    """

    def __init__(self, inner, batch_size: int = 50, max_attempts: int = 8, wait_s: int = 30):
        self.inner        = inner
        self.batch_size   = batch_size
        self.max_attempts = max_attempts
        self.wait_s       = wait_s
        self.model        = getattr(inner, "model", "unknown")

    def _with_retry(self, fn, *args):
        import time
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(*args)
            except Exception as e:
                quota_hit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if not quota_hit or attempt == self.max_attempts:
                    raise
                print(f"   ⏳ Embeddings chạm quota, chờ {self.wait_s}s (lần {attempt}) ...")
                time.sleep(self.wait_s)

    def embed_documents(self, texts):
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._with_retry(self.inner.embed_documents, texts[i:i + self.batch_size]))
        return vectors

    def embed_query(self, text):
        return self._with_retry(self.inner.embed_query, text)


def _make_quota_fallback_chat(models: list, names: list):
    """
    Tạo chat model "xoay vòng theo quota": gọi model đầu tiên; khi model đó hết quota
    NGÀY (free tier Gemini tính quota theo từng model) thì chuyển hẳn sang model kế tiếp.
    Lỗi tạm thời (quota/phút, 503) → chờ rồi thử lại cùng model.
    """
    import threading
    import time
    from typing import Any
    from langchain_core.language_models.chat_models import BaseChatModel

    lock = threading.Lock()

    class QuotaFallbackChat(BaseChatModel):
        models: list[Any]
        names: list[str]
        current: int = 0
        wait_s: int = 20
        max_transient: int = 6

        @property
        def _llm_type(self) -> str:
            return "quota-fallback-chat"

        @property
        def active_model(self) -> str:
            return self.names[min(self.current, len(self.names) - 1)]

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            transient = 0
            per_day_hits = 0
            while True:
                idx = self.current
                if idx >= len(self.models):
                    raise RuntimeError("Tất cả judge models đều đã hết quota ngày.")
                try:
                    return self.models[idx]._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                except Exception as e:
                    msg = str(e)
                    if "PerDay" in msg:
                        # API đôi khi trả lỗi quota ngày "nhầm" 1 lần → chỉ chuyển model
                        # khi gặp lại lần thứ 2 liên tiếp.
                        per_day_hits += 1
                        if per_day_hits < 2:
                            time.sleep(5)
                            continue
                        per_day_hits = 0
                        with lock:
                            if self.current == idx:
                                self.current += 1
                                nxt = self.names[self.current] if self.current < len(self.names) else "—"
                                print(f"\n   🔁 '{self.names[idx]}' hết quota ngày → chuyển sang '{nxt}'")
                        transient = 0
                        continue
                    if any(code in msg for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL",
                                                   "504", "DEADLINE_EXCEEDED")):
                        transient += 1
                        if transient > self.max_transient:
                            raise
                        time.sleep(self.wait_s)
                        continue
                    raise

    return QuotaFallbackChat(models=models, names=names, rate_limiter=_get_gemini_rate_limiter())


def get_eval_llm():
    """
    LLM dùng làm giám khảo cho RAGAS (temperature=0).

    - gemini: xoay vòng qua GEMINI_EVAL_MODELS khi một model hết quota ngày.
    - provider khác: giống get_llm(temperature=0).
    """
    if config.PROVIDER != "gemini":
        return get_llm(temperature=0)

    from langchain_google_genai import ChatGoogleGenerativeAI
    names = config.GEMINI_EVAL_MODELS
    models = [
        ChatGoogleGenerativeAI(model=n, google_api_key=config.GOOGLE_API_KEY, temperature=0,
                               max_retries=0, timeout=120)
        for n in names
    ]
    return _make_quota_fallback_chat(models, names)


def get_llm(provider: str = None, temperature: float = 0.0):
    """
    Trả về BaseChatModel tương ứng với provider được chọn.

    Args:
        provider    : "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                      Mặc định: đọc PROVIDER từ .env (config.PROVIDER)
        temperature : độ ngẫu nhiên (0.0 = tất định, 1.0 = sáng tạo)

    Returns:
        BaseChatModel instance sẵn sàng sử dụng

    Raises:
        ValueError nếu provider không hợp lệ
        ImportError nếu package tương ứng chưa được cài đặt
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
            timeout=90,                          # tránh request treo vô hạn
            max_retries=6,                       # tự thử lại khi gặp 429/503
            rate_limiter=_get_gemini_rate_limiter(),
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    elif provider == "openrouter":
        # OpenRouter dùng OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
            # OpenRouter giữ trước credit theo max_tokens; mặc định của model (16k) dễ bị 402.
            max_tokens=2048,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance tương ứng với provider được chọn.

    Lưu ý quan trọng:
        - Anthropic KHÔNG có Embeddings API → tự động fallback về OpenAI embeddings
        - OpenRouter cũng dùng OpenAI embeddings (không có API embeddings riêng)
        - Ollama cần model embedding riêng (mặc định: nomic-embed-text)
          Cài đặt: ollama pull nomic-embed-text

    Args:
        provider: "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                  Mặc định: đọc PROVIDER từ .env

    Returns:
        Embeddings instance sẵn sàng sử dụng
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openrouter" and not config.OPENAI_API_KEY:
        # Không có OpenAI key → gọi embeddings qua endpoint OpenAI-compatible của OpenRouter.
        # check_embedding_ctx_length=False: gửi text thô thay vì token ids của tiktoken.
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENROUTER_EMBEDDING_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            check_embedding_ctx_length=False,
        )

    if provider in ("openai", "openrouter"):
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return _RetryingEmbeddings(GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        ))

    elif provider == "anthropic":
        # Anthropic không cung cấp Embeddings API → dùng OpenAI thay thế
        print("⚠️  Anthropic không có Embeddings API — đang dùng OpenAI embeddings thay thế.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )
