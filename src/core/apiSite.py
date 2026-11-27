"""Located LLM API sites, their buckets and protocols, and the LangChain call names they rely on."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

MODEL_FACTORY = "langchain.chat_models.init_chat_model"
MODEL_WRAPPERS = {"bind", "bind_tools", "with_config", "with_retry", "with_structured_output"}
INVOKE_METHODS = {"invoke", "ainvoke", "batch", "abatch", "stream", "astream"}
ASYNC_METHODS = {"ainvoke", "abatch", "astream"}
SOURCE_SUFFIXES = {".py", ".ts", ".tsx"}
SKIP_DIRECTORIES = {"__pycache__", "check", "node_modules", ".git"}


class ApiBucket(Enum):
    FRAMEWORK = "framework"
    PROTOCOL = "protocol"
    MODEL = "model"


class ApiProtocol(Enum):
    LANGCHAIN_INVOKE = "langchain-invoke"
    LANGCHAIN_FACTORY = "langchain-factory"
    DEEPAGENT_CREATE = "deepagent-create"
    CLIENT_GENERATE = "client-generate"
    OPENAI_CHAT = "openai-chat"
    AI_SDK_STREAM = "ai-sdk-stream"
    MODEL_LOCATED = "model-located"


@dataclass(frozen=True)
class ApiSite:
    file: str
    path: str
    line: int
    column: int
    symbol: str
    protocol: str
    bucket: str

    def as_dict(self) -> dict:
        return {"file": self.file, "line": self.line, "column": self.column,
                "symbol": self.symbol, "protocol": self.protocol, "bucket": self.bucket}


def make_site(file: Path, line: int, column: int, symbol: str, protocol: ApiProtocol, bucket: ApiBucket) -> ApiSite:
    """Build a site whose symbol is qualified by the file name."""
    return ApiSite(file.name, str(file), line, column, f"{file.name}:{symbol}", protocol.value, bucket.value)


def iter_source_files(path):
    """Yield Python and TypeScript files under ``path``, skipping check material and caches."""
    root = Path(path)
    if root.is_file():
        yield root
        return
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_DIRECTORIES for part in candidate.parts):
            continue
        yield candidate
