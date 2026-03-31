from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # database
    database_url: str = "postgresql+asyncpg://cartograph:cartograph@localhost:5432/cartograph"
    database_url_sync: str = "postgresql://cartograph:cartograph@localhost:5432/cartograph"

    # parser
    parser_bin: str = "npx ts-node ../parser/src/index.ts parse"
    max_project_size_mb: int = 100

    # GraphSAGE
    sage_hidden_dim: int = 128
    sage_output_dim: int = 64
    sage_num_layers: int = 3
    sage_neighbor_sizes: list[int] = [25, 10, 5]
    sage_dropout: float = 0.3
    sage_lr: float = 0.001
    sage_epochs: int = 200
    sage_batch_size: int = 512

    # GAT
    gat_hidden_dim: int = 64
    gat_output_dim: int = 32
    gat_num_heads: int = 8
    gat_num_layers: int = 2
    gat_dropout: float = 0.4
    gat_lr: float = 0.0005
    gat_epochs: int = 150
    gat_temporal_decay: float = 0.95  # weight decay for older co-change edges

    # clustering
    min_modules: int = 3
    max_modules: int = 12
    silhouette_threshold: float = 0.25

    # risk detection
    risk_auth_patterns: list[str] = [
        "auth", "authenticate", "authorize", "session", "token",
        "jwt", "middleware", "protect", "guard", "verify",
    ]
    risk_secret_patterns: list[str] = [
        r"(?i)(api[_-]?key|secret|password|token|credential)\s*[:=]\s*['\"][^'\"]{8,}",
        r"(?i)(sk|pk)[-_][a-z0-9]{20,}",
        r"(?i)bearer\s+[a-z0-9]{20,}",
    ]

    # SBERT
    sbert_model: str = "all-MiniLM-L6-v2"
    sbert_dim: int = 384

    # LLM (for semantic zoom descriptions)
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"

    class Config:
        env_prefix = "CARTOGRAPH_"
        env_file = ".env"


settings = Settings()
