# fd-quant 计算面 image — 柏讯(wire) 确定性计算 MCP。
#
# 发布通道：GitHub Actions -> TCR（.github/workflows/image.yml，配方 docs/IMAGE-RELEASE.md）。
# 硬性要求（IMAGE-RELEASE 六步清单）：上游源为默认（美国 runner 绕道国内镜像会把
# 构建拖成小时级），国内镜像源一律 ARG 且默认空；单层 <300MB。
#
# 依赖分两段：基础依赖（numpy/pandas/jsonschema）为计算面本身所需，装进主层；
# fastmcp 属传输层（extras `server`）——计算面七族只依赖 numpy/标准库，
# 单独列出是为了让"无框架"这条边界在镜像里也成立可查。
FROM python:3.12-slim

ARG PIP_INDEX_URL=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY fd_quant ./fd_quant

RUN if [ -n "$PIP_INDEX_URL" ]; then pip install --no-cache-dir -i "$PIP_INDEX_URL" ".[server]"; else pip install --no-cache-dir ".[server]"; fi \
 && useradd -m -u 1000 appuser

USER appuser

EXPOSE 8080

# 纯计算面：无状态、不取数、不读库、不联网。默认 HTTP 传输；
# stdio 形态由 MCP 客户端以 `python -m fd_quant.compute.server` 直接拉起。
CMD ["python", "-m", "fd_quant.compute.server", "--http", "8080", "--host", "0.0.0.0"]
