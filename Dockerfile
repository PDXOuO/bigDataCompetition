FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.12-slim-bookworm

# 1. 安装编译依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# 2. 编译安装 TA-Lib 底层库 (保持原样，这部分写得很好)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 3. 引入 uv 神器
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 4. 拷贝依赖文件并安装 (利用 uv 的极速与锁死机制)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# 5. 拷贝你的所有代码和跑出来的模型权重
# 务必确保你的 best_model.pth 放在 model/ 文件夹里一起被拷进去
COPY . .

# 6. 设置环境变量
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

# 7. 【终极修改】删掉 sleep，改为自动执行你的预测脚本
# 这样容器一启动就会自动跑 test.py，跑完自动退出交卷
CMD ["python", "code/src/test.py"]