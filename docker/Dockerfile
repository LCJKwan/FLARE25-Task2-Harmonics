# ---------- builder: install into a venv ----------
FROM python:3.12.5-slim AS builder

WORKDIR /workspace
COPY requirements.txt .

# create venv and install deps
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------- final image ----------
FROM python:3.12.5-slim

WORKDIR /workspace

# copy the venv from builder
COPY --from=builder /opt/venv /opt/venv

# ensure the venv python is on PATH
ENV PATH="/opt/venv/bin:$PATH"

# copy your code, model, and the launcher script
COPY inference.py inference_config.json predict.sh model_cpu.pth /workspace/

# make sure predict.sh is executable
RUN chmod +x predict.sh

# (no ENTRYPOINT or CMD needed: the evaluation command
# explicitly does `bash -c "sh predict.sh"`)
