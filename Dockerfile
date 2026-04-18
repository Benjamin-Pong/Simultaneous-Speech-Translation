FROM python:3.11

WORKDIR /home/simulstream_agent

COPY --from=simulstream_base config ./config
COPY --from=simulstream_base simulstream ./simulstream
COPY --from=simulstream_base pyproject.toml ./

# Install first — these get cached
RUN pip install torch --index-url https://download.pytorch.org/whl/cu130
RUN pip install -e .[canary]

# Copy your custom files after — changes here won't invalidate pip cache
COPY dynamic_sliding_window.py ./simulstream/server/speech_processors/
COPY model_sm4t.py ./simulstream/server/speech_processors/
COPY dynamic_sliding_window.yaml ./config/

EXPOSE 8080

CMD [ \
    "python", \
    "-m", \
    "simulstream.server.speech_processors.remote.http_speech_processor_server", \
    "--speech-processor-config", \
    "config/dynamic_sliding_window.yaml" \
]