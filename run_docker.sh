docker run -it --mount "type=bind,source=$(pwd),target=/app/dt" --entrypoint /bin/bash --gpus=all dt_experiments 
