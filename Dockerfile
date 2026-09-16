# The live feed, for ECS Fargate. The app is not in here and does not
# belong in here: it reads 1.25 GB of history that stays on the machine
# doing the reading, while this process publishes about 2 MB a session.
FROM python:3.12-slim

# Unbuffered, or CloudWatch shows nothing until a buffer happens to fill
# and a crash loop looks like a silent one. This is the single most useful
# line in the file when something goes wrong at 09:14.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first, in their own layer, so editing a module does not
# reinstall pandas on every build.
COPY requirements-feed.txt .
RUN pip install --no-cache-dir -r requirements-feed.txt

# Source only. .dockerignore is an ALLOWLIST: everything is excluded and
# then Python modules, the feed's requirements and the instrument snapshot
# are named back in. So this copies every module - adding an import to
# live_feed cannot produce an image that builds and then dies on start -
# while no data file ships unless someone asks for it by name. The context
# is a few MB, not the 2 GB the working directory holds.
COPY . .

# prewarm writes prior sessions into bar_cache at startup. On Fargate that
# is the task's own ephemeral disk - 20 GB by default, against the ~700 MB
# this needs - and it is discarded when the task stops, which is why the
# task must start well before the open rather than at it.
RUN mkdir -p bar_cache live_bars && \
    useradd --create-home --uid 10001 feed && \
    chown -R feed:feed /app
USER feed

# No port and no health check on purpose: this process only makes outbound
# connections, to Kite and to S3, so the service needs no load balancer
# and nothing should be able to reach it.
CMD ["python", "-u", "live_feed.py"]
