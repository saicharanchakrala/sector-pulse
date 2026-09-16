# Running the live feed on ECS

The feed streams Kite ticks, buckets them into bars, and publishes about
2 MB a session to S3. The app stays on your laptop and reads those bars
from S3 instead of from local disk. Nothing else moves: the 754 MB
`bar_cache` and 559 MB `forecast_cache` are history the app reads, and
they stay next to the app that reads them.

Everything AWS-side is named **zone-pulse**, in account `322535271012`,
region `ap-southeast-2`, cluster `avsp-cluster`.

| Resource | Name |
|---|---|
| ECR repository | `zone-pulse` |
| ECS service and task family | `zone-pulse` |
| Task and execution role | `Innomesh-ecs-task-execution-role` (existing, shared) |
| Log group | `/ecs/zone-pulse` |
| S3 prefix | `s3://original-image-test/zone-pulse/` |
| SSM parameter | `/zone-pulse/kite-access-token` |

## One-time setup

```powershell
powershell -ExecutionPolicy Bypass -File deploy\setup.ps1
powershell -ExecutionPolicy Bypass -File deploy\push_image.ps1   # needs Docker Desktop running
. deploy\env.ps1
.venv\Scripts\python publish_turnover.py
```

`env.ps1` is **dot-sourced** - the leading `. ` matters. It reads
`deploy/env.vars`, a plain KEY=VALUE file that `run_nightly.cmd` parses
too: cmd cannot dot-source a PowerShell script, and two copies of the
same four values is how the nightly job came to run with the bucket
unset, silently skipping the two steps that publish to S3. It sets the three
`SECTOR_PULSE_S3_*` variables plus `AWS_PROFILE` in your own session.
Setting them inline instead is easy to get wrong, and getting it wrong is
not loud: `publish_turnover.py` refuses outright, but the app just carries
on reading local disk and shows a feed frozen at whatever a local run last
wrote. `. deploy\env.ps1 -Off` puts it back.

`setup.ps1` registers the task definition and creates the service at
**desiredCount 0**, so nothing starts and nothing costs anything until you
scale it. Safe to re-run.

Two one-time pieces are NOT in it any more, because they are already done
and re-running them every time only adds ways to fail:

* the **bucket policy** granting `Innomesh-ecs-task-execution-role`
  `s3:GetObject` on `s3://original-image-test/zone-pulse/*` - the one
  permission that role was missing. `deploy/bucket-policy.json` still
  holds it if it ever needs recreating.
* the **`/ecs/zone-pulse` log group**, 30-day retention.

There is no manual credential step. `setup.ps1` reads your API key from
`.kite_session.json` and inlines it into the task definition, so the
repository never carries it.

## Every trading morning

Kite access tokens die around 06:00 IST and a new one needs an interactive
login with your password and 2FA. There is no browser on Fargate, so the
login stays here and only the token travels.

1. Log in through the app as you always have.
2. `powershell -ExecutionPolicy Bypass -File deploy\push_token.ps1`

That stores the token and scales the service to 1.

**Start it by about 08:45.** A container begins with an empty `bar_cache`,
so `prewarm` refetches roughly 2,500 symbols of prior sessions from Kite at
three requests a second - about 14 minutes. Starting at 09:15 means the
feed is still downloading when the session opens.

After the close: `aws ecs update-service --cluster avsp-cluster --service zone-pulse --desired-count 0 --profile innomesh-dev`

## Nightly, on this machine

```powershell
.venv\Scripts\python fetch_tail.py          # fold the day into the daily store
.venv\Scripts\python publish_turnover.py    # publish the ranking the feed streams on
```

Measured 2026-09-15: 2,504 symbols, 49 KB - against the 559 MB store it
came from.

`run_nightly.cmd` does both. The second one matters: `live_feed.by_turnover`
ranks the streaming universe on 20-session turnover from the 559 MB daily
store, which a container does not have. `publish_turnover.py` uploads just
the ranking - about 60 KB. Without it the feed falls back to the ~216 F&O
underlyings and quietly streams a far narrower universe than you think.

## Pointing the app at S3

```powershell
. deploy\env.ps1
.venv\Scripts\streamlit run app.py
```

which sets:

```
SECTOR_PULSE_S3_BUCKET=original-image-test
SECTOR_PULSE_S3_PREFIX=zone-pulse
SECTOR_PULSE_S3_REGION=ap-southeast-2
AWS_PROFILE=innomesh-dev
```

`AWS_PROFILE` is in there because boto3 on this machine has no default
profile, and without it the failure is a `NoCredentialsError` surfacing as
a StorageError that reads like a broken bucket rather than a missing
profile.

Leave them unset and everything behaves exactly as it did before any of
this existed, reading and writing local disk. That is what lets the whole
test suite go on describing local behaviour.

## Cost

Roughly $0.05/hour for 1 vCPU and 2 GB while running. At about 7 hours a
weekday that is **$7-8/month**, plus a few cents of S3. Scaled to zero
overnight and at weekends it costs nothing but storage.

Watch the S3 egress if you widen the autoscan: the app re-reads the day's
parquet every 20 seconds, which is roughly 2.3 GB a session pulled back
across the network.

## How the credentials are split, and what it costs you

**The API key is an identifier, not a credential.** It travels in Kite's
own login URL and in the websocket URL, and on its own it can do nothing -
every call needs the access token as well. So it rides as a plain
environment variable in the task definition.

**The access token is the real credential** and stays a SecureString in
SSM. It places orders.

It is encrypted under the account's default `aws/ssm` key, which has a
consequence worth stating plainly: the shared
`Innomesh-ecs-task-execution-role` carries `AmazonSSMReadOnlyAccess` on
`"*"`, so anything else in this account running under that role can read
the token back. An earlier version of this setup used a dedicated KMS key
and dedicated roles to close that off; reusing the shared role trades that
isolation for one less moving part.

If that trade stops feeling right, the fix is a dedicated task role plus a
customer-managed key - about $1/month and one extra step in `setup.ps1`.

**`KITE_API_SECRET` never reaches AWS at all.** It is only needed to
exchange a request token for an access token, which happens on this
machine.

## Things worth knowing

**The shared role was missing exactly one permission.** Verified with
`simulate-principal-policy`: `s3:PutObject` allowed, `s3:ListBucket`
allowed, `s3:GetObject` implicitDeny. The feed reads exactly one object -
`turnover.parquet` - so `setup.ps1` grants that with a **bucket policy**
rather than by editing the role. Nothing shared changes, and the blast
radius is one prefix of one bucket. `setup.ps1` refuses to proceed if the
bucket ever acquires a policy of its own, rather than replacing it.

**The cluster is in Sydney, not Mumbai.** Kite's servers are in India, so
every request carries an extra 150-200ms. That is irrelevant to 3-minute
bars and to a persistent websocket, but it does stretch the prewarm. The
open question is whether Kite is reachable from an Australian IP at all -
validate that on the first run before relying on it.

**The file lock is skipped on Fargate.** `live_feed` guards against two
feeds with a lock file, which is meaningless on an ephemeral container
disk. ECS gives the real guarantee instead: `desiredCount 1` plus
`maximumPercent=100` on the service, so the scheduler stops the old task
before starting a new one. Without that second half the default would
briefly run two feeds. Do not run a local feed at the same time - Kite
allows three sockets per key and two feeds double the tick load for
nothing.

## Troubleshooting

```powershell
aws logs tail /ecs/zone-pulse --follow --profile innomesh-dev
aws ecs describe-services --cluster avsp-cluster --services zone-pulse --profile innomesh-dev
```

| Symptom | Cause |
|---|---|
| Task stops immediately, `ResourceNotFoundException` | No access token stored yet - run `push_token.ps1` |
| Feed logs "No turnover.parquet published yet" | Run `publish_turnover.py` with the S3 env vars set |
| `StorageError ... AccessDenied` on startup | The bucket policy is missing - re-run `setup.ps1` step 1 |
| Feed logs "Running on ECS with no object store configured" | `SECTOR_PULSE_S3_BUCKET` missing from the task definition |
| App shows no live bars but the feed is running | Check the three env vars above are set where Streamlit launched |
| `InvalidParameterException ... logGroupName` from a Git Bash shell | MSYS rewrites `/ecs/zone-pulse` into `C:/Program Files/Git/ecs/...`. Run these in PowerShell, or prefix with `MSYS_NO_PATHCONV=1` |
| `Error parsing parameter 'cli-input-json': Invalid JSON received` | A UTF-8 BOM. `Set-Content -Encoding utf8` writes one on PowerShell 5.1; the scripts use `[System.IO.File]::WriteAllText` with `UTF8Encoding($false)` instead |
| A script prints success over a blank value | A failing native command does not throw even under `ErrorActionPreference = Stop`. The scripts check `$LASTEXITCODE` after every `aws` call |
| `ZoneInfoNotFoundError` | `tzdata` missing from the image; it is in `requirements-feed.txt` for exactly this reason |
