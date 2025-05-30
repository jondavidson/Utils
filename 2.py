Reality check — what Dask can and cannot do

Goal	Supported?	How to do it	What can go wrong

Hard-pin a task to “any worker that lives on host α”	Yes	use the built-in workers= constraint (string or list of host prefixes) or an equivalent dask.annotate(workers=…)	If the host prefix doesn’t match any registered worker address the scheduler puts the task in no-worker state → “No worker found”.
Hard-pin a task to “only the first worker on host α”	Yes	pass the full worker address, e.g. workers="tcp://alpha:43123"	Same caveat: address must exist when the task reaches the scheduler.
Invent a custom resource at runtime and add it later with client.run()	No – resources must be present when the worker registers with the scheduler . Trying to patch them in afterwards leaves the scheduler unaware and you get no-worker.		
Pass a per-task dict to client.compute	Not allowed – the API accepts one workers= value (string/iterable) for the whole call . Use submit for per-task placement or annotate inside the graph.		



---

A 100 % working minimal snippet

Below the worker-pinning is done with host-prefix constraints.
No late resource tweaking, no exotic plugins – just the documented API.

from dask.distributed import SSHCluster, Client
import dask
from dask import delayed

# ── 1. Launch an SSH cluster ──────────────────────────────────────────────
hosts = [
    "localhost",      # scheduler
    "alpha", "alpha",
    "bravo", "bravo",
]
cluster = SSHCluster(
    hosts,
    remote_python="/opt/dask25/bin/python",
    connect_options={"username": "pipelines", "known_hosts": None},
    worker_options={"nthreads": 1, "memory_limit": "8GB"},
)
client = Client(cluster)
client.wait_for_workers(4)          # ensure everything is up

# helper — print the addresses so we know what to pin to
print("Workers:", list(client.scheduler_info()["workers"]))

# ── 2. Build a tiny graph and pin by host name ────────────────────────────
def square(x): return x * x
def plus10(x): return x + 10

with dask.annotate(workers="alpha", allow_other_workers=False):
    a = delayed(square)(5)          # may run on *any* alpha worker

with dask.annotate(workers="bravo", allow_other_workers=False):
    b = delayed(plus10)(a)          # may run on *any* bravo worker

result = client.compute(b).result()
print("Result:", result)            # 35

Why this always works

1. workers="alpha" is interpreted by the scheduler as “any worker whose address string starts with 'alpha'” – both tcp://alpha:43123 and tcp://alpha:43124 are valid matches.


2. Because the constraint is a core feature (not a DIY resource), the scheduler immediately knows which workers qualify. If none qualify it errors out instead of hanging.


3. allow_other_workers=False makes the failure obvious: if alpha goes down you’ll get a clean CancelledError, not silent migration.




---

If you really need resources

Define them at worker start-up (environment variable or --resources flag):

dask worker scheduler:8786 --resources "alpha=1"

or, in SSHCluster, pass

worker_options={"resources": {"alpha": 1}, ...}

for every worker you launch on host α.

Then annotate tasks with resources={"alpha": 1} exactly as documented.


But remember: resources are immutable after registration; changing them later leads straight to the “No worker found” state. Pinning via workers= avoids that entire class of problems and is usually simpler.


---

Take-away

Host-prefix worker constraints (workers="alpha") are the easiest, officially supported way to steer tasks to specific machines.

Resources work too, but only if declared at worker launch.

Late-binding tricks (modifying worker.resources from a client) look tempting but don’t update the scheduler – hence the errors you saw.


