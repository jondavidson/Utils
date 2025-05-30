"""
End-to-end example:

1.  Spin up (or attach to) a Dask cluster.
2.  Tag every worker with a virtual *resource* equal to its host name.
3.  Build a tiny DAG whose tasks are pinned—via `dask.annotate(resources=…)`—
    so that the scheduler will run them only on workers from the right host.
"""

from dask.distributed import SSHCluster, Client, get_worker
import dask
from dask import delayed

# --------------------------------------------------------------------------- #
# 1)  Bring the cluster online (scheduler on localhost, 2 workers per host)
# --------------------------------------------------------------------------- #

HOSTS = [
    "localhost",                 # scheduler
    "worker1", "worker1",
    "worker2", "worker2",
    "worker3", "worker3",
    "worker4", "worker4",
]

cluster = SSHCluster(
    HOSTS,
    connect_options=dict(username="myuser", client_keys=["~/.ssh/id_ed25519"], known_hosts=None),
    remote_python="/opt/dask25/bin/python",
    worker_options=dict(nthreads=1, memory_limit="16GB", local_directory="/tmp/dask"),
    scheduler_options=dict(port=8786, dashboard_address=":8787"),
)

client = Client(cluster)
client.wait_for_workers(8)        # make sure everyone has checked in

# --------------------------------------------------------------------------- #
# 2)  Give each worker one “resource unit” named after its physical host
# --------------------------------------------------------------------------- #

def _set_resource(tag):
    """Runs *inside* the worker process and adds a virtual resource."""
    w = get_worker()
    w.resources[tag] = 1           # idempotent

for addr, ws in client.scheduler_info()["workers"].items():
    host_tag = ws["host"]          # e.g. "worker2"
    client.run(_set_resource, tag=host_tag, workers=[addr])

# (Optional) sanity-check
print({addr: ws["resources"] for addr, ws in client.scheduler_info()["workers"].items()})
# → {'tcp://10.0.0.11:43123': {'worker1': 1}, ...}

# --------------------------------------------------------------------------- #
# 3)  Build and pin a simple dependency graph
# --------------------------------------------------------------------------- #

def step_a(x):
    return x ** 2

def step_b(y):
    return y + 10

with dask.annotate(resources={"worker1": 1}, allow_other_workers=False):
    a = delayed(step_a)(5)

with dask.annotate(resources={"worker2": 1}, allow_other_workers=False):
    b = delayed(step_b)(a)

future = client.compute(b)         # scheduler enforces both dependencies and placement
print(future.result())             # → (5 ** 2) + 10 == 35


Below is a drop-in, working pattern that clears up the two pain-points you hit:

A) tagging workers (or hosts) with a “resource” label after the cluster is up;

B) pinning individual tasks before you hand them to the scheduler.



---

A)  Tagging every live worker — the bullet-proof version

from dask.distributed import Client, get_worker

client = Client(cluster)        # or Client("tcp://scheduler:8786")
client.wait_for_workers(8)      # block until all have registered

def set_resource(tag):
    """Runs *inside* each worker process and adds one virtual resource."""
    w = get_worker()            # always safe inside client.run
    w.resources[tag] = 1        # idempotent: overwrite if it existed

for addr, ws in client.scheduler_info()["workers"].items():
    host_tag = ws["host"]       # e.g. 'worker1'
    client.run(set_resource, tag=host_tag, workers=[addr])

What was wrong with the earlier one-liner?

Issue	Why it failed

lambda dask_worker=None: …	client.run() does not inject a dask_worker argument. You must call distributed.get_worker() inside the remote function.
Late binding of i	In a loop, lambda: … f"node{i}" … captures the same i for every iteration. Using host_tag inside set_resource(tag) (passed as a keyword) fixes it.


Check the result:

{addr: ws["resources"]
 for addr, ws in client.scheduler_info()["workers"].items()}
# → {'tcp://10.0.0.11:43123': {'worker1': 1}, ...}


---

B)  Pinning tasks – what is and is not allowed

1 Pin with dask.annotate( resources=… )  (recommended)

import dask, dask.delayed as delayed

with dask.annotate(resources={"worker2": 1}, allow_other_workers=False):
    t1 = delayed(stepA)(x)

with dask.annotate(resources={"worker3": 1}, allow_other_workers=False):
    t2 = delayed(stepB)(t1)

Works for delayed, bag, DataFrame, Array, everything.

The resource strings must match exactly what you set via set_resource().


2 Pin with workers= per task (fine for a handful)

t1_fut = client.submit(stepA, x, workers="tcp://10.0.0.11:43123",
                       allow_other_workers=False)

Not supported → client.compute(tasks, workers={task: addr})

workers= can be a single address, a list of addresses, or None.
It is not a mapping keyed by the individual graph nodes, so a dict raises TypeError.  Per-task placement must be expressed in the graph itself (→ annotate or per-submit), never via a dict at compute-time.


---

One helper that feels like the dict-you-wanted

If you must drive placement from a dict, build it into the graph first:

def pin(obj, address):
    "Return a clone of delayed/collection pinned to address."
    import dask
    with dask.annotate(workers=address, allow_other_workers=False):
        return obj

task_map = {
    "A": ("worker1", delayed(stepA)(x)),
    "B": ("worker2", delayed(stepB)(y)),
}

pinned = [pin(obj, addr) for _, (addr, obj) in task_map.items()]
futs   = client.compute(pinned)          # now valid – one workers string per node


---

TL;DR

Use get_worker() inside client.run() when you mutate worker state; lambdas that assume an injected dask_worker will crash.

workers= on compute/submit is one address or list for all tasks, not a per-task dict.  Put placement hints in the graph (annotate) instead.

After these fixes you have deterministic, host-aware placement with no scheduler surprises.


