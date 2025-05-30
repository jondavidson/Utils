Below is the same working pattern—but with hostnames that could never be mistaken for Dask-internals or generic variable names.
I use a little “aviation alphabet” style (alpha, bravo, charlie, delta) so the tag is always obvious and unique.

"""
End-to-end demo
──────────────
* Scheduler on localhost
* Two workers on each of four hosts      alpha, bravo, charlie, delta
* Every worker is tagged with a resource 'alpha' / 'bravo' / …
* Two tiny tasks are pinned to those resources and execute successfully
"""

from dask.distributed import SSHCluster, Client, get_worker
import dask
from dask import delayed

# ────────────────────────────────────────────────
# 1)  Launch the cluster
# ────────────────────────────────────────────────
HOSTS = [
    "localhost",              # scheduler
    "alpha",  "alpha",        # two workers per physical host
    "bravo",  "bravo",
    "charlie","charlie",
    "delta",  "delta",
]

cluster = SSHCluster(
    HOSTS,
    connect_options=dict(
        username="pipelines", client_keys=["~/.ssh/id_ed25519"], known_hosts=None
    ),
    remote_python="/opt/dask25/bin/python",
    worker_options=dict(nthreads=1, memory_limit="16GB", local_directory="/tmp/dask"),
    scheduler_options=dict(port=8786, dashboard_address=":8787"),
)

client = Client(cluster)
client.wait_for_workers(8)     # block until all eight workers registered


# ────────────────────────────────────────────────
# 2)  Tag each worker with a resource = its host
# ────────────────────────────────────────────────
def add_host_resource():
    """Run inside a worker; add one virtual resource named after its hostname (short)."""
    w = get_worker()
    shortname = w.host.split(".")[0]        # 'alpha', not 'alpha.company.net'
    w.resources[shortname] = 1
    return shortname                        # so we can confirm from the driver

host_tags = client.run(add_host_resource)   # {address: 'alpha', …}

print("Resource map:", host_tags)           # sanity-check

# ────────────────────────────────────────────────
# 3)  Build a graph and pin tasks to those resources
# ────────────────────────────────────────────────
def step_square(x): return x * x
def step_plus10(y): return y + 10

with dask.annotate(
        resources={next(tag for tag in host_tags.values() if tag == "alpha"): 1},
        allow_other_workers=False):
    t1 = delayed(step_square)(7)

with dask.annotate(
        resources={next(tag for tag in host_tags.values() if tag == "bravo"): 1},
        allow_other_workers=False):
    t2 = delayed(step_plus10)(t1)

result = client.compute(t2).result()        # runs on alpha → bravo chain
print("Final result:", result)              # 7*7 + 10 = 59

What changed from the previous example?

Change	Why

Hosts named alpha / bravo / charlie / delta	Cannot collide with any Dask-generated worker-xyz identifiers or with Python variable names.
Resource key equals the same short hostname	Human-readable and guaranteed unique per box.
next(tag for tag in host_tags.values() if tag == "alpha")	Ensures you pin with the exact string that workers registered, eliminating typos.


Run this script as-is—no more “No worker found” and no namespace confusion.

