#!/usr/bin/env python
from dask.distributed import SSHCluster, Client

HOSTS = [
    "localhost",          # scheduler
    "worker1", "worker1",
    "worker2", "worker2",
    "worker3", "worker3",
    "worker4", "worker4",
]

REMOTE_PY = "/home/myuser/miniconda3/envs/dask25/bin/python"

CONNECT_OPTS = dict(
    username="myuser",
    client_keys=["~/.ssh/id_ed25519"],
    known_hosts=None,       # relax host-key checking
)

cluster = SSHCluster(
    HOSTS,
    remote_python=REMOTE_PY,
    connect_options=CONNECT_OPTS,
    scheduler_options=dict(port=8786, dashboard_address=":8787"),
    worker_options=dict(nthreads=4, memory_limit="16GB", local_directory="/tmp/dask"),
)

client = Client(cluster)
print("Dashboard:", cluster.dashboard_link)

# ---------- Tag workers by physical host so tasks can say "needs hostX" ----------
client.wait_for_workers(8)
sched_info = client.scheduler_info()

for addr, ws in sched_info["workers"].items():
    host_tag = ws["host"]            # e.g. 'worker1'
    client.run(
        lambda dask_worker=None, tag=host_tag: dask_worker.resources.update({tag: 1}),
        workers=[addr],
    )

print("Worker resources:", {k: w["resources"] for k, w in client.scheduler_info()["workers"].items()})


#!/usr/bin/env python
"""
Turn a dict-of-dicts into delayed tasks, respect depends_on,
optionally pin to a *host* (not an individual worker).
"""

import json, dask
from dask.distributed import Client
from collections import defaultdict, deque

# -------------------------------------------------------------------
# Example tasks.json
# [
#   {"key": "A", "func": "module.foo", "kwargs": {"x": 1}, "depends_on": [],      "host": "worker1"},
#   {"key": "B", "func": "module.bar", "kwargs": {"y": 2}, "depends_on": ["A"],   "host": "worker2"},
#   {"key": "C", "func": "module.baz", "kwargs": {},      "depends_on": ["A","B"]}
# ]
# -------------------------------------------------------------------

def load_func(path):
    """Import 'module.sub:f' or 'module.f' lazily"""
    modpath, _, fname = path.rpartition(".")
    if ":" in path: modpath, _, fname = path.partition(":")
    mod = __import__(modpath, fromlist=[fname])
    return getattr(mod, fname)

def build_graph(task_list):
    # Build adjacency + indegree for topological walk
    children = defaultdict(list)
    indeg = defaultdict(int)
    tasks = {t["key"]: t for t in task_list}
    for t in task_list:
        for dep in t.get("depends_on", []):
            children[dep].append(t["key"])
            indeg[t["key"]] += 1

    # Topological ordering (Kahn)
    q = deque([k for k in tasks if indeg[k] == 0])
    order = []
    while q:
        k = q.popleft(); order.append(k)
        for child in children[k]:
            indeg[child] -= 1
            if indeg[child] == 0: q.append(child)

    delayed_objs = {}
    for key in order:
        t = tasks[key]
        func = load_func(t["func"])
        parents = [delayed_objs[p] for p in t.get("depends_on", [])]

        ann = (
            dask.annotate(resources={t["host"]: 1})
            if t.get("host") else dask.annotate()
        )
        with ann:
            delayed_objs[key] = dask.delayed(func)(*parents, **t.get("kwargs", {}))

    return [delayed_objs[k] for k in order]  # leaves last

if __name__ == "__main__":
    client = Client("tcp://localhost:8786")
    tasks = json.load(open("tasks.json"))
    leaves = build_graph(tasks)
    futs = client.compute(leaves, sync=False)   # dependencies & host constraints are honoured
    print("Submitted", [f.key for f in futs])
