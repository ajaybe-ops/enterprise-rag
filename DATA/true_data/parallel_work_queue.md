# Parallel Processing Using Work Queues

## Overview

In this example you will show how to run a parallel processing application
using a work queue. The example uses one Kubernetes Job to run multiple pods
that coordinate through a Redis service.

OVERVIEW OF STEPS
1. Start a storage service to hold the work queue. (We will use Redis).
2. Create a queue, and fill it with messages.
3. Start a Job that works on tasks from the queue.

## Starting the work queue service

For this example, you will start a single instance of Redis.

Run the following commands:

    kubectl apply -f https://k8s.io/examples/application/job/redis/redis-pod.yaml
    kubectl apply -f https://k8s.io/examples/application/job/redis/redis-service.yaml

The first command creates a Pod running Redis. The second creates a Service
that gives the Pod a stable DNS name inside the cluster, so the worker Pods
can reach it at `redis` regardless of the Pod IP.

Verify the service is up before continuing:

    kubectl get pods -l app=redis
    kubectl get svc redis

## Filling the queue with tasks

Start a temporary interactive Pod for running the Redis CLI:

    kubectl run -i --tty temp --image redis --command "/bin/sh"

Once the prompt appears, connect to Redis and fill the queue (named "job2"):

    redis-cli -h redis

    # Inside the redis prompt:
    rpush job2 "apple"
    rpush job2 "banana"
    rpush job2 "cherry"

    # Check the list:
    lrange job2 0 -1

`rpush` appends to the tail of the list, which preserves FIFO order. `lrange job2 0 -1`
returns every element, which is how you confirm the queue contents before
starting the workers.

## Starting the workers

Now run the Job:

    kubectl apply -f https://k8s.io/examples/application/job/parallel-work-queue.yaml

The manifest defines two Jobs, `job-wq-1` and `job-wq-2`:

    spec:
      parallelism: 2
      template:
        metadata:
          name: job-wq-2
        spec:
          containers:
          - name: c
            image: gcr.io/myproject/job-wq-2
          restartPolicy: OnFailure

The `parallelism` field is set to 2 in the job-wq-2 manifest, meaning a maximum
of 2 Pods run simultaneously. Each pod picks up one unit of work from the Redis
queue, processes it, and repeats until the queue is empty.

In this example, as each pod is created, it picks up one unit of work from a
task queue, processes it, and repeats until the end of the queue is reached.

Note on Parallelism:
In this example, each pod works on several items from the queue and then exits.
Workers signal the queue is empty by exiting with success.

## Why a work queue instead of one big Pod

Splitting work across Pods gives you independent restarts, horizontal scale
without rewriting the application, and natural batching. The queue is the only
shared state, so a lost Pod loses only the item it was holding — the remaining
items are still in Redis and another Pod will pick them up.
