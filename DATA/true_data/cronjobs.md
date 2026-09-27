# CronJobs and Job Specifications

## Overview

A CronJob creates and manages Jobs on a schedule. CronJob is a Kubernetes
resource in the `batch` API group that wraps a Job template and a schedule.

    apiVersion: batch/v1
    kind: CronJob
    metadata:
      name: hello
    spec:
      schedule: "*/1 * * * *"
      jobTemplate:
        spec:
          template:
            spec:
              containers:
              - name: hello
                image: busybox
                command: ["/bin/sh", "-c", "date"]
              restartPolicy: OnFailure

The schedule field uses the standard cron syntax with an optional time zone
suffix, for example `*/1 * * * *` or `H 3 * * 0`. The controller creates a Job
named `<cronjob-name>-<hash>` on each tick. Failed runs count toward
`startingDeadlineSeconds`; once that budget is exhausted, the CronJob stops
creating new Jobs.

## Pod restart policies

Jobs have strict requirements for Pod restart policies:

OnFailure: Kubelet restarts the container within the same Pod if it crashes.
Never: Entire Pod is marked as failed, Job controller creates a new Pod.
Always: Not allowed (would prevent Pod from ever completing).

## Completions and parallelism

Two critical fields control how Jobs execute:

Completions
Specifies the number of Pods that must successfully complete:

    spec:
      completions: 5  # Job completes after 5 successful Pod completions

Parallelism
Controls maximum concurrent Pods:

    spec:
      parallelism: 3  # Max 3 Pods running simultaneously

`completions: 5` with `parallelism: 3` means the Job starts at most three Pods
at a time and keeps replacing finished ones until five have exited successfully.

Other Job spec fields worth knowing:

- `completionMode: Indexed` — each Pod gets an index, so work can be sharded
  deterministically across Pods without coordination.
- `backoffLimit` — how many retries before the Job is marked failed. Default 6.
- `activeDeadlineSeconds` — wall-clock ceiling for the whole Job.
- `ttlSecondsAfterFinished` — how long the finished Job object is retained.

## API Group and Immutability

2. API Group and Immutability
Belongs to the batch/v1 API group (stable).
Pod template cannot be modified after Job creation.
Immutable by design for consistency.

This is a common source of confusion when a Job appears stuck: you cannot patch
the pod template. You must delete the Job and create a new one with the
corrected template.

## Suspending a CronJob

Set `spec.suspend: true` to stop new Jobs from being scheduled without deleting
the CronJob. In-flight Jobs continue to completion.
