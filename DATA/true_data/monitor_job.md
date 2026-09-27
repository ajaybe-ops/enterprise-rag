# Monitoring a Job

## Starting a local cluster

Start Minikube to create a local Kubernetes cluster:

    minikube start --driver=docker

This command will take a few minutes to complete. You'll see a series of
messages as Minikube downloads necessary components and starts the cluster.

Once complete, you should see a message like:

    Done! kubectl is now configured to use "minikube" cluster and "default" namespace by default

The `--driver=docker` flag selects the Docker driver. Other common drivers are
`virtualbox`, `hyperv` and `none` (existing cluster). On Windows, the Docker
driver is usually the fastest option when Docker Desktop is already running.

## Verifying core components

Check that the core Kubernetes components are running:

    kubectl get pods -n kube-system

You should see several pods running, including components like kube-apiserver,
kube-controller-manager, and others.

## Checking job status

The primary tool for monitoring Kubernetes resources is kubectl. Let's use it to
check the status of our job:

    kubectl get jobs

You should see output similar to this:

    NAME        COMPLETIONS   DURATION   AGE
    hello-job   1/1           10s        30s

For more detailed information:

    kubectl describe jobs/hello-job

`describe` is where you find the events block. Scheduling failures
(`0/1 nodes are available: insufficient cpu`), image pull back-offs and
`BackoffLimitExceeded` all show up there.

To view logs from a completed pod:

    kubectl logs pods/<pod-name>

To follow logs live, add `-f`. To get logs from every pod of a Job:

    kubectl logs -l job-name=<job-name> --all-containers

## Watching a Job in a loop

For a quick status refresh without re-typing the command:

    watch -n 1 kubectl get jobs

On Windows PowerShell:

    while ($true) { kubectl get jobs; Start-Sleep -Seconds 1 }

## Cleaning up

    kubectl delete job hello-job
    kubectl delete cronjob hello
    minikube stop
    minikube delete
