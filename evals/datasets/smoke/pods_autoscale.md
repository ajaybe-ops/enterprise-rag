# Pod Autoscaling in Kubernetes

Autoscaling in Kubernetes ensures that your application can handle varying loads
by automatically adjusting the number of running pods or the resources allocated
to them. This dynamic adjustment is crucial for maintaining application
performance, optimizing resource usage, and reducing operational costs.

## Horizontal Pod Autoscaling (HPA)

Horizontal Pod Autoscaling (HPA) automatically scales the number of pods in a
replication controller, deployment, or replica set based on observed CPU
utilization or other select metrics.

HPA adds or removes pods. It never changes the resource requests or limits of an
existing pod — it changes how many pods exist.

HPA is available in the `autoscaling/v2` API group. A minimal configuration:

    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata:
      name: php-apache
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: php-apache
      minReplicas: 1
      maxReplicas: 10
      metrics:
      - type: Resource
        resource:
          name: cpu
          target:
            type: Utilization
            averageUtilization: 50

## Vertical Pod Autoscaling (VPA)

Vertical Pod Autoscaling adjusts the CPU and memory resource requests and limits
of existing pods. It resizes the resources of existing pods rather than
creating or deleting pods.

The difference in one line: HPA adds or removes pods; VPA resizes the resources
of existing pods.

VPA has two operating modes:

- `Auto` — the controller evicts pods so they can be resized. Requires a restart
  of the pod, so it does not suit latency-sensitive workloads.
- `Recreate` — only applies recommendations at pod creation time.

VPA recommendations can be surfaced without acting on them, by leaving the
update mode unset. That is a common first step: collect recommendations for a
week, then decide.

## Installing the Metrics Server

The Metrics Server is the component HPA reads from. It aggregates resource usage
from the kubelet and exposes the `metrics.k8s.io` API.

Use kubectl apply to download and apply the Components manifest directly from
the latest release of the Metrics Server:

    $ kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

This command fetches the YAML manifest for the latest release of the Metrics
Server from its GitHub repository and applies it to your Kubernetes cluster.

After applying the manifest, verify that the Metrics Server pods are running successfully.
You can check the pods in the kube-system namespace:

    kubectl get pods -n kube-system | grep metrics-server

You should see pods related to the Metrics Server running and ready.

## Verifying that metrics are being collected

Once the Metrics Server is up and running, you can confirm that it's collecting
metrics by querying the API. For example, you can retrieve the CPU and memory
usage metrics for nodes and pods:

    $ kubectl top nodes
    $ kubectl top pods --all-namespaces

If the Metrics Server is properly installed and functioning, you should see CPU
and memory usage metrics for nodes and pods in your cluster.

An HPA that reports `<unknown>` for its target metric almost always means the
Metrics Server is not installed, or the kubelet was not started with
`--horizontal-pod-autoscaler-sync-period` set high enough to be scraped.
