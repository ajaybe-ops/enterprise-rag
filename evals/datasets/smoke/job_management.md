# Job Management — Databricks

Databricks job management can be handled through three developer tools. The
choice depends on what you are optimising for: one-off shell work, a long-lived
application, or something the SDKs do not cover yet.

## The three developer tools

The following table compares the Databricks CLI, the Databricks SDKs, and the REST API:

Databricks CLI: Access Databricks functionality using the command-line interface
(CLI), which wraps the REST API. Use the CLI for one-off tasks such as
experimentation, shell scripting, and invoking the REST API directly.

Databricks SDKs: Develop applications and create custom Databricks workflows
using a Databricks SDK, available for Python, Java, Go, or R.

Databricks REST API: Use directly for use cases where an SDK is not available.

If none of the above options work for your specific use case, you can use the
Databricks REST API directly. Use the REST API directly for use cases such as
automating processes where an SDK in your preferred programming language is not
currently available.

## Installing the CLI

    curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

Authenticate once, and the CLI stores a token for subsequent calls:

    databricks configure
    databricks auth login --host https://<workspace-host>

Every command accepts `--host` and `--profile`, so CI jobs should use a profile
rather than an interactive login.

## Retrieving a single job

To print information about an individual job in a workspace, run the following command:

    databricks jobs get <job-id>

    # Example
    databricks jobs get 478701692316314

This command returns JSON with fields like job_id, settings, creator_user_name,
and task configuration.

To list every job in the workspace, including whether it is currently running:

    databricks jobs list

## Creating and running jobs

    databricks jobs create --json @job-spec.json
    databricks jobs run <job-id>

To follow a run to completion and stream its logs:

    databricks jobs run-now <job-id> --watch

## When to use the REST API directly

Use the REST API directly when none of the supported SDK languages (Python,
Java, Go, R) are available for your use case, or when you need to automate
processes for which no SDK currently exists in your preferred programming
language.

The REST API base path is:

    https://<workspace-host>/api/2.1

Jobs endpoints are prefixed accordingly, e.g. `POST /api/2.1/jobs/run-now` with
a JSON body of `{"job_id": <id>}`.
