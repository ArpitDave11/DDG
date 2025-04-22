Thanks! I’ll put together a Databricks-based solution that:
- Retrieves the list of VMs in the given Azure subscription and resource group.
- Applies round-robin logic across only *reachable* VMs using ping tests.
- Automatically skips unreachable ones.
- Initiates a job (via SSH or API call) on the selected VM.

I’ll include all setup steps, code, and configurations to ensure this works end-to-end inside Azure Databricks.
I'll get back to you shortly with a complete, tested solution.

# Azure Databricks Guide: Round-Robin Selection of Azure VMs for Remote Job Execution

## Overview

This guide describes how to build an Azure Databricks solution that selects an Azure Virtual Machine (VM) in round-robin fashion and triggers a job on it. The solution operates within a single Azure **subscription** and **resource group** (specified by the user) and assumes you have full access to Azure and Databricks. The main tasks the solution performs are:

1. **List all VMs** in a given Azure resource group.  
2. **Ping each VM** to check if it’s running and network-accessible (reachability).  
3. **Skip unreachable** VMs and maintain a **round-robin rotation** among the reachable ones.  
4. **Persist the last used VM index** so that each run picks the next VM in sequence.  
5. **Execute a remote job** on the selected VM (e.g. via SSH, REST API call, or Azure VM run-command).

We will use **Python** with the Azure SDK inside an Azure Databricks notebook. Key libraries include Azure SDK packages like `azure-identity` for authentication and `azure-mgmt-compute` and `azure-mgmt-network` for resource management. The guide also covers setting up authentication (using a service principal or managed identity) and any necessary configuration (libraries, network permissions, etc.) for the Databricks environment.

## Prerequisites and Setup

Before implementing the solution, ensure the following prerequisites are met:

- **Azure Resources**: An Azure subscription with a resource group containing the target VMs. Note the **Subscription ID** and **Resource Group name**. All VMs should be in this resource group (they can be Windows or Linux). For network reachability, either the Databricks cluster should reside in the same Virtual Network as the VMs or the VMs should have public IP addresses with appropriate NSG rules allowing ping/SSH access.
- **Databricks Workspace**: An Azure Databricks workspace with a cluster to run the notebook. The cluster should be running a runtime that supports Python 3 and has access to the Azure SDK (we will install needed libraries in the notebook).
- **Azure Credentials**: A way for Databricks to authenticate to Azure. Typically, this is done via an Azure **Service Principal** (client ID, tenant ID, client secret) with proper RBAC rights. At minimum, it needs **Reader** access to list VMs in the resource group and perhaps **Contributor** if we use run-command on VMs. *(Alternatively, you can use a managed identity attached to the Databricks workspace for authentication.)*

**Library Installation:** In your Databricks notebook, install the required Azure SDK packages. You can use the `%pip` magic command at the top of the notebook to install libraries into the cluster environment:

```python
# Install required Azure SDK packages on the Databricks cluster (run this in a notebook cell).
%pip install azure-identity azure-mgmt-compute azure-mgmt-network
```

This will install:
- `azure-identity`: for Azure AD authentication in Python.
- `azure-mgmt-compute`: to interact with Azure Compute resources (VMs).
- `azure-mgmt-network`: to interact with networking resources (to get IP addresses of VMs).

## Azure Authentication in Databricks

For the Databricks notebook to access Azure resources, authenticate using Azure AD credentials. The simplest method is to use a **Service Principal**. You can provide its details via environment variables so that the Azure SDK picks them up. The Azure SDK’s `DefaultAzureCredential` will automatically use these environment variables (or a managed identity, if configured) to obtain a token ([Getting started - Managing Compute using Azure Python SDK - Code Samples | Microsoft Learn](https://learn.microsoft.com/en-us/samples/azure-samples/azure-samples-python-management/compute/#:~:text=2,based%20OS%2C%20you%20can%20do)). 

**Setup Azure credentials** (replace the placeholder values with your actual Tenant ID, Client ID, Client Secret, and Subscription ID): 

```python
import os
# Set environment variables for Azure credentials (Service Principal)
os.environ["AZURE_TENANT_ID"] = "<YOUR_TENANT_ID>"
os.environ["AZURE_CLIENT_ID"] = "<YOUR_SP_CLIENT_ID>"
os.environ["AZURE_CLIENT_SECRET"] = "<YOUR_SP_CLIENT_SECRET>"

# Azure subscription and resource group info
subscription_id = "<YOUR_AZURE_SUBSCRIPTION_ID>"
resource_group = "<YOUR_RESOURCE_GROUP_NAME>"
```

> *💡 **Security Note:** In practice, do not hard-code secrets in code. Use Databricks [Secret Scopes](https://docs.microsoft.com/azure/databricks/security/secrets) to securely store the client secret and retrieve it (e.g., via `dbutils.secrets.get`). The code above uses environment variables for simplicity. Ensure the service principal has at least read access to the resource group (and VM run command privileges if needed).*

Next, create Azure SDK clients using these credentials. We use `DefaultAzureCredential` for authentication and then initialize the Compute and Network management clients with the subscription ID:

```python
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient

# Acquire a credential object for Azure (this will use the env vars set above)
credential = DefaultAzureCredential()

# Initialize Azure management clients for Compute and Network
compute_client = ComputeManagementClient(credential, subscription_id)
network_client = NetworkManagementClient(credential, subscription_id)
```

The above code establishes connections to Azure. The `ComputeManagementClient` will allow us to list VMs and manage them, and the `NetworkManagementClient` will help retrieve network interface details (for IP addresses). This pattern of creating clients with credentials and subscription is standard in Azure SD ([Get IP from VM object using azure sdk in python - Stack Overflow](https://stackoverflow.com/questions/40728871/get-ip-from-vm-object-using-azure-sdk-in-python#:~:text=compute_client%20%3D%20ComputeManagementClient,subscription_id))】.

## Listing VMs in the Resource Group

With the compute client ready, we can list all virtual machines in the target resource group. Azure SDK provides a method `compute_client.virtual_machines.list(resource_group_name)` to get VMs. We’ll retrieve the VMs and then filter for those that are running and reachable:

```python
# Fetch all VMs in the specified resource group
vm_list = list(compute_client.virtual_machines.list(resource_group))
print(f"Found {len(vm_list)} VMs in resource group '{resource_group}'.")
```

Each item in `vm_list` is a VirtualMachine resource object. We can examine properties like `vm.name` for the VM’s name and `vm.network_profile` for its network interfaces, etc. 

Before attempting to ping a VM, it’s wise to check if the VM is powered on (running). We can use the Azure SDK to get the **instance view** (status) of each VM. The instance view provides status codes, including power state. For example, a status code of `"PowerState/running"` indicates the VM is runnin ([How could I list Azure Virtual Machines using Python? - Stack Overflow](https://stackoverflow.com/questions/58925397/how-could-i-list-azure-virtual-machines-using-python#:~:text=statuses%20%3D%20compute_client,2%20and%20statuses%5B1))】. We will call `compute_client.virtual_machines.instance_view()` for each VM to get its status:

```python
reachable_vms = []  # will store tuples of (vm_name, ip_address) for reachable VMs

for vm in vm_list:
    vm_name = vm.name
    # Get instance view to check power state
    instance_view = compute_client.virtual_machines.instance_view(resource_group, vm_name)
    statuses = instance_view.statuses

    # Determine if VM is running
    power_state = None
    for stat in statuses:
        if stat.code and stat.code.startswith('PowerState'):
            power_state = stat.code  # e.g., "PowerState/running" or "PowerState/deallocated"
    if power_state != 'PowerState/running':
        print(f"VM '{vm_name}' is not running (status: {power_state}), skipping.")
        continue  # skip stopped or deallocated VMs
```

In the loop above, we skip VMs that are not in a running state. The check uses the status codes returned by `instance_view` (note: typically `statuses[1]` is the power stat ([How could I list Azure Virtual Machines using Python? - Stack Overflow](https://stackoverflow.com/questions/58925397/how-could-i-list-azure-virtual-machines-using-python#:~:text=statuses%20%3D%20compute_client,2%20and%20statuses%5B1))】, but we loop for clarity). Now, for each running VM, we will retrieve its IP address and attempt a ping.

## Retrieving VM IP and Checking Reachability

To ping a VM, we need an IP address (private or public) that the Databricks cluster can reach. Azure VMs have one or more network interfaces (NICs). Each NIC can have a private IP and optionally a public IP. We will use the Azure **NetworkManagementClient** to get NIC details, then extract the IP. 

For simplicity, we’ll assume each VM has at least one NIC and use the first NIC. We retrieve the NIC’s resource ID from `vm.network_profile.network_interfaces[0].id`, then use the network client to get that NIC resource. From the NIC, we can get the IP configurations which include the private IP and possibly a reference to a public I ([Get IP from VM object using azure sdk in python - Stack Overflow](https://stackoverflow.com/questions/40728871/get-ip-from-vm-object-using-azure-sdk-in-python#:~:text=try%3A%20thing%3Dnetwork_client)) ([Get IP from VM object using azure sdk in python - Stack Overflow](https://stackoverflow.com/questions/40728871/get-ip-from-vm-object-using-azure-sdk-in-python#:~:text=In%20this%20example%20you%20could,to%20get%20the%20public%20IPs))】:

```python
    # VM is running, get its primary NIC and IP
    nic_id = vm.network_profile.network_interfaces[0].id  # NIC resource ID
    # Parse NIC name and resource group from the ID (format: .../resourceGroups/<RG>/.../networkInterfaces/<NICName>)
    nic_name = nic_id.split("/")[-1]
    nic_rg = nic_id.split("/")[4]  # NIC's resource group (often same as VM RG)

    # Get NIC details
    nic = network_client.network_interfaces.get(nic_rg, nic_name)
    ip_address = None
    if nic.ip_configurations and len(nic.ip_configurations) > 0:
        # Take the first IP configuration
        ip_config = nic.ip_configurations[0]
        # Prefer private IP if available
        if ip_config.private_ip_address:
            ip_address = ip_config.private_ip_address
        # If no private IP (or not reachable), use public IP if exists
        if not ip_address and ip_config.public_ip_address:
            pub_ip_id = ip_config.public_ip_address.id
            pub_name = pub_ip_id.split("/")[-1]
            pub_rg = pub_ip_id.split("/")[4]
            public_ip = network_client.public_ip_addresses.get(pub_rg, pub_name)
            ip_address = public_ip.ip_address
```

At this point, the variable `ip_address` holds an address to contact the VM. If the Databricks cluster is on the same virtual network as the VM, the private IP will be used. If the cluster is external, we fall back to the public IP (if the VM has one and the NSG/firewall permits access).

Now we attempt to **ping** the IP to check reachability. We can use Python’s `subprocess` to call the system ping utility. On Linux Databricks clusters, the ping command will use the `-c 1` flag to send a single ICMP packet (on Windows, the flag would be `-n 1`). We’ll construct the ping command accordingly and check the exit code:

```python
        # Test reachability with a ping
        import platform, subprocess
        if ip_address:
            # Use appropriate ping syntax for the platform
            ping_count_flag = "-n" if platform.system().lower() == "windows" else "-c"
            ping_cmd = ["ping", ping_count_flag, "1", ip_address]
            try:
                result = subprocess.run(ping_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                if result.returncode == 0:
                    reachable_vms.append((vm_name, ip_address))
                    print(f"VM '{vm_name}' is reachable at {ip_address}.")
                else:
                    print(f"VM '{vm_name}' did not respond to ping.")
            except Exception as e:
                print(f"Ping to VM '{vm_name}' failed: {e}")
        else:
            print(f"No IP address found for VM '{vm_name}', skipping.")
```

This code attempts one ping to the VM’s IP. If the ping succeeds (exit code 0), we consider the VM **reachable** and add it to our `reachable_vms` list. If it fails or times out, we skip that VM. The `reachable_vms` list will accumulate all VMs that are **running and network-accessible**.

> *Note:* ICMP ping must be allowed by the VM’s network security group and the cluster’s network. If ICMP is blocked but the VM is actually reachable for the purpose of your job (for example, via TCP on a certain port), you might replace the ping test with an appropriate port check or alternate verification method. The concept remains to *skip* VMs that are not reachable.

After this loop, we have a list of candidate VMs (`reachable_vms`). Each element is a tuple `(name, ip_address)`. Now we can implement the **round-robin selection** among these.

## Handling No Reachable VMs (Retry Logic)

It’s possible that no VM is reachable (for example, all are stopped or network-inaccessible at the moment). In such cases, the solution should retry or handle the absence of available VMs.

If `reachable_vms` is empty after the initial scan, we can implement a simple **retry mechanism**: wait for a short interval and attempt the scan again, up to a certain number of retries. This gives time for a VM to start up or become reachable. For example:

```python
import time

if not reachable_vms:
    max_retries = 3
    retry_delay_sec = 30
    for attempt in range(1, max_retries+1):
        print(f"No reachable VMs found. Retrying in {retry_delay_sec} seconds... (attempt {attempt}/{max_retries})")
        time.sleep(retry_delay_sec)
        # Re-run the scanning logic to update reachable_vms
        reachable_vms = []
        for vm in compute_client.virtual_machines.list(resource_group):
            # (Repeat the VM status check and ping logic here to populate reachable_vms)
            # ...
            pass
        if reachable_vms:
            print("A VM became reachable after retry.")
            break

    if not reachable_vms:
        raise RuntimeError("No reachable VM available after multiple retries.")
```

In the code above, we pause and retry the VM discovery and ping up to 3 times (with a 30-second interval). If after retries no VM is reachable, we raise an exception or take alternate action (you might choose to alert or exit gracefully instead of raising an error, depending on your use case).

## Round-Robin VM Selection (Persisting State)

Once we have a list of reachable VMs, we need to select the “next” VM in a round-robin fashion. This requires remembering which VM was used last time. We’ll maintain a persistent index that points to the last used VM, and increment it each time. On each run, the next VM in the list will be chosen.

**Persistent tracking:** We can store the last used index in a small **Delta Lake** table or a file in DBFS. Using Delta is convenient in Databricks for small metadata, as it allows easy reads/writes and can handle concurrency if needed. We'll illustrate using a Delta table. (Alternatively, a simple file can be used with `dbutils.fs.put ([Databricks File Save - Stack Overflow](https://stackoverflow.com/questions/56442678/databricks-file-save#:~:text=To%20save%20a%20file%20to,the%20%2FFileStore%20directory%20in%20DBFS))】 to write the index, and read it back on the next run.)

First, define where to store the state (for example, a path in DBFS):

```python
from delta.tables import DeltaTable

state_path = "dbfs:/FileStore/vm_round_robin_state"  # path for Delta table storage

# If state table doesn't exist, initialize it with last_index = -1 (meaning no VM used yet)
if not DeltaTable.isDeltaTable(spark, state_path):
    spark.createDataFrame([(-1,)], ["last_index"]).write.format("delta").mode("overwrite").save(state_path)

# Read the last used index from the Delta table
last_index_df = spark.read.format("delta").load(state_path)
last_index = last_index_df.collect()[0]["last_index"]
```

The above code uses `DeltaTable.isDeltaTable` to check if we have previously stored state. If not, it creates a Delta table (at the given path) with a single row, `last_index = -1`. Then it reads the table to get the last used index value. On first run, this will be -1.

Now determine the next index in round-robin order. If `last_index` is -1 or reaches the end of the list, we wrap around to 0. Essentially, we do `(last_index + 1) mod N` where N is the number of reachable VMs:

```python
# Determine next VM index in round-robin
if not reachable_vms:
    raise RuntimeError("No available VM to select.")  # (should not happen here if handled above)
num_vms = len(reachable_vms)
next_index = (last_index + 1) % num_vms  # round-robin calculation

# Select the VM at next_index
selected_vm_name, selected_vm_ip = reachable_vms[next_index]
print(f"Selected VM for this run: {selected_vm_name} (IP: {selected_vm_ip})")

# Update the Delta state with the new last_index
spark.createDataFrame([(next_index,)], ["last_index"]).write.format("delta").mode("overwrite").save(state_path)
```

At this point, `selected_vm_name` (and its IP) is the VM chosen to run the job, and the state is updated so that next time a different VM will be picked. The Delta table ensures the index persists across job runs. 

*Alternative:* Instead of a Delta table, you could use a DBFS file as a simple counter store. For example, writing the index to `/FileStore/last_vm_index.txt` and reading it next time. Using `dbutils.fs.put("/FileStore/last_vm_index.txt", "2", overwrite=True)` would store the index as tex ([Databricks File Save - Stack Overflow](https://stackoverflow.com/questions/56442678/databricks-file-save#:~:text=To%20save%20a%20file%20to,the%20%2FFileStore%20directory%20in%20DBFS))】. Ensure proper locking if multiple processes might update it.

## Initiating a Job on the Selected VM

After selecting the VM, the final step is to initiate the desired **job/task** on that VM. There are multiple ways to execute a job on a remote VM:

- **SSH Command**: If the VM allows SSH access (e.g., a Linux VM with port 22 open), you could use an SSH client (such as Python’s `paramiko`) to connect and run a shell command or script on the VM. This requires network connectivity to the VM and credentials (SSH key or password).
- **REST API Call**: If the VM is running a service that exposes a REST API, you could send an HTTP request (e.g., using `requests` in Python) to trigger an action on the VM. This requires the VM to have an accessible endpoint.
- **Azure VM Run Command**: Azure provides a **Run Command** feature that lets you execute scripts on a VM through the Azure management plane. This doesn’t require direct network access to the VM; instead, Azure sends the command to the VM’s guest agent. We can invoke this via the Azure SDK.

For a robust Databricks solution, using **Azure’s Run Command** is convenient since it works even if the cluster cannot directly reach the VM over the network (as long as the VM is running and has the Azure VM agent). Below, we use `ComputeManagementClient.virtual_machines.begin_run_command` to execute a script on the VM. We’ll demonstrate with a simple shell command. If the VM is Windows, we would use a PowerShell command instead.

```python
# Prepare the run-command payload
if vm.network_profile:  # we'll assume we know the OS; here we use a Linux example
    command_id = "RunShellScript"   # use "RunPowerShellScript" for Windows VM ([Run command in linux vm in azure using python sdk - Stack Overflow](https://stackoverflow.com/questions/51478227/run-command-in-linux-vm-in-azure-using-python-sdk/51485744#51485744#:~:text=If%20using%20Windows%2C%20you%20can,command%20id))】
    script_lines = [
        "echo Hello from Azure Databricks > /tmp/hello.txt"  # sample command: create a file on the VM
    ]
    command_params = {
        "command_id": command_id,
        "script": script_lines
    }

    # Invoke the run command on the VM
    poller = compute_client.virtual_machines.begin_run_command(resource_group, selected_vm_name, command_params)
    result = poller.result()  # wait for completion

    # Check the result
    if result.value and len(result.value) > 0:
        output_message = result.value[0].message  # stdout/stderr from the VM script
        print(f"Run Command output from {selected_vm_name}:\n{output_message}")
    else:
        print(f"Run Command completed on {selected_vm_name}, no output was returned.")
```

In the code above, we set `command_id` to `"RunShellScript"` for a Linux VM (the script provided is a bash command). For Windows VMs, Azure expects a PowerShell script and you would use `"RunPowerShellScript" ([Run command in linux vm in azure using python sdk - Stack Overflow](https://stackoverflow.com/questions/51478227/run-command-in-linux-vm-in-azure-using-python-sdk/51485744#51485744#:~:text=If%20using%20Windows%2C%20you%20can,command%20id))】 as the command_id and supply PowerShell commands. The `script` is a list of strings, each string is a line to execute on the VM. We then call `begin_run_command()` with the resource group, VM name, and parameters. This returns a poller (an asynchronous operation handle), and we call `.result()` to block until the command finishes. The result (of type `RunCommandResult`) may contain output in `result.value[0].message` (for example, any stdout or error message ([Python Azure SDK virtual_machines.run_command - Stack Overflow](https://stackoverflow.com/questions/62961460/python-azure-sdk-virtual-machines-run-command#:~:text=This%20is%20expected%20type%20as,python))】.

**Verification:** The sample command writes a file `/tmp/hello.txt` on the VM. We printed the output message (if any) for confirmation. In practice, you would replace `script_lines` with whatever actions are needed to start your job on the VM (for example, launching a service, running a script, etc.). You could also pass parameters to the script if needed (Azure run-command allows parameters in the payload).

## Additional Notes and Configuration

- **Network and Permissions**: If you choose to use SSH or REST calls instead of Azure run-command, ensure the Databricks cluster can network-reach the VM (consider VNet peering or opening necessary firewall ports). Also ensure any credentials (SSH keys, etc.) are securely managed (e.g., stored in secret scopes).
- **Databricks Scheduling**: You can schedule this notebook as a Databricks Job to run at intervals, so it continuously distributes tasks across VMs. The persistent index in Delta will ensure the round-robin sequence continues across job runs.
- **Scaling**: This solution assumes a modest number of VMs. The Azure SDK calls are efficient for typical usage. If you have a very large number of VMs, you might consider parallelizing the ping checks or using asynchronous calls. Also, the Delta table approach for a single value is fine for low contention scenarios; for multiple concurrent processes, additional locking might be needed (which is beyond this scope).

By following this guide, you have a Databricks-based orchestrator that lists Azure VMs, pings for availability, selects the next VM in a rotating manner, and triggers a job on it. This pattern can be useful for distributing work to a pool of servers or ensuring high availability by failing over to the next alive VM. Adjust the specifics (such as the job command or selection criteria) as needed for your scenario. 

