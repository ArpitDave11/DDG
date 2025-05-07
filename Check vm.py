The following Python script removes the Delta table and round-robin logic. It pings two specified VMs and selects one based on reachability as per the requirements:
Define a list of two VM hosts.
Ping both VMs using the is_vm_reachable(vm_host) function (assumed to be defined elsewhere).
If both VMs are reachable, pick one at random using random.choice().
If only one VM is reachable, raise an exception indicating only one VM is available.
If neither VM is reachable, raise an exception indicating no VM was found.
Print which VMs were reachable and what selection was made (if any).
Implementation

Below is the complete Python script implementing the above logic:
import random

# List of VM hosts to check
vms = ['10.191.86.77', '10.191.86.78']

# Ping each VM to check reachability (is_vm_reachable is assumed to be defined elsewhere)
reachable_vms = [vm for vm in vms if is_vm_reachable(vm)]

# Print which VMs were reachable
if reachable_vms:
    print("Reachable VMs:", ", ".join(reachable_vms))
else:
    print("Reachable VMs: None")

# Select a VM or raise exceptions based on reachability
if len(reachable_vms) == 2:
    # Both VMs are reachable, select one randomly
    selected_vm = random.choice(reachable_vms)
    print(f"Both VMs are reachable. Randomly selected VM: {selected_vm}")
elif len(reachable_vms) == 1:
    # Only one VM is reachable, raise an exception
    vm = reachable_vms[0]
    raise Exception(f"Only one VM is available ({vm}). Random selection failed.")
else:
    # No VM is reachable, raise an exception
    raise Exception("No reachable VM found in the list.")
