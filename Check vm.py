import itertools
import subprocess
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient

def get_vm_list(subscription_id):
    """
    Retrieve all VMs in the given Azure subscription.
    Returns a list of VM objects (SDK models or simple dicts).
    """
    client = ComputeManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id
    )
    return list(client.virtual_machines.list_all())

def ping_test(vm):
    """
    Ping the given VM’s IP address.
    Returns (ip_address, vm_name) if ping succeeds, else (None, None).
    """
    # Support both SDK model and plain dict
    vm_info = vm.as_dict() if hasattr(vm, "as_dict") else vm
    ip_address = vm_info.get("properties", {}).get("ipAddress")
    vm_name    = vm_info.get("name") or getattr(vm, "name", None)

    if not ip_address:
        print(f"[WARN] No IP for VM {vm_name}")
        return None, None

    proc = subprocess.Popen(
        ["ping", "-c", "4", ip_address],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, _ = proc.communicate()
    text = stdout.decode("utf-8", errors="ignore")

    if "0% packet loss" in text:
        print(f"[INFO] Ping OK: {vm_name} ({ip_address})")
        return ip_address, vm_name
    else:
        print(f"[INFO] Ping FAIL: {vm_name} ({ip_address})")
        return None, None

def select_reachable_vm(vm_list):
    """
    Round‑robin through vm_list, pinging each one once.
    Returns the first VM object whose ping succeeds.
    Raises Exception if none respond.
    """
    rr = itertools.cycle(vm_list)
    for _ in range(len(vm_list)):
        vm = next(rr)
        ip, name = ping_test(vm)
        if ip and name:
            print(f"[INFO] Selected VM {name} ({ip})")
            return vm
    raise Exception("No reachable VM found.")

def main():
    # -------------------------------
    # Replace `key` with your own variable
    # or load it from environment, e.g.:
    #   import os
    #   subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    # -------------------------------
    subscription_id = key

    vms = get_vm_list(subscription_id)
    if not vms:
        print("No VMs found in subscription.")
        return

    try:
        vm = select_reachable_vm(vms)
        # print out name/IP of selected VM
        name = getattr(vm, "name", vm.get("name"))
        print(f"Reachable VM → {name}")
    except Exception as e:
        print("❌", e)

if __name__ == "__main__":
    main()
