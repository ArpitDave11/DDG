import os
import itertools
import subprocess

from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient


def get_vm_list(subscription_id):
    """
    Return a list of all VM models in the subscription.
    """
    compute_client = ComputeManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id
    )
    return list(compute_client.virtual_machines.list_all())


def extract_nic_ips(vm, subscription_id, network_client=None):
    """
    Given a VM model, fetch its NIC resources and pull out all IPs:
      - public IPs (if any)
      - private IPs
    Returns a list of (ip_address:str, nic_name:str) tuples.
    """
    if network_client is None:
        network_client = NetworkManagementClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id
        )

    # Parse the VM's resource group from its ARM ID
    parts = vm.id.split('/')
    rg = parts[parts.index('resourceGroups') + 1]

    ips = []
    for nic_ref in vm.network_profile.network_interfaces:
        # nic_ref.id == /.../resourceGroups/<rg>/providers/Microsoft.Network/networkInterfaces/<nic_name>
        nic_parts = nic_ref.id.split('/')
        nic_rg   = nic_parts[nic_parts.index('resourceGroups') + 1]
        nic_name = nic_parts[-1]

        nic = network_client.network_interfaces.get(nic_rg, nic_name)

        for ip_conf in nic.ip_configurations:
            # 1) Public IP
            if ip_conf.public_ip_address:
                pip_parts = ip_conf.public_ip_address.id.split('/')
                pip_rg   = pip_parts[pip_parts.index('resourceGroups') + 1]
                pip_name = pip_parts[-1]
                public_ip = network_client.public_ip_addresses.get(pip_rg, pip_name)
                ips.append((public_ip.ip_address, nic_name))

            # 2) Private IP
            elif ip_conf.private_ip_address:
                ips.append((ip_conf.private_ip_address, nic_name))

    return ips


def ping_ip(ip, count=4, timeout=5):
    """
    Ping a single IP; return True if 0% packet loss.
    """
    proc = subprocess.Popen(
        ["ping", "-c", str(count), "-W", str(timeout), ip],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, _ = proc.communicate()
    text = out.decode("utf-8", errors="ignore")
    return "0% packet loss" in text


def select_reachable_vm(subscription_id):
    """
    Round‐robin through all VMs → for each VM, try each extracted IP → return
    the first (vm, ip, nic_name) that responds to ping.
    """
    vms = get_vm_list(subscription_id)
    if not vms:
        raise RuntimeError("No VMs found in subscription.")

    net_client = NetworkManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id
    )

    seen = set()
    rr = itertools.cycle(vms)
    for _ in range(len(vms)):
        vm = next(rr)
        if vm.id in seen:
            continue
        seen.add(vm.id)

        ips = extract_nic_ips(vm, subscription_id, network_client=net_client)
        print(f"[INFO] VM {vm.name} has IPs: {ips}")

        for ip, nic in ips:
            print(f"[INFO] Pinging {vm.name}/{nic} @ {ip}...")
            if ping_ip(ip):
                print(f"[SUCCESS] {vm.name}/{nic} ({ip})")
                return vm, ip, nic

        print(f"[WARN] All IPs failed for VM {vm.name}")

    raise RuntimeError("No reachable VM found.")


def main():
    # load from env or replace here
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID") or "<YOUR-SUBSCRIPTION-ID>"

    try:
        vm, ip, nic = select_reachable_vm(subscription_id)
        print(f"\n🔎 Selected VM: {vm.name}\n    NIC: {nic}\n    IP : {ip}")
    except Exception as e:
        print("\n❌", e)


if __name__ == "__main__":
    main()
