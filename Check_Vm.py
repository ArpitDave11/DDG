#!/usr/bin/env python3
import requests
import time
import logging
import re
from azure.identity import DefaultAzureCredential

# -----------------------------------------------------------------------------
# CONFIGURATION — fill in or set via environment for DefaultAzureCredential
# -----------------------------------------------------------------------------
SUBSCRIPTION_ID     = "<YOUR_SUBSCRIPTION_ID>"
RESOURCE_GROUP_NAME = "<YOUR_RESOURCE_GROUP>"
VM_NAME             = "<YOUR_VM_NAME>"
API_VERSION         = "2021-04-01"   # appropriate Compute RunCommand API version

# -----------------------------------------------------------------------------
# AUTHENTICATION
# -----------------------------------------------------------------------------
# Uses azure-identity DefaultAzureCredential, which supports:
#  • Environment variables (AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET)
#  • Managed identity if running in Azure VM/App Service/etc.
credential   = DefaultAzureCredential()
token        = credential.get_token("https://management.azure.com/.default")
ACCESS_TOKEN = token.token

# -----------------------------------------------------------------------------
# ENDPOINT & HEADERS
# -----------------------------------------------------------------------------
RUN_COMMAND_URL = (
    f"https://management.azure.com/subscriptions/{SUBSCRIPTION_ID}"
    f"/resourceGroups/{RESOURCE_GROUP_NAME}"
    f"/providers/Microsoft.Compute/virtualMachines/{VM_NAME}"
    f"/runCommand?api-version={API_VERSION}"
)

HEADERS = {
    "Content-Type":  "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s %(levelname)s %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S"
)

# -----------------------------------------------------------------------------
# CORE HELPER: send a single shell-script command via RunCommand and poll till done
# -----------------------------------------------------------------------------
def send_and_wait(cmd_text, timeout=600, poll_interval=5):
    """
    1) POSTs to Azure VM RunCommand with RunShellScript and your `cmd_text`
    2) If 202 Accepted, polls the Azure-AsyncOperation URL until status == 'Succeeded'
    3) Raises Exception on HTTP errors, timeouts, or 'Failed' statuses
    """
    body = {
        "commandId": "RunShellScript",
        "script":    [cmd_text]
    }
    logging.info(f"→ Sending RunCommand: {cmd_text!r}")
    resp = requests.post(RUN_COMMAND_URL, headers=HEADERS, json=body)
    if resp.status_code in (200, 201):
        logging.info("✔ Command completed synchronously.")
        return

    if resp.status_code != 202:
        raise RuntimeError(f"RunCommand failed [{resp.status_code}]: {resp.text!r}")

    # async case: extract polling URL
    poll_url = resp.headers.get("Azure-AsyncOperation") or resp.headers.get("Location")
    if not poll_url:
        raise RuntimeError("Missing Azure-AsyncOperation/Location header for polling.")

    logging.info(f"↻ Polling async status at {poll_url}")
    start = time.time()
    while True:
        poll_resp = requests.get(poll_url, headers=HEADERS)
        if poll_resp.status_code not in (200, 202):
            raise RuntimeError(f"Polling error [{poll_resp.status_code}]: {poll_resp.text!r}")

        status = poll_resp.json().get("status", "").lower()
        logging.info(f"  • status = {status.capitalize()}")
        if status == "succeeded":
            logging.info("✔ Async operation succeeded.")
            return
        if status in ("failed", "canceled"):
            raise RuntimeError(f"✖ Async operation {status}. Details: {poll_resp.text!r}")

        if time.time() - start > timeout:
            raise RuntimeError("✖ Timeout waiting for RunCommand to complete.")

        time.sleep(poll_interval)


# -----------------------------------------------------------------------------
# PRIMARY ENTRY: handles the special DATEFLE-first logic, then delegates to send_and_wait()
# -----------------------------------------------------------------------------
def execute_commands(command_str):
    """
    1) If command_str contains the special marker, split it into individual commands,
       extract + run the DATEFLE one first; wait for success.
    2) Then run the rest in one go.
    3) If no special marker, run command_str as-is.
    """
    SPECIAL = "WMALESL_5481_DEV_DMSH_PROSSNG_DATEFLE"

    if SPECIAL in command_str:
        logging.info("🔎 Detected DATEFLE job — running it first.")
        # Split on semicolons or newlines, strip blanks
        parts = [p.strip() for p in re.split(r"[;\n]+", command_str) if p.strip()]
        # Pull out the DATEFLE command
        datefle_cmd = next(p for p in parts if SPECIAL in p)
        remaining = "; ".join(p for p in parts if p != datefle_cmd)

        # Run DATEFLE first
        send_and_wait(datefle_cmd)

        # Then the rest (if any)
        if remaining:
            logging.info("▶ Running remaining commands.")
            send_and_wait(remaining)
        else:
            logging.info("✅ No other commands to run after DATEFLE.")

    else:
        logging.info("▶ No DATEFLE job found — running all commands together.")
        send_and_wait(command_str)


# -----------------------------------------------------------------------------
# MAIN GUARD
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: replace this with your actual multi-command string
    all_cmds = """
      sudo -H -u eisladmin bash -c 'source /mnt/ingestion/autosys/es1.env;
      sendevent -E STARTJOB -J WMALESL_5481_DEV_DMSH_PRCSSNG_LSPT0100;
      WMALESL_5481_DEV_DMSH_PROSSNG_DATEFLE --option xyz;
      sendevent -E CHANGE_STATUS -S FAILURE -J WMALESL_5481_DEV_DMSH_PRCSSNG_MNF1200;
      sendevent -E STARTJOB -J WMALESL_5481_DEV_DMSH_PRCSSNG_PDBT931Z;
      '
    """
    try:
        execute_commands(all_cmds)
    except Exception as e:
        logging.error(f"❌ Execution failed: {e}")
        raise





import re

# … your existing imports, getToken(), variables, etc. …

def send_and_wait(cmd):
    """
    This is your existing block that does:
      • builds `body = {"commandId": …, "script": [cmd]}`, 
      • calls requests.post(...),
      • polls Azure-AsyncOperation until succeeded/failed,
      • raises on failure.
    Keep this exactly as you have it now.
    """
    # <— paste your existing request/post + polling code here,
    # using `cmd` in place of your current `command` variable.
    pass  # <<— replace with your logic

# … right before you build `body` and call your existing code …  
# assume `command` is your full multi-cmd string

# 1) Detect and extract the DATEFLE job
if "WMALESL_5481_DEV_DMSH_PROSSNG_DATEFLE" in command:
    # split on semicolons or newlines
    parts = [p.strip() for p in re.split(r"[;\n]+", command) if p.strip()]
    # isolate the DATEFLE command
    datefle = next(p for p in parts if "WMALESL_5481_DEV_DMSH_PROSSNG_DATEFLE" in p)
    # re-join the rest
    others = "; ".join(p for p in parts if p != datefle)

    # optional: preserve your chmod logic if DATEFLE needs it
    if "DATEFLE" in datefle:
        datefle = "chmod 777 /mnt/ingestion/autosys/inactive.sh; " + datefle

    # 2) Run DATEFLE first
    send_and_wait(datefle)

    # 3) Once that succeeds, run everything else (if any)
    if others:
        # if your existing code also prepends chmod on any DATEFLE in the rest, you can reapply here
        send_and_wait(others)

else:
    # no DATEFLE: behave exactly as before
    send_and_wait(command)




###########
import os
import itertools
import subprocess

from azure.identity import DefaultAzureCredential
from azure.mgmt.network import NetworkManagementClient


def get_backend_pool_ips(
    subscription_id: str,
    resource_group: str,
    load_balancer_name: str,
    pool_name: str = None
):
    """
    Returns a list of tuples: (vm_name, nic_name, ip_address)
    for every IP config tied into the specified LB backend pool.
    """
    net_client = NetworkManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id
    )

    # 1) Fetch the LB
    lb = net_client.load_balancers.get(resource_group, load_balancer_name)

    # 2) Pick the pool (by name or first one)
    pools = lb.backend_address_pools or []
    if not pools:
        raise RuntimeError(f"Load Balancer {load_balancer_name} has no backend pools.")
    if pool_name:
        pool = next((p for p in pools if p.name == pool_name), None)
        if pool is None:
            raise RuntimeError(f"No pool named {pool_name} in LB {load_balancer_name}")
    else:
        pool = pools[0]

    results = []
    # 3) Each .backend_ip_configurations is a reference to a NIC's IP config
    for ip_conf_ref in pool.backend_ip_configurations or []:
        # parse: /subscriptions/.../resourceGroups/rg/providers/Microsoft.Network/
        #          networkInterfaces/{nic_name}/ipConfigurations/{conf_name}
        parts = ip_conf_ref.id.split("/")
        rg_idx = parts.index("resourceGroups") + 1
        nic_idx = parts.index("networkInterfaces") + 1
        conf_idx = parts.index("ipConfigurations") + 1

        nic_rg      = parts[rg_idx]
        nic_name    = parts[nic_idx]
        config_name = parts[conf_idx]

        # 4) Retrieve the full NIC
        nic = net_client.network_interfaces.get(nic_rg, nic_name)

        # 5) Find that specific ipConfiguration on the NIC
        ip_conf = next((c for c in nic.ip_configurations if c.name == config_name), None)
        if not ip_conf:
            continue

        # 6a) Private IP
        if ip_conf.private_ip_address:
            results.append((nic.virtual_machine.id.split("/")[-1], nic_name, ip_conf.private_ip_address))

        # 6b) Public IP (if attached)
        if ip_conf.public_ip_address:
            pip_id   = ip_conf.public_ip_address.id
            pip_parts= pip_id.split("/")
            pip_rg   = pip_parts[pip_parts.index("resourceGroups")+1]
            pip_name = pip_parts[-1]
            public_ip = net_client.public_ip_addresses.get(pip_rg, pip_name)
            if public_ip.ip_address:
                results.append((nic.virtual_machine.id.split("/")[-1], nic_name, public_ip.ip_address))

    return results


def ping_test(ip: str, count: int = 4, timeout: int = 5) -> bool:
    """
    Returns True if ping → “0% packet loss” in the output.
    """
    proc = subprocess.Popen(
        ["ping", "-c", str(count), "-W", str(timeout), ip],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, _ = proc.communicate()
    return b"0% packet loss" in out


def select_reachable_from_lb(
    subscription_id: str,
    resource_group: str,
    load_balancer_name: str,
    pool_name: str = None
):
    """
    Round‐robin through all backend pool IPs; 
    returns the first (vm_name, nic_name, ip) that responds, or raises if none do.
    """
    entries = get_backend_pool_ips(
        subscription_id, resource_group, load_balancer_name, pool_name
    )
    if not entries:
        raise RuntimeError("No NIC‑IP entries found in that backend pool.")

    # cycle through VM/NIC/IP tuples
    for vm_name, nic_name, ip in itertools.cycle(entries):
        print(f"[INFO] Trying {vm_name}/{nic_name} @ {ip}…")
        if ping_test(ip):
            print(f"[SUCCESS] {vm_name}/{nic_name} ({ip}) is reachable.")
            return vm_name, nic_name, ip

    # in practice you’ll never exit the for‐loop without a break unless entries empty
    raise RuntimeError("No reachable IP found in backend pool.")


def main():
    sub_id  = os.getenv("AZURE_SUBSCRIPTION_ID", "<YOUR-SUB-ID>")
    rg      = os.getenv("AZURE_RESOURCE_GROUP", "<YOUR-RG>")
    lb_name = os.getenv("AZURE_LOAD_BALANCER", "<YOUR-LB-NAME>")
    pool    = os.getenv("AZURE_LB_POOL", None)  # optional

    try:
        vm, nic, ip = select_reachable_from_lb(sub_id, rg, lb_name, pool)
        print(f"\n→ Selected: VM={vm}, NIC={nic}, IP={ip}")
    except Exception as e:
        print("❌", e)


if __name__ == "__main__":
    main()
