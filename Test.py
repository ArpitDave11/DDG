# Part 1: Round-Robin VM Selection

from datetime import datetime
import subprocess
from delta.tables import DeltaTable
from pyspark.sql import functions as F

# Example list of VMs behind the load balancer (replace with actual hostnames or IPs)
vm_list = ["vm1.mycompany.net", "vm2.mycompany.net", "vm3.mycompany.net"]

# Initialize the Delta table for round-robin index if it doesn't exist.
# The table will have a single row storing the last used index (starting at -1 so the first VM used will be index 0).
if not spark.catalog.tableExists("default.vm_round_robin"):
    (spark.createDataFrame([(-1,)], ["last_index"])
         .write.format("delta").mode("overwrite").saveAsTable("default.vm_round_robin"))

def is_vm_reachable(host: str) -> bool:
    """Ping the given host once to check if it's reachable. Returns True if reachable, False if not."""
    try:
        # '-c 1' sends a single ping, '-W 3' sets a 3-second timeout for the ping reply
        subprocess.check_output(["ping", "-c", "1", "-W", "3", host])
        return True
    except subprocess.CalledProcessError:
        return False

def select_next_reachable_vm(vm_list: list) -> str:
    """
    Selects the next VM in round-robin order from vm_list that is reachable.
    Updates the Delta table with the chosen index and returns the VM host as a string.
    Throws an exception if no VM is reachable.
    """
    # Fetch the last used index from the Delta table
    last_index = spark.table("default.vm_round_robin").select("last_index").head()[0]
    if last_index is None:
        last_index = -1
    # Compute the next index in round-robin fashion
    start_index = (last_index + 1) % len(vm_list)
    
    chosen_index = None
    # Try each VM in the list (up to length of list attempts) to find a reachable one
    for offset in range(len(vm_list)):
        idx = (start_index + offset) % len(vm_list)
        vm_host = vm_list[idx]
        # Ping the VM to check reachability
        if is_vm_reachable(vm_host):
            chosen_index = idx
            break
    if chosen_index is None:
        # No VM responded to ping
        raise Exception("No reachable VM found in the list.")
    
    # Update the Delta table with the new last-used index (chosen_index)
    vm_index_dt = DeltaTable.forName(spark, "default.vm_round_robin")
    vm_index_dt.update(condition="true", set={"last_index": F.lit(chosen_index)})
    return vm_list[chosen_index]

