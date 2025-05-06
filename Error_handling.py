import requests
import time
import json

# Configuration: Azure REST endpoint and authentication (token should be obtained beforehand)
subscription_id = "<YOUR_SUBSCRIPTION_ID>"
resource_group = "<YOUR_RESOURCE_GROUP>"
vm_name = "<YOUR_VM_NAME>"
command = "RunShellScript"  # or other commandId
script_text = "autosys_command_to_trigger_job.sh arg1 arg2"  # The bash command or script to run

url = (f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/"
       f"{resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}/runCommand?api-version=2024-07-01")
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer <ACCESS_TOKEN>"  # Azure AD access token with appropriate scope
}
payload = {
    "commandId": "RunShellScript",
    "script": [script_text]
}

result = {
    "status": None,
    "message": None,
    "commandId": None
}
# Include errorCode in result only if a failure occurs.

try:
    # 1. Invoke the Run Command on the VM
    response = requests.post(url, headers=headers, json=payload, timeout=10)
except requests.exceptions.RequestException as e:
    # Network or request error (DNS failure, refused connection, etc.)
    result["status"] = "failure"
    result["message"] = f"Failed to invoke RunCommand: {str(e)}"
    result["commandId"] = None
    result["errorCode"] = "HTTP_REQUEST_FAILED"
    print(json.dumps(result))
    exit(1)

# Check immediate HTTP response status
if response.status_code not in (200, 202):
    # Azure returned an error (e.g., 4xx or 5xx). Include Azure's message if available.
    result["status"] = "failure"
    result["commandId"] = None
    try:
        error_body = response.json().get("error", {})
    except ValueError:
        error_body = {}
    err_message = error_body.get("message") or response.text or "Unknown error"
    err_code = error_body.get("code") or f"HTTP_{response.status_code}"
    result["message"] = f"Azure RunCommand API call failed: {err_message}"
    result["errorCode"] = err_code
    print(json.dumps(result))
    exit(1)

# If status_code is 200, the RunCommand completed synchronously (rare for long scripts).
# If 202, it's running asynchronously.
operation_url = response.headers.get("Azure-AsyncOperation") or response.headers.get("Location")
command_id = None
if operation_url:
    # Extract a command execution ID from the operation URL if possible
    # Typically the operation GUID is the last part of the URL
    try:
        command_id = operation_url.split("/")[-1].split("?")[0]
    except Exception:
        command_id = operation_url
    result["commandId"] = command_id
else:
    # If no operation URL, use a fallback ID (or None)
    result["commandId"] = None

# 2. Poll the Azure-AsyncOperation for the result (if async)
max_attempts = 30           # Limit the number of polling attempts (to avoid endless loop)
poll_interval = 2           # starting poll interval in seconds
status = "InProgress"
poll_attempt = 0
final_response = None

while status == "InProgress" and poll_attempt < max_attempts:
    try:
        poll_resp = requests.get(operation_url, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        # If a polling request fails, wait and retry
        time.sleep(poll_interval)
        poll_attempt += 1
        poll_interval = min(poll_interval * 2, 30)  # exponential backoff, cap at 30s
        continue
    # If we got here, we have a poll_resp
    if poll_resp.status_code == 202:
        # Still running; increase backoff and loop again
        status = "InProgress"
    elif poll_resp.status_code == 200:
        # Completed (Succeeded or Failed)
        try:
            final_response = poll_resp.json()
        except ValueError:
            # If response body is not JSON (should not happen for 200), break with failure
            final_response = {"status": "Failed", "error": {"code": "InvalidJSON", "message": "Invalid JSON in response"}}
        status = final_response.get("status") or final_response.get("Status") or "Unknown"
    else:
        # Unexpected HTTP status code; treat as failure
        final_response = {"status": "Failed", "error": {"code": f"HTTP_{poll_resp.status_code}", 
                                                        "message": poll_resp.text}}
        status = "Failed"
    # Wait before next poll if still in progress
    if status == "InProgress":
        time.sleep(poll_interval)
        poll_attempt += 1
        poll_interval = min(poll_interval * 2, 30)

# If we exited loop due to reaching max_attempts while still InProgress, mark as timeout
if status == "InProgress":
    result["status"] = "failure"
    result["errorCode"] = "POLLING_TIMEOUT"
    result["message"] = "RunCommand execution did not complete within the expected time."
    print(json.dumps(result))
    exit(1)

# 3. Parse the final response from Azure (when status is Succeeded or Failed)
# Azure final_response should contain either an 'error' or an 'output'
if final_response is None:
    # This should not happen, but handle just in case
    result["status"] = "failure"
    result["errorCode"] = "NO_RESPONSE"
    result["message"] = "No response received from Azure RunCommand."
    print(json.dumps(result))
    exit(1)

azure_status = final_response.get("status") or final_response.get("Status")
if azure_status and azure_status.lower() == "failed":
    # Azure indicates the run command failed at the infrastructure level
    err_info = final_response.get("error", {}) or final_response.get("properties", {}).get("error", {})
    err_code = err_info.get("code", "RunCommandFailed")
    err_msg = err_info.get("message", "Run Command execution failed.")
    result["status"] = "failure"
    result["errorCode"] = err_code
    result["message"] = f"Azure RunCommand failed: {err_msg}"
    # We can further inspect output if any was captured before failure
    output_val = final_response.get("properties", {}).get("output", {}).get("value")
    if output_val:
        # Merge any output messages to provide more context
        output_messages = " ".join(item.get("message", "") for item in output_val if item.get("message"))
        if output_messages:
            result["message"] += " Output: " + output_messages
    print(json.dumps(result))
    exit(1)

# If Azure status is Succeeded or not provided (meaning 200 sync response perhaps)
# Extract stdout/stderr from the output
output_text = ""
exit_code = None

# Azure might return output in different formats; handle accordingly:
#  - final_response["properties"]["output"]["value"] as list of {code, message}
#  - or final_response["value"] as list of statuses (older API versions)
output_entries = []
if "properties" in final_response and isinstance(final_response["properties"].get("output", {}), dict):
    output_entries = final_response["properties"]["output"].get("value", [])
elif "value" in final_response:
    # If 'value' is present at top-level, assume it's list of status messages
    output_entries = final_response["value"]
# Concatenate any stdout/stderr messages
for entry in output_entries:
    msg = entry.get("message")
    code = entry.get("code", "")
    # Typically, code might contain "StdOut" or "StdErr"
    if msg:
        output_text += msg + "\n"
    # Check for exit code or error code if present in Azure statuses
    if "ExitCode" in code or "exitcode" in code.lower():
        try:
            exit_code = int(msg.strip())
        except Exception:
            exit_code = None

# Also check if instanceView data present (in case of managed run command)
instance_view = final_response.get("properties", {}).get("instanceView")
if instance_view:
    # If instanceView is available, it has direct fields for output, exitCode, etc.
    exit_code = instance_view.get("exitCode", exit_code)
    if instance_view.get("output"):
        # instanceView.output might contain the combined output
        output_text += instance_view["output"] + "\n"
    # Check error field in instanceView
    inst_error = instance_view.get("error")
    if inst_error:
        # Append instanceView error message if present
        err_msg = inst_error.get("message") or str(inst_error)
        output_text += err_msg + "\n"

# 4. Determine final status based on exit_code and output content
if exit_code is not None and exit_code != 0:
    # Non-zero exit code indicates script failure
    result["status"] = "failure"
else:
    # Assume success unless we find error patterns in output
    result["status"] = "success"

# Scan output_text for Autosys-specific errors or other failures
text_lower = output_text.lower()
error_detected = None
if "failed to get initial configuration" in text_lower or "cauajm_e_10062" in text_lower:
    error_detected = ("AUTOSYS_CONFIG_FAILURE", "Autosys Application Server configuration failure.")
elif "communication attempt" in text_lower and "failed!" in text_lower:
    # Matches "Communication attempt ... has failed!"
    error_detected = ("AUTOSYS_TIMEOUT", "Autosys Application Server communication failed or timed out.")
elif "exhausted list of available application server" in text_lower:
    error_detected = ("AUTOSYS_TIMEOUT", "Autosys Application Server not reachable (exhausted all servers).")
elif "timeout" in text_lower:
    # A generic timeout keyword catch (if present elsewhere in output)
    error_detected = ("AUTOSYS_TIMEOUT", "Autosys job timed out.")

if error_detected:
    code, msg = error_detected
    result["status"] = "failure"
    result["errorCode"] = code
    # If the output text contains a more detailed Autosys error message, include it
    # (e.g., include the exact Autosys error line for clarity)
    detailed = None
    if code == "AUTOSYS_CONFIG_FAILURE":
        # Find the line with 'Failed to get initial configuration'
        for line in output_text.splitlines():
            if "failed to get initial configuration" in line.lower():
                detailed = line.strip()
                break
    elif code == "AUTOSYS_TIMEOUT":
        # Find the first line that indicates the comm failure or timeout
        for line in output_text.splitlines():
            if "communication attempt" in line or "exhausted list of available" in line or "timeout" in line:
                detailed = line.strip()
                break
    if detailed:
        result["message"] = f"{msg} Details: {detailed}"
    else:
        result["message"] = msg
elif result["status"] == "failure":
    # A non-specific failure (exit code non-zero without known pattern)
    result["errorCode"] = result.get("errorCode", "GENERAL_FAILURE")
    # Provide either the output or a generic message
    if output_text:
        result["message"] = f"Command failed. Output: {output_text.strip()}"
    else:
        result["message"] = "Command execution failed with an unknown error."
else:
    # Success path
    result["message"] = "Command executed successfully."
    # We could include a portion of output in message if needed, but keep it brief
    # e.g., result["message"] = f"Command executed successfully. Output: {output_text.strip()}"
    # (Omitting full output in message for brevity)

# 5. Print the resulting structured JSON
print(json.dumps(result))
