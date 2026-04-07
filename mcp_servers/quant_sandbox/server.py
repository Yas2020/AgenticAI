import os
import subprocess
import logging
import sys
from pathlib import Path
import uuid
from mcp.server.fastmcp import FastMCP

# Configure logging to show internal MCP message routing
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

# You can specifically target the MCP server logger for less noise
logging.getLogger("mcp.server").setLevel(logging.DEBUG)
logging.getLogger("fastmcp").setLevel(logging.DEBUG)


# Create the server
mcp = FastMCP("QuantSandbox", host="0.0.0.0", port=3001)

FORBIDDEN_IMPORTS = [
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "multiprocessing",
    "threading",
]

def contains_forbidden_imports(code: str):
    for word in FORBIDDEN_IMPORTS:
        if f"import {word}" in code or f"from {word}" in code:
            return word
    return None


@mcp.tool()
async def execute_quant_code(code: str):
    # 1. Protection (Keep your forbidden imports check)
    bad = contains_forbidden_imports(code)
    if bad: 
        return {"status": "error", "message": f"Forbidden: {bad}"}

    # 2. Setup Directory
    run_id = str(uuid.uuid4())[:8]
    # Ensure this is the folder MAPPED in docker-compose
    base_artifacts_path = Path("/app/artifacts") 
    run_dir = base_artifacts_path / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

    os.chmod(run_dir, 0o777)
    
    script_path = run_dir / "script.py"
    
    # THE INJECTION:
    # We define a constant in the beginning of the code so it doesn't have to import 'os'
    path_injection = f"ARTIFACT_DIR = '{str(run_dir)}'\n"
    final_code = path_injection + code
    script_path.write_text(final_code)

    # 3. Environment Injection
    # Crucial so the LLM code knows WHERE to save plots
    env = os.environ.copy()
    env["ARTIFACT_DIR"] = str(run_dir)

    try:
        result = subprocess.run(
            ["python", "-u", str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(run_dir),
            env=env # <--- Pass the environment here
        )

        # 4. Artifact Discovery (Scanning for files)
        # We return PATHS, not base64, for the Senior Volume Strategy
        generated_files = []
        for file in run_dir.glob("*.png"):
            generated_files.append({
                "name": file.name,
                "rel_path": f"run_{run_id}/{file.name}" # Path for Notebook
            })

        return {
            "status": "success" if result.returncode == 0 else "failed",
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "artifacts": generated_files,
            "run_id": run_id
        }

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Timeout"}
    
    # REMOVED: shutil.rmtree(run_dir) 
    # Logic: We keep the files for the Notebook to read.

    # finally:
    #     # Clean up execution directory
    #     shutil.rmtree(run_dir, ignore_errors=True)
        
        
if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
    )
    
    
    
# @mcp.list_tools()
# async def handle_list_tools():
#     return {
#         "name": "execute_quant_code",
#         "description": "Execute Python code for quantitative analysis",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "code": {"type": "string"}
#             },
#             "required": ["code"]
#         }
#     }
