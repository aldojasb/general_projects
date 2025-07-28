# Introduction: Why Use a Dev Container?

Modern Data Science and software teams require consistency, reproducibility, and fast on-boarding. A **Dev Container** (via VS Code’s Dev Containers or GitHub Codespaces) packages your code, dependencies, and tooling into a single, versioned environment. This approach brings benefits on three levels:

### Benefits for Data Science & Development Teams

- **Consistency Across Machines**
   Every team member runs in the **exact same** container image, so “it works on my machine” becomes a thing of the past.

- **Low Setup Friction**
   New hires or contributors simply open the repo in VS Code and the container spins up with all runtimes, libraries, and tools pre-installed.

- **Isolation & Safety**
   You keep project dependencies separate from host-OS packages—no risk of polluting your laptop or server with conflicting libraries.

- **Easy Experimentation**
   Spin up multiple containers side-by-side (e.g. one per project or feature) and tear them down cleanly when you’re done.

  

### Benefits for the Business

- **Faster Time to Market**
   Developers and analysts spend less time wrestling with environment issues and more time delivering features and insights.
   
- **Lower Support Overhead**
   Fewer “environment setup” tickets for IT or DevOps, and predictable deployments in staging and production.
   
- **Reproducible Pipelines**
   Models, reports, and demos run identically in local, CI, and cloud, reducing bugs and data-drift surprises.
   
- **Stronger Security & Compliance**
   Containers can be locked down (fixed base images, vulnerability scanning) and audited, aligning with enterprise governance.
   
- **Scalable Collaboration**
   Cross-functional teams (Data Science, Engineering, QA, Product) share a single source of truth—no more “which Python version?” debates.
   
   

------

## Considerations & Trade-Offs

While Dev Containers offer consistency and reproducibility, they also introduce some overhead and complexity. It’s important to understand these downsides so you can make an informed choice:

### Technical Considerations

- **Container Build & Startup Time**
   Each time you rebuild or open the container, VS Code must spin up Docker layers, install dependencies, and initialize services. For large images or frequent rebuilds, this can add minutes to your workflow.
   
- **Resource Consumption**
   Running Docker containers—especially multiple in parallel—consumes CPU, memory, and disk I/O. On resource­-constrained machines, this may slow down not only your containers but also other local applications.
   
- **Debugging Container Issues**
   When something goes wrong (e.g. networking, volume mounting, permission errors), you need Docker-specific troubleshooting skills, which can steepen the learning curve for data scientists or analysts unfamiliar with container tooling.
   
- **Dependency on Docker & VS Code**
    Your team must install and maintain compatible versions of Docker Desktop (or Docker Engine) and the VS Code “Dev Containers” extension. Contributors using other editors will miss out or need an alternative workflow.
   
   

------

## Prerequisites

1. **Docker Desktop for Mac**
    Download and install Docker Desktop for Mac (Intel or Apple Silicon) following the official guide [Docker Documentation](https://docs.docker.com/desktop/setup/install/mac-install/?utm_source=chatgpt.com).

   - Make sure you meet the system requirements (macOS version, ≥4 GB RAM).

2. **Visual Studio Code**
    Install VS Code from https://code.visualstudio.com/.

3. **Dev Containers extension**
    In VS Code, install **Dev Containers** (formerly “Remote – Containers”) from the Extensions marketplace:

   ```bash
   ms-vscode-remote.remote-containers
   ```

4. **Git**
    Ensure `git` is available (macOS Command Line Tools or Homebrew).

   ```bash
   git --version
   ```


------

## Quickstart: Clone & Run the Dev Container

1. **Clone the repository**

   ```bash
   clone https://github.com/aldojasb/my_devcontainer.git
   cd my_devcontainer
   ```

2. **Open in VS Code**

   ```bash
   code .
   ```

   VS Code will detect the `devcontainer.json` in your workspace.

3. **Rebuild (or Reopen) the Container**

   - When prompted in the lower right, click **“Reopen in Container”**.
   - **Or**, open the Command Palette, type **Dev Containers: Rebuild Container**, and press Enter.

4. **Wait for the build to complete**
    VS Code will pull your base image, install dependencies (Python, Zsh, Oh My Zsh, Poetry, pyenv, etc.), and mount your workspace.

5. **Verify**

   - Open a new integrated terminal (Terminal → New Terminal).

   - Ensure you’re in Zsh with Oh My Zsh loaded.

   - Check that you can run your project’s Poetry venv:

     ```bash
     general_projects/simulators_fsm
     poetry run python --version
     ```

   - You should see `Python 3.11.10` (or your configured version) coming from `.venv`.


### Helpful Links

- Docker Desktop for Mac install guide: https://docs.docker.com/desktop/install/mac-install/ [Docker Documentation](https://docs.docker.com/desktop/setup/install/mac-install/?utm_source=chatgpt.com)
- VS Code download: https://code.visualstudio.com/
- Dev Containers extension: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers

That’s it! Your team can now clone the repo, rebuild the container, and have a consistent, ready-to-code environment in minutes.



------

## Get My Portfolio Locally

Follow these steps to clone and run the **general_projects** portfolio on your machine.

### 1. Clone the repo
```bash
git clone https://github.com/aldojasb/general_projects.git
cd general_projects
```


### 2. Install all dependencies
```bash
poetry install
```


---

## **How to Use Create a New Project by using Cookiecutter Template**

To create a new project using the Cookiecutter template, follow these steps:

1. **Navigate to the Repository Root:**
   ```bash
   cd /path/to/general_projects
   ```

2. **Run the Cookiecutter Command:**
   ```bash
   cookiecutter ./cookiecutter_template
   ```

3. **Provide Input:**
   - When prompted, enter values such as `project_name`. For example:
     ```
     [1/1] project_name (default: project_test): MyNewProject
     ```

4. **Project Generation:**
   - A new folder will be created in the `general_projects` directory with the name you specified (e.g., `MyNewProject`).
   - The folder will include pre-configured files and a standardized structure.

5. **Start Development:**
   - Navigate to the new project folder:
     ```bash
     cd MyNewProject
     ```
   - Install dependencies with Poetry:
     ```bash
     poetry install
     ```
   - Activate the virtual environment:
     ```bash
     poetry shell
     ```
   - Start coding!


---


## Troubleshooting: Jupyter Kernel in VS Code Dev Container

If your notebook isn’t picking up the in-project virtual environment, follow these steps to verify and register the kernel.

### 1. List all registered kernels
If your notebook isn’t picking up the in-project virtual environment, follow these steps to verify and register the kernel.

### 1. List all registered kernels
Ensure Jupyter can see existing kernels:

```bash
poetry run jupyter kernelspec list
```

**What it does**: Uses Poetry’s venv to run `jupyter kernelspec list`, which outputs all kernel names and paths.

**Example output**:

```bash
Available kernels:
python3            /home/vscode/.local/share/jupyter/kernels/python3
MyNewProject     /home/vscode/.local/share/jupyter/kernels/MyNewProject
```

### 2. Register your simulator’s virtualenv as a new kernel

From your project root (e.g. `MyNewProject/`), run:

```bash
poetry run python -m ipykernel install --user --name MyNewProject --display-name "MyNewProject_python_venv"
```

**`poetry run`**: Executes the following command inside the project’s `.venv`.

**`python -m ipykernel install`**: Installs a new Jupyter kernel spec.

**Flags**:

- `--user` → install under your home directory (no root needed).
- `--name MyNewProject` → the internal ID (folder name under `kernels/`).
- `--display-name "MyNewProject_python_venv"` → the name shown in VS Code’s kernel picker

### 3. Reload VS Code

After registering:

1. Open the **Command Palette** (`Ctrl+Shift+P` / `⌘+Shift+P`).
2. Type **Developer: Reload Window** and hit **Enter**.

This forces VS Code to refresh its kernel list.

------

### 4. Select your new kernel

1. Open your `.ipynb` file.
2. In the top-right corner, click the current kernel name.
3. Choose **MyNewProject_python_venv**.

Now your notebook will run on the exact Python interpreter inside `MyNewProject/.venv`.

**Tip**: To confirm, run in a code cell:

```python
import sys
print(sys.executable)
```

It should point to:

```bash
/workspace/general_projects/MyNewProject/.venv/bin/python
```

------
