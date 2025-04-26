# General Projects Repository

This repository serves as a structured foundation for developing **proof of concepts** and **personal projects**.

It provides a standardized project architecture and essential tools to enhance **scalability**, **consistency**, and **collaboration** within development teams.

---

## **Features**

- **Predefined Project Template:** Utilize the **Cookiecutter** template to streamline project setup and ensure a uniform structure across all projects.
- **Dependency Management with Poetry:** Simplifies dependency management and virtual environment creation for a seamless development experience.
- **Scalability & Standardization:** Ensures projects follow best practices, reducing setup time and improving maintainability.
- **Flexibility for Experimentation:** Ideal for prototyping new ideas while maintaining a structured approach.

---

## **How to Use the Cookiecutter Template**

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

Happy coding and experimenting!



## Troubleshooting: Jupyter Kernel in VS Code Dev Container

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



