# General Projects Repository

This repository is designed to host **proof of concepts** and **personal projects**, providing a space for experimentation, learning, and development. The structure and tools included aim to make starting new projects quick and seamless.

---

## **Features**

- **Centralized Workspace:** Organize all your general projects and proofs of concept in one place.
- **Cookiecutter Template:** A pre-defined project template to kickstart new projects efficiently.
- **Reusable Components:** Shared scripts, notebooks, and configurations for rapid development.

---

## **Benefits of Using the Cookiecutter Template**

The Cookiecutter template included in this repository is tailored for this workspace. Here are some of the benefits:

1. **Consistency:** Standardizes the structure across projects, ensuring uniformity and ease of navigation.
2. **Time-saving:** Automates the creation of boilerplate files and directories, so you can focus on development.
3. **Pre-configured Tools:** Includes default configurations for tools like Poetry, Jupyter, and Python virtual environments.
4. **Ease of Use:** Simple commands to create new projects without repetitive manual setup.

---

## **How to Use the Cookiecutter Template**

To create a new project using the Cookiecutter template, follow these steps:

1. **Navigate to the Repository Root:**
   ```bash
   cd /path/to/general_projects
   ```

2. **Run the Cookiecutter Command:**
   ```bash
   cookiecutter ./cookiecutter_template_v1
   ```

3. **Provide Input:**
   - When prompted, enter values such as `project_name`. For example:
     ```
     [1/1] project_name (default: project_test_01): MyNewProject
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

