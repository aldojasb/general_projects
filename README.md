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

