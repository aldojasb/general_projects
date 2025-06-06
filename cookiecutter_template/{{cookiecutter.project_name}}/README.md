# {{cookiecutter.project_name}}


# Description


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
│
├── notebooks          <- Jupyter notebooks.
│
├── pyproject.toml     <- Project configuration file with package metadata for {{ cookiecutter.project_name.lower().replace(' ', '/').replace('-', '/').replace('_', '/') }}
│                         and configuration for tools like black
│
└── src/{{ cookiecutter.project_name.lower().replace(' ', '/').replace('-', '/').replace('_', '/') }} <- Source code for use in this project.
    │
    ├── main.py          <- execute the code
    │
    ├── get_data.py      <- get data
    │
    ├── helper.py        <- extra functions to enhance the code
    │
    ├── predict.py       <- predict data
    │
    ├── process_data.py  <- process data
    │
    └──  train.py        <- train data    


```
