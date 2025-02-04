# Database Generator

## Overview

The `database_generator.py` module provides a structured and scalable approach to generating synthetic datasets for industrial applications. It supports:
- Standard data generation for industrial pump operations.
- Introduction of different types of anomalies.
- A factory pattern to construct databases from multiple datasets.

This module follows **SOLID principles**, ensuring maintainability, flexibility, and code reusability.

---

## Features

- **Abstract Base Classes (ABCs)**: Ensures a clear contract for data generation and anomaly introduction.
- **Standard Data Generation**: Simulates industrial pump data with configurable parameters.
- **Anomaly Injection**: Introduces **exponential** and **intermittent spike** anomalies to datasets.
- **Database Factory**: Merges multiple datasets while prioritizing anomalous records for analysis.
- **Logging**: Integrated logging for tracking data creation and anomalies.

---

## Installation

To install this module using **Poetry**, run:

```bash
poetry add git+https://github.com/aldojasb/general_projects.git
```
## Usage

A full demo can be found in [notebook](https://github.com/aldojasb/general_projects/tree/creating-methods-in-dbtoolks/database_generator/notebooks)

1. Generating Standard Data
Create a dataset representing industrial pump operations.

```python
from datetime import datetime
from database_generator import IndustrialPumpData

# Define parameters
data_generator = IndustrialPumpData(
    start_datetime=datetime(2023, 1, 1),
    end_datetime=datetime(2023, 1, 2),
    frequency="1h",
    seed_for_random=42
)

# Generate data
standard_data = data_generator.generate_standard_data()
print(standard_data.head())
```

2. Introducing Anomalies
Apply an exponential anomaly to simulate increasing deviations.

```python
from database_generator import ExponentialAnomaly

anomaly_generator = ExponentialAnomaly(
    start_datetime=datetime(2023, 1, 1, 6),
    end_datetime=datetime(2023, 1, 1, 12),
    variable_to_insert_anomalies="temperature_c",
    standard_data=standard_data
)

anomalous_data = anomaly_generator.introduce_anomaly()
print(anomalous_data.head())
```

Apply an intermittent spike anomaly to simulate sudden outliers.

```python
from database_generator import IntermittentSpikeAnomaly

spike_anomaly = IntermittentSpikeAnomaly(
    start_datetime=datetime(2023, 1, 1, 8),
    end_datetime=datetime(2023, 1, 1, 14),
    variable_to_insert_anomalies="pressure_mpa",
    standard_data=anomalous_data
)

spiked_data = spike_anomaly.introduce_anomaly()
print(spiked_data.head())
```

3. Creating a Combined Database
Merge multiple datasets while prioritizing anomalous records.

```python
from database_generator import SimpleDatabaseFactory

database_factory = SimpleDatabaseFactory(
    list_of_df=[standard_data, anomalous_data, spiked_data],
    flag_column="flag_normal_data"
)

final_database = database_factory.create_database()
print(final_database.head())
```

---

## SOLID Principles Applied
1. Single Responsibility Principle (SRP)
Each class is responsible for a single, well-defined functionality:

- `IndustrialPumpData` → Generates standard data.
- `ExponentialAnomaly` / `IntermittentSpikeAnomaly` → Introduce specific types of anomalies.
- `SimpleDatabaseFactory` → Combines multiple datasets into a single database.

2. Open-Closed Principle (OCP)
The design allows easy extension without modifying existing code:

- New anomaly types can be introduced by creating a subclass of AnomalyGenerator.
- Additional data sources can be integrated without modifying SimpleDatabaseFactory.

3. Liskov Substitution Principle (LSP)
All subclasses `(IndustrialPumpData, ExponentialAnomaly, IntermittentSpikeAnomaly)` correctly implement the behavior defined in their respective abstract base classes `(StandardDataGenerator, AnomalyGenerator)`. This ensures that any subclass can replace its parent class without altering the program's correctness.

4. Interface Segregation Principle (ISP)
Separate interfaces `(StandardDataGenerator, AnomalyGenerator, DatabaseFactory)` ensure that classes only implement methods relevant to their purpose. This avoids forcing unrelated functionalities into a single class.

5. Dependency Inversion Principle (DIP)
High-level classes depend on abstract base classes rather than concrete implementations:

- `ExponentialAnomaly` and `IntermittentSpikeAnomaly` depend on `AnomalyGenerator`, not on specific data sources.
- `SimpleDatabaseFactory` expects a list of pd.DataFrame objects rather than being tightly coupled to a specific data structure.