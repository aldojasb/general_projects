# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: database_generator
#     language: python
#     name: database_generator
# ---

# # Developing SQLAlchemy methods
#
# ### Methods to Build:
#
# - Setup: Connecting to the Database
# - Create a Table for Pump Data
# - Insert Data into the Table
# - Query Data from the Table
# - Update Data in the Table
# - Delete Data from the Table
# - Use SQLAlchemy ORM to Define Models and Perform CRUD Operations

# +
import os
from typing import Optional
import logging
from datetime import datetime
import pytz  # For timezone conversion

# Get the logger for this module
logger = logging.getLogger(__name__)

from database_generator.logging_configuration import initialize_logging_configuration
initialize_logging_configuration()


import pandas as pd
from sqlalchemy import (
    create_engine, text,
    MetaData, Table, Column, Integer, Engine,
    inspect, insert, desc, select, and_
)
from sqlalchemy.exc import SQLAlchemyError  # Import SQLAlchemyError for error handling

from sqlalchemy.types import (
    Float as SQLAlchemyFloat,
    TIMESTAMP as SQLAlchemyTimestamp,
    Integer as SQLAlchemyInteger,
    String as SQLAlchemyString,
    Boolean as SQLAlchemyBoolean,

    )



# +

def create_sql_alchemy_engine(
    user: Optional[str] = None,
    password: Optional[str] = None,
    host: Optional[str] = "localhost",
    port: Optional[int] = 5432,  # PostgreSQL default port
    dbname: Optional[str] = None,
) -> Engine:
    """
    Create a SQLAlchemy engine for connecting to a PostgreSQL database.

    Parameters:
    user (str): Database username. If not provided, will use the 'DATABASE_USER' environment variable.
    password (str): Database password. If not provided, will use the 'DATABASE_PASSWORD' environment variable.
    host (str): Database host. Defaults to 'localhost'.
    port (int): Database port. Defaults to 5432.
    dbname (str): Database name. If not provided, will use the 'DATABASE_NAME' environment variable.

    Returns:
    Engine: A SQLAlchemy Engine instance for database operations.
    """
    # Retrieve from environment variables if not provided
    user = user or os.getenv("DATABASE_USER")
    password = password or os.getenv("DATABASE_PASSWORD")
    host = host or os.getenv("DATABASE_HOST", "localhost")
    port = port or int(os.getenv("DATABASE_PORT", 5432))  # Default to PostgreSQL port 5432
    dbname = dbname or os.getenv("DATABASE_NAME")

    if not user or not password or not dbname:
        logging.error('Database credentials and name must be provided.')
        raise ValueError("Database credentials and name must be provided.")

    # Construct the connection string
    connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"

    # Create the SQLAlchemy engine
    return create_engine(connection_string)


# -

# ### Understanding the Syntax
#
# ### 1. with engine.connect() as connection:
# Purpose: This line is using a context manager (with statement) to create a new database connection from the SQLAlchemy engine.
#
# engine.connect():
#
# This method is used to create a new Connection object. The Connection object represents an active database connection. It provides a way to execute SQL statements, manage transactions, and interact with the database.
# When you call engine.connect(), SQLAlchemy establishes a connection to the database from the connection pool maintained by the engine.
# with Statement (Context Manager):
#
# The with statement ensures that resources are properly managed. When the code block inside the with statement is done executing, it automatically releases the connection back to the pool (or closes it if it's no longer needed). This prevents resource leaks and ensures efficient use of connections.
# It also handles any exceptions that might occur within the block, ensuring the connection is properly closed even if an error occurs.
# Benefit of Using with:
#
# Automatic cleanup: You don’t need to manually close the connection. It’s done automatically when the block is exited, either after successful execution or an error.
# Reduces boilerplate code: You don’t need to write explicit try/finally blocks to ensure cleanup.
#
# ### 2. connection.execute()
# Purpose: This method is used to execute a SQL statement on the database.
#
# How It Works:
#
# connection.execute(...) takes an Executable object (like text()), a SQL expression, or a SQLAlchemy statement object (e.g., select(), insert(), update(), delete()), and sends it to the database for execution.
# Why Use .execute()?:
#
# It abstracts the complexity of sending SQL commands to the database, making it easier to work with different database backends (e.g., PostgreSQL, MySQL, SQLite) without needing to change your code.
# .execute() is a powerful function that supports a wide range of SQLAlchemy constructs, making it versatile for both raw SQL execution and ORM-based queries.
#
# ### 3. .fetchone()
# Purpose: This method fetches a single row from the result set of the executed SQL query.
#
# How It Works:
#
# When you execute a SQL query that returns data (like SELECT), the execute() method returns a Result object.
# Calling .fetchone() on the Result object retrieves the next row of the result set as a tuple.
# If there are no more rows available, .fetchone() returns None.
# Why Use .fetchone()?:
#
# Efficient Memory Usage: If you only need one row from the result set, .fetchone() is more memory-efficient than .fetchall(), which retrieves all rows at once.
# Useful for Single Row Queries: If you know your query is designed to return only one row (e.g., SELECT 1), .fetchone() is appropriate.
#
# ### Summary
# - engine.connect(): Establishes a connection to the database.
# - with ... as ...:: A context manager to handle resource cleanup automatically.
# - connection.execute(...): Executes a SQL statement or SQLAlchemy expression.
# - fetchone(): Retrieves the next row from the result set of the executed SQL statement.

# +
def map_dtype_to_sqlalchemy(dtype: str):
    """
    Map pandas DataFrame dtype to SQLAlchemy data type.
    """
    if pd.api.types.is_float_dtype(dtype):
        return SQLAlchemyFloat
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        # Ensure TIMESTAMP WITH TIME ZONE for datetime types
        return SQLAlchemyTimestamp(timezone=True)
    elif pd.api.types.is_bool_dtype(dtype):
        return SQLAlchemyBoolean
    elif pd.api.types.is_string_dtype(dtype):
        return SQLAlchemyString
    elif pd.api.types.is_integer_dtype(dtype):
        return SQLAlchemyInteger
    # Add more mappings as needed (Integer, String, etc.)
    else:
        logging.error(f"Unsupported dtype: {dtype}."
                      "Please provide a mapping for this type.")
        raise ValueError(f"Unsupported dtype: {dtype}"
                         "Please provide a mapping for this type.")


def create_table_from_dataframe(df: pd.DataFrame, table_name: str, engine: Engine) -> None:
    """
    Create a table in the database from a pandas DataFrame, using the DataFrame's index as the primary key.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data structure.
    table_name (str): The name of the table to be created in the database.
    engine (Engine): An SQLAlchemy Engine instance connected to the database.
    """
    # Reset the index and make it a column named 'index'
    df_reset = df.reset_index(names=['Timestamp_id'])

    # Define metadata
    metadata = MetaData()

    # Define the table schema dynamically based on DataFrame columns and dtypes
    columns = []  # Initialize an empty list of columns

    # Loop through DataFrame columns and create SQLAlchemy columns
    for column_name, dtype in df_reset.dtypes.items():
        sqlalchemy_type = map_dtype_to_sqlalchemy(dtype)
        # Set the 'index' column as the primary key
        if column_name == 'Timestamp_id':
            columns.append(Column(column_name, sqlalchemy_type, primary_key=True))
        else:
            columns.append(Column(column_name, sqlalchemy_type))

    # Create the table schema
    table = Table(table_name, metadata, *columns)

    # Use Inspector to check if the table already exists
    inspector = inspect(engine)

    if inspector.has_table(table_name):
        logging.warning(f"Table '{table_name}' already exists. Skipping creation. "
                        "If you want to create a table with the same name, please, consider deleting the current one.")
    else:
        # Create the table in the database
        metadata.create_all(engine)
        logging.info(f"Table '{table_name}' created from DataFrame.")

    return table


# -

# ### 1. metadata = MetaData()
#
# - MetaData in SQLAlchemy:
#
# MetaData is a container object in SQLAlchemy that holds information about the database schema (i.e., tables, columns, constraints).
# It acts as a central registry that stores all the schema constructs, such as tables, columns, and other schema elements.
# When you create a MetaData object, you're essentially creating a blank registry that will hold the structure of your tables.
# Purpose in the Function:
#
# In the context of the create_table_from_dataframe function, metadata = MetaData() is used to define a new, empty metadata object where we can register our table definitions. This object will then be used to generate SQL commands to create the tables in the actual database.
#
# ### 2. table = Table(table_name, metadata, *columns)
#
# - What is Table in SQLAlchemy?
#
# Table is a SQLAlchemy class that represents a database table. It defines the table's name, the columns it contains, the data types for each column, and any other metadata such as primary keys and foreign keys.
# How Table is Defined in the Code:
#
# table = Table(table_name, metadata, *columns):
# table_name: The name of the table you want to create in the database.
# metadata: The MetaData instance where this table's schema will be registered.
# *columns: A list of Column objects that define the structure of the table (e.g., column names and their types).
# By defining the table this way, we are dynamically creating a schema based on the provided DataFrame.
# Where table is Used:
#
# The table object itself is not directly used later in the function. Instead, its definition is registered in the metadata object. When we call metadata.create_all(engine), it uses all the table definitions registered in metadata to create the tables in the database.
#
# Why It Seems Unused:
# Although it looks like table is not being used, it is indeed crucial for creating the schema. The table definition is stored in metadata when we define it with Table(table_name, metadata, *columns).
#
# ### 3. metadata.create_all(engine)
#
# - How metadata.create_all(engine) Works:
#
# metadata.create_all(engine) is a method that generates SQL CREATE TABLE statements for all the table objects registered with the MetaData instance (metadata) and executes them against the provided database engine.
# The engine represents the connection to the database. When you call create_all(engine), SQLAlchemy translates the table definitions in metadata into the appropriate SQL commands for the specific database dialect (e.g., PostgreSQL, MySQL) and runs them to create the tables.
# What Happens Under the Hood:
#
# For each Table object registered in metadata, SQLAlchemy generates the SQL command for creating that table.
# If you have multiple tables defined within metadata, it will create all of them in the database.
# It also checks if the table already exists in the database. If it does, it will skip creating that table (unless specified otherwise).
#
# ### Summary of the Process
# metadata = MetaData(): Creates a container to hold all the table definitions.
# table = Table(table_name, metadata, *columns): Defines a table schema dynamically and registers it with the metadata object.
# metadata.create_all(engine): Generates and executes the SQL commands to create all tables registered in metadata in the connected database.
#

# ### Explanation of Inspector Usage
# The Inspector is a class in SQLAlchemy that provides a generalized interface to database schema information. It's a powerful tool for introspecting (i.e., examining) the schema of a database.
#
# - How the Inspector Works:
#
# 1. What is Inspector:
#
# Inspector is part of SQLAlchemy's sqlalchemy.engine.reflection module.
#
# It provides methods to inspect database schema details such as tables, columns, indexes, constraints, etc.
# Creating an Inspector Instance:
#
# - inspector = inspect(engine):
# inspect() is a function that returns an instance of the Inspector class for a given Engine or Connection.
#
# The engine parameter is the SQLAlchemy Engine instance connected to the database.
# When you call inspect(engine), SQLAlchemy constructs an Inspector object that provides methods to interact with the database schema.
# Checking for Table Existence with Inspector:
#
# - inspector.has_table(table_name):
# This method checks if a table with the specified name (table_name) exists in the current database schema.
# If the table exists, it returns True; otherwise, it returns False.
# This is useful for ensuring that a table is not created if it already exists, preventing potential conflicts or errors.
#
# 2. Other Common Inspector Methods:
#
# - get_table_names(): Returns a list of all table names in the current schema.
# - get_columns(table_name): Returns a list of column names and their details for a specified table.
# - get_primary_keys(table_name): Returns a list of primary key column names for a specified table.
# - get_foreign_keys(table_name): Returns a list of foreign keys and their details for a specified table.
# - get_indexes(table_name): Returns a list of indexes and their details for a specified table.

def insert_dataframe_to_table(df: pd.DataFrame, table: Table, engine: Engine) -> None:
    """
    Insert data from a pandas DataFrame into a specified SQLAlchemy table.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be inserted.
    table (Table): The SQLAlchemy Table instance where data will be inserted.
    engine (Engine): An SQLAlchemy Engine instance connected to the database.

    Returns:
    None
    """
    # Reset the index and make it a column named 'index'
    df_reset = df.reset_index(names=['Timestamp_id'])

    # Convert DataFrame to a list of dictionaries for insertion
    data_to_insert = df_reset.to_dict(orient='records')

    # Use the engine to connect and perform bulk insert
    with engine.connect() as connection:
        # Start a transaction
        trans = connection.begin()
        try:
            # Prepare the insert statement
            insert_stmt = insert(table)

            # Execute the bulk insert
            connection.execute(insert_stmt, data_to_insert)

            # Commit the transaction
            trans.commit()
            logging.info(f"Data from DataFrame inserted successfully into '{table.name}' table.")

        except SQLAlchemyError as e:
            # Rollback the transaction in case of an error
            trans.rollback()
            logging.error(f"Error occurred while inserting data into '{table.name}': {e}")
            raise e



# ### Explanation:
#
# 1. Convert DataFrame to List of Dictionaries:
# - df.to_dict(orient='records'): This converts the DataFrame into a list of dictionaries, where each dictionary represents a row of data. This is compatible with SQLAlchemy's bulk insert operations.
# Prepare the Insert Statement:
#
#
# 2. insert(table): Creates an INSERT SQL statement for the specified table object.
# Execute Bulk Insert:
#
#
# 3. connection.execute(insert_stmt, data_to_insert): Executes the INSERT statement using the connection object. This performs a bulk insert of all the rows in one operation.
#
#
# 4. Commit the Transaction:
# - connection.commit(): Commits the transaction to save the changes to the database.
#
# ### Executing the Insert Statement:
#
# - connection.execute(insert_stmt, data_to_insert) is the core of the operation. Here’s what happens under the hood:
# SQLAlchemy Engine Prepares the SQL Statement: The insert_stmt object, along with data_to_insert, is handed over to SQLAlchemy's engine and connection objects.
#
# - Generating Parameterized SQL: SQLAlchemy generates a parameterized SQL query. Parameterized queries are used to safely insert data, protecting against SQL injection and improving performance. For example:
# sql
#
# - Example:
#
# INSERT INTO sensor_data (index_and_id, Temperature_C, Pressure_MPa, ...)
#
# VALUES (:index_and_id_1, :Temperature_C_1, :Pressure_MPa_1, ...),
#        (:index_and_id_2, :Temperature_C_2, :Pressure_MPa_2, ...),
#        ...
#
# The placeholders (:index_and_id_1, :Temperature_C_1, etc.) represent the parameterized query placeholders that will be replaced by the actual values from data_to_insert.
#
# - Bulk Insertion:
# SQLAlchemy automatically bundles the multiple rows from data_to_insert and inserts them in a single bulk operation.
# This approach is more efficient than inserting rows one by one, as it reduces the number of database round-trips.
# The actual database adapter (in this case, psycopg2 for PostgreSQL) handles the underlying bulk insertion logic, and SQLAlchemy provides a convenient interface for interacting with it.
# Committing the Transaction:
#
# - After executing the execute() method, you call connection.commit() to commit the transaction. This makes all the changes (i.e., inserted rows) persistent in the database.
# If you were to omit commit(), the changes would not be saved to the database.

def get_last_timestamp(engine: Engine, table: Table, timezone: str = 'UTC') -> datetime:
    """
    Retrieve the last datetime from the specified table and return it as a timezone-aware datetime object.

    Parameters:
    engine (Engine): An SQLAlchemy Engine instance connected to the database.
    table (Table): The SQLAlchemy Table instance from which to retrieve the datetime.
    timezone (str): The timezone to use for the datetime object. Defaults to 'UTC'.

    Returns:
    datetime: The last datetime value from the table as a timezone-aware datetime object.
    """
    # Use the engine to connect to the database
    with engine.connect() as connection:
        try:
            # Select the last Timestamp_id from the table
            select_stmt = select(table.c.Timestamp_id).order_by(desc(table.c.Timestamp_id)).limit(1)

            # Execute the query and fetch the result
            result = connection.execute(select_stmt).scalar()

            if result is None:
                logging.warning(f"No rows found in the table '{table.name}'. Returning None.")
                return None

            # Convert the result to a timezone-aware datetime object
            last_timestamp = result.astimezone(pytz.timezone(timezone))
            logging.info(f"Last datetime retrieved from '{table.name}' table: {last_timestamp}")

            return last_timestamp

        except Exception as e:
            logging.error(f"Error retrieving the last datetime from '{table.name}': {e}")
            raise e



def query_data_by_datetime(engine: Engine, table: Table, start_time: datetime, end_time: datetime, timezone: str = 'UTC') -> pd.DataFrame:
    """
    Query data from the specified table within a given datetime range and return it as a pandas DataFrame.

    Parameters:
    engine (Engine): An SQLAlchemy Engine instance connected to the database.
    table (Table): The SQLAlchemy Table instance from which to query data.
    start_time (datetime): The start datetime for the query range (can be a timezone-aware datetime object or a string in ISO format).
    end_time (datetime): The end datetime for the query range (can be a timezone-aware datetime object or a string in ISO format).
    timezone (str): The timezone to use for the datetime index in the DataFrame. Defaults to 'UTC'.

    Returns:
    pd.DataFrame: A DataFrame containing the queried data, with a timezone-aware 'Timestamp_id' as the index.
    """
    # Check if start_time and end_time are already timezone-aware datetime objects and in UTC
    if not isinstance(start_time, datetime) or start_time.tzinfo is None or start_time.tzinfo != pytz.UTC:
        start_datetime = pd.to_datetime(start_time).astimezone(pytz.timezone(timezone))
    else:
        start_datetime = start_time

    if not isinstance(end_time, datetime) or end_time.tzinfo is None or end_time.tzinfo != pytz.UTC:
        end_datetime = pd.to_datetime(end_time).astimezone(pytz.timezone(timezone))
    else:
        end_datetime = end_time

    # Use the engine to connect to the database
    with engine.connect() as connection:
        try:
            # Build the SQL query to select data within the specified range
            query = select(table).where(
                and_(
                    table.c.Timestamp_id >= start_datetime,
                    table.c.Timestamp_id <= end_datetime
                )
            )

            # Execute the query and fetch the result as a DataFrame
            result = connection.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

            # If the DataFrame is empty, log a warning and return it as is
            if df.empty:
                logging.warning(f"No data found in the table '{table.name}' for the specified datetime range.")
                return df

            # Convert 'Timestamp_id' to a timezone-aware datetime index if not already
            if not isinstance(df['Timestamp_id'].dtype, pd.DatetimeTZDtype):
                df['Timestamp_id'] = pd.to_datetime(df['Timestamp_id']).dt.tz_localize('UTC').dt.tz_convert(timezone)
            df.set_index('Timestamp_id', inplace=True)

            logging.info(f"Data retrieved successfully from '{table.name}' table for the specified datetime range.")
            return df

        except Exception as e:
            logging.error(f"Error querying data from '{table.name}': {e}")
            raise e




# +
# example:

from database_generator.helpers import (
    get_config_path,
    load_and_process_params,
)

from database_generator.get_data import (
    generate_stable_toy_data,
    introduce_exponential_anomalies,
    simulate_broken_sensor,
)

from database_generator.evaluate import (
    overlaid_plots_with_plotly,
)

# get the path to the .json file from the environment

path_for_the_json_file = get_config_path()
path_for_the_json_file

config_dict = load_and_process_params(path_for_the_json_file)

start_date_for_the_toy_dataset = config_dict['start_date_for_the_toy_dataset']
number_of_rows_for_stable_toy_data = config_dict['number_of_rows_for_stable_toy_data']
seed_for_the_stable_dataset = config_dict['seed_for_the_stable_dataset']

# Example usage
df_stable = generate_stable_toy_data(number_of_rows=number_of_rows_for_stable_toy_data, start_date=start_date_for_the_toy_dataset, seed_for_random=42)

# Create an engine using environment variables or specified parameters
engine = create_sql_alchemy_engine(
    # user='my_user',
    # password='my_secrets',
    # host='localhost',
    # port=5432,
    # dbname='data_generator_v1'
)

# Create the table from the DataFrame
table = create_table_from_dataframe(df_stable, "sensor_data", engine)

# Insert data into the new table
insert_dataframe_to_table(df_stable, table, engine)

# -

df_stable.head(10)


df_stable.tail()

last_timestamp = get_last_timestamp(engine, table, timezone='UTC')
print("Last timestamp: ", last_timestamp)
print('last timestemp format: ', type(last_timestamp))

start_time = '2024-08-23 10:00:00+00:00'
end_time = '2024-08-23 10:45:00+00:00'
queried_df_01 = query_data_by_datetime(engine, table, start_time, end_time, timezone='UTC')

queried_df_01



# # theoretical stuff
#
# ### The Engine in SQLAlchemy is a core object that represents the interface to the database. Here’s a breakdown of what the Engine does:
#
# - Connection Pooling: The Engine manages a pool of database connections. When you execute a query, the Engine provides a connection from this pool, making it efficient to execute multiple queries without needing to establish a new connection each time.
#
# - Database Dialect: The Engine is configured with a dialect that is specific to the type of database you're using (PostgreSQL in this case). This dialect translates SQLAlchemy commands into the appropriate SQL for your database system.
#
# - Execution Context: The Engine provides the execution context for SQL queries. It takes SQL expressions and translates them into the SQL string that is sent to the database.
#
# - Thread-Safe: The Engine is designed to be shared among multiple threads, and it's safe to use concurrently. This is especially useful for web applications where multiple requests need to interact with the database.
