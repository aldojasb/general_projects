import os
from typing import Optional
from datetime import datetime
import pytz  # For timezone conversion
from database_generator.helpers import get_config_path, load_and_process_params

# Get the logger for this module
import logging
from database_generator.logging_configuration import setup_logging_for_this_script
setup_logging_for_this_script()
# Get the logger for this module
logger = logging.getLogger(__name__)

import pandas as pd

from sqlalchemy import (
    create_engine, text,
    MetaData, Table, Column, Integer, Engine,
    inspect, insert as generic_insert, desc, select, and_
)
from sqlalchemy.dialects.postgresql import insert as insert_for_postgre

from sqlalchemy.exc import SQLAlchemyError, IntegrityError  # Import SQLAlchemyError for error handling

from sqlalchemy.types import (
    Float as SQLAlchemyFloat,
    TIMESTAMP as SQLAlchemyTimestamp,
    Integer as SQLAlchemyInteger,
    String as SQLAlchemyString,
    Boolean as SQLAlchemyBoolean,

    )


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
        logger.error('Database credentials and name must be provided.')
        raise ValueError("Database credentials and name must be provided.")

    # Construct the connection string
    connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"

    # Create the SQLAlchemy engine
    return create_engine(connection_string)

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
        logger.error(f"Unsupported dtype: {dtype}."
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
    df_reset = df.reset_index(names=['t_stamp'])

    # Convert all column names to lowercase
    df_reset.columns = [col.lower() for col in df_reset.columns]

    # Define metadata
    metadata = MetaData()

    # Define the table schema dynamically based on DataFrame columns and dtypes
    columns = []  # Initialize an empty list of columns

    # Loop through DataFrame columns and create SQLAlchemy columns
    for column_name, dtype in df_reset.dtypes.items():
        sqlalchemy_type = map_dtype_to_sqlalchemy(dtype)
        # Set the 'index' column as the primary key
        if column_name == 't_stamp':
            columns.append(Column(column_name, sqlalchemy_type, primary_key=True))
        else:
            columns.append(Column(column_name, sqlalchemy_type))

    # Create the table schema
    table = Table(table_name, metadata, *columns)

    # Use Inspector to check if the table already exists
    inspector = inspect(engine)

    if inspector.has_table(table_name):
        logger.warning(f"Table '{table_name}' already exists. Skipping creation. "
                        "If you want to create a table with the same name, please, consider deleting the current one.")
    else:
        # Create the table in the database
        metadata.create_all(engine)
        logger.info(f"Table '{table_name}' created from DataFrame.")

    return table


def add_missing_columns_to_table(df: pd.DataFrame, table_name: str, engine: Engine) -> None:
    """
    Check if there are columns in the DataFrame that do not exist in the table schema,
    and add them to the table if necessary.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be inserted.
    table_name (str): The name of the table to be checked and altered if necessary.
    engine (Engine): An SQLAlchemy Engine instance connected to the database.

    Returns:
    None
    """
    # Use SQLAlchemy's Inspector to get the columns of the table
    inspector = inspect(engine)
    # Get table columns in lowercase for consistency
    table_columns = {col['name'].lower() for col in inspector.get_columns(table_name)}

    # Identify columns in the DataFrame that are not in the table
    extra_columns = {col.lower() for col in df.columns} - table_columns
    if not extra_columns:
        logger.info(f"From add_missing_columns_to_table, No extra columns found in DataFrame for table '{table_name}'.")
        return

    # Add missing columns to the table
    with engine.connect() as connection:
        # Start a transaction
        trans = connection.begin()
        try:
            for column in extra_columns:
                # Determine SQLAlchemy type for each extra column
                # Ensure the original case of the DataFrame column is preserved
                original_column_name = next(col for col in df.columns if col.lower() == column)
                sqlalchemy_type = map_dtype_to_sqlalchemy(df[original_column_name].dtype)

                # Generate SQL command to add column
                add_column_sql = f"ALTER TABLE {table_name} ADD COLUMN {column} {sqlalchemy_type.__visit_name__.upper()};"
                connection.execute(text(add_column_sql))
                logger.info(f"Added column '{original_column_name}' of type '{sqlalchemy_type}' to table '{table_name}'.")

            # Commit the transaction after adding all missing columns
            trans.commit()

        except SQLAlchemyError as e:
            # Rollback the transaction in case of an error
            trans.rollback()
            logger.error(f"From add_missing_columns_to_table, Error occurred while adding columns to table '{table_name}': {e}")
            raise e


def insert_dataframe_to_table(df: pd.DataFrame, table: Table, engine: Engine) -> str:
    """
    Insert data from a pandas DataFrame into a specified SQLAlchemy table.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be inserted.
    table (Table): The SQLAlchemy Table instance where data will be inserted.
    engine (Engine): An SQLAlchemy Engine instance connected to the database.

    Returns:
    str: A message indicating the status of the insertion.
    """
    # Reset the index and make it a column named 't_stamp'
    df_reset = df.reset_index(names=['t_stamp'])

    # Convert all column names to lowercase to match table columns
    df_reset.columns = [col.lower() for col in df_reset.columns]

    # Add missing columns to the table if needed
    add_missing_columns_to_table(df_reset, table.name, engine)

    # Convert DataFrame to a list of dictionaries for insertion
    data_to_insert = df_reset.to_dict(orient='records')

    # Use the engine to connect and perform bulk insert
    with engine.connect() as connection:
        # Start a transaction
        trans = connection.begin()
        try:
            # Prepare the insert statement with ON CONFLICT DO NOTHING using PostgreSQL insert
            insert_stmt = insert_for_postgre(table).values(data_to_insert)
            insert_stmt = insert_stmt.on_conflict_do_nothing(index_elements=['t_stamp'])

            # Execute the bulk insert
            connection.execute(insert_stmt)

            # Commit the transaction
            trans.commit()
            logger.info(f"Data from DataFrame inserted successfully into '{table.name}' table (duplicates ignored).")
            return "success"

        except IntegrityError as e:
            # Log the detailed error message provided by the IntegrityError
            logger.error(f"Integrity error while inserting data into '{table.name}': {e.orig}")

            # Rollback the transaction
            trans.rollback()
            return "integrity_error"

        except SQLAlchemyError as e:
            # Rollback the transaction in case of any other SQLAlchemy error
            trans.rollback()
            logger.error(f"Error occurred while inserting data into '{table.name}': {e}")
            raise e

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
            select_stmt = select(table.c.t_stamp).order_by(desc(table.c.t_stamp)).limit(1)

            # Execute the query and fetch the result
            result = connection.execute(select_stmt).scalar()

            if result is None:
                logger.warning(f"No rows found in the table '{table.name}'. Returning None.")
                return None

            # Convert the result to a timezone-aware datetime object
            last_timestamp = result.astimezone(pytz.timezone(timezone))
            logger.info(f"Last datetime retrieved from '{table.name}' table: {last_timestamp}")

            return last_timestamp

        except Exception as e:
            logger.error(f"Error retrieving the last datetime from '{table.name}': {e}")
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
                    table.c.t_stamp >= start_datetime,
                    table.c.t_stamp <= end_datetime
                )
            )

            # Execute the query and fetch the result as a DataFrame
            result = connection.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

            # If the DataFrame is empty, log a warning and return it as is
            if df.empty:
                logger.warning(f"No data found in the table '{table.name}' for the specified datetime range.")
                return df

            # Convert 'Timestamp_id' to a timezone-aware datetime index if not already
            if not isinstance(df['t_stamp'].dtype, pd.DatetimeTZDtype):
                df['t_stamp'] = pd.to_datetime(df['t_stamp']).dt.tz_localize('UTC').dt.tz_convert(timezone)
            df.set_index('t_stamp', inplace=True)

            logger.info(f"Data retrieved successfully from '{table.name}' table for the specified datetime range.")
            return df

        except Exception as e:
            logger.error(f"Error querying data from '{table.name}': {e}")
            raise e

def store_pandas_dataframe_into_postegre(
    df: pd.DataFrame, 
    columns_to_store: list[str] = None
) -> None:
    """
    Stores pandas DataFrame into a PostgreSQL database.

    This function filters the DataFrame for specific columns (if provided), creates a table in the PostgreSQL database if it doesn't exist, and inserts the data into the created table.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing anomaly data to be stored.
    columns_to_store : list[str], optional
        List of column names to be stored in the database. If None, all columns are stored. Defaults to None.

    Returns:
    -------
    None
    """
    # 1- Read the parameters (JSON file)
    path_for_the_json_file = get_config_path()
    config_dict = load_and_process_params(path_for_the_json_file)

    # Retrieve configuration parameters from the config_dict
    table_name = config_dict['table_name_to_be_created_on_postgresql']

    # Filter the DataFrame to store only specific columns, if provided
    if columns_to_store:
        df = df[columns_to_store].copy()

    # Create engine to access the database
    sqlalchemy_engine = create_sql_alchemy_engine()

    # Create or use an existing table from the DataFrame
    table = create_table_from_dataframe(df, table_name, sqlalchemy_engine)

    # Insert data into the table
    insert_dataframe_to_table(df, table, sqlalchemy_engine)