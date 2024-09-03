import os
from typing import Optional
import logging
from datetime import datetime
import pytz  # For timezone conversion

# Get the logger for this module
logger = logging.getLogger(__name__)


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

