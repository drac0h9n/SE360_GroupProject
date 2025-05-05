# db.py
# Database operations module
# Modified delete_result function to decrement last_run_index in run_counters
# when deleting the record with the highest run_index.
# ** Changed all variables/column names related to coverage count from 'y' to 'c' **

import sqlite3
import json
import os
from datetime import datetime

# --- Constants ---
DB_FILE = "k6_results.db"  # Database filename

# --- Database Setup ---
def setup_database(db_file=DB_FILE):
    """
    Initializes the database connection and creates required tables if they don't exist.
    Includes `results` table and `run_counters` table.

    Args:
        db_file (str): Path to the database file.
    """
    print(f"Checking/Creating database: {os.path.abspath(db_file)}") # TRANSLATED (print absolute path for easier debugging)
    conn = None  # Initialize connection variable
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # --- Create results table (!!! column y_condition changed to c_condition !!!) ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                m INTEGER NOT NULL,
                n INTEGER NOT NULL,
                k INTEGER NOT NULL,
                j INTEGER NOT NULL,
                s INTEGER NOT NULL,
                run_index INTEGER NOT NULL,
                num_results INTEGER NOT NULL,
                c_condition INTEGER, -- !!! Modified: y_condition -> c_condition
                algorithm TEXT,
                time_taken REAL,
                universe TEXT,
                sets_found TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(m, n, k, j, s, run_index)
            )
        ''')
        print("Database table 'results' is ready. (Note: c_condition column)") # TRANSLATED

        # --- Create run_counters table ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS run_counters (
                m INTEGER NOT NULL,
                n INTEGER NOT NULL,
                k INTEGER NOT NULL,
                j INTEGER NOT NULL,
                s INTEGER NOT NULL,
                last_run_index INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (m, n, k, j, s)
            )
        ''')
        print("Database table 'run_counters' is ready.") # TRANSLATED

        conn.commit() # Commit all changes
        print(f"Database setup complete.") # TRANSLATED

    except sqlite3.Error as e:
        print(f"Database setup error: {e}") # TRANSLATED
    finally:
        if conn:
            conn.close() # Ensure connection is closed

# --- Get and Increment Run Index ---
def get_and_increment_run_index(m, n, k, j, s, db_file=DB_FILE):
    """
    Gets the next run index (starting from 1) for the given parameter combination
    and updates the database counter.
    This function is thread/process safe (safe for calls within a single Python process).

    Args:
        m, n, k, j, s: Parameter combination.
        db_file (str): Path to the database file.

    Returns:
        int: The next available run index (starting from 1).
        None: If a database error occurs.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file, timeout=10) # Increase timeout for busy database
        cursor = conn.cursor()

        # Use transaction for atomicity
        conn.execute("BEGIN") # Start transaction

        # 1. Query the current last_run_index
        cursor.execute("SELECT last_run_index FROM run_counters WHERE m=? AND n=? AND k=? AND j=? AND s=?", (m, n, k, j, s))
        result = cursor.fetchone()

        next_index = 1 # Default to 1 (first run)
        if result:
            # If record found, next index is current index + 1
            current_index = result[0]
            next_index = current_index + 1
            # 2. Update the record
            # print(f"DEBUG: Updating counter for {m,n,k,j,s} from {current_index} to {next_index}") # Debug
            cursor.execute("UPDATE run_counters SET last_run_index = ? WHERE m=? AND n=? AND k=? AND j=? AND s=?", (next_index, m, n, k, j, s))
        else:
            # 2. If no record found, insert a new record, last_run_index is 1
            # print(f"DEBUG: Inserting new counter for {m,n,k,j,s} with index {next_index}") # Debug
            cursor.execute("INSERT INTO run_counters (m, n, k, j, s, last_run_index) VALUES (?, ?, ?, ?, ?, ?)", (m, n, k, j, s, next_index))

        conn.commit() # Commit transaction
        print(f"Database: Assigned and recorded run_index for parameters ({m},{n},{k},{j},{s}): {next_index}") # TRANSLATED
        return next_index

    except sqlite3.Error as e:
        print(f"Database error (getting/incrementing run index for {m},{n},{k},{j},{s}): {e}") # TRANSLATED
        if conn:
            try:
                conn.rollback() # Rollback transaction if error occurred
                print("Database: Get/increment index transaction rolled back.") # TRANSLATED
            except sqlite3.Error as rb_err:
                print(f"Database error: Rollback failed: {rb_err}") # TRANSLATED
        return None # Indicate failure to get index
    finally:
        if conn:
            conn.close()

# --- Data Saving ---
def save_result(result_data, db_file=DB_FILE):
    """
    Saves the result of a single run to the database.

    Args:
        result_data (dict): Dictionary containing the run result. Keys should correspond to table column names.
                             Must include 'm', 'n', 'k', 'j', 's', 'run_index', etc.
                             Should now use 'c_condition' instead of 'y_condition'.
        db_file (str): Path to the database file.

    Returns:
        bool: True if save was successful, False if an error occurred.
    """
    conn = None
    required_keys = {'m', 'n', 'k', 'j', 's', 'run_index', 'num_results'}
    if not required_keys.issubset(result_data.keys()):
         print(f"Error: Missing required keys when saving result. Needed: {required_keys}, Provided: {result_data.keys()}") # TRANSLATED
         return False

    try:
        conn = sqlite3.connect(db_file, timeout=10)
        cursor = conn.cursor()

        # Get dictionary keys and values, ensuring consistent order
        columns = list(result_data.keys())
        values = [result_data[col] for col in columns]

        # Create SQL statement
        cols_str = ', '.join(f'"{col}"' for col in columns) # Add quotes to column names for keyword safety
        placeholders = ', '.join('?' * len(values))
        sql = f'INSERT INTO results ({cols_str}) VALUES ({placeholders})'

        cursor.execute(sql, values) # Use parameterized query
        conn.commit()
        print(f"Successfully saved result (Params: {result_data.get('m')}-{result_data.get('n')}-{result_data.get('k')}-{result_data.get('j')}-{result_data.get('s')}, run={result_data.get('run_index')}) to database {db_file}") # TRANSLATED
        return True
    except sqlite3.IntegrityError:
        print(f"Warning: Attempted to insert duplicate result record into `results` table (m={result_data.get('m')}, ..., run={result_data.get('run_index')}).") # TRANSLATED
        return False
    except sqlite3.Error as e:
        print(f"Database save error (saving to results table): {e}") # TRANSLATED
        import traceback
        traceback.print_exc() # Print detailed error
        return False
    finally:
        if conn:
            conn.close()

# --- Data Query Functions (Unchanged, but note returned dict keys are now c_condition) ---
def get_all_results(db_file=DB_FILE):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows] # Note: This will automatically include the c_condition column
    except sqlite3.Error as e:
        print(f"Database query error (querying results table): {e}") # TRANSLATED
        return []
    finally:
        if conn: conn.close()

def get_all_counters(db_file=DB_FILE):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM run_counters ORDER BY m, n, k, j, s")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"Database query error (querying run_counters table): {e}") # TRANSLATED
        return []
    finally:
        if conn: conn.close()

def get_results_summary(db_file=DB_FILE):
    conn = None
    results = []
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Select required columns, including c_condition
        cursor.execute("""
            SELECT id, m, n, k, j, s, run_index, num_results, c_condition, timestamp
            FROM results
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        # print(f"Database Query: Found {len(results)} result summaries.") # Reduce log noise
    except sqlite3.Error as e:
        print(f"Database query error (querying results summary): {e}") # TRANSLATED
        results = []
    finally:
        if conn: conn.close()
    return results

def get_result_details(result_id, db_file=DB_FILE):
    conn = None
    details = None
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results WHERE id = ?", (result_id,)) # Select all columns
        row = cursor.fetchone()
        if row:
            details = dict(row) # details['c_condition'] will exist (if DB has this column)
            # Try parsing JSON fields
            try:
                details['sets_found_parsed'] = json.loads(details['sets_found']) if details.get('sets_found') else []
                details['universe_parsed'] = json.loads(details['universe']) if details.get('universe') else []
                # print(f"Database Query: Successfully retrieved details for ID={result_id}.") # Reduce log noise
            except json.JSONDecodeError as json_err:
                print(f"Database Warning: Failed to parse JSON field for ID={result_id}: {json_err}") # TRANSLATED
                details['sets_found_parsed'] = f"JSON Parsing Error: {details.get('sets_found')}"
                details['universe_parsed'] = f"JSON Parsing Error: {details.get('universe')}"
            except Exception as parse_err:
                 print(f"Database Error: Error parsing fields for ID={result_id}: {parse_err}") # TRANSLATED
                 details['sets_found_parsed'] = f"Error during parsing: {details.get('sets_found')}"
                 details['universe_parsed'] = f"Error during parsing: {details.get('universe')}"
        else:
            print(f"Database Query: Record with ID={result_id} not found.") # TRANSLATED

    except sqlite3.Error as e:
        print(f"Database query error (getting details for ID={result_id}): {e}") # TRANSLATED
    finally:
        if conn: conn.close()
    # Note: The returned 'details' dictionary will now contain the 'c_condition' key (if the column exists in the DB)
    # Example format string would now use 'c_condition'
    # f"Coverage condition (c): {details.get('c_condition', 'N/A')}\n"
    return details

# --- Data Deletion (Core Modification) ---
def delete_result(result_id, db_file=DB_FILE):
    """
    Deletes a record from the database based on its ID.
    **Modification**: If the deleted record was the latest run for its parameter combination
    (run_index equals last_run_index in run_counters), decrement run_counters accordingly.

    Args:
        result_id (int): The database ID of the result to delete.
        db_file (str): Path to the database file.

    Returns:
        bool: True if deletion (and necessary counter update) was successful, False if an error occurred or the record was not found.
    """
    conn = None
    success = False
    try:
        conn = sqlite3.connect(db_file, timeout=10)
        cursor = conn.cursor()

        # --- Start Transaction ---
        conn.execute("BEGIN")

        # 1. Get details of the record to delete (m, n, k, j, s, run_index)
        cursor.execute("SELECT m, n, k, j, s, run_index FROM results WHERE id = ?", (result_id,))
        result_to_delete = cursor.fetchone()

        if not result_to_delete:
            print(f"Database Operation: Record to delete with ID={result_id} not found.") # TRANSLATED
            conn.rollback() # Rollback transaction
            return False # Return False because not found

        m, n, k, j, s, deleted_run_index = result_to_delete
        print(f"Database Operation: Preparing to delete ID={result_id} (Params: {m}-{n}-{k}-{j}-{s}, RunIndex: {deleted_run_index})") # TRANSLATED

        # 2. Get the current last_run_index for the parameter combination
        cursor.execute("SELECT last_run_index FROM run_counters WHERE m=? AND n=? AND k=? AND j=? AND s=?", (m, n, k, j, s))
        counter_result = cursor.fetchone()

        # If counter doesn't exist (shouldn't happen ideally, but defensive programming)
        if not counter_result:
             print(f"Database Warning: Counter record for parameters {m}-{n}-{k}-{j}-{s} not found, but corresponding result ID={result_id} exists. Data might be inconsistent. Will only delete the result.") # TRANSLATED
             last_run_index_in_counter = -1 # Set to a value that cannot be equal
        else:
             last_run_index_in_counter = counter_result[0]
             print(f"Database Operation: Current last_run_index for parameters {m}-{n}-{k}-{j}-{s} is {last_run_index_in_counter}.") # TRANSLATED

        # 3. Execute the delete operation
        cursor.execute("DELETE FROM results WHERE id = ?", (result_id,))
        rows_affected = cursor.rowcount

        if rows_affected > 0:
            print(f"Database Operation: Successfully deleted ID={result_id} ({rows_affected} rows) from 'results' table.") # TRANSLATED

            # 4. **Conditionally update the counter**
            # Check if the deleted run_index was the highest recorded index
            if deleted_run_index == last_run_index_in_counter:
                # It was the latest, need to decrement the counter
                new_counter_value = last_run_index_in_counter - 1
                 # Ensure counter doesn't go below 0 (theoretically won't, as run_index starts at 1)
                new_counter_value = max(0, new_counter_value)
                print(f"Database Operation: Deleting the latest record, updating run_counters to {new_counter_value}...") # TRANSLATED
                cursor.execute("UPDATE run_counters SET last_run_index = ? WHERE m=? AND n=? AND k=? AND j=? AND s=?",
                               (new_counter_value, m, n, k, j, s))
                if cursor.rowcount > 0:
                     print(f"Database Operation: Successfully updated run_counters.") # TRANSLATED
                else:
                     # Update failed? This is strange, could mean counter record was deleted concurrently?
                     print(f"Database Warning: Attempt to update run_counters failed (no matching row found), despite finding it earlier.") # TRANSLATED
                     # Decide whether to rollback; choosing not to here since the result is already deleted
            else:
                # Deleted was not the latest, no need to update counter
                print(f"Database Operation: Deleted run_index ({deleted_run_index}) is not the latest ({last_run_index_in_counter}), no need to update run_counters.") # TRANSLATED

            # --- Commit Transaction ---
            conn.commit()
            print(f"Database Operation: Deletion and related operations committed.") # TRANSLATED
            success = True
        else:
            # Deletion failed (maybe deleted by another operation between getting info and actual delete)
            print(f"Database Operation: ID={result_id} not found during DELETE execution (possibly deleted concurrently).") # TRANSLATED
            conn.rollback() # Rollback transaction
            success = False

    except sqlite3.Error as e:
        print(f"Database delete error (deleting ID={result_id}): {e}") # TRANSLATED
        if conn:
            try:
                conn.rollback()
                print("Database: Delete operation transaction rolled back.") # TRANSLATED
            except sqlite3.Error as rb_err:
                print(f"Database error: Rollback delete failed: {rb_err}") # TRANSLATED
        success = False
    finally:
        if conn:
            conn.close()
    return success

# --- (Ensure if __name__ == '__main__' part doesn't interfere with imports) ---
if __name__ == '__main__':
    print("Running db.py directly to set up/check the database...") # TRANSLATED
    setup_database()

    print("\n--- Database Operation Test (including delete logic) ---") # TRANSLATED

    # Mock scenario
    test_params = {'m': 45, 'n': 9, 'k': 6, 'j': 5, 's': 5}
    test_db_file = DB_FILE

    def run_and_save_mock(params, run_idx, num_results=10):
        mock_data = params.copy()
        mock_data.update({
            'run_index': run_idx,
            'num_results': num_results,
            'c_condition': params['s'], # !!! Modified: y_condition -> c_condition (just using s to simulate, actual c should come from outside)
            'algorithm': 'MOCK',
            'time_taken': 0.1,
            'universe': json.dumps(list(range(1, params['n']+1))),
            'sets_found': json.dumps([list(range(i+1, i+1+params['k'])) for i in range(num_results)])
        })
        save_result(mock_data, db_file=test_db_file)
        # Get the ID just inserted (assuming it's the last one)
        summary = get_results_summary(db_file=test_db_file)
        return summary[0]['id'] if summary else None

    print("\n--- Scenario Simulation Start ---") # TRANSLATED
    print(f"Parameters: {test_params}") # TRANSLATED

    # 1. Run first time, get run_index 1
    idx1 = get_and_increment_run_index(**test_params, db_file=test_db_file)
    id1 = run_and_save_mock(test_params, idx1) if idx1 else None
    print(f"First run: run_index={idx1}, DB ID={id1}") # TRANSLATED

    # 2. Run second time, get run_index 2
    idx2 = get_and_increment_run_index(**test_params, db_file=test_db_file)
    id2 = run_and_save_mock(test_params, idx2) if idx2 else None
    print(f"Second run: run_index={idx2}, DB ID={id2}") # TRANSLATED

    # 3. Run third time, get run_index 3
    idx3 = get_and_increment_run_index(**test_params, db_file=test_db_file)
    id3 = run_and_save_mock(test_params, idx3) if idx3 else None
    print(f"Third run: run_index={idx3}, DB ID={id3}") # TRANSLATED

    print("\nCurrent counter status:") # TRANSLATED
    counters = get_all_counters(db_file=test_db_file)
    for counter in counters: # Use clearer variable name
        if counter['m'] == test_params['m'] and counter['s'] == test_params['s']: # Simple filter
            print(f"  Params: {counter['m']}-{counter['n']}-{counter['k']}-{counter['j']}-{counter['s']}, LastIndex: {counter['last_run_index']}")

    # 4. Delete record with ID id3 (run_index=3)
    if id3:
        print(f"\n--- Deleting ID={id3} (latest run_index={idx3}) ---") # TRANSLATED
        deleted = delete_result(id3, db_file=test_db_file)
        print(f"Delete operation result: {deleted}") # TRANSLATED

        print("\nCounter status after deletion:") # TRANSLATED
        counters = get_all_counters(db_file=test_db_file)
        for counter in counters:
             if counter['m'] == test_params['m'] and counter['s'] == test_params['s']:
                print(f"  Params: {counter['m']}-{counter['n']}-{counter['k']}-{counter['j']}-{counter['s']}, LastIndex: {counter['last_run_index']} (Expected: {idx3-1})") # TRANSLATED (added Expected)
    else:
        print("\nSkipping delete test because previous insertion failed.") # TRANSLATED

    # 5. Run calculation again, expect run_index to be 3 again
    print("\n--- Running again, expecting run_index reuse ---") # TRANSLATED
    next_idx = get_and_increment_run_index(**test_params, db_file=test_db_file)
    next_id = run_and_save_mock(test_params, next_idx) if next_idx else None
    print(f"Run again: run_index={next_idx} (Expected: {idx3}), DB ID={next_id}") # TRANSLATED (added Expected)

    print("\nFinal counter status:") # TRANSLATED
    counters = get_all_counters(db_file=test_db_file)
    for counter in counters:
         if counter['m'] == test_params['m'] and counter['s'] == test_params['s']:
            print(f"  Params: {counter['m']}-{counter['n']}-{counter['k']}-{counter['j']}-{counter['s']}, LastIndex: {counter['last_run_index']} (Expected: {idx3})") # TRANSLATED (added Expected)

    # 6. (Optional) Test deleting old record (id2, run_index=2)
    if id2:
        print(f"\n--- (Optional) Deleting old record ID={id2} (run_index={idx2}) ---") # TRANSLATED
        deleted_old = delete_result(id2, db_file=test_db_file)
        print(f"Delete old record result: {deleted_old}") # TRANSLATED
        print("\nCounter status after deleting old record:") # TRANSLATED
        counters = get_all_counters(db_file=test_db_file)
        for counter in counters:
             if counter['m'] == test_params['m'] and counter['s'] == test_params['s']:
                 # Counter should still be 3 (from step 5) at this point
                print(f"  Params: {counter['m']}-{counter['n']}-{counter['k']}-{counter['j']}-{counter['s']}, LastIndex: {counter['last_run_index']} (Expected: {idx3}, unchanged)") # TRANSLATED (added Expected, unchanged)

    print("\n--- Scenario Simulation End ---") # TRANSLATED

    print("\nDatabase module test finished.") # TRANSLATED