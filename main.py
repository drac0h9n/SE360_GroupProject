# main.py
# Build a Graphical User Interface (GUI) using Flet.
# Handle user input, call backend calculations, and display results.
# When K=6, the results are saved to an SQLite database.
# Includes options for manually inputting the Universe and specifying the coverage condition c.
# Removed the global RunCounter, using db.py to manage the persistent run_index.
# **Added database result management features, allowing users to view, display details of, and delete records from the database.**
# **Modification: Removed the secondary confirmation dialog for delete operations.**
# **Modification: Added print preview functionality.**

import flet as ft
import random
import time
import json  # For serializing/deserializing lists to store/retrieve from the database
import sys   # For checking command-line arguments
import os    # For path operations
import threading  # For executing calculations in the background to avoid UI blocking
import functools  # For functools.partial (though perhaps not directly used in this version, kept for potential future use)
from datetime import datetime # For formatting timestamps

# Import backend logic and database operations
from backend import Sample, comb, HAS_ORTOOLS  # Import Sample class, comb function, HAS_ORTOOLS
import db  # Import the database operations module (db.py)

# --- Global Variables ---
random_seed = int(time.time())  # Use current time as the random seed
random.seed(random_seed)
print(f"Using random seed: {random_seed}") # Output: Using random seed: ...

# --- Main Application Function ---
def main(page: ft.Page):
    """The main function to build and run the Flet application"""
    page.title = "An Optimal Samples Selection System"  # Window title
    page.theme_mode = ft.ThemeMode.LIGHT  # Theme mode
    page.vertical_alignment = ft.MainAxisAlignment.START  # Page vertical alignment
    page.window_width = 900   # Initial window width
    page.window_height = 850  # Initial window height

    # --- State Variables (for sharing data across functions) ---
    # Using ft.Ref to simplify updating controls or variable values across functions
    selected_db_result_id = ft.Ref[int]()  # Stores the ID of the currently selected record in the database list
    db_results_list_data = ft.Ref[list]() # Stores the list of result summaries loaded from the database [{id:.., m:.., ...}, ...]
    db_results_list_data.current = []     # Initialize as an empty list

    # --- ========================== ---
    # --- UI Control Definitions (Calculation Part) ---
    # --- ========================== ---

    # ... (Control definitions for the calculation part remain unchanged) ...
    # --- Input Controls ---
    txt_m = ft.TextField(label="M (Base Set)", hint_text="Example: 45", width=100, value="45")
    txt_n = ft.TextField(label="N (Universe)", hint_text="Example: 8", width=100, value="8")
    txt_k = ft.TextField(label="K (Block Size)", hint_text="Example: 6", width=100, value="6")
    txt_j = ft.TextField(label="J (Subset)", hint_text="Example: 4", width=100, value="4")
    txt_s = ft.TextField(label="S (Intersection >=)", hint_text="Example: 4", width=100, value="4")
    txt_timeout = ft.TextField(label="Timeout (s)", hint_text="Example: 60", width=100, value="60")

    # --- Universe Input Options ---
    chk_manual_univ = ft.Checkbox(label="Manual Universe Input", value=False, on_change=None) # Checkbox, initially unchecked
    txt_manual_univ = ft.TextField(
        label="Enter N numbers (space-separated, range 1~M)",  # Text field for manual Universe input
        visible=False,  # Initially hidden
        width=600,      # Text field width
        hint_text="Example: 1 5 10 15 20 25 30 35"
    )

    # --- Coverage Condition c Input Options --- # MODIFIED: y -> c
    chk_specify_c = ft.Checkbox(label="Specify c Manually (Coverage Count)", value=False, on_change=None) # Checkbox # MODIFIED: y -> c
    txt_specify_c = ft.TextField( # MODIFIED: y -> c
        label="Enter c value (Range 1 ~ C(j,s))", # Text field for manual c input # MODIFIED: y -> c
        visible=False,  # Initially hidden
        width=250,      # Text field width
        hint_text="Enter positive integer c" # MODIFIED: y -> c
    )

    # --- Output/Information Display Controls ---
    theoretical_c_info = ft.Text("Theoretical Coverage (C(k,s)C(n-k,j-s)): ...", size=12) # Display theoretical c value # MODIFIED: y -> c
    max_single_j_coverage = ft.Text("Max Coverage per J-subset (C(j,s)): ...", size=12) # Display C(j,s)
    # Calculation result display area, allows text selection, sets max lines, allows content overflow for scrolling
    sample_result_info = ft.Text(
        "Calculation results will be displayed here...", # Output: Calculation results will be displayed here...
        size=12,
        selectable=True,       # Allow user to select text
        max_lines=25,          # Limit maximum display lines (scrolling needed if exceeded)
        overflow=ft.TextOverflow.VISIBLE # Content is visible when overflowing (requires external container for scrolling)
    )
    # Log output area, using ListView for automatic scrolling
    log_output = ft.ListView(
        expand=True,           # Allow the list to expand and fill space
        spacing=5,             # Spacing between lines
        auto_scroll=True,      # Automatically scroll to the bottom
        height=200             # Give the log area a fixed height
    )

    # --- Buttons ---
    submit_button = ft.ElevatedButton(text="Start Calculation", on_click=None, icon=ft.icons.PLAY_ARROW) # Trigger calculation
    clear_log_button = ft.ElevatedButton(text="Clear Log", on_click=lambda _: clear_log(), icon=ft.icons.CLEAR_ALL) # Clear log area

    # --- Progress Indicator ---
    progress_ring = ft.ProgressRing(visible=False, width=20, height=20, stroke_width = 3) # Displayed during calculation

    # --- ============================= ---
    # --- UI Control Definitions (Database Management Part) ---
    # --- ============================= ---

    # --- Database Results List and Details Controls ---
    db_results_list_view = ft.ListView(expand=True, spacing=5) # List view to display database record summaries
    # Text area to display details of the selected record
    db_result_details_view = ft.Text(
        "Please select a record above, then click 'Show Details'.", # Output: Please select a record above, then click 'Show Details'.
        selectable=True,
        max_lines=20, # Limit lines
        overflow=ft.TextOverflow.VISIBLE
    )
    # Use RadioGroup to manage list selection, ensuring only one can be selected at a time
    db_results_radio_group = ft.RadioGroup(
        content=db_results_list_view, # Use ListView as the content of RadioGroup
        on_change=None # Event handler for selection change will be bound later
    )

    # --- Database Management Buttons ---
    show_db_view_button = ft.ElevatedButton("View/Manage Database Results", icon=ft.icons.STORAGE, on_click=None) # Switch to database view
    refresh_db_button = ft.ElevatedButton("Refresh List", icon=ft.icons.REFRESH, on_click=None) # Reload the database list
    # Button to display details of the selected item, initially disabled until a user selects an item
    display_details_button = ft.ElevatedButton(
        "Show Details",
        icon=ft.icons.VISIBILITY,
        on_click=None,
        disabled=True # Initially disabled
    )
    # Button to delete the selected item, initially disabled, red color as a warning
    delete_selected_button = ft.ElevatedButton(
        "Delete Selected",
        icon=ft.icons.DELETE_FOREVER,
        on_click=None, # <-- **Modification**: on_click will be directly bound to execute_delete
        color=ft.colors.RED, # Button text color is red
        disabled=True # Initially disabled
    )
    # /// NEW: Add Print button
    print_selected_button = ft.ElevatedButton(
        "Print",
        icon=ft.icons.PRINT,
        on_click=None, # Bind later
        disabled=True # Initially disabled
    )
    # Button to return from the database view to the main calculation interface
    back_to_main_button = ft.ElevatedButton("Back to Calculation Interface", icon=ft.icons.ARROW_BACK, on_click=None)

    # --- Database Result Details Container ---
    db_details_container = ft.Container(
        content=db_result_details_view, # Wrap the details text control
        border=ft.border.all(1, ft.colors.BLACK26), # Add a border
        border_radius=ft.border_radius.all(5),      # Rounded border corners
        padding=10,                                # Inner padding
        margin=ft.margin.only(top=10),             # Top outer margin
        expand=True,                               # Allow container to expand and fill vertical space
    )

    # --- Overall Container for Database Result Management View (Initially hidden) ---
    db_management_view = ft.Column(
        [
            ft.Text("Database Results List", size=16, weight=ft.FontWeight.BOLD), # Title
            # /// MODIFIED: Add Print button to this row
            ft.Row(
                [refresh_db_button, display_details_button, delete_selected_button, print_selected_button, back_to_main_button],
                spacing=10, # Spacing between buttons
                wrap=True   # Allow buttons to wrap to the next line if space is insufficient
            ),
            ft.Text("Select a record to operate on:", size=12), # Hint text
            # Container for the database results list, set border and fixed height
            ft.Container(
                content=db_results_radio_group, # Contains RadioGroup (which includes ListView)
                border=ft.border.all(1, ft.colors.BLACK12),
                border_radius=ft.border_radius.all(5),
                padding=5,
                height=300 # Give the list area a fixed height
            ),
            ft.Container( # Use a Container to wrap the Text and apply margin
                content=ft.Text("Selected Record Details", size=14, weight=ft.FontWeight.BOLD), # Text no longer has margin property
                margin=ft.margin.only(top=10) # Apply margin to the Container holding the Text
            ),
            db_details_container # Details display container
        ],
        visible=False, # Database management view is initially hidden
        expand=True    # Allow this view to expand vertically
    )

    # --- ======================== ---
    # /// NEW: UI Control Definitions (Print Part) ---
    # --- ======================== ---
    print_details_display = ft.Text( # Text control to display content for printing
        "...",
        selectable=True,
        size=12,
        overflow=ft.TextOverflow.VISIBLE # Allow content overflow
    )

    print_back_button = ft.ElevatedButton( # Button to return from print view to database view
        "Back to Database List",
        icon=ft.icons.ARROW_BACK,
        on_click=None # Bind later
    )

    print_preview_view = ft.Column( # Overall container for the print preview view
        [
            ft.Row(
                [
                    ft.Text("Print Preview - Record Details", size=16, weight=ft.FontWeight.BOLD),
                    print_back_button # Place the back button next to the title
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN # Distribute title and button to opposite ends
            ),
            ft.Divider(),
            ft.Container( # Container to display content, with scrollbar
                ft.Column(
                    [print_details_display], # Contains the Text displaying content
                    scroll=ft.ScrollMode.ADAPTIVE # Allow scrolling when content is excessive
                ),
                border=ft.border.all(1, ft.colors.BLACK26),
                border_radius=ft.border_radius.all(5),
                padding=10,
                margin=ft.margin.only(top=10),
                expand=True # Allow the content area to expand and fill vertical space
            )
        ],
        visible=False, # Print preview view initially hidden
        expand=True    # Allow this view to expand vertically
    )

    # --- ========================== ---
    # --- Helper Functions and Event Handlers ---
    # --- ========================== ---

    # ... (log_message, clear_log, show_info_message, set_busy, validate_and_get_int, update_c_related_info, on_manual_univ_change, on_specify_c_change etc. functions remain unchanged) ...
    # --- Logging Function ---
    def log_message(message: str, is_error: bool = False):
        """Adds a timestamped message to the log area (log_output)"""
        timestamp = time.strftime("%H:%M:%S", time.localtime()) # Get current time
        color = ft.colors.RED if is_error else ft.colors.BLACK87 # Use red for error messages
        log_output.controls.append(ft.Text(f"[{timestamp}] {message}", size=11, selectable=True, color=color))
        # Flet is reactive, but sometimes explicitly calling page.update() ensures immediate UI refresh
        page.update()

    def clear_log():
        """Clears the log area"""
        log_output.controls.clear()
        log_message("Log cleared.") # Output: Log cleared.
        page.update()

    # --- Helper Function to Display Info/Error Messages ---
    def show_info_message(text_control: ft.Text, message: str, is_error: bool = False):
        """Updates the specified Text control (like sample_result_info or db_result_details_view) to display information or errors"""
        text_control.value = message # Set the text content
        text_control.color = ft.colors.RED if is_error else ft.colors.BLACK # Set the text color
        page.update()

    # --- Function to Control UI Busy State ---
    def set_busy(busy: bool):
        """Controls the enabled/disabled state of buttons and the progress ring, and the disabled state of input controls"""
        submit_button.disabled = busy        # Disable/enable submit button
        progress_ring.visible = busy         # Show/hide progress ring
        show_db_view_button.disabled = busy  # Disable switching to database view during calculation

        # Disable/enable calculation-related input fields and checkboxes
        for ctrl in [txt_m, txt_n, txt_k, txt_j, txt_s, txt_timeout, chk_manual_univ, txt_manual_univ, chk_specify_c, txt_specify_c]: # MODIFIED: y -> c
            ctrl.disabled = busy

        # Note: The enabled/disabled state of database management view buttons is managed by their own logic (e.g., enabled only after selection)
        # But they can be handled uniformly here if needed (e.g., completely disable database operations during calculation)
        refresh_db_button.disabled = busy
        is_selection_made = selected_db_result_id.current is not None
        display_details_button.disabled = busy or not is_selection_made # Combine with existing logic
        delete_selected_button.disabled = busy or not is_selection_made # Combine with existing logic
        print_selected_button.disabled = busy or not is_selection_made # /// MODIFIED: Control Print button

        back_to_main_button.disabled = busy # Disallow returning during calculation

        page.update() # Apply state changes

    # --- Input Validation and Processing ---
    def validate_and_get_int(field: ft.TextField, name: str, min_val: int = 0) -> int | None:
        """Validates if the value in the input field (field) is an integer and not less than min_val.
           Returns the integer value if valid; otherwise returns None, logs a message, and sets the field's error text."""
        field.error_text = None # Clear previous error state
        field.update()
        try:
            value = int(field.value.strip()) # Get value, remove leading/trailing spaces, try converting to int
            if value < min_val: # Check if less than minimum value
                msg = f"Error: {name} ({value}) cannot be less than {min_val}." # Output: Error: ... cannot be less than ...
                log_message(msg, is_error=True)
                show_info_message(sample_result_info, f"Input Error: {name} cannot be less than {min_val}.", is_error=True) # Display error in main result area # Output: Input Error: ... cannot be less than ...
                field.error_text = f"Cannot be less than {min_val}" # Set small error hint next to the input field # Output: Cannot be less than ...
                field.update()
                return None # Return None indicating validation failure
            return value # Validation passed, return integer value
        except ValueError: # Catch exception if integer conversion fails
            msg = f"Error: {name} ('{field.value}') must be a valid integer." # Output: Error: ... must be a valid integer.
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, f"Input Error: {name} must be an integer.", is_error=True) # Output: Input Error: ... must be an integer.
            field.error_text = "Must be an integer" # Output: Must be an integer
            field.update()
            return None

    # --- Dynamically Update c-Related Information --- # MODIFIED: y -> c
    def update_c_related_info(e=None): # MODIFIED: y -> c
        """Recalculates and updates the display of the theoretical c value (Theoretical Coverage) and C(j,s) (Max Coverage per J-subset) when N, K, J, S input fields change.""" # MODIFIED: y -> c
        # Safely get input values, None if invalid
        n_str, k_str, j_str, s_str = txt_n.value, txt_k.value, txt_j.value, txt_s.value
        n, k, j, s = None, None, None, None
        try: n = int(n_str.strip()) if n_str.strip() else None
        except ValueError: pass
        try: k = int(k_str.strip()) if k_str.strip() else None
        except ValueError: pass
        try: j = int(j_str.strip()) if j_str.strip() else None
        except ValueError: pass
        try: s = int(s_str.strip()) if s_str is not None and s_str.strip() != "" else None # s=0 is valid
        except ValueError: pass

        # --- Update C(j,s) display ---
        max_single_cov_val_str = "Waiting for valid J, S..." # Default hint # Output: Waiting for valid J, S...
        max_single_j_coverage.color = ft.colors.GREY # Default grey
        max_c_limit = 0 # Store the result of C(j,s) calculation for the range hint of manual c input # MODIFIED: y -> c

        if j is not None and s is not None and j >= 0 and s >= 0: # Ensure j and s are valid non-negative integers
            try:
                if j < s:
                    max_single_cov_val = 0
                    max_single_cov_val_str = f"C({j},{s}): 0 (because j < s)" # Output: C(...,...): 0 (because j < s)
                    max_single_j_coverage.color = ft.colors.ORANGE # Orange hint
                else:
                    max_single_cov_val = comb(j, s) # Call backend comb function to calculate combinations
                    max_single_cov_val_str = f"C({j},{s}): {max_single_cov_val}"
                    max_single_j_coverage.color = ft.colors.BLACK # Normal black display
                    max_c_limit = max_single_cov_val # Update the upper limit for c # MODIFIED: y -> c
                # Update the label of the manual c input field to hint the valid range # MODIFIED: y -> c
                txt_specify_c.label = f"Enter c value (1 ~ {max_c_limit})" if max_c_limit > 0 else "Enter c value (invalid or not needed)" # MODIFIED: y -> c twice # Output: Enter c value (1 ~ ...) or Enter c value (invalid or not needed)
                txt_specify_c.update() # MODIFIED: y -> c
            except ValueError as combo_err: # Catch parameter errors from comb function
                 max_single_cov_val_str = f"C({j},{s}): Calculation error ({combo_err})" # Output: C(...,...): Calculation error (...)
                 max_single_j_coverage.color = ft.colors.RED # Red error hint
                 txt_specify_c.label = "Enter c value (requires valid J,S)" # MODIFIED: y -> c # Output: Enter c value (requires valid J,S)
                 txt_specify_c.update() # MODIFIED: y -> c
            except Exception as calc_err: # Catch other possible calculation errors
                 max_single_cov_val_str = f"C({j},{s}): Unknown error ({calc_err})" # Output: C(...,...): Unknown error (...)
                 max_single_j_coverage.color = ft.colors.RED
                 txt_specify_c.label = "Enter c value (calculation error)" # MODIFIED: y -> c # Output: Enter c value (calculation error)
                 txt_specify_c.update() # MODIFIED: y -> c
        else: # If j or s is invalid
             txt_specify_c.label = "Enter c value (requires valid J,S)" # MODIFIED: y -> c # Output: Enter c value (requires valid J,S)
             txt_specify_c.update() # MODIFIED: y -> c
        # Update the text control displaying C(j,s)
        max_single_j_coverage.value = f"Max Coverage per J-subset (C(j,s)): {max_single_cov_val_str}" # Output: Max Coverage per J-subset (C(j,s)): ...

        # --- Update theoretical c = C(k,s)*C(n-k,j-s) display --- # MODIFIED: y -> c
        theoretical_c_val_str = "Waiting for valid N, K, J, S..." # Default hint # MODIFIED: y -> c # Output: Waiting for valid N, K, J, S...
        theoretical_c_info.color = ft.colors.GREY       # Default grey # MODIFIED: y -> c

        # Ensure n, k, j, s are all valid non-negative integers
        if n is not None and k is not None and j is not None and s is not None and all(v >= 0 for v in [n,k,j,s]):
            try:
                 # Check if parameters for combinations are valid
                 term1_valid = (k >= s)
                 term2_valid = (n - k >= j - s) and (j >= s) # Note: (n-k) and (j-s) must both be non-negative
                 params_logic_valid = (n >= k and n >= j) # Basic logic check

                 if not params_logic_valid:
                      theoretical_c_val_str = "Invalid parameters (N must be >= K and >= J)" # MODIFIED: y -> c # Output: Invalid parameters (N must be >= K and >= J)
                      theoretical_c_info.color = ft.colors.ORANGE # MODIFIED: y -> c
                 elif not term1_valid or not term2_valid: # If any combination parameter is invalid
                     theoretical_c_value = 0 # MODIFIED: y -> c
                     invalid_term = ""
                     if not term1_valid: invalid_term += " C(k,s)"
                     if not term2_valid: invalid_term += " C(n-k,j-s)"
                     theoretical_c_val_str = f"0 (due to invalid combination term(s):{invalid_term.strip()})" # MODIFIED: y -> c # Output: 0 (due to invalid combination term(s):...)
                     theoretical_c_info.color = ft.colors.BLACK # Result is 0, but reason is clear, use black # MODIFIED: y -> c
                 else:
                    comb1 = comb(k, s)       # Calculate C(k,s)
                    comb2 = comb(n - k, j - s) # Calculate C(n-k, j-s)
                    theoretical_c_value = comb1 * comb2 # Calculate theoretical c value # MODIFIED: y -> c twice
                    theoretical_c_val_str = f"{theoretical_c_value} (C({k},{s})={comb1} * C({n-k},{j-s})={comb2})" # MODIFIED: y -> c twice
                    theoretical_c_info.color = ft.colors.BLACK # Normal black # MODIFIED: y -> c
            except ValueError as combo_error: # Catch errors during combination calculation
                 theoretical_c_val_str = f"Calculation error ({combo_error})" # MODIFIED: y -> c # Output: Calculation error (...)
                 theoretical_c_info.color = ft.colors.RED # MODIFIED: y -> c
            except Exception as calc_error:   # Catch other calculation errors
                 theoretical_c_val_str = f"Unknown error ({calc_error})" # MODIFIED: y -> c # Output: Unknown error (...)
                 theoretical_c_info.color = ft.colors.RED # MODIFIED: y -> c
        # Update the text control displaying the theoretical c value # MODIFIED: y -> c
        theoretical_c_info.value = f"Theoretical Coverage (C(k,s)C(n-k,j-s)): {theoretical_c_val_str}" # MODIFIED: y -> c twice # Output: Theoretical Coverage (C(k,s)C(n-k,j-s)): ...

        page.update() # Update page display

    # --- Bind on_change events for N, K, J, S input fields ---
    txt_n.on_change = update_c_related_info # MODIFIED: y -> c
    txt_k.on_change = update_c_related_info # MODIFIED: y -> c
    txt_j.on_change = update_c_related_info # MODIFIED: y -> c
    txt_s.on_change = update_c_related_info # MODIFIED: y -> c

    # --- Checkbox Event Handlers ---
    def on_manual_univ_change(e):
        """Toggles the visibility of the corresponding text field when the 'Manual Universe Input' checkbox state changes"""
        is_manual = chk_manual_univ.value
        txt_manual_univ.visible = is_manual # Set text field visibility
        if not is_manual: # If unchecked
            txt_manual_univ.value = ""        # Clear text field content
            txt_manual_univ.error_text = None # Clear error hint
            txt_manual_univ.update()          # Update text field
        # For better layout, we put the text field in a Row and toggle the Row's visibility
        manual_univ_row = txt_manual_univ.parent # Get the parent Row containing the text field
        if manual_univ_row:
            manual_univ_row.visible = is_manual # Set Row visibility
        page.update() # Update page

    def on_specify_c_change(e): # MODIFIED: y -> c
        """Toggles the visibility of the corresponding text field when the 'Specify c Manually' checkbox state changes""" # MODIFIED: y -> c
        is_manual = chk_specify_c.value # MODIFIED: y -> c
        txt_specify_c.visible = is_manual # MODIFIED: y -> c
        if not is_manual:
            txt_specify_c.value = "" # MODIFIED: y -> c
            txt_specify_c.error_text = None # MODIFIED: y -> c
            txt_specify_c.update() # MODIFIED: y -> c
        # Similarly, toggle the visibility of the Row containing the text field
        specify_c_row = txt_specify_c.parent # MODIFIED: y -> c twice
        if specify_c_row: # MODIFIED: y -> c
            specify_c_row.visible = is_manual # MODIFIED: y -> c
        page.update()

    # --- Bind on_change events for Checkboxes ---
    chk_manual_univ.on_change = on_manual_univ_change
    chk_specify_c.on_change = on_specify_c_change # MODIFIED: y -> c twice

    # --- ================================ ---
    # --- Logic Functions for Database Result Management ---
    # --- ================================ ---

    def format_result_summary(result_item: dict) -> str:
        """Formats a single result summary dictionary fetched from the database into a readable string for list display.
           **Modified Format**: ID: X | M-N-K-J-S-RunIndex-Y | Timestamp
        """
        # ... (This function remains unchanged) ...
        try:
            # Try parsing ISO format timestamp
            timestamp_dt = datetime.fromisoformat(result_item['timestamp']) if result_item.get('timestamp') else datetime.now()
        except ValueError:
            # Provide an alternative display or log error if parsing fails
            timestamp_dt = datetime.now() # Or None
            print(f"Warning: Could not parse timestamp '{result_item.get('timestamp')}' for ID {result_item.get('id')}") # Output: Warning: Could not parse timestamp '...' for ID ...

        # Format the timestamp
        time_str = timestamp_dt.strftime('%Y-%m-%d %H:%M') if timestamp_dt else "N/A"

        # Build the string segment containing M, N, K, J, S, RunIndex, and NumResults (Y)
        params_run_num_str = (f"{result_item.get('m','?')}-"
                              f"{result_item.get('n','?')}-"
                              f"{result_item.get('k','?')}-"
                              f"{result_item.get('j','?')}-"
                              f"{result_item.get('s','?')}-"
                              f"{result_item.get('run_index','?')}-"  # Append RunIndex
                              f"{result_item.get('num_results','?')}") # Directly append NumResults (Y)

        # Build the final summary string using the modified format
        return (f"ID: {result_item.get('id', 'N/A')} | "
                f"{params_run_num_str} | " # Use the new combined string segment
                f"{time_str}")

    def update_db_list_view():
        """Updates the database results list UI (ListView) based on the global state `db_results_list_data.current`"""
        # ... (Basically unchanged, just resets print preview details when selection is cleared) ...
        db_results_list_view.controls.clear() # Clear existing list items
        if not db_results_list_data.current: # If data is empty
            db_results_list_view.controls.append(ft.Text("No result records found in the database.")) # Output: No result records found in the database.
        else:
            # Iterate through data, create a Radio button for each record
            for item in db_results_list_data.current:
                summary_text = format_result_summary(item) # Get formatted summary text
                # Each list item is a Radio button, its value stores the record's database ID (as a string)
                db_results_list_view.controls.append(
                    ft.Radio(value=str(item['id']), label=summary_text)
                )
        # Clear previous selection state and details display
        db_results_radio_group.value = None        # Clear RadioGroup's selected value
        selected_db_result_id.current = None       # Clear globally stored selected ID
        db_result_details_view.value = "Please select a record above, then click 'Show Details'." # Reset details area hint # Output: Please select a record above, then click 'Show Details'.
        print_details_display.value = ""           # /// NEW: Clear print preview area
        display_details_button.disabled = True     # Disable 'Show Details' button
        delete_selected_button.disabled = True     # Disable 'Delete Selected' button
        print_selected_button.disabled = True      # /// NEW: Disable 'Print' button
        page.update() # Update UI

    def load_db_results(e=None):
        """Loads the result summary list from the database and updates the UI"""
        # ... (This function remains unchanged) ...
        log_message("Loading result list from database...") # Output: Loading result list from database...
        try:
            results = db.get_results_summary() # Call db module function to get summary list
            db_results_list_data.current = results # Update global state variable
            update_db_list_view() # Update ListView interface with new data
            log_message(f"Successfully loaded {len(results)} result summaries.") # Output: Successfully loaded ... result summaries.
        except Exception as ex: # Catch potential database or other errors
            msg = f"Error loading database result list: {ex}" # Output: Error loading database result list: ...
            log_message(msg, is_error=True)
            # Can display error message in the details area
            show_info_message(db_result_details_view, f"Error: {msg}", is_error=True) # Output: Error: ...
            db_results_list_data.current = [] # Clear data on error
            update_db_list_view() # Update UI to empty list state
        finally:
             page.update() # Ensure page is finally updated

    def on_db_result_select(e):
        """Callback function triggered when a user selects a Radio button in the database results list"""
        # /// MODIFIED: Add control for the Print button
        selected_id_str = db_results_radio_group.value # Get the currently selected value of RadioGroup (i.e., the record ID string)
        if selected_id_str: # If an item is selected
            try:
                selected_id_int = int(selected_id_str) # Convert ID string to integer
                selected_db_result_id.current = selected_id_int # Update global state variable
                display_details_button.disabled = False # Enable 'Show Details' button
                delete_selected_button.disabled = False # Enable 'Delete Selected' button
                print_selected_button.disabled = False # /// NEW: Enable 'Print' button
                log_message(f"Selected database record ID: {selected_id_int}") # Output: Selected database record ID: ...
            except (ValueError, TypeError): # Catch integer conversion failure or other type errors
                selected_db_result_id.current = None
                display_details_button.disabled = True
                delete_selected_button.disabled = True
                print_selected_button.disabled = True # /// NEW: Disable 'Print' button
                log_message(f"Invalid selection value: '{selected_id_str}'.", is_error=True) # Output: Invalid selection value: '...'.
        else: # If no item is selected (e.g., list is empty or selection cleared)
            selected_db_result_id.current = None
            display_details_button.disabled = True
            delete_selected_button.disabled = True
            print_selected_button.disabled = True # /// NEW: Disable 'Print' button
        # Prompt user for the next action
        db_result_details_view.value = "Please click 'Show Details' to view the selected record." # Output: Please click 'Show Details' to view the selected record.
        print_details_display.value = "" # /// NEW: Clear print preview area
        page.update() # Update button states and details area text

    # --- Bind database list selection event ---
    db_results_radio_group.on_change = on_db_result_select

    def display_selected_details(e):
        """Triggered when the 'Show Details' button is clicked, fetches and displays detailed information of the selected record"""
        # ... (This function remains unchanged) ...
        if selected_db_result_id.current is None: # Check if a record is selected
            log_message("No record selected to show details.", is_error=True) # Output: No record selected to show details.
            return

        target_id = selected_db_result_id.current
        log_message(f"Fetching details for ID={target_id}...") # Output: Fetching details for ID=...
        # Update details area to show loading status
        show_info_message(db_result_details_view, f"Loading details for ID={target_id}...") # Output: Loading details for ID=...
        page.update() # Immediately show loading hint

        try:
            # Call db module function to get full details
            details = db.get_result_details(target_id)
            if details: # If details were successfully fetched
                # --- Format details information for display ---
                params_str = f"{details.get('m','?')}-{details.get('n','?')}-{details.get('k','?')}-{details.get('j','?')}-{details.get('s','?')}-{details.get('run_index','?')}"
                # Get parsed Universe and Sets (if parsing failed, db.py stores an error string)
                universe_disp = details.get('universe_parsed', 'N/A')
                sets_list = details.get('sets_found_parsed', [])
                num_sets = len(sets_list) if isinstance(sets_list, list) else 0 # Ensure sets_list is a list before calculating length

                # Build the text to display
                details_text = (
                    f"ID: {details.get('id')}\n"
                    f"Parameters (M-N-K-J-S-RunIdx): {params_str}\n" # Output: Parameters (M-N-K-J-S-RunIdx): ...
                    f"Timestamp: {details.get('timestamp')}\n" # Output: Timestamp: ...
                    f"Algorithm: {details.get('algorithm', 'N/A')}\n" # Output: Algorithm: ...
                    f"Time Taken: {details.get('time_taken', 0):.2f} seconds\n" # Format float # Output: Time Taken: ... seconds
                    f"Coverage Condition (c): {details.get('c_condition', 'N/A')}\n" # MODIFIED: y -> c (twice) # Output: Coverage Condition (c): ...
                    f"Universe ({len(universe_disp) if isinstance(universe_disp, list) else 'N/A'} items): {universe_disp}\n" # Output: Universe (... items): ...
                    f"Found Sets ({num_sets} groups):\n" # Output: Found Sets (... groups):
                )

                # --- Format display of the sets list (sets_list) ---
                MAX_SETS_TO_DISPLAY_DB = 100 # Can display more sets in the details view
                if num_sets > 0 and isinstance(sets_list, list):
                    sets_to_display = sets_list[:MAX_SETS_TO_DISPLAY_DB]
                    sets_lines = []
                    sets_per_line = 4 # How many sets to display per line
                    for i in range(0, len(sets_to_display), sets_per_line):
                         # Sort each set, convert to string, join with |
                         line = " | ".join([str(sorted(s)) for s in sets_to_display[i:i+sets_per_line]])
                         sets_lines.append(f"  {line}") # Add indentation
                    details_text += "\n".join(sets_lines) # Combine all lines
                    if num_sets > MAX_SETS_TO_DISPLAY_DB: # If too many sets to fully display
                        details_text += f"\n  ... ({num_sets - MAX_SETS_TO_DISPLAY_DB} more not shown)" # Output: ... (... more not shown)
                elif isinstance(sets_list, str): # If JSON parsing failed, sets_list will be an error string
                     details_text += f"  (Could not parse sets: {sets_list})" # Output: (Could not parse sets: ...)
                else: # If no sets
                    details_text += "  (None)" # Output: (None)

                # Update the details display area
                show_info_message(db_result_details_view, details_text)
                log_message(f"Displayed details for ID={target_id}.") # Output: Displayed details for ID=...
            else: # If db.get_result_details returned None (record not found)
                msg = f"Details for ID={target_id} not found in the database." # Output: Details for ID=... not found in the database.
                log_message(msg, is_error=True)
                show_info_message(db_result_details_view, msg, is_error=True)
        except Exception as ex: # Catch other errors during fetching or formatting
            msg = f"Error displaying details (ID={target_id}): {ex}" # Output: Error displaying details (ID=...): ...
            log_message(msg, is_error=True)
            show_info_message(db_result_details_view, f"Error: {msg}", is_error=True) # Output: Error: ...
        finally:
             page.update() # Ensure page is finally updated

    def execute_delete(e):
        """Executes the actual delete operation (called directly by the 'Delete Selected' button)"""
        # ... (This function remains unchanged) ...
        if selected_db_result_id.current is None: # Double check if an item is selected
            log_message("No record selected for deletion.", is_error=True) # Output: No record selected for deletion.
            return

        target_id = selected_db_result_id.current
        log_message(f"Attempting to directly delete record ID={target_id}...") # Output: Attempting to directly delete record ID=...
        try:
            success = db.delete_result(target_id) # Call db module's delete function
            if success:
                log_message(f"Record ID={target_id} successfully deleted from the database. Refreshing list...") # Output: Record ID=... successfully deleted from the database. Refreshing list...
                # After successful deletion, reload the database list to reflect changes
                load_db_results() # load_db_results will automatically update UI and clear selection
            else:
                # If db.delete_result returns False (possibly because record not found, or sqlite error without exception)
                # db.py should have logged the specific reason internally
                log_message(f"Delete operation for record ID={target_id} completed, but it might not have been actually deleted (please check logs).", is_error=True) # Output: Delete operation for record ID=... completed, but it might not have been actually deleted (please check logs).
        except Exception as ex: # Catch exceptions during the deletion process
            msg = f"Error occurred while deleting record ID={target_id}: {ex}" # Output: Error occurred while deleting record ID=...: ...
            log_message(msg, is_error=True)
            show_info_message(db_result_details_view, f"Deletion Error: {msg}", is_error=True) # Display error in details area # Output: Deletion Error: ...
        finally:
             page.update() # Ensure UI is updated

    # /// NEW: Format details for printing (with numbering)
    def format_details_for_printing(details: dict) -> str:
        """Formats record details into a string suitable for printing, with numbered sets."""
        if not details:
            return "Error: Could not fetch record details." # Output: Error: Could not fetch record details.

        try:
            params_str = f"{details.get('m', '?')}-{details.get('n', '?')}-{details.get('k', '?')}-{details.get('j', '?')}-{details.get('s', '?')}-{details.get('run_index', '?')}"
            universe_disp = details.get('universe_parsed', 'N/A')
            sets_list = details.get('sets_found_parsed', [])
            num_sets = len(sets_list) if isinstance(sets_list, list) else 0
            timestamp_str = details.get('timestamp', 'N/A')
            # Try for a more user-friendly time format
            try:
                 timestamp_dt = datetime.fromisoformat(timestamp_str)
                 timestamp_str = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                 pass # Keep the original string

            # Build basic information
            print_text = (
                f"Record ID: {details.get('id')}\n" # Output: Record ID: ...
                f"Parameters (M-N-K-J-S-RunIdx): {params_str}\n" # Output: Parameters (M-N-K-J-S-RunIdx): ...
                f"Saved Time: {timestamp_str}\n" # Output: Saved Time: ...
                f"Algorithm: {details.get('algorithm', 'N/A')}\n" # Output: Algorithm: ...
                f"Calculation Time: {details.get('time_taken', 0):.2f} seconds\n" # Output: Calculation Time: ... seconds
                f"Coverage Condition (c): {details.get('c_condition', 'N/A')}\n" # Output: Coverage Condition (c): ...
                f"Universe Used ({len(universe_disp) if isinstance(universe_disp, list) else 'N/A'} items):\n  {universe_disp}\n" # Output: Universe Used (... items):
                "========================================\n"
                f"Found Sets ({num_sets} groups):\n" # Output: Found Sets (... groups):
                "========================================\n"
            )

            # --- Add numbered list of sets ---
            if num_sets > 0 and isinstance(sets_list, list):
                set_lines = []
                for idx, s_item in enumerate(sets_list, start=1):
                    # Sort each set and convert to string
                    set_str = str(sorted(s_item))
                    set_lines.append(f"{idx}. {set_str}") # Add number
                print_text += "\n".join(set_lines) # Combine all numbered lines
            elif isinstance(sets_list, str): # If JSON parsing failed
                print_text += f"  (Could not parse sets: {sets_list})" # Output: (Could not parse sets: ...)
            else: # If no sets
                print_text += "  (None)" # Output: (None)

            return print_text

        except Exception as e:
            log_message(f"Error formatting text for printing: {e}", is_error=True) # Output: Error formatting text for printing: ...
            return f"Error: Error formatting details for record {details.get('id','N/A')}.\n{e}" # Output: Error: Error formatting details for record ... .

    # /// NEW: Function to show the print preview interface
    def show_print_preview(e=None):
        """Fetches selected record details, formats them, and displays in the print preview view."""
        if selected_db_result_id.current is None:
            log_message("No record selected for printing.", is_error=True) # Output: No record selected for printing.
            # Could display error in print preview view, but this shouldn't normally happen (button disabled)
            print_details_display.value = "Error: No record selected." # Output: Error: No record selected.
            print_details_display.color = ft.colors.RED
            page.update()
            return

        target_id = selected_db_result_id.current
        log_message(f"Preparing print preview for ID={target_id}...") # Output: Preparing print preview for ID=...
        print_details_display.value = f"Loading data for ID={target_id} for print preview..." # Output: Loading data for ID=... for print preview...
        print_details_display.color = ft.colors.BLACK # Reset color
        # Force update to show loading message
        main_computation_view.visible = False
        db_management_view.visible = False
        print_preview_view.visible = True
        page.update()

        try:
            # Fetch full details
            details = db.get_result_details(target_id)
            if details:
                # Format details for printing
                formatted_print_text = format_details_for_printing(details)
                print_details_display.value = formatted_print_text
                log_message(f"Generated print preview content for ID={target_id}.") # Output: Generated print preview content for ID=...
            else:
                msg = f"Error: Could not fetch details for ID={target_id} for printing." # Output: Error: Could not fetch details for ID=... for printing.
                log_message(msg, is_error=True)
                print_details_display.value = msg
                print_details_display.color = ft.colors.RED
        except Exception as ex:
            msg = f"Error generating print preview (ID={target_id}): {ex}" # Output: Error generating print preview (ID=...): ...
            log_message(msg, is_error=True)
            print_details_display.value = f"Error: {msg}" # Output: Error: ...
            print_details_display.color = ft.colors.RED

        # Ensure the final interface is the updated print preview view
        main_computation_view.visible = False
        db_management_view.visible = False
        print_preview_view.visible = True
        page.update()

    # --- Bind database management button events ---
    refresh_db_button.on_click = load_db_results       # Refresh button -> Load data
    display_details_button.on_click = display_selected_details # Show Details button -> Display details
    delete_selected_button.on_click = execute_delete     # Delete button -> Execute delete directly
    print_selected_button.on_click = show_print_preview # /// NEW: Print button -> Show print preview

    # --- ================= ---
    # --- View Switching Logic ---
    # --- ================= ---

    # --- Main Computation View Container (Organizes the original calculation interface controls in a Column) ---
    main_computation_view = ft.Column(
        [
            # Parameter input title
            ft.Text("Parameter Input", size=16, weight=ft.FontWeight.BOLD),
            # Row for M, N, K, J, S, Timeout input fields
            ft.Row(
                [txt_m, txt_n, txt_k, txt_j, txt_s, txt_timeout],
                spacing=10, alignment=ft.MainAxisAlignment.START, wrap=True
            ),
            # Row for checkboxes
            ft.Row(
                [chk_manual_univ, chk_specify_c], # MODIFIED: y -> c
                alignment=ft.MainAxisAlignment.START, spacing=20, vertical_alignment=ft.CrossAxisAlignment.CENTER
            ),
            # Row for manual Universe input (visibility controlled by Row)
            ft.Row([txt_manual_univ], visible=chk_manual_univ.value),
            # Row for manual c input (visibility controlled by Row) # MODIFIED: y -> c
            ft.Row([txt_specify_c], visible=chk_specify_c.value), # MODIFIED: y -> c twice
            # Divider line
            ft.Divider(height=10),
            # Row displaying theoretical c and C(j,s) # MODIFIED: y -> c
            ft.Row([theoretical_c_info, max_single_j_coverage], alignment=ft.MainAxisAlignment.SPACE_AROUND), # MODIFIED: y -> c
            # Divider line
            ft.Divider(height=10),
            # Row for action buttons (Calculate, Clear Log, View Database)
            ft.Row(
                [submit_button, progress_ring, clear_log_button, show_db_view_button], # Added show_db_view_button
                alignment=ft.MainAxisAlignment.START, spacing=15, vertical_alignment=ft.CrossAxisAlignment.CENTER
            ),
            # Log area title
            ft.Text("Run Log", size=14, weight=ft.FontWeight.BOLD),
            # Log output container (fixed height)
            ft.Container(
                content=log_output,
                border=ft.border.all(1, ft.colors.BLACK12),
                border_radius=ft.border_radius.all(5),
                padding=5,
                expand=False # Log area does not automatically expand to fill space
            ),
             # Calculation results area title
            ft.Text("Calculation Results", size=14, weight=ft.FontWeight.BOLD),
             # Calculation results display container (allows expansion)
            ft.Container(
                content=sample_result_info,
                border=ft.border.all(1, ft.colors.BLACK26),
                border_radius=ft.border_radius.all(5),
                padding=10,
                margin=ft.margin.only(top=10),
                expand=True # Result area can expand to fill remaining vertical space
            )
        ],
        visible=True, # Main computation view is visible by default
        expand=True   # Allow this view to expand vertically
    )

    # /// MODIFIED: Update view switching logic to include print view
    def switch_view(view_name: str):
        """Switches the visible view: 'main', 'db', 'print'"""
        main_computation_view.visible = (view_name == 'main')
        db_management_view.visible = (view_name == 'db')
        print_preview_view.visible = (view_name == 'print')

        if view_name == 'db':
            # When switching to database view, refresh the list if not coming from print preview
            # (If just returned from print preview, list content remains the same, no need to refresh)
            if not print_preview_view.visible: # Simple check, may need more precise state
                load_db_results()
            log_message("Switched to Database Results Management view.") # Output: Switched to Database Results Management view.
        elif view_name == 'main':
            log_message("Returned to the main Calculation interface.") # Output: Returned to the main Calculation interface.
        elif view_name == 'print':
            # show_print_preview function handles content loading and logging
             pass # Logging done in show_print_preview

        page.update()

    # --- Bind view switching button events ---
    # Click 'View/Manage Database Results' button
    show_db_view_button.on_click = lambda e: switch_view('db')
    # Click 'Back to Calculation Interface' button in database view
    back_to_main_button.on_click = lambda e: switch_view('main')
    # /// NEW: Click 'Back to Database List' button in print preview
    print_back_button.on_click = lambda e: switch_view('db')

    # --- ========================== ---
    # --- Core Logic Function for Submitting Calculation ---
    # --- ========================== ---
    # ... (on_submit and run_computation functions remain unchanged) ...
    def on_submit(e):
        """Handles the submit button click event: validates input, gets run_index, starts background calculation thread."""
        log_message("Calculation request received...") # Output: Calculation request received...
        show_info_message(sample_result_info, "Processing input parameters...") # Output: Processing input parameters...

        # --- 1. Clear old error hints ---
        for field in [txt_m, txt_n, txt_k, txt_j, txt_s, txt_timeout, txt_manual_univ, txt_specify_c]: # MODIFIED: y -> c
            field.error_text = None
            field.update()

        error_occurred = False # Flag to track if an error occurred

        # --- 2. Get and validate basic parameters M, N, K, J, S, Timeout ---
        m_processed = validate_and_get_int(txt_m, "M", 1)
        n_processed = validate_and_get_int(txt_n, "N", 1)
        k_processed = validate_and_get_int(txt_k, "K", 1)
        j_processed = validate_and_get_int(txt_j, "J", 1)
        s_processed = validate_and_get_int(txt_s, "S", 0) # S can be 0
        timeout_val = validate_and_get_int(txt_timeout, "Timeout", 1) # Changed name for translation consistency

        if None in [m_processed, n_processed, k_processed, j_processed, s_processed, timeout_val]:
            log_message("Basic parameter validation failed.", is_error=True) # Output: Basic parameter validation failed.
            error_occurred = True

        if error_occurred: return # Stop if basic parameter validation failed

        timeout_seconds = timeout_val

        # --- 3. Further check logical relationships between parameters ---
        validation_errors = []
        if n_processed > m_processed: validation_errors.append(f"N ({n_processed}) cannot be greater than M ({m_processed}).")
        if k_processed > n_processed: validation_errors.append(f"K ({k_processed}) cannot be greater than N ({n_processed}).")
        if j_processed > n_processed: validation_errors.append(f"J ({j_processed}) cannot be greater than N ({n_processed}).")
        if s_processed > k_processed: validation_errors.append(f"S ({s_processed}) cannot be greater than K ({k_processed}).")
        if s_processed > j_processed: validation_errors.append(f"S ({s_processed}) cannot be greater than J ({j_processed}).")
        if s_processed < 0: validation_errors.append("S cannot be less than 0.")

        if validation_errors: # If logical errors exist
            error_msg = "\n".join(validation_errors)
            log_message(f"Parameter logic error: {error_msg}", is_error=True) # Output: Parameter logic error: ...
            show_info_message(sample_result_info, f"Error:\n{error_msg}", is_error=True) # Output: Error:\n...
            return # Stop execution

        # --- 4. Handle Universe (based on checkbox state) ---
        is_manual_univ = chk_manual_univ.value # Get checkbox state
        log_message(f"Manual Universe option: {'Checked' if is_manual_univ else 'Unchecked'}") # Output: Manual Universe option: Checked/Unchecked
        univ = [] # Initialize Universe list
        if is_manual_univ: # If manual input is checked
            manual_str = txt_manual_univ.value.strip() # Get manually entered value
            txt_manual_univ.error_text = None # Clear old error
            if not manual_str: # Input is empty
                msg = "Error: Manual Universe input checked, but the input field is empty." # Output: Error: Manual Universe input checked, but the input field is empty.
                log_message(msg, is_error=True)
                show_info_message(sample_result_info, msg, is_error=True)
                txt_manual_univ.error_text = "Input cannot be empty" # Output: Input cannot be empty
                txt_manual_univ.update()
                error_occurred = True
            else: # Input is not empty, try parsing
                try:
                    num_strs = manual_str.split() # Split by spaces
                    univ_nums_temp = []
                    for num_str in num_strs: # Convert one by one
                        try: univ_nums_temp.append(int(num_str))
                        except ValueError: raise ValueError(f"Input '{num_str}' is not a valid integer.")

                    # Check quantity, duplicates, range
                    if len(univ_nums_temp) != n_processed: raise ValueError(f"Expected {n_processed} numbers, but got {len(univ_nums_temp)}.")
                    if len(set(univ_nums_temp)) != len(univ_nums_temp): raise ValueError("Input numbers contain duplicates.")
                    invalid_nums = [x for x in univ_nums_temp if not (1 <= x <= m_processed)]
                    if invalid_nums: raise ValueError(f"Numbers must be between 1 and {m_processed}. Invalid: {invalid_nums}")

                    univ = sorted(univ_nums_temp) # Sort and store in univ
                    log_message(f"Using manually entered Universe: {univ}") # Output: Using manually entered Universe: [...]
                    txt_manual_univ.update() # Update to clear possible old error hints
                except ValueError as ve: # Catch errors during parsing or validation
                    log_message(f"Manual Universe input error: {ve}", is_error=True) # Output: Manual Universe input error: ...
                    show_info_message(sample_result_info, f"Manual Universe input error: {ve}", is_error=True) # Output: Manual Universe input error: ...
                    txt_manual_univ.error_text = str(ve) # Display error next to the text field
                    txt_manual_univ.update()
                    error_occurred = True
        else: # If manual input is not checked, generate randomly
            try:
                univ = sorted(random.sample(range(1, m_processed + 1), n_processed)) # Random sampling
                log_message(f"Randomly generated Universe: {univ}") # Output: Randomly generated Universe: [...]
            except ValueError as e: # Catch sampling errors due to M < N, etc.
                 msg = f"Cannot generate random Universe (M={m_processed}, N={n_processed}): {e}" # Output: Cannot generate random Universe (M=..., N=...): ...
                 log_message(msg, is_error=True)
                 show_info_message(sample_result_info, msg, is_error=True)
                 error_occurred = True

        if error_occurred: return # Stop if Universe handling failed

        # --- 5. Handle coverage condition c (based on checkbox state) --- # MODIFIED: y -> c
        is_manual_c = chk_specify_c.value # Get checkbox state # MODIFIED: y -> c
        log_message(f"Manual C option: {'Checked' if is_manual_c else 'Unchecked'}") # MODIFIED: Y -> C, is_manual_y -> is_manual_c # Output: Manual C option: Checked/Unchecked
        condition_processed = None # Final c value to use # MODIFIED: y -> c
        max_c_single_j = -1 # Store the result of C(j,s) calculation # MODIFIED: y -> c
        try: # Calculate C(j,s)
            if j_processed >= s_processed >= 0: max_c_single_j = comb(j_processed, s_processed) # MODIFIED: y -> c
            else: max_c_single_j = 0 # MODIFIED: y -> c
        except ValueError: # Theoretically, previous validation should prevent this error, but add check just in case
             msg = f"Error: Invalid parameters when calculating C(j={j_processed}, s={s_processed})." # Output: Error: Invalid parameters when calculating C(j=..., s=...).
             log_message(msg, is_error=True)
             show_info_message(sample_result_info, msg, is_error=True)
             error_occurred = True

        if not error_occurred: # If calculating C(j,s) didn't cause an error
            if is_manual_c: # If manually specifying c is checked # MODIFIED: y -> c
                c_str = txt_specify_c.value.strip() # Get input value # MODIFIED: y -> c twice
                txt_specify_c.error_text = None # Clear old error # MODIFIED: y -> c
                if not c_str: # Input is empty # MODIFIED: y -> c (variable name only)
                    msg = "Error: Specify c manually checked, but the input field is empty." # MODIFIED: y -> c # Output: Error: Specify c manually checked, but the input field is empty.
                    log_message(msg, is_error=True)
                    show_info_message(sample_result_info, msg, is_error=True)
                    txt_specify_c.error_text = "Input cannot be empty" # MODIFIED: y -> c # Output: Input cannot be empty
                    txt_specify_c.update() # MODIFIED: y -> c
                    error_occurred = True
                else: # Input is not empty, try parsing and validating
                    try:
                        specified_c = int(c_str) # Convert to integer # MODIFIED: y -> c twice
                        # Validate c's range # MODIFIED: y -> c
                        if specified_c < 1: raise ValueError("c value must be at least 1.") # MODIFIED: y -> c twice
                        # Check if c exceeds C(j,s) # MODIFIED: y -> c
                        # Note: max_c_single_j might be 0 (e.g., j<s), in which case any positive c is invalid # MODIFIED: y -> c twice
                        if max_c_single_j >= 0 and specified_c > max_c_single_j: # MODIFIED: y -> c twice
                            raise ValueError(f"c ({specified_c}) cannot exceed C({j_processed},{s_processed})={max_c_single_j}.") # MODIFIED: y -> c 3 times
                        elif max_c_single_j == 0 and specified_c > 0: # MODIFIED: y -> c twice
                             raise ValueError(f"C(j,s) is 0, cannot specify a positive c ({specified_c}).") # MODIFIED: y -> c twice

                        condition_processed = specified_c # Validation passed, use user-specified value # MODIFIED: y -> c
                        log_message(f"Using manually specified coverage condition c = {condition_processed}") # MODIFIED: y -> c # Output: Using manually specified coverage condition c = ...
                        txt_specify_c.update() # Update to clear possible old error hints # MODIFIED: y -> c
                    except ValueError as ve: # Catch conversion or range validation errors
                        log_message(f"Manual c value input error: {ve}", is_error=True) # MODIFIED: y -> c # Output: Manual c value input error: ...
                        show_info_message(sample_result_info, f"Manual c value input error: {ve}", is_error=True) # MODIFIED: y -> c # Output: Manual c value input error: ...
                        txt_specify_c.error_text = str(ve) # MODIFIED: y -> c
                        txt_specify_c.update() # MODIFIED: y -> c
                        error_occurred = True
            else: # If manually specifying c is not checked, automatically use C(j,s) # MODIFIED: y -> c
                 condition_processed = max_c_single_j # MODIFIED: y -> c
                 log_message(f"Using automatically calculated coverage condition c = C(j,s) = {condition_processed}") # MODIFIED: y -> c # Output: Using automatically calculated coverage condition c = C(j,s) = ...
                 # Backend algorithm should handle c=0 (meaning no coverage requirement) # MODIFIED: y -> c
                 if condition_processed <= 0:
                      log_message(f"Warning: Calculated c = {condition_processed}. Backend will proceed accordingly.") # MODIFIED: y -> c # Output: Warning: Calculated c = .... Backend will proceed accordingly.

        if error_occurred: return # Stop if c handling failed # MODIFIED: y -> c

        # --- All input parameters processed and validated ---
        log_message(f"Final parameters: M={m_processed}, N={n_processed}, K={k_processed}, J={j_processed}, S={s_processed}, Using c={condition_processed}, Timeout={timeout_seconds}s") # MODIFIED: y -> c # Output: Final parameters: M=..., N=..., K=..., J=..., S=..., Using c=..., Timeout=...s
        show_info_message(sample_result_info, f"Parameter validation passed.\nUniverse: {univ}\nc={condition_processed}\nPreparing to start calculation (Timeout: {timeout_seconds} seconds)...") # MODIFIED: y -> c # Output: Parameter validation passed.\nUniverse: [...]\nc=...\nPreparing to start calculation (Timeout: ... seconds)...

        # --- 6. Get persistent run index (x) ---
        log_message("Fetching next run index from the database...") # Output: Fetching next run index from the database...
        current_run_idx = db.get_and_increment_run_index(m_processed, n_processed, k_processed, j_processed, s_processed)

        if current_run_idx is None: # If getting index failed
            msg = "Error: Could not get or update run index from the database! Calculation cancelled." # Output: Error: Could not get or update run index from the database! Calculation cancelled.
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, msg, is_error=True)
            return # Stop execution

        log_message(f"Run index for this execution: {current_run_idx}") # Output: Run index for this execution: ...

        # --- 7. **Disable UI**, prepare for time-consuming operation ---
        set_busy(True)

        # --- 8. Create Sample instance and prepare to run calculation ---
        sample = None
        try:
            # Create Sample instance, passing all parameters
            sample = Sample(m=m_processed, n=n_processed, k=k_processed, j=j_processed, s=s_processed,
                            c=condition_processed,    # Pass the final processed c value # MODIFIED: y -> c twice
                            run_idx=current_run_idx,  # Pass the run_index obtained from the database
                            timeout=timeout_seconds,
                            rand_instance=random)     # Pass the current random number generator instance
            sample.univ = univ # Set the previously prepared Universe to the instance
            log_message(f"Sample instance (Run {current_run_idx}) created successfully, starting background calculation...") # Output: Sample instance (Run ...) created successfully, starting background calculation...

            # --- 9. Start background calculation thread ---
            # Run the time-consuming sample.run() method in a separate thread
            computation_thread = threading.Thread(
                target=run_computation,   # Target function for the thread to execute
                args=(sample,),           # Pass the Sample instance to the target function
                daemon=True               # Set as daemon thread, so it exits when the main program exits
            )
            computation_thread.start() # Start the thread
            # UI remains disabled; run_computation function will call set_busy(False) to re-enable UI upon completion

        except ValueError as e: # Catch parameter errors potentially thrown during Sample initialization
            msg = f"Parameter error when creating calculation instance (Run {current_run_idx}): {e}" # Output: Parameter error when creating calculation instance (Run ...): ...
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, msg, is_error=True)
            set_busy(False) # Error occurred, re-enable UI
        except Exception as e: # Catch other unexpected errors
            msg = f"Unexpected error during calculation startup (Run {current_run_idx}): {e}" # Output: Unexpected error during calculation startup (Run ...): ...
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, f"Runtime error: {e}", is_error=True) # Output: Runtime error: ...
            set_busy(False) # Error occurred, re-enable UI
            print(f"--- Uncaught error during Sample instance creation or startup (Run {current_run_idx}) ---")
            import traceback
            traceback.print_exc() # Print detailed error stack to console
            print(f"--- Error end ---")

    # --- Function to run calculation in a background thread and update UI ---
    def run_computation(sample_instance: Sample):
        """Runs Sample.run() in a separate thread, processes results, updates UI, and restores UI state upon completion."""
        result_text = f"Calculating (Run {sample_instance.run_idx})..." # Initial result text
        is_final_error = False # Flag if the final result indicates an error state
        run_idx_local = getattr(sample_instance, 'run_idx', 'N/A') # Get run_index for use in logs

        try:
            # --- Execute core calculation ---
            log_message(f"Background thread started executing calculation (Run {run_idx_local})...") # Output: Background thread started executing calculation (Run ...)...
            sample_instance.run() # Call the Sample object's run method, which blocks the current thread until completion or timeout
            log_message(f"Background calculation finished (Run {run_idx_local}).") # Output: Background calculation finished (Run ...).

            # --- Process calculation results ---
            log_message(f"Processing calculation results (Run {run_idx_local}). Result identifier: {sample_instance.ans}") # Output: Processing calculation results (Run ...). Result identifier: ...

            # Check if the result returned by backend is valid
            if sample_instance.result and 'alg' in sample_instance.result:
                # Extract information from the result dictionary
                alg = sample_instance.result.get('alg', 'N/A')
                sets_list = sample_instance.sets if isinstance(sample_instance.sets, list) else []
                num_sets = len(sets_list)
                time_taken = sample_instance.result.get('time', 0)
                status = sample_instance.result.get('status', 'Unknown')
                final_run_idx = sample_instance.result.get('run_index', run_idx_local) # Confirm run_index
                c_used = sample_instance.result.get('coverage_target', sample_instance.c) # Get the actually used c # MODIFIED: y -> c twice

                log_message(f"Solver Algorithm (Run {final_run_idx}): {alg}, Status: {status}, Time: {time_taken:.2f}s, Sets found: {num_sets}") # Output: Solver Algorithm (Run ...): ..., Status: ..., Time: ...s, Sets found: ...

                # --- Build text to display in the main result area ---
                result_text = (
                    f"Result ID: {str(sample_instance.ans)}\n"
                    f"Run Index: {final_run_idx}\n"
                    f"Universe ({len(sample_instance.univ)} items): {sample_instance.univ}\n"
                    f"Algorithm: {alg}\n"
                    f"Status: {status}\n"
                    f"Using c: {c_used}\n" # MODIFIED: y -> c twice
                    f"Time Taken: {time_taken:.2f} seconds\n"
                    f"Found Sets ({num_sets} groups):\n"
                )
                # Determine if status indicates an error/failure state
                is_final_error = status not in ('OPTIMAL', 'FEASIBLE', 'SUCCESS', 'INFEASIBLE')

                # --- Format display of found sets list ---
                MAX_SETS_TO_DISPLAY = 50 # Max number of sets to display in the main result area
                if num_sets > 0 :
                    sets_to_display = sets_list[:MAX_SETS_TO_DISPLAY]
                    sets_lines = []
                    sets_per_line = 3 # How many sets to display per line
                    for i in range(0, len(sets_to_display), sets_per_line):
                         line = " | ".join([str(sorted(s)) for s in sets_to_display[i:i+sets_per_line]])
                         sets_lines.append(f"  {line}") # Add indentation
                    result_text += "\n".join(sets_lines)
                    if num_sets > MAX_SETS_TO_DISPLAY: # If too many sets
                        result_text += f"\n  ... ({num_sets - MAX_SETS_TO_DISPLAY} more not shown)" # Output: ... (... more not shown)
                elif status == 'INFEASIBLE': # If status is infeasible
                     result_text += "  (Problem proven infeasible)" # Output: (Problem proven infeasible)
                else: # Other cases (including success with 0 sets, or error states)
                     result_text += "  (None)" # Output: (None)

                # --- Database Storage (only if K=6) ---
                if sample_instance.k == 6:
                    log_message(f"K=6 (Run {final_run_idx}), attempting to save results to the database...") # Output: K=6 (Run ...), attempting to save results to the database...
                    try:
                        # Convert Universe and Sets list to JSON strings for storage
                        universe_str = json.dumps(sample_instance.univ)
                        found_sets_str = json.dumps(sets_list)

                        # Prepare data dictionary to store in the database
                        result_data = {
                            'm': sample_instance.m, 'n': sample_instance.n, 'k': sample_instance.k,
                            'j': sample_instance.j, 's': sample_instance.s, 'run_index': final_run_idx,
                            'num_results': num_sets, 'c_condition': c_used, 'algorithm': alg, # MODIFIED: y -> c twice
                            'time_taken': time_taken, 'universe': universe_str, 'sets_found': found_sets_str
                            # timestamp will be added automatically by the database
                        }
                        # Call db module's save function
                        save_success = db.save_result(result_data)
                        if save_success:
                            log_message(f"Results (Run {final_run_idx}) successfully saved to the database.") # Output: Results (Run ...) successfully saved to the database.
                        else:
                            # save_result logs errors internally, just log a general message here
                            log_message(f"Problem encountered while saving results (Run {final_run_idx}) to the database (possibly duplicate or error, please check logs).", is_error=True) # Output: Problem encountered while saving results (Run ...) to the database (possibly duplicate or error, please check logs).
                            is_final_error = True # Mark as error state
                    except Exception as db_err: # Catch other exceptions during saving
                        error_msg = f"Error: Failed to save results (Run {final_run_idx}) to the database: {db_err}" # Output: Error: Failed to save results (Run ...) to the database: ...
                        log_message(error_msg, is_error=True)
                        is_final_error = True
                        print(error_msg) # Print to console
                        import traceback; traceback.print_exc() # Print stack trace
                else:
                    log_message(f"K={sample_instance.k} != 6, results (Run {final_run_idx}) were not saved to the database.") # Output: K=... != 6, results (Run ...) were not saved to the database.
            else: # If sample_instance.result is invalid or missing 'alg' key
                 error_msg = f"Calculation finished (Run {run_idx_local}), but could not retrieve valid algorithm result details." # Output: Calculation finished (Run ...), but could not retrieve valid algorithm result details.
                 log_message(error_msg, is_error=True)
                 result_text = error_msg
                 is_final_error = True

        except Exception as compute_err: # Catch top-level errors during run_computation
            error_msg = f"Error during calculation execution or result processing (Run {run_idx_local}): {compute_err}" # Output: Error during calculation execution or result processing (Run ...): ...
            log_message(error_msg, is_error=True)
            result_text = f"Runtime error (Run {run_idx_local}):\n{compute_err}" # Output: Runtime error (Run ...): ...
            is_final_error = True
            print(f"--- Uncaught error in calculation thread (Run {run_idx_local}) ---")
            import traceback; traceback.print_exc() # Print stack trace to console
            print(f"--- Error end ---")

        finally:
            # --- Regardless of success or failure, finally update UI and restore control states ---
             try:
                  # Update main result display area
                  show_info_message(sample_result_info, result_text, is_error=is_final_error)
                  # !! Crucial: Restore availability of UI controls !!
                  set_busy(False)
                  log_message(f"Calculation and result processing flow finished (Run {run_idx_local}). UI restored.") # Output: Calculation and result processing flow finished (Run ...). UI restored.
             except Exception as ui_update_err:
                  # Catch potential errors during UI update (though Flet should handle thread safety well)
                  print(f"Critical Error: Error updating UI from calculation thread: {ui_update_err}") # Output: Critical Error: Error updating UI from calculation thread: ...
                  # Try logging the error
                  try: log_message(f"!!! UI Update Error: {ui_update_err}", is_error=True) # Output: !!! UI Update Error: ...
                  except: pass
                  # Try restoring button states again, just in case
                  try: set_busy(False)
                  except: print("!!! Emergency: Failed to restore UI control state! Application might be unresponsive.") # Output: !!! Emergency: Failed to restore UI control state! Application might be unresponsive.
             finally:
                    page.update() # Ensure the page gets a final update

    # --- Bind the on_submit function to the submit button's click event ---
    submit_button.on_click = on_submit

    # --- ============ ---
    # --- Final Page Layout ---
    # --- ============ ---
    # /// MODIFIED: Add print_preview_view to the layout
    page.add(
        ft.Container( # Use a top-level container to wrap all content
            content=ft.Column(
                [
                    main_computation_view, # Main calculation view (initially visible)
                    db_management_view,    # Database management view (initially hidden)
                    print_preview_view     # /// NEW: Print preview view (initially hidden)
                ],
                expand=True # Allow the inner Column to expand
            ),
            expand=True, # Allow the top-level container to fill the entire page space
            padding = 10 # Add padding to the overall content
        )
    )
    # Allow the entire page to scroll if content overflows
    page.scroll = ft.ScrollMode.ADAPTIVE

    # --- ============ ---
    # --- Application Initialization ---
    # --- ============ ---
    log_message("Application starting...") # Output: Application starting...
    log_message(f"Database file path: {os.path.abspath(db.DB_FILE)}") # Display database file location # Output: Database file path: ...
    log_message(f"Google OR-Tools available: {'Yes' if HAS_ORTOOLS else 'No'}") # Display OR-Tools status # Output: Google OR-Tools available: Yes/No

    # Trigger initial calculation and display of c-related info # MODIFIED: y -> c
    update_c_related_info() # MODIFIED: y -> c

    # Try initializing the database (create table if it doesn't exist)
    try:
        db.setup_database()
        log_message("Database initialization check completed.") # Output: Database initialization check completed.
    except Exception as init_db_err:
        msg = f"Critical Error: Database initialization failed: {init_db_err}" # Output: Critical Error: Database initialization failed: ...
        log_message(msg, is_error=True)
        # sample_result_info might not be fully loaded initially, printing to console is more reliable
        print(msg)
        import traceback; traceback.print_exc()
        # Can display error hint on the main interface
        show_info_message(sample_result_info, msg, is_error=True)

    # Update the page once after initialization
    page.update()
    log_message("Application interface loaded, waiting for user actions.") # Output: Application interface loaded, waiting for user actions.

# --- Application Entry Point ---
if __name__ == "__main__":
     # --- Handle command-line arguments to specify database path (optional) ---
    if len(sys.argv) > 1: # If command-line arguments are provided
        custom_db_path = sys.argv[1] # Get the first argument
        # Simple check if it looks like a path or filename
        if os.sep in custom_db_path or custom_db_path.endswith(".db"):
            db_dir = os.path.dirname(os.path.abspath(custom_db_path)) # Get the directory part
            # If an existing directory is provided
            if os.path.isdir(custom_db_path):
                 db.DB_FILE = os.path.join(custom_db_path, "k6_results.db") # Use default filename in that directory
                 db_dir = custom_db_path
            else: # If a file path is provided
                 db.DB_FILE = custom_db_path # Use the provided path directly
                 db_dir = os.path.dirname(db.DB_FILE) # Get its directory

            print(f"Info: Using database path specified via command line: {os.path.abspath(db.DB_FILE)}") # Output: Info: Using database path specified via command line: ...
            # Try creating the directory where the database resides (if it doesn't exist)
            if db_dir and not os.path.exists(db_dir): # Check if db_dir is not empty
                try:
                    os.makedirs(db_dir, exist_ok=True) # exist_ok=True prevents error if directory already exists
                    print(f"Info: Created database directory: {db_dir}") # Output: Info: Created database directory: ...
                except OSError as e:
                    print(f"Error: Could not create database directory '{db_dir}': {e}. Using default path '{db.DB_FILE}'.") # Output: Error: Could not create database directory '...': .... Using default path '...'.
                    db.DB_FILE = "k6_results.db" # Revert to default
            elif not db_dir: # Handle case where db_dir is empty (e.g., just a filename was given)
                print("Info: Database will be created/used in the current working directory.") # Output: Info: Database will be created/used in the current working directory.

        else:
             print(f"Warning: Invalid database path argument '{custom_db_path}'. Using default path '{db.DB_FILE}'.") # Output: Warning: Invalid database path argument '...'. Using default path '...'.

    # Start the Flet application
    # view=ft.AppView.WEB_BROWSER can make the app open in a browser
    ft.app(target=main)