# backend.py
# Contains core computation logic: Greedy algorithm and CP-SAT (ILP) solver
# ** Version modified to support the "each j-subset covered at least c times" requirement (c replaces y). **
# ** Removed RunCounter class as run_index management moved to database db.py **

import random
import time
import math
import multiprocessing as mp
import queue # For inter-process communication
from itertools import combinations, chain
from collections import Counter, defaultdict # Import defaultdict

# --- OR-Tools CP-SAT Related Imports ---
try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    print("Warning: Failed to import ortools.sat.python.cp_model.") # TRANSLATED
    print("CP-SAT (ILP) solver will be unavailable. Please ensure Google OR-Tools is installed:") # TRANSLATED
    print("python -m pip install --upgrade --user ortools")
    HAS_ORTOOLS = False
# --- OR-Tools Imports End ---

# --- Helper Functions ---

def comb(n, k):
    """Calculates combinations C(n, k)"""
    if k < 0 or k > n:
        return 0
    try:
        # math.comb might raise ValueError in some edge cases (e.g., very large numbers)
        return math.comb(n, k)
    except ValueError:
        print(f"Warning: comb({n}, {k}) calculation overflow or invalid arguments, returning 0.") # TRANSLATED
        return 0 # Or raise exception, depending on how to handle

def ksubsets(items, k):
    """Generates all subsets of size `k` from `items`"""
    return list(combinations(items, k))

# --- RunCounter class has been removed ---
# --- Greedy Algorithm (modified to support c-coverage) ---

def greedy_cover(args):
    """
    Greedy algorithm to solve the covering problem, requiring each j-subset to be covered at least c times.
    Args: (q, univ, k, j, s, c, run_idx)
         q: Result queue
         univ: Universe of elements (n elements)
         k: Covering block size
         j: Size of subsets to be covered
         s: Intersection size threshold (j-subset intersection k-block >= s)
         c: **Required coverage count** (each j-subset needs to be covered by at least c qualifying k-blocks)
         run_idx: Run index (provided by caller, managed via database)
    """
    q, univ, k, j, s, c, run_idx = args # Unpack args (c replaces y)
    n = len(univ)
    start_time = time.time()
    # !!! Modification: coverage_target uses c
    result = {'alg': 'Greedy', 'status': 'INIT', 'sets': [], 'time': 0, 'j_subsets_total': 0, 'j_subsets_covered': 0, 'coverage_target': c, 'run_index': run_idx}

    print(f"[Greedy-{run_idx}] Starting Greedy algorithm (coverage target c={c})...") # TRANSLATED

    # Basic parameter check
    if c <= 0:
        print(f"[Greedy-{run_idx}] Error: Coverage count c ({c}) must be greater than 0.") # TRANSLATED
        result['status'] = 'ERROR_INVALID_C'
        result['error_message'] = f"Coverage count c ({c}) must be greater than 0." # TRANSLATED (Error Message)
        result['time'] = time.time() - start_time
        q.put(result)
        return

    try:
        # 1. Generate all possible k-subsets (potential covering blocks)
        all_k_subsets_tuples = list(combinations(univ, k))
        all_k_subsets = [set(subset) for subset in all_k_subsets_tuples]
        print(f"[Greedy-{run_idx}] Generated {len(all_k_subsets)} candidate k-subsets.") # TRANSLATED

        # 2. Generate all j-subsets that need to be covered
        target_j_subsets_tuples = list(combinations(univ, j))
        target_j_subsets = [set(subset) for subset in target_j_subsets_tuples]
        num_j_subsets = len(target_j_subsets)
        result['j_subsets_total'] = num_j_subsets
        print(f"[Greedy-{run_idx}] Need to cover {num_j_subsets} j-subsets, each at least {c} times.") # TRANSLATED

        if num_j_subsets == 0:
             print(f"[Greedy-{run_idx}] No j-subsets to cover, completing directly.") # TRANSLATED
             result['status'] = 'SUCCESS' # No target, considered success
             result['time'] = time.time() - start_time
             q.put(result)
             return

        # 3. Precompute which j-subsets each k-subset can cover (based on intersection size >= s).
        k_subset_covers_j_indices = defaultdict(list)
        j_subset_covered_by_k_indices = defaultdict(list)
        for idx_j, j_subset in enumerate(target_j_subsets):
            for idx_k, k_subset in enumerate(all_k_subsets):
                if len(j_subset.intersection(k_subset)) >= s:
                    k_subset_covers_j_indices[idx_k].append(idx_j)
                    j_subset_covered_by_k_indices[idx_j].append(idx_k)
        print(f"[Greedy-{run_idx}] Precomputation of covering relationships completed.") # ALREADY ENGLISH

        # Check if there exists a j-subset that cannot be covered c times at all.
        for idx_j in range(num_j_subsets):
             potential_covers = len(j_subset_covered_by_k_indices[idx_j])
             if potential_covers < c:
                 print(f"[Greedy-{run_idx}] Error: j-subset {idx_j} ({target_j_subsets[idx_j]}) can be covered by at most {potential_covers} k-subsets, which does not meet the requirement of c={c}. Problem is infeasible.") # ALREADY ENGLISH
                 result['status'] = 'INFEASIBLE_C_TARGET'
                 result['error_message'] = f"j-subset {idx_j} cannot be covered {c} times." # ALREADY ENGLISH (Error Message)
                 result['time'] = time.time() - start_time
                 q.put(result)
                 return

        # 4. Greedy selection process (oriented towards c-coverage)
        selected_k_subset_indices = [] # Store indices of selected k-subsets
        # Track current coverage count for each j-subset / j_subset_coverage_count is a list of num_j_subsets zeros [0, 0, 0, ... 0, 0]
        j_subset_coverage_count = [0] * num_j_subsets
        # Track which j-subsets still need more coverage (coverage count < c)

        needs_more_coverage_j_indices = set(range(num_j_subsets))
        print(f"[Greedy-{run_idx}] Starting greedy selection, target c={c}...") # TRANSLATED

        iteration = 0
        # Loop until the coverage count of all j-subsets reaches c.
        while needs_more_coverage_j_indices:
            iteration += 1
            best_k_subset_idx = -1
            max_coverage_increase_count = -1 # How many additional covers would selecting this k-block provide for *unsatisfied* j-subsets?

            # Find the k-subset that covers the most currently *unsatisfied* j-subsets.
            # If multiple k-subsets cover the same maximum number, this simple loop picks the first one found.
            # More sophisticated tie-breaking (e.g., fewest total elements) could be added but isn't here.
            potential_candidates = []
            for idx_k in range(len(all_k_subsets)):
                # Optimization: If this k_subset was already selected, skip?
                # No, because a k_subset might be needed multiple times if c > 1,
                # although the current logic *doesn't re-select* the same block index.
                # This greedy approach selects a *new* block in each step.
                # If c > 1, it relies on *different* blocks contributing to the count.

                # Calculate which *still-to-be-covered* j-subsets the current k-subset can provide coverage for.
                relevant_j_indices = set(k_subset_covers_j_indices.get(idx_k, [])).intersection(needs_more_coverage_j_indices)
                current_coverage_increase_count = len(relevant_j_indices)

                # Greedy strategy: Select the k-subset that provides the most coverage
                # for the currently unsatisfied j-subsets.
                if current_coverage_increase_count > max_coverage_increase_count:
                    max_coverage_increase_count = current_coverage_increase_count
                    best_k_subset_idx = idx_k
                # Simple tie-breaking: if counts are equal, prefer lower index (implicitly done)

            if best_k_subset_idx == -1:
                # This means no available k-subset can cover any more *unsatisfied* j-subsets.
                # Check if this is expected (all covered) or an issue.
                if needs_more_coverage_j_indices:
                    # If there are still j-subsets needing coverage, but no k-subset helps,
                    # it implies either the problem is infeasible from the start (should have been caught earlier)
                    # or the greedy choice got stuck.
                    print(f"[Greedy-{run_idx}] Error: In iteration {iteration}, although {len(needs_more_coverage_j_indices)} j-subsets still do not meet c={c} coverage, no k-block could be found to increase their coverage. Problem might be infeasible or greedy strategy failed.") # TRANSLATED
                    result['status'] = 'FAILED_INCOMPLETE_COVER' # Indicate failure to cover all
                    result['error_message'] = "Greedy search could not find a k-subset to cover remaining j-subsets." # ALREADY ENGLISH
                else:
                     # This case shouldn't be reached if the while loop condition is correct.
                     # If needs_more_coverage_j_indices is empty, the loop should have terminated.
                     print(f"[Greedy-{run_idx}] Logic Warning: In iteration {iteration}, best_k_subset_idx is -1, but needs_more_coverage_j_indices is empty.") # TRANSLATED
                break # Exit the loop

            # Pick the best k-subset found in this iteration.
            # Note: This basic greedy doesn't prevent selecting the same k-subset index multiple times
            # if it remains the best option in subsequent iterations. This is valid for c>1.
            # However, the *current implementation* uses list.append, implicitly building a list of *distinct steps*,
            # not necessarily distinct blocks if the same block is best multiple times.
            # For c>1, a better greedy might track remaining potential contribution per block.
            # Sticking to the current simple approach:
            selected_k_subset_indices.append(best_k_subset_idx)
            chosen_k_subset = all_k_subsets[best_k_subset_idx] # Get the set itself for logging
            # Logging the chosen block's index and content
            print(f"[Greedy-{run_idx}] Iteration {iteration}: Selected block index {best_k_subset_idx} {list(sorted(chosen_k_subset))}, "
                  f"potentially increasing coverage for {max_coverage_increase_count} unsatisfied j-subsets.") # ALREADY ENGLISH

            # Update the coverage count of the affected j-subsets
            newly_satisfied_count = 0
            # Find which j_indices this selected k_subset can cover (precomputed)
            j_indices_affected_this_round = k_subset_covers_j_indices.get(best_k_subset_idx, [])

            for idx_j in j_indices_affected_this_round:
                # Only update counts for j-subsets that *still need* more coverage
                if idx_j in needs_more_coverage_j_indices:
                    j_subset_coverage_count[idx_j] += 1
                    # Check if this j-subset has now reached the target coverage 'c'
                    if j_subset_coverage_count[idx_j] >= c:
                        needs_more_coverage_j_indices.remove(idx_j) # Remove from the set of unsatisfied j-subsets
                        newly_satisfied_count += 1

            print(f"[Greedy-{run_idx}]   -> After this round, {newly_satisfied_count} j-subsets newly reached the c={c} target.") # ALREADY ENGLISH
            print(f"[Greedy-{run_idx}]   -> {len(needs_more_coverage_j_indices)} j-subsets still need more coverage.") # ALREADY ENGLISH

            # Add an iteration limit to prevent potential infinite loops in edge cases
            # A reasonable upper bound might be num_j_subsets * c, but len(all_k_subsets) * c is safer if blocks are limited.
            max_iterations = len(all_k_subsets) * c if len(all_k_subsets) > 0 else num_j_subsets * c
            max_iterations = max(max_iterations, num_j_subsets) # Ensure at least num_j_subsets iterations if c=1
            if iteration > max_iterations and max_iterations > 0: # Check only if max_iterations is positive
                  print(f"[Greedy-{run_idx}] Warning: Exceeded maximum iterations ({iteration} > {max_iterations}), possibly stuck in a loop or slow convergence. Terminating early.") # TRANSLATED
                  result['status'] = 'FAILED_ITERATION_LIMIT'
                  break # Exit loop if iteration limit is reached

        # 5. Results
        # The selected_k_subset_indices contains the *indices* of the blocks chosen in each step.
        # Convert these indices back to the actual sets (sorted lists).
        chosen_sets_list = [list(sorted(list(all_k_subsets[idx]))) for idx in selected_k_subset_indices]
        result['sets'] = chosen_sets_list
        result['j_subsets_covered'] = num_j_subsets - len(needs_more_coverage_j_indices) # The number of j-subsets that met the c requirement

        # Determine final status based on whether all j-subsets were covered
        if not needs_more_coverage_j_indices: # If the set of unsatisfied j-subsets is empty
            result['status'] = 'SUCCESS'
            print(f"[Greedy-{run_idx}] Greedy algorithm successfully completed c={c} coverage for all {num_j_subsets} j-subsets, selecting a total of {len(chosen_sets_list)} sets.") # ALREADY ENGLISH
        else:
            # If loop exited but needs_more_coverage_j_indices is not empty, coverage is incomplete.
            # Check if status was already set to an error/limit state inside the loop.
            if result['status'] == 'INIT': # If status hasn't been set by an earlier error
                result['status'] = 'FAILED_INCOMPLETE_COVER' # Default failure status

            print(f"[Greedy-{run_idx}] Greedy algorithm finished, status: {result['status']}. {result['j_subsets_covered']}/{num_j_subsets} j-subsets met c={c} coverage. Selected {len(chosen_sets_list)} sets.") # TRANSLATED

    except MemoryError:
        print(f"[Greedy-{run_idx}] Memory Error: Insufficient memory during calculation.") # TRANSLATED
        result['status'] = 'ERROR_MEMORY'
        result['error_message'] = 'Memory error during execution.' # ALREADY ENGLISH
    except Exception as e:
        print(f"[Greedy-{run_idx}] Error during Greedy algorithm execution: {e}") # TRANSLATED
        import traceback
        traceback.print_exc()
        result['status'] = 'ERROR'
        result['error_message'] = str(e)

    finally:
        # Calculate time taken and put into queue
        result['time'] = time.time() - start_time
        try:
            q.put(result)
            print(f"[Greedy-{run_idx}] Result placed in queue. Time taken: {result['time']:.2f} seconds.") # TRANSLATED
        except Exception as qe:
             print(f"[Greedy-{run_idx}] Error: Could not put result into queue: {qe}") # TRANSLATED


# --- CP-SAT (ILP) Solver (modified to support c-coverage) ---

def cpsat_cover(args):
    """
    Uses OR-Tools CP-SAT solver to solve the covering problem, requiring each j-subset to be covered at least c times.
    Args: (q, univ, k, j, s, c, run_idx, timeout_solver)
         q: Result queue
         univ: Universe of elements
         k: Covering block size
         j: Size of subsets to be covered
         s: Intersection size threshold (>= s)
         c: **Required coverage count**
         run_idx: Run index (provided by caller, managed via database)
         timeout_solver: Internal timeout for the CP-SAT solver (seconds)
    """
    if not HAS_ORTOOLS:
        q.put({'alg': 'ILP', 'status': 'ERROR_MISSING_ORTOOLS', 'sets': [], 'time': 0, 'coverage_target': args[5], 'run_index': args[6]}) # Include run_idx
        return


    q, univ, k, j, s, c, run_idx, timeout_solver = args # Unpack args
    n = len(univ)
    start_time = time.time()
    # !!! Modification: coverage_target uses c
    result = {'alg': 'ILP', 'status': 'INIT', 'sets': [], 'time': 0, 'coverage_target': c, 'run_index': run_idx}

    print(f"[ILP-{run_idx}] Starting CP-SAT solver execution (coverage target c={c})... Timeout={timeout_solver}s") # TRANSLATED

    # Basic parameter check
    if c <= 0:
        print(f"[ILP-{run_idx}] Error: Coverage count c ({c}) must be greater than 0.") # TRANSLATED
        result['status'] = 'ERROR_INVALID_C'
        result['error_message'] = f"Coverage count c ({c}) must be greater than 0." # TRANSLATED (Error Message)
        result['time'] = time.time() - start_time
        q.put(result)
        return

    try:
        # 1. Generate all possible k-subsets (variables)
        all_k_subsets_tuples = list(combinations(univ, k))
        all_k_subsets = [frozenset(subset) for subset in all_k_subsets_tuples]
        num_k_subsets = len(all_k_subsets)
        k_subset_indices = {subset: i for i, subset in enumerate(all_k_subsets)} # Map subset to index
        print(f"[ILP-{run_idx}] Generated {num_k_subsets} candidate k-subsets (variables).") # TRANSLATED

        # 2. Generate all j-subsets that need to be covered (constraints)
        target_j_subsets_tuples = list(combinations(univ, j))
        target_j_subsets = [frozenset(subset) for subset in target_j_subsets_tuples]
        num_j_subsets = len(target_j_subsets)
        print(f"[ILP-{run_idx}] Need to cover {num_j_subsets} j-subsets, each at least {c} times.") # TRANSLATED

        if num_j_subsets == 0:
            print(f"[ILP-{run_idx}] No j-subsets to cover, completing directly.") # TRANSLATED
            result['status'] = 'OPTIMAL' # Unconstrained problem, objective is 0, solution is empty set
            result['time'] = time.time() - start_time
            q.put(result)
            return

        # 3. Create the CP-SAT model
        model = cp_model.CpModel()

        # 4. Define variables: Whether each k-subset is selected (0 or 1)
        # x[i] = 1 if the i-th k-subset (all_k_subsets[i]) is selected, 0 otherwise.
        x = [model.NewBoolVar(f'x_{i}') for i in range(num_k_subsets)]

        # 5. Define constraints: Each j-subset must be covered by at least c satisfying k-subsets
        print(f"[ILP-{run_idx}] Starting to add constraints (each j-subset >= {c} times coverage)...") # ALREADY ENGLISH
        constraints_added = 0
        feasible = True # Mark whether the model is potentially feasible

        # Precompute which k-subsets can potentially cover each j-subset to speed up constraint building
        j_subset_potential_covers = defaultdict(list) # Map: j_subset_index -> [k_subset_index1, k_subset_index2, ...]
        for idx_k, k_subset in enumerate(all_k_subsets):
             for idx_j, j_subset in enumerate(target_j_subsets):
                 if len(j_subset.intersection(k_subset)) >= s:
                     j_subset_potential_covers[idx_j].append(idx_k)

        # Add constraints for each target j-subset
        for idx_j, j_subset in enumerate(target_j_subsets):
            # Find the indices of all k_subsets whose intersection with the current j_subset is >= s
            # These are the k-subsets that *can* contribute to covering this j-subset.
            covering_k_indices = j_subset_potential_covers.get(idx_j, [])

            # Check feasibility constraint: Can this j-subset *ever* be covered c times?
            if len(covering_k_indices) < c:
                print(f"[ILP-{run_idx}] Error: j-subset {idx_j} ({set(j_subset)}) can be covered by at most {len(covering_k_indices)} k-subsets (intersection>=s), cannot satisfy the requirement c={c}. Problem is infeasible.") # TRANSLATED
                result['status'] = 'INFEASIBLE' # Directly mark model as infeasible
                result['error_message'] = f"j-subset {idx_j} cannot be covered {c} times (max possible: {len(covering_k_indices)})." # ALREADY ENGLISH
                feasible = False
                break # No need to add more constraints, problem is already known to be infeasible

            # Add the core constraint: the sum of selected covering k-subsets must be >= c
            # Only add constraint if there are potential covering blocks and c >= 1
            if covering_k_indices and c >= 1:
                # Constraint: sum(x[i] for i in covering_k_indices) >= c
                model.Add(sum(x[i] for i in covering_k_indices) >= c)
                constraints_added += 1
            elif c < 1:
                 # If c is 0 or negative (already checked at start, but as defense), no constraint needed.
                 pass
            # elif not covering_k_indices and c >= 1: # Handled by the feasibility check above

        if not feasible:
             # If pre-check found infeasibility, record time and exit
             result['time'] = time.time() - start_time
             q.put(result)
             return

        print(f"[ILP-{run_idx}] Added coverage constraints for {constraints_added}/{num_j_subsets} j-subsets requiring coverage (c={c}).") # ALREADY ENGLISH

        # 6. Define the objective function: Minimize the total number of selected k-subsets
        model.Minimize(sum(x))

        # 7. Create the solver and set parameters
        solver = cp_model.CpSolver()
        # Set the time limit provided by the caller
        solver.parameters.max_time_in_seconds = float(timeout_solver)
        # Optional: Log search progress (can be very verbose)
        # solver.parameters.log_search_progress = True
        # Try using multiple workers if CPU allows, often speeds up search
        try:
            num_workers = mp.cpu_count()
            # Avoid using too many workers if CPU count is low or OS limits apply
            solver.parameters.num_search_workers = max(1, num_workers // 2 if num_workers > 1 else 1)
            if solver.parameters.num_search_workers > 1:
                 print(f"[ILP-{run_idx}] Solving with {solver.parameters.num_search_workers} workers...") # ALREADY ENGLISH
            else:
                 print(f"[ILP-{run_idx}] Solving with default number of workers (1)...") # ALREADY ENGLISH
        except NotImplementedError:
             print(f"[ILP-{run_idx}] Could not detect CPU count, solving with default workers.") # ALREADY ENGLISH
             solver.parameters.num_search_workers = 1 # Default fallback

        # 8. Solve the model
        print(f"[ILP-{run_idx}] Starting CP-SAT solver...") # ALREADY ENGLISH
        solve_start_time = time.time()
        status = solver.Solve(model)
        solve_end_time = time.time()
        solver_wall_time = solve_end_time - solve_start_time
        print(f"[ILP-{run_idx}] Solver finished in {solver_wall_time:.2f} seconds.") # ALREADY ENGLISH


        # 9. Process Results
        status_map = {
            cp_model.OPTIMAL: 'OPTIMAL',        # Found the optimal solution.
            cp_model.FEASIBLE: 'FEASIBLE',      # Found a feasible solution, but optimality not proven (often due to timeout).
            cp_model.INFEASIBLE: 'INFEASIBLE',    # Proven that no solution exists.
            cp_model.MODEL_INVALID: 'MODEL_INVALID',# The model formulation itself is invalid.
            cp_model.UNKNOWN: 'UNKNOWN'         # Solver stopped without a conclusive status (e.g., timeout before finding feasible solution).
        }
        result['status'] = status_map.get(status, f'UNMAPPED_STATUS_{status}') # Get mapped status or raw status code

        # Get objective value (number of sets) if a solution was found
        obj_value = float('inf')
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            try:
                 # solver.ObjectiveValue() gives the minimized sum(x)
                 obj_value = solver.ObjectiveValue()
            except OverflowError: # Handle potential overflow for very large objectives
                 print(f"[ILP-{run_idx}] Warning: Objective value calculation resulted in overflow.") # ALREADY ENGLISH
                 obj_value = float('inf') # Treat as infinite if overflow

        print(f"[ILP-{run_idx}] Solving finished. Status: {result['status']}, Objective value (number of sets): {int(obj_value) if obj_value != float('inf') else 'N/A'}") # TRANSLATED

        # Extract the solution (selected k-subsets) if optimal or feasible
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            selected_indices = [i for i, var in enumerate(x) if solver.Value(var) == 1]
            chosen_sets = [all_k_subsets[i] for i in selected_indices] # Get the frozensets
            # Convert to list of lists (sorted elements) for consistent output
            result['sets'] = [list(sorted(list(s))) for s in chosen_sets]
            print(f"[ILP-{run_idx}] Found {len(result['sets'])} sets forming the solution.") # TRANSLATED
            # Sanity check: verify the number of sets matches the objective value
            if len(result['sets']) != int(obj_value):
                print(f"[ILP-{run_idx}] Warning: Number of sets found ({len(result['sets'])}) does not match the reported objective value ({int(obj_value)})!") # TRANSLATED
                # This might indicate an issue with solution extraction or the solver's reporting.
        elif status == cp_model.UNKNOWN:
             print(f"[ILP-{run_idx}] Solver could not find a feasible solution or prove infeasibility within {timeout_solver} seconds. Status UNKNOWN, often due to timeout or problem complexity.") # TRANSLATED
             result['error_message'] = f"Solver timed out ({timeout_solver}s) or stopped with UNKNOWN status." # ALREADY ENGLISH
        elif status == cp_model.INFEASIBLE:
             print(f"[ILP-{run_idx}] Model proven infeasible, no solution exists satisfying c={c} coverage condition.") # TRANSLATED
             # Error message might have been set during constraint check, or can be set here.
             if 'error_message' not in result:
                 result['error_message'] = "The problem is proven to be infeasible by the solver." # ALREADY ENGLISH
        else: # Handle MODEL_INVALID or other unmapped statuses
             print(f"[ILP-{run_idx}] Solver returned an unexpected status: {result['status']}") # TRANSLATED
             result['error_message'] = f"Solver returned unexpected status: {result['status']}" # ALREADY ENGLISH

    except FileNotFoundError as fnf_err: # Specifically catch potential OR-Tools dependency issues
        print(f"[ILP-{run_idx}] CP-SAT File Error: {fnf_err}. Confirm OR-Tools installation is complete and paths are correct.") # TRANSLATED
        result['status'] = 'ERROR_ORTOOLS_FILE'
        result['error_message'] = str(fnf_err)
        # This might happen if OR-Tools installation is broken or has missing native libraries.
    except MemoryError:
        print(f"[ILP-{run_idx}] Memory Error: Insufficient memory during CP-SAT calculation.") # TRANSLATED
        result['status'] = 'ERROR_MEMORY'
        result['error_message'] = 'Memory error during CP-SAT execution.' # ALREADY ENGLISH
    except Exception as e:
        print(f"[ILP-{run_idx}] Error occurred during CP-SAT execution: {e}") # TRANSLATED
        import traceback
        traceback.print_exc()
        result['status'] = 'ERROR'
        result['error_message'] = str(e)

    finally:
        result['time'] = time.time() - start_time # Ensure final time is recorded (including model build and solve)
        try:
            q.put(result)
            print(f"[ILP-{run_idx}] Result placed in queue. Total time taken: {result['time']:.2f} seconds.") # TRANSLATED
        except Exception as qe:
            print(f"[ILP-{run_idx}] Error: Could not put result into queue: {qe}") # TRANSLATED


# --- Sample Class: Organizes computation tasks ---
class Sample:
    """
    Represents a computation instance, manages parameters, runs algorithms, collects results.
    """
    def __init__(self, m, n, k, j, s, c, run_idx, timeout=60, rand_instance=None):
        """
        Initializes a Sample instance.

        Args:
            m (int): Size of the base set {1, ..., m}
            n (int): Size of the selected subset (universe)
            k (int): Size of each covering block (Set)
            j (int): Size of the subset to be covered
            s (int): Minimum number of elements a j-subset must share with a covering block (Set) (>=s)
            c (int): **Coverage requirement** (each j-subset to be covered >= c times)
            run_idx (int): Unique run identifier (obtained from database, persistent across sessions)
            timeout (int): **ILP priority wait time (seconds)** (following 'wait' logic from code B).
                           *Note: The semantics of timeout have been adjusted as requested to prioritize ILP wait time*
            rand_instance (random.Random, optional): Random number generator instance for universe generation
        """
        self.m = m
        self.n = n
        self.k = k
        self.j = j
        self.s = s
        self.c = c # Store the actual coverage requirement used (c)
        self.run_idx = run_idx # Persistent index
        self.timeout = timeout # timeout now represents ILP priority wait time
        self.rand = rand_instance if rand_instance else random

        # --- Result attributes initialization ---
        self.univ = []        # Universe of n elements to be used (might be overwritten by manual input)
        self.q = mp.Queue()   # Queue to receive algorithm results
        self.result = {}      # Dictionary to store the result of the selected algorithm
        self.sets = []        # List to store the covering sets found by the selected algorithm
        self.ans = None       # Final formatted result string

        # Print using c
        print(f"[Sample-{self.run_idx}] Initializing instance: M={m}, N={n}, K={k}, J={j}, S={s}, Target_C={c}, RunIndex={run_idx}, ILP_WaitTimeout={timeout}s") # TRANSLATED (shows run_idx and adjusted timeout meaning)

        # 1. Generate Universe - check logic remains the same
        if n > m:
            raise ValueError(f"Error: N ({n}) cannot be greater than M ({m})") # TRANSLATED (Error Message)
        if not (n > 0 and m >= n):
            print(f"[Sample-{self.run_idx}] M ({m}) or N ({n}) is invalid. Universe must be provided externally or generated in run().") # TRANSLATED
            if n <= 0 or m <= 0:
                 raise ValueError("M and N must be positive.") # TRANSLATED (Error Message)
        # (Actual generation/setting of Universe is usually done by main.py before calling run)


    def run(self):
        """
        Starts Greedy and CP-SAT (if available) algorithms for parallel computation.
        Modified result selection logic: Prioritizes waiting for ILP to return an OPTIMAL/FEASIBLE solution within the specified `self.timeout` seconds.
        If ILP fails to return successfully within this time, it waits for and uses the Greedy SUCCESS solution.
        If neither succeeds, uses fallback logic (prioritizes ILP failure status, then Greedy failure status).
        """
        print(f"[RUN-{self.run_idx}] Starting parallel computation (target c={self.c})... ILP priority wait: {self.timeout}s") # TRANSLATED

        # Check if Universe is valid - logic remains the same
        if not self.univ or len(self.univ) != self.n:
             if self.n > 0 and self.m >= self.n:
                 self.univ = sorted(self.rand.sample(range(1, self.m + 1), self.n))
                 print(f"[RUN-{self.run_idx}] Warning: Valid Universe not provided externally, generating internally: {self.univ}") # TRANSLATED
             else:
                 print(f"[RUN-{self.run_idx}] Error: Universe is invalid or size mismatch with N, and cannot generate default. Univ: {self.univ}, N: {self.n}") # TRANSLATED
                 self.ans = f"Fail: Invalid Universe"
                 self.result = {'status': 'Error', 'alg': 'Preprocessing', 'sets': [], 'time': 0, 'error': 'Invalid Universe', 'run_index': self.run_idx}
                 self.sets = []
                 return

        processes = []
        results_map = {} # To store received results
        selected_result = None
        time_start_run = time.time() # Record run method start time

        # -- Prepare argument tuple --
        common_args = (self.q, self.univ, self.k, self.j, self.s, self.c, self.run_idx)

        # -- Start Greedy process --
        args_greedy = common_args
        p_greedy = mp.Process(target=greedy_cover, args=(args_greedy,), daemon=True)
        processes.append(p_greedy)
        p_greedy.start()
        print(f"[RUN-{self.run_idx}] Started Greedy process (PID: {p_greedy.pid}, target c={self.c})") # TRANSLATED

        # -- Start CP-SAT process (if available) --
        p_ilp = None
        if HAS_ORTOOLS:
            # The internal timeout for the CP-SAT solver can be set slightly longer or equal to the priority wait time,
            # as we primarily care if it returns a *successful* result within the priority time.
            # Here, set to self.timeout, consistent with the priority wait time.
            timeout_solver_internal = max(1.0, float(self.timeout)) # Ensure at least 1 second

            args_ilp = common_args + (timeout_solver_internal,) # Add solver internal timeout
            p_ilp = mp.Process(target=cpsat_cover, args=(args_ilp,), daemon=True)
            processes.append(p_ilp)
            p_ilp.start()
            print(f"[RUN-{self.run_idx}] Started CP-SAT (ILP) process (PID: {p_ilp.pid}, target c={self.c}), internal timeout {timeout_solver_internal:.1f}s") # TRANSLATED
        else:
            print(f"[RUN-{self.run_idx}] CP-SAT (ILP) solver unavailable, running only Greedy.") # TRANSLATED

        # ==============================================================
        # ========== Modified Result Selection Logic (mimics Code B) Start ==========
        # ==============================================================
        ilp_completed_successfully_within_timeout = False
        greedy_completed = False
        greedy_result_data = None
        ilp_result_data = None

        # Prioritize waiting for ILP to complete and return results within self.timeout
        print(f"[RUN-{self.run_idx}] Prioritizing wait for ILP result, max {self.timeout:.1f} seconds...") # TRANSLATED
        wait_start_time = time.time()
        time_waited = 0
        max_wait = self.timeout

        try:
            while time_waited < max_wait:
                remaining_time = max_wait - time_waited
                if remaining_time <= 0: break # Timeout

                try:
                    # Try non-blocking or short-blocking get
                    res = self.q.get(timeout=min(0.1, remaining_time)) # Wait briefly, avoid long blocks
                    res_alg = res.get('alg')
                    res_status = res.get('status')
                    res_run_idx = res.get('run_index') # Check run_index match

                    if res_run_idx != self.run_idx:
                        print(f"[RUN-{self.run_idx}] Warning: Received result from different run ({res_run_idx}), ignoring.") # TRANSLATED
                        continue # Ignore mismatching result

                    if res_alg == 'ILP':
                        ilp_result_data = res
                        print(f"[RUN-{self.run_idx}] Received ILP result at {time.time() - wait_start_time:.2f}s, status: {res_status}") # TRANSLATED
                        # Check if it's a successful ILP result
                        if res_status in ('OPTIMAL', 'FEASIBLE') and res.get('sets') is not None:
                            ilp_completed_successfully_within_timeout = True
                            selected_result = ilp_result_data # **Immediately select ILP result**
                            print(f"[RUN-{self.run_idx}] **Selected ILP result** (Status: {res_status}, Set count: {len(selected_result['sets'])}) as it returned successfully within priority time.") # TRANSLATED
                            break # Found satisfactory ILP result, stop waiting
                        else:
                             # ILP finished but not successfully (INFEASIBLE, UNKNOWN, ERROR etc.)
                             # Continue waiting to see if Greedy finishes, or until timeout
                             pass
                    elif res_alg == 'Greedy':
                        greedy_result_data = res
                        greedy_completed = True
                        print(f"[RUN-{self.run_idx}] Received Greedy result at {time.time() - wait_start_time:.2f}s, status: {res_status}") # TRANSLATED
                        # Don't select Greedy immediately, continue waiting for ILP or timeout
                    else:
                         print(f"[RUN-{self.run_idx}] Warning: Received result from unknown algorithm ({res_alg}), ignoring.") # TRANSLATED

                    # If both processes have finished (and we haven't selected ILP), can exit wait early
                    if ilp_result_data is not None and greedy_result_data is not None and not ilp_completed_successfully_within_timeout:
                        break

                except queue.Empty:
                    # Queue is empty, continue waiting
                    pass
                except Exception as q_err:
                     print(f"[RUN-{self.run_idx}] Error getting result from queue: {q_err}") # TRANSLATED
                     break # Error occurred, stop waiting

                # Update wait time
                time_waited = time.time() - wait_start_time

            # --- Decision after priority wait ends ---
            if selected_result: # If ILP was already selected
                print(f"[RUN-{self.run_idx}] ILP returned successfully within priority time and was selected.") # TRANSLATED
            else:
                # ILP did not return successfully within priority time
                print(f"[RUN-{self.run_idx}] ILP did not return successfully within the {self.timeout:.1f}s priority wait time. Now checking/waiting for Greedy result...") # TRANSLATED

                # Check if Greedy has already finished
                if greedy_completed and greedy_result_data and greedy_result_data.get('status') == 'SUCCESS' and greedy_result_data.get('sets') is not None:
                    selected_result = greedy_result_data # Select the completed successful Greedy result
                    print(f"[RUN-{self.run_idx}] **Selected Greedy result** (Status: SUCCESS, Set count: {len(selected_result['sets'])}) because ILP did not succeed first.") # TRANSLATED
                else:
                    # Greedy hasn't finished, or finished unsuccessfully
                    # Need to continue waiting for Greedy (possibly beyond self.timeout)
                    if not greedy_completed and p_greedy.is_alive():
                        print(f"[RUN-{self.run_idx}] Greedy process is still running, continuing to wait for its completion...") # TRANSLATED
                        remaining_overall_timeout = 3600 # Set a long overall timeout to avoid infinite wait
                        try:
                             # Loop get until Greedy result is obtained or timeout
                             while greedy_result_data is None:
                                 res = self.q.get(timeout=remaining_overall_timeout) # Wait longer
                                 if res.get('alg') == 'Greedy' and res.get('run_index') == self.run_idx:
                                     greedy_result_data = res
                                     greedy_completed = True
                                     print(f"[RUN-{self.run_idx}] Greedy result finally received, status: {res.get('status')}") # TRANSLATED
                                     break
                                 elif res.get('alg') == 'ILP' and ilp_result_data is None and res.get('run_index') == self.run_idx:
                                     # If ILP result arrives late, record it too
                                     ilp_result_data = res
                                     print(f"[RUN-{self.run_idx}] Late ILP result received, status: {res.get('status')}") # TRANSLATED
                                 # Ignore other irrelevant results
                        except queue.Empty:
                             print(f"[RUN-{self.run_idx}] Error: Still did not receive Greedy result within the extra wait time.") # TRANSLATED
                             # greedy_result_data remains None
                        except Exception as q_err_long:
                             print(f"[RUN-{self.run_idx}] Error getting result from queue while waiting extra time for Greedy: {q_err_long}") # TRANSLATED

                    # Re-check if Greedy result is available and successful
                    if greedy_result_data and greedy_result_data.get('status') == 'SUCCESS' and greedy_result_data.get('sets') is not None:
                         selected_result = greedy_result_data
                         print(f"[RUN-{self.run_idx}] **Selected Greedy result** (Status: SUCCESS, Set count: {len(selected_result['sets'])}) received after extra waiting.") # TRANSLATED
                    else:
                         # Greedy unsuccessful or result not received, enter fallback logic
                         print(f"[RUN-{self.run_idx}] Greedy did not succeed or result could not be obtained. Entering fallback selection logic...") # TRANSLATED
                         # Fallback: Prefer any ILP result (even failed), then any Greedy result
                         if ilp_result_data: # Prefer ILP (even INFEASIBLE, UNKNOWN, ERROR)
                             selected_result = ilp_result_data
                             print(f"[RUN-{self.run_idx}] **Selected ILP result as fallback** (Status: {selected_result.get('status', 'N/A')})") # TRANSLATED
                         elif greedy_result_data: # Then Greedy (even FAILED, ERROR)
                             selected_result = greedy_result_data
                             print(f"[RUN-{self.run_idx}] **Selected Greedy result as fallback** (Status: {selected_result.get('status', 'N/A')})") # TRANSLATED
                         else:
                              # Not even one result received
                              print(f"[RUN-{self.run_idx}] Error: Could not obtain any valid result from the queue.") # TRANSLATED
                              total_wait_time = time.time() - wait_start_time
                              self.ans = f"Fail: No valid result obtained."
                              self.result = {'status': 'NoResult', 'alg': 'None', 'sets': [], 'time': total_wait_time, 'run_index': self.run_idx}
                              self.sets = []
                              # (subsequent finally block handles process termination)

        except (queue.Empty) as toe:
             # This Empty exception should primarily trigger during initial wait (though inner loops might too)
             total_wait_time = time.time() - wait_start_time
             print(f"[RUN-{self.run_idx}] Error: Queue empty or timed out while waiting for results ({type(toe).__name__}). Total run time: {time.time() - time_start_run:.2f}s.") # TRANSLATED
             self.ans = f"Fail: Timeout or Queue Empty ({self.timeout:.1f}s priority wait)"
             # Check if partial results were recorded
             if ilp_result_data:
                 selected_result = ilp_result_data
                 if 'status' not in selected_result or selected_result.get('status') in ('INIT', 'UNKNOWN'): selected_result['status'] = 'TIMEOUT_PARTIAL'
                 print(f"[RUN-{self.run_idx}] **Selected partial ILP result (after timeout)** (Status: {selected_result.get('status')})") # TRANSLATED
             elif greedy_result_data:
                 selected_result = greedy_result_data
                 if 'status' not in selected_result or selected_result.get('status') in ('INIT'): selected_result['status'] = 'TIMEOUT_PARTIAL'
                 print(f"[RUN-{self.run_idx}] **Selected partial Greedy result (after timeout)** (Status: {selected_result.get('status')})") # TRANSLATED
             else: # If really no result at all
                 self.result = {'status': 'Timeout', 'alg': 'None', 'sets': [], 'time': total_wait_time, 'run_index': self.run_idx}
             if selected_result: self.result = selected_result # Use partial result
             self.sets = self.result.get('sets', []) if self.result else []


        except Exception as e:
            total_time = time.time() - time_start_run
            print(f"[RUN-{self.run_idx}] Unexpected error occurred while processing results: {e}") # TRANSLATED
            import traceback
            traceback.print_exc()
            self.ans = f"Error: Exception in result processing"
            # Try to recover info from partial results
            part_res = ilp_result_data or greedy_result_data
            err_alg = part_res.get('alg', 'Error') if part_res else 'Error'
            self.result = {'status': 'RuntimeError', 'alg': err_alg, 'sets': [], 'time': total_time, 'error': str(e), 'run_index': self.run_idx}
            self.sets = []
        # ============================================================
        # ========== Modified Result Selection Logic (mimics Code B) End ==========
        # ============================================================

        finally:
            # Ensure all processes are terminated - logic remains the same
            print(f"[RUN-{self.run_idx}] Attempting to terminate child processes...") # TRANSLATED
            for p in processes:
                pid_str = f"PID {p.pid}" if p.pid else "process" # TRANSLATED "进程" -> "process"
                try:
                    if p.is_alive():
                        print(f"  - Terminating {pid_str}...") # TRANSLATED
                        p.terminate() # Send SIGTERM
                        p.join(timeout=1.0) # Wait 1 sec
                        if p.is_alive(): # If still running
                            print(f"  - {pid_str} failed to terminate normally, forcing termination...") # TRANSLATED
                            p.kill() # Send SIGKILL
                            p.join(timeout=0.5) # Wait briefly
                        print(f"  - {pid_str} terminated.") # TRANSLATED
                    else:
                        print(f"  - {pid_str} already finished.") # TRANSLATED
                except Exception as term_err:
                    print(f"  - Error terminating {pid_str}: {term_err}") # TRANSLATED
            # Clean up queue - logic remains the same
            print(f"[RUN-{self.run_idx}] Cleaning up result queue...") # TRANSLATED
            while not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: break
                except Exception as q_clean_err:
                    print(f"  - Error cleaning queue: {q_clean_err}") # TRANSLATED
                    break # Avoid infinite loop
            print(f"[RUN-{self.run_idx}] Run method result processing and cleanup completed.") # TRANSLATED

        # --- Set final result attributes --- (Logic remains the same, but based on selected_result)
        if selected_result: # Check if selected_result was successfully assigned
            self.result = selected_result
            # Ensure run_index is in the final result
            if 'run_index' not in self.result or self.result['run_index'] != self.run_idx:
                 print(f"[RUN-{self.run_idx}] Warning: Final selected result is missing the correct run_index, forcing it to {self.run_idx}. Result: {self.result.get('run_index')}") # TRANSLATED
                 self.result['run_index'] = self.run_idx

            self.sets = self.result.get('sets', []) # Get list of sets, default to empty
            # Ensure self.sets is a list
            if not isinstance(self.sets, list):
                 print(f"[RUN-{self.run_idx}] Warning: 'sets' in the final result is not a list (Type: {type(self.sets)}), resetting to empty list.") # TRANSLATED
                 self.sets = []

            num_results = len(self.sets)

            # Build standard answer string: m-n-k-j-s-run_idx-num_sets
            self.ans = f"{self.m}-{self.n}-{self.k}-{self.j}-{self.s}-{self.run_idx}-{num_results}"

            final_alg = self.result.get('alg', 'N/A')
            final_status = self.result.get('status', 'N/A')
            final_time = self.result.get('time', 0) # Time reported internally by the algorithm
            cov_target = self.result.get('coverage_target', self.c) # Confirm c value
            final_run_idx = self.result.get('run_index', self.run_idx) # Confirm again

            print(f"[RUN-{final_run_idx}] ---- Final Result Summary ----") # ALREADY ENGLISH
            print(f"[RUN-{final_run_idx}] Selected Algorithm: {final_alg}") # ALREADY ENGLISH
            print(f"[RUN-{final_run_idx}] Status: {final_status}") # ALREADY ENGLISH
            print(f"[RUN-{final_run_idx}] Coverage Target (c): {cov_target}") # ALREADY ENGLISH
            print(f"[RUN-{final_run_idx}] Algorithm Time: {final_time:.2f}s") # ALREADY ENGLISH (internal algorithm time)
            print(f"[RUN-{final_run_idx}] Total Run Method Time: {time.time() - time_start_run:.2f}s") # ALREADY ENGLISH (total time for run method)
            print(f"[RUN-{final_run_idx}] Sets Found: {num_results}") # ALREADY ENGLISH
            print(f"[RUN-{final_run_idx}] Result ID (ans): {self.ans}") # ALREADY ENGLISH
            print(f"[RUN-{final_run_idx}] -----------------------------") # ALREADY ENGLISH

        elif self.ans is None: # If selected_result is None and ans was not set in an exception
             # self.result might have been set during exception handling, or remains empty
             fail_status = self.result.get('status', 'UnknownFailure') if hasattr(self, 'result') and self.result else 'SetupFailure'
             num_results = 0 # Number of sets is 0 on failure
             self.ans = f"Fail({fail_status}):{self.m}-{self.n}-{self.k}-{self.j}-{self.s}-{self.run_idx}-{num_results}"
             # Ensure self.result exists and contains basic info
             if not hasattr(self, 'result') or not self.result:
                 self.result = {'status': fail_status, 'alg': 'None', 'sets': [], 'time': time.time() - time_start_run, 'run_index': self.run_idx}
             elif 'run_index' not in self.result:
                  self.result['run_index'] = self.run_idx # Add run_index
             if not hasattr(self, 'sets'): # Ensure self.sets exists
                self.sets = []
             print(f"[RUN-{self.run_idx}] Ultimately failed to select a valid result. Ans set to: {self.ans}") # TRANSLATED

        # Return before end of run method, ensuring main.py can continue execution
        return


# --- Main Program Entry Point (for testing backend.py directly) ---
if __name__ == '__main__':

    print("backend.py executed directly. Performing c-coverage test...") # TRANSLATED
    print("Note: Running backend.py directly now depends on db.py to get the run_index.") # TRANSLATED
    print("Will use default database file 'k6_results.db' to get/update the index.") # TRANSLATED

    # Import db module (if not already imported)
    try:
        import db as test_db
        test_db.setup_database() # Ensure database and table exist (column name c_condition)
    except ImportError:
        print("Error: Cannot import db.py. Please ensure db.py is in the same directory or Python path.") # TRANSLATED
        exit()
    except Exception as db_setup_err:
        print(f"Error: Error setting up database: {db_setup_err}") # TRANSLATED
        exit()

    # Set test parameters
    test_m = 10
    test_n = 7
    test_k = 4
    test_j = 3
    test_s = 2
    test_c = 2 # Use c
    test_ilp_wait_timeout = 15 # ** Modification: timeout now represents ILP priority wait time **


    print(f"\nTest Parameters: M={test_m}, N={test_n}, K={test_k}, J={test_j}, S={test_s}, C={test_c}, ILP_WaitTimeout={test_ilp_wait_timeout}s") # TRANSLATED

    try:
        # Get persistent run_index
        test_run_idx = test_db.get_and_increment_run_index(test_m, test_n, test_k, test_j, test_s)
        if test_run_idx is None:
             print("Error: Could not retrieve run index from the database.") # TRANSLATED
             exit()
        print(f"Run index obtained from database for this run: {test_run_idx}") # TRANSLATED

        test_random_instance = random.Random(0) # Fixed seed
        # Instantiate Sample using c and adjusted timeout
        sample_instance = Sample(test_m, test_n, test_k, test_j, test_s, test_c,
                                 test_run_idx,
                                 test_ilp_wait_timeout, # Pass ILP priority wait time
                                 test_random_instance)

        # Manually set a Universe
        sample_instance.univ = sorted(test_random_instance.sample(range(1, test_m + 1), test_n))
        print(f"Setting Universe: {sample_instance.univ}") # TRANSLATED

        sample_instance.run()

        print("\n--- Test Run Results ---") # TRANSLATED
        print(f"Final Result Identifier (ans): {sample_instance.ans}") # TRANSLATED
        print(f"Selected Algorithm Result (result): {sample_instance.result}") # TRANSLATED
        print(f"Sets Found (sets):") # TRANSLATED
        if sample_instance.sets:
            MAX_SETS_PRINT = 20
            for i, found_set in enumerate(sample_instance.sets[:MAX_SETS_PRINT]):
                print(f"  Set {i+1}: {found_set}")
            if len(sample_instance.sets) > MAX_SETS_PRINT:
                print(f"  ... (and {len(sample_instance.sets) - MAX_SETS_PRINT} more sets not printed)") # TRANSLATED
        else:
            print("  No sets were found.") # TRANSLATED

    except ValueError as ve:
         print(f"Parameter Error: {ve}") # TRANSLATED
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}") # TRANSLATED
        import traceback
        traceback.print_exc()

    print("\nBackend test finished.") # TRANSLATED