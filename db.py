# db.py
# 数据库操作模块
# 修改 delete_result 函数，使其在删除具有最高 run_index 的记录时，
# 相应地递减 run_counters 表中的 last_run_index。

import sqlite3
import json
import os
from datetime import datetime

# --- 常量 ---
DB_FILE = "k6_results.db"  # 数据库文件名

# --- 数据库设置 ---
def setup_database(db_file=DB_FILE):
    """
    初始化数据库连接并创建所需表（如果不存在）。
    包括 `results` 表和 `run_counters` 表。

    Args:
        db_file (str): 数据库文件的路径。
    """
    print(f"正在检查/创建数据库: {os.path.abspath(db_file)}") # 打印绝对路径方便调试
    conn = None  # 初始化连接变量
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # --- 创建 results 表 ---
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
                y_condition INTEGER,
                algorithm TEXT,
                time_taken REAL,
                universe TEXT,
                sets_found TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(m, n, k, j, s, run_index)
            )
        ''')
        print("数据库表 'results' 已准备就绪。")

        # --- 创建 run_counters 表 ---
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
        print("数据库表 'run_counters' 已准备就绪。")

        conn.commit() # 提交所有更改
        print(f"数据库设置完成。")

    except sqlite3.Error as e:
        print(f"数据库设置错误: {e}")
    finally:
        if conn:
            conn.close() # 确保关闭连接

# --- 获取并增加运行索引 ---
def get_and_increment_run_index(m, n, k, j, s, db_file=DB_FILE):
    """
    获取给定参数组合的下一个运行索引 (从 1 开始)，并更新数据库计数。
    这个函数是线程/进程安全的（对于单个 Python 进程内的调用是安全的）。

    Args:
        m, n, k, j, s: 参数组合。
        db_file (str): 数据库文件路径。

    Returns:
        int: 下一个可用的运行索引 (从 1 开始)。
        None: 如果发生数据库错误。
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file, timeout=10) # 增加超时以防数据库繁忙
        cursor = conn.cursor()

        # 使用事务确保原子性
        conn.execute("BEGIN") # 开始事务

        # 1. 查询当前的 last_run_index
        cursor.execute("SELECT last_run_index FROM run_counters WHERE m=? AND n=? AND k=? AND j=? AND s=?", (m, n, k, j, s))
        result = cursor.fetchone()

        next_index = 1 # 默认为 1 (第一次运行)
        if result:
            # 如果找到了记录，下一个索引是当前索引 + 1
            current_index = result[0]
            next_index = current_index + 1
            # 2. 更新记录
            # print(f"DEBUG: Updating counter for {m,n,k,j,s} from {current_index} to {next_index}") # Debug
            cursor.execute("UPDATE run_counters SET last_run_index = ? WHERE m=? AND n=? AND k=? AND j=? AND s=?", (next_index, m, n, k, j, s))
        else:
            # 2. 如果没找到记录，插入新记录，last_run_index 就是 1
            # print(f"DEBUG: Inserting new counter for {m,n,k,j,s} with index {next_index}") # Debug
            cursor.execute("INSERT INTO run_counters (m, n, k, j, s, last_run_index) VALUES (?, ?, ?, ?, ?, ?)", (m, n, k, j, s, next_index))

        conn.commit() # 提交事务
        print(f"数据库：为参数 ({m},{n},{k},{j},{s}) 分配并记录 run_index: {next_index}")
        return next_index

    except sqlite3.Error as e:
        print(f"数据库错误 (获取/增加运行索引 for {m},{n},{k},{j},{s}): {e}")
        if conn:
            try:
                conn.rollback() # 如果出错，回滚事务
                print("数据库：获取/增加索引事务已回滚。")
            except sqlite3.Error as rb_err:
                print(f"数据库错误：回滚失败: {rb_err}")
        return None # 表示获取索引失败
    finally:
        if conn:
            conn.close()

# --- 数据保存 ---
def save_result(result_data, db_file=DB_FILE):
    """
    将单次运行的结果保存到数据库中。

    Args:
        result_data (dict): 包含运行结果的字典，键应与表列名对应。
                             必须包含 'm', 'n', 'k', 'j', 's', 'run_index' 等。
        db_file (str): 数据库文件的路径。

    Returns:
        bool: True 如果保存成功, False 如果发生错误。
    """
    conn = None
    required_keys = {'m', 'n', 'k', 'j', 's', 'run_index', 'num_results'}
    if not required_keys.issubset(result_data.keys()):
         print(f"错误: 保存结果时缺少必要键。需要: {required_keys}, 实际提供: {result_data.keys()}")
         return False

    try:
        conn = sqlite3.connect(db_file, timeout=10)
        cursor = conn.cursor()

        # 获取字典的键和值，确保顺序一致
        columns = list(result_data.keys())
        values = [result_data[col] for col in columns]

        # 创建 SQL 语句
        cols_str = ', '.join(f'"{col}"' for col in columns) # 给列名加引号以防关键字冲突
        placeholders = ', '.join('?' * len(values))
        sql = f'INSERT INTO results ({cols_str}) VALUES ({placeholders})'

        cursor.execute(sql, values) # 使用参数化查询
        conn.commit()
        print(f"成功将结果 (参数: {result_data.get('m')}-{result_data.get('n')}-{result_data.get('k')}-{result_data.get('j')}-{result_data.get('s')}, run={result_data.get('run_index')}) 保存到数据库 {db_file}")
        return True
    except sqlite3.IntegrityError:
        print(f"警告: 尝试插入重复的结果记录到 `results` 表 (m={result_data.get('m')}, ..., run={result_data.get('run_index')})。")
        return False
    except sqlite3.Error as e:
        print(f"数据库保存错误 (保存到 results 表): {e}")
        import traceback
        traceback.print_exc() # 打印详细错误
        return False
    finally:
        if conn:
            conn.close()

# --- 数据查询函数 (保持不变) ---
def get_all_results(db_file=DB_FILE):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"数据库查询错误 (查询 results 表): {e}")
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
        print(f"数据库查询错误 (查询 run_counters 表): {e}")
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
        cursor.execute("""
            SELECT id, m, n, k, j, s, run_index, num_results, timestamp
            FROM results
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        # print(f"数据库查询：找到 {len(results)} 条结果摘要。") #减少日志噪音
    except sqlite3.Error as e:
        print(f"数据库查询错误 (查询 results 摘要): {e}")
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
        cursor.execute("SELECT * FROM results WHERE id = ?", (result_id,))
        row = cursor.fetchone()
        if row:
            details = dict(row)
            # 尝试解析 JSON 字段
            try:
                details['sets_found_parsed'] = json.loads(details['sets_found']) if details.get('sets_found') else []
                details['universe_parsed'] = json.loads(details['universe']) if details.get('universe') else []
                # print(f"数据库查询：获取 ID={result_id} 的详细信息成功。") #减少日志噪音
            except json.JSONDecodeError as json_err:
                print(f"数据库警告：解析 ID={result_id} 的 JSON 字段失败: {json_err}")
                details['sets_found_parsed'] = f"JSON 解析错误: {details.get('sets_found')}"
                details['universe_parsed'] = f"JSON 解析错误: {details.get('universe')}"
            except Exception as parse_err:
                 print(f"数据库错误：解析 ID={result_id} 的字段时出错: {parse_err}")
                 details['sets_found_parsed'] = f"解析时出错: {details.get('sets_found')}"
                 details['universe_parsed'] = f"解析时出错: {details.get('universe')}"
        else:
            print(f"数据库查询：未找到 ID={result_id} 的记录。")

    except sqlite3.Error as e:
        print(f"数据库查询错误 (获取详情 ID={result_id}): {e}")
    finally:
        if conn: conn.close()
    return details

# --- 数据删除 (核心修改) ---
def delete_result(result_id, db_file=DB_FILE):
    """
    根据结果的 ID 从数据库删除记录。
    **修改**: 如果删除的记录是其参数组合的最新运行 (run_index 等于
    run_counters 中的 last_run_index)，则相应地递减 run_counters。

    Args:
        result_id (int): 要删除的结果的数据库 ID。
        db_file (str): 数据库文件路径。

    Returns:
        bool: True 如果删除（和必要的计数器更新）成功, False 如果发生错误或未找到记录。
    """
    conn = None
    success = False
    try:
        conn = sqlite3.connect(db_file, timeout=10)
        cursor = conn.cursor()

        # --- 开始事务 ---
        conn.execute("BEGIN")

        # 1. 获取要删除记录的详细信息 (m, n, k, j, s, run_index)
        cursor.execute("SELECT m, n, k, j, s, run_index FROM results WHERE id = ?", (result_id,))
        result_to_delete = cursor.fetchone()

        if not result_to_delete:
            print(f"数据库操作：未找到要删除的记录 ID={result_id}。")
            conn.rollback() # 回滚事务
            return False # 返回 False 因为未找到

        m, n, k, j, s, deleted_run_index = result_to_delete
        print(f"数据库操作：准备删除 ID={result_id} (Params: {m}-{n}-{k}-{j}-{s}, RunIndex: {deleted_run_index})")

        # 2. 获取当前参数组合的 last_run_index
        cursor.execute("SELECT last_run_index FROM run_counters WHERE m=? AND n=? AND k=? AND j=? AND s=?", (m, n, k, j, s))
        counter_result = cursor.fetchone()

        # 如果 counter 不存在（理论上不应发生，但要做防御性编程）
        if not counter_result:
             print(f"数据库警告：未找到参数 {m}-{n}-{k}-{j}-{s} 的计数器记录，但存在对应的结果 ID={result_id}。数据可能不一致。将仅删除结果。")
             last_run_index_in_counter = -1 # 设为一个不可能相等的值
        else:
             last_run_index_in_counter = counter_result[0]
             print(f"数据库操作：参数 {m}-{n}-{k}-{j}-{s} 的当前 last_run_index 为 {last_run_index_in_counter}。")

        # 3. 执行删除操作
        cursor.execute("DELETE FROM results WHERE id = ?", (result_id,))
        rows_affected = cursor.rowcount

        if rows_affected > 0:
            print(f"数据库操作：成功从 'results' 表删除 ID={result_id} ({rows_affected} 行)。")

            # 4. **条件性地更新计数器**
            # 检查被删除的 run_index 是否是当前记录的最高索引
            if deleted_run_index == last_run_index_in_counter:
                # 是最新的，需要递减计数器
                new_counter_value = last_run_index_in_counter - 1
                 # 确保计数器不会小于0 （虽然理论上不会，因为 run_index 从1开始）
                new_counter_value = max(0, new_counter_value)
                print(f"数据库操作：删除的是最新记录，将 run_counters 更新为 {new_counter_value}...")
                cursor.execute("UPDATE run_counters SET last_run_index = ? WHERE m=? AND n=? AND k=? AND j=? AND s=?",
                               (new_counter_value, m, n, k, j, s))
                if cursor.rowcount > 0:
                     print(f"数据库操作：成功更新 run_counters。")
                else:
                     # 更新失败？这很奇怪，可能意味着计数器记录同时被删除了？
                     print(f"数据库警告：尝试更新 run_counters 失败（未找到匹配行），尽管之前找到了。")
                     # 决定是否回滚；这里选择不回滚，因为结果已经删除了
            else:
                # 删除的不是最新的，不需要更新计数器
                print(f"数据库操作：删除的 run_index ({deleted_run_index}) 不是最新的 ({last_run_index_in_counter})，无需更新 run_counters。")

            # --- 提交事务 ---
            conn.commit()
            print(f"数据库操作：删除及相关操作已提交。")
            success = True
        else:
            # 删除失败（可能在获取信息和实际删除之间被其他操作删除了）
            print(f"数据库操作：执行 DELETE 时未找到 ID={result_id}（可能已被并发删除）。")
            conn.rollback() # 回滚事务
            success = False

    except sqlite3.Error as e:
        print(f"数据库删除错误 (删除 ID={result_id}): {e}")
        if conn:
            try:
                conn.rollback()
                print("数据库：删除操作事务已回滚。")
            except sqlite3.Error as rb_err:
                print(f"数据库错误：回滚删除失败: {rb_err}")
        success = False
    finally:
        if conn:
            conn.close()
    return success

# --- (确保 if __name__ == '__main__' 部分不会干扰导入) ---
if __name__ == '__main__':
    print("正在直接运行 db.py 来设置/检查数据库...")
    setup_database()

    print("\n--- 数据库操作测试 (含删除逻辑) ---")

    # 模拟场景
    test_params = {'m': 45, 'n': 9, 'k': 6, 'j': 5, 's': 5}
    test_db_file = DB_FILE

    def run_and_save_mock(params, run_idx, num_results=10):
        mock_data = params.copy()
        mock_data.update({
            'run_index': run_idx,
            'num_results': num_results,
            'y_condition': params['s'], # 模拟
            'algorithm': 'MOCK',
            'time_taken': 0.1,
            'universe': json.dumps(list(range(1, params['n']+1))),
            'sets_found': json.dumps([list(range(i+1, i+1+params['k'])) for i in range(num_results)])
        })
        save_result(mock_data, db_file=test_db_file)
        # 获取刚插入的 ID (假设是最后一个)
        summary = get_results_summary(db_file=test_db_file)
        return summary[0]['id'] if summary else None

    print("\n--- 场景模拟开始 ---")
    print(f"参数: {test_params}")

    # 1. 运行第一次，获取 run_index 1
    idx1 = get_and_increment_run_index(**test_params, db_file=test_db_file)
    id1 = run_and_save_mock(test_params, idx1) if idx1 else None
    print(f"第一次运行: run_index={idx1}, DB ID={id1}")

    # 2. 运行第二次，获取 run_index 2
    idx2 = get_and_increment_run_index(**test_params, db_file=test_db_file)
    id2 = run_and_save_mock(test_params, idx2) if idx2 else None
    print(f"第二次运行: run_index={idx2}, DB ID={id2}")

    # 3. 运行第三次，获取 run_index 3
    idx3 = get_and_increment_run_index(**test_params, db_file=test_db_file)
    id3 = run_and_save_mock(test_params, idx3) if idx3 else None
    print(f"第三次运行: run_index={idx3}, DB ID={id3}")

    print("\n当前计数器状态:")
    counters = get_all_counters(db_file=test_db_file)
    for c in counters:
        if c['m'] == test_params['m'] and c['s'] == test_params['s']: # 简单过滤
            print(f"  Params: {c['m']}-{c['n']}-{c['k']}-{c['j']}-{c['s']}, LastIndex: {c['last_run_index']}")

    # 4. 删除 ID 为 id3 (run_index=3) 的记录
    if id3:
        print(f"\n--- 删除 ID={id3} (最新的 run_index={idx3}) ---")
        deleted = delete_result(id3, db_file=test_db_file)
        print(f"删除操作结果: {deleted}")

        print("\n删除后计数器状态:")
        counters = get_all_counters(db_file=test_db_file)
        for c in counters:
             if c['m'] == test_params['m'] and c['s'] == test_params['s']:
                print(f"  Params: {c['m']}-{c['n']}-{c['k']}-{c['j']}-{c['s']}, LastIndex: {c['last_run_index']} (预期: {idx3-1})")
    else:
        print("\n跳过删除测试，因为之前的插入失败。")

    # 5. 再次运行计算，期望 run_index 再次为 3
    print("\n--- 再次运行，期望 run_index 复用 ---")
    next_idx = get_and_increment_run_index(**test_params, db_file=test_db_file)
    next_id = run_and_save_mock(test_params, next_idx) if next_idx else None
    print(f"再次运行: run_index={next_idx} (预期: {idx3}), DB ID={next_id}")

    print("\n最终计数器状态:")
    counters = get_all_counters(db_file=test_db_file)
    for c in counters:
         if c['m'] == test_params['m'] and c['s'] == test_params['s']:
            print(f"  Params: {c['m']}-{c['n']}-{c['k']}-{c['j']}-{c['s']}, LastIndex: {c['last_run_index']} (预期: {idx3})")

    # 6. (可选) 测试删除旧记录 (id2, run_index=2)
    if id2:
        print(f"\n--- (可选) 删除旧记录 ID={id2} (run_index={idx2}) ---")
        deleted_old = delete_result(id2, db_file=test_db_file)
        print(f"删除旧记录结果: {deleted_old}")
        print("\n删除旧记录后计数器状态:")
        counters = get_all_counters(db_file=test_db_file)
        for c in counters:
             if c['m'] == test_params['m'] and c['s'] == test_params['s']:
                 # 此时计数器应该仍然是 3 (来自步骤 5)
                print(f"  Params: {c['m']}-{c['n']}-{c['k']}-{c['j']}-{c['s']}, LastIndex: {c['last_run_index']} (预期: {idx3}, 不变)")

    print("\n--- 场景模拟结束 ---")

    print("\n数据库模块测试结束。")