# db.py
# 数据库操作模块
# 新增 run_counters 表和 get_and_increment_run_index 函数来管理持久化的运行索引

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
    包括 `results` 表和新增的 `run_counters` 表。

    Args:
        db_file (str): 数据库文件的路径。
    """
    print(f"正在检查/创建数据库: {os.path.abspath(db_file)}") # 打印绝对路径方便调试
    conn = None  # 初始化连接变量
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # --- 创建 results 表 ---
        # (保持原样)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                m INTEGER NOT NULL,
                n INTEGER NOT NULL,
                k INTEGER NOT NULL,
                j INTEGER NOT NULL,
                s INTEGER NOT NULL,
                run_index INTEGER NOT NULL,         -- 现在是从 run_counters 获取的持久化索引
                num_results INTEGER NOT NULL,
                y_condition INTEGER,                -- 输入的覆盖条件 y 值
                algorithm TEXT,
                time_taken REAL,
                universe TEXT,
                sets_found TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(m, n, k, j, s, run_index)    -- 唯一性约束仍然重要
            )
        ''')
        print("数据库表 'results' 已准备就绪。")

        # --- 创建 run_counters 表 (新增) ---
        # 用于存储每个参数组合的最后一个 run_index
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS run_counters (
                m INTEGER NOT NULL,
                n INTEGER NOT NULL,
                k INTEGER NOT NULL,
                j INTEGER NOT NULL,
                s INTEGER NOT NULL,
                last_run_index INTEGER NOT NULL DEFAULT 0, -- 记录最后使用的索引
                PRIMARY KEY (m, n, k, j, s)        -- 参数组合是主键，确保唯一性
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

# --- 获取并增加运行索引 (核心修改) ---
def get_and_increment_run_index(m, n, k, j, s, db_file=DB_FILE):
    """
    获取给定参数组合的下一个运行索引 (从 1 开始)，并更新数据库计数。
    这个函数是线程/进程安全的（对于单个 Python 进程内的调用是安全的，
    因为 SQLite 连接和事务是针对每个连接的）。

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
            cursor.execute("UPDATE run_counters SET last_run_index = ? WHERE m=? AND n=? AND k=? AND j=? AND s=?", (next_index, m, n, k, j, s))
        else:
            # 2. 如果没找到记录，插入新记录，last_run_index 就是 1
            cursor.execute("INSERT INTO run_counters (m, n, k, j, s, last_run_index) VALUES (?, ?, ?, ?, ?, ?)", (m, n, k, j, s, next_index))

        conn.commit() # 提交事务
        print(f"数据库：为参数 ({m},{n},{k},{j},{s}) 分配并记录 run_index: {next_index}")
        return next_index

    except sqlite3.Error as e:
        print(f"数据库错误 (获取/增加运行索引 for {m},{n},{k},{j},{s}): {e}")
        if conn:
            try:
                conn.rollback() # 如果出错，回滚事务
                print("数据库：事务已回滚。")
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
        # print(f"DEBUG SQL: {sql}") # 调试用
        # print(f"DEBUG Values: {values}") # 调试用

        cursor.execute(sql, values) # 使用参数化查询
        conn.commit()
        print(f"成功将结果 (m={result_data.get('m')}, n={result_data.get('n')}, k={result_data.get('k')}, j={result_data.get('j')}, s={result_data.get('s')}, run={result_data.get('run_index')}) 保存到数据库 {db_file}")
        return True
    except sqlite3.IntegrityError:
        # 捕获唯一性约束冲突错误
        # 由于 run_index 现在持久化递增，理论上不应该再频繁触发此错误，除非手动修改了 run_counters 表
        print(f"警告: 尝试插入重复的结果记录到 `results` 表 (m={result_data.get('m')}, n={result_data.get('n')}, k={result_data.get('k')}, j={result_data.get('j')}, s={result_data.get('s')}, run={result_data.get('run_index')})。这可能表示 run_index 管理出现问题或数据被外部修改。")
        return False
    except sqlite3.Error as e:
        print(f"数据库保存错误 (保存到 results 表): {e}")
        import traceback
        traceback.print_exc() # 打印详细错误
        return False
    finally:
        if conn:
            conn.close()

# --- (可选) 数据查询函数 ---
def get_all_results(db_file=DB_FILE):
    """
    (可选) 从数据库获取所有结果。

    Args:
        db_file (str): 数据库文件的路径。

    Returns:
        list: 包含所有结果记录的列表 (每个记录是一个字典)。
              如果出错则返回空列表。
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row # 让结果可以按列名访问
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows] # 转换为字典列表
    except sqlite3.Error as e:
        print(f"数据库查询错误 (查询 results 表): {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_all_counters(db_file=DB_FILE):
    """
    (可选) 从数据库获取所有运行计数器的当前状态。

    Args:
        db_file (str): 数据库文件的路径。

    Returns:
        list: 包含所有计数器记录的列表 (每个记录是一个字典)。
              如果出错则返回空列表。
    """
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
        if conn:
            conn.close()

def get_results_summary(db_file=DB_FILE):
    """
    从数据库获取所有结果的摘要信息，用于列表显示。

    Args:
        db_file (str): 数据库文件的路径。

    Returns:
        list: 包含结果摘要字典的列表。每个字典包含 'id', 'm', 'n', 'k', 'j', 's', 'run_index', 'num_results', 'timestamp'。
              如果出错则返回空列表。
    """
    conn = None
    results = []
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row # 按列名访问
        cursor = conn.cursor()
        # 选择需要的列，并按时间戳降序排序
        cursor.execute("""
            SELECT id, m, n, k, j, s, run_index, num_results, timestamp
            FROM results
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        print(f"数据库查询：找到 {len(results)} 条结果摘要。")
    except sqlite3.Error as e:
        print(f"数据库查询错误 (查询 results 摘要): {e}")
        results = [] # 出错时返回空列表
    finally:
        if conn:
            conn.close()
    return results

def get_result_details(result_id, db_file=DB_FILE):
    """
    根据结果的 ID 从数据库获取详细信息。

    Args:
        result_id (int): 要获取详情的结果的数据库 ID。
        db_file (str): 数据库文件路径。

    Returns:
        dict: 包含所选结果所有字段的字典。'sets_found' 和 'universe' 需要被 JSON 解析。
              如果找不到记录或发生错误，返回 None。
    """
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
                if details.get('sets_found'):
                    details['sets_found_parsed'] = json.loads(details['sets_found'])
                else:
                     details['sets_found_parsed'] = []
                if details.get('universe'):
                    details['universe_parsed'] = json.loads(details['universe'])
                else:
                    details['universe_parsed'] = []
                print(f"数据库查询：获取 ID={result_id} 的详细信息成功。")
            except json.JSONDecodeError as json_err:
                print(f"数据库警告：解析 ID={result_id} 的 JSON 字段失败: {json_err}")
                # 保留原始字符串，并标记解析失败
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
        if conn:
            conn.close()
    return details

def delete_result(result_id, db_file=DB_FILE):
    """
    根据结果的 ID 从数据库删除记录。

    Args:
        result_id (int): 要删除的结果的数据库 ID。
        db_file (str): 数据库文件路径。

    Returns:
        bool: True 如果删除成功, False 如果发生错误或未找到记录。
    """
    conn = None
    success = False
    try:
        conn = sqlite3.connect(db_file, timeout=10)
        cursor = conn.cursor()
        # 执行删除操作
        cursor.execute("DELETE FROM results WHERE id = ?", (result_id,))
        # 检查是否有行被删除
        if cursor.rowcount > 0:
            conn.commit()
            print(f"数据库操作：成功删除 ID={result_id} 的记录。")
            success = True
        else:
            print(f"数据库操作：未找到要删除的记录 ID={result_id}。")
            # 未找到也算“成功”完成操作，但没有实际删除
            success = True # 或者可以返回 False 表示未删除任何东西
    except sqlite3.Error as e:
        print(f"数据库删除错误 (删除 ID={result_id}): {e}")
        if conn:
            try:
                conn.rollback()
                print("数据库：删除操作已回滚。")
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

    print("\n--- 数据库操作测试 ---")
    # 添加一些模拟数据用于测试 (如果需要)
    # ...

    print("\n测试 get_results_summary:")
    summary = get_results_summary()
    if summary:
        print(f"找到 {len(summary)} 条摘要:")
        for item in summary[:3]: # 打印前3条
            print(f"  ID: {item['id']}, Params: {item['m']}-{item['n']}-{item['k']}-{item['j']}-{item['s']}-{item['run_index']}, Sets: {item['num_results']}, Time: {item['timestamp']}")
    else:
        print("摘要列表为空。")

    # 假设数据库中存在 ID=1 的记录
    test_id = 1
    if summary: test_id = summary[0]['id'] # 获取最新记录的ID测试
    print(f"\n测试 get_result_details (ID={test_id}):")
    details = get_result_details(test_id)
    if details:
        print(f"  获取到详情: M={details.get('m')}, N={details.get('n')}, K={details.get('k')}, Run={details.get('run_index')}")
        print(f"  Universe (parsed): {details.get('universe_parsed')}")
        print(f"  Sets Found ({len(details.get('sets_found_parsed',[]))}) (parsed, 前5个): {details.get('sets_found_parsed', [])[:5]}")
        # print(f"  原始 Sets Found: {details.get('sets_found')[:100]}...") # 打印原始 JSON (截断)
    else:
        print(f"  未能获取 ID={test_id} 的详情。")

    # print(f"\n测试 delete_result (ID={test_id}):")
    # # 警告：取消注释下一行将删除数据！
    # # deleted = delete_result(test_id)
    # # print(f"删除操作结果: {deleted}")
    # print("(删除测试已注释掉，防止意外删除数据)")

    print("\n数据库模块测试结束。")