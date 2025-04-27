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

if __name__ == '__main__':
    # 作为脚本运行时，执行数据库设置（用于初始化或测试）
    print("正在直接运行 db.py 来设置/检查数据库...")
    setup_database()

    print("\n测试获取并增加运行索引 (例: m=10, n=5, k=3, j=2, s=1):")
    m, n, k, j, s = 10, 5, 3, 2, 1
    idx1 = get_and_increment_run_index(m, n, k, j, s)
    print(f"第一次获取索引: {idx1}")
    idx2 = get_and_increment_run_index(m, n, k, j, s)
    print(f"第二次获取索引: {idx2}")
    idx3 = get_and_increment_run_index(m, n, k, j, s)
    print(f"第三次获取索引: {idx3}")
    print("---")
    m, n, k, j, s = 12, 6, 4, 3, 2 # 不同参数
    idx_other = get_and_increment_run_index(m, n, k, j, s)
    print(f"获取另一组参数的索引: {idx_other}")

    print("\n测试查询 run_counters 表:")
    counters = get_all_counters()
    if counters:
        for counter in counters:
            print(counter)
    else:
        print("run_counters 表中尚无记录。")

    print("\n测试查询 results 表:")
    results = get_all_results()
    if results:
        print(f"共找到 {len(results)} 条结果记录。打印最新的 5 条:")
        for res in results[:5]: # 打印最多5条记录
            # 为了简洁，只打印部分字段
            print(f"  ID:{res['id']}, Params:({res['m']},{res['n']},{res['k']},{res['j']},{res['s']}), Run:{res['run_index']}, NumSets:{res['num_results']}, Alg:{res['algorithm']}, Time:{res['time_taken']:.2f}s, Timestamp:{res['timestamp']}")
            # print(res) # 打印完整记录
    else:
        print("results 表中尚无记录。")