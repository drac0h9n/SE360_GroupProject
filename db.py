# db.py
import sqlite3
import json
import os
from datetime import datetime

# --- 常量 ---
DB_FILE = "k6_results.db"  # 数据库文件名

# --- 数据库设置 ---
def setup_database(db_file=DB_FILE):
    """
    初始化数据库连接并创建结果表（如果不存在）。

    Args:
        db_file (str): 数据库文件的路径。
    """
    print(f"正在检查/创建数据库: {os.path.abspath(db_file)}") # 打印绝对路径方便调试
    conn = None  # 初始化连接变量
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # 创建表的 SQL 语句
        # 添加了详细的列和 UNIQUE 约束
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT, -- 记录的唯一ID
                m INTEGER NOT NULL,                 -- 参数 M
                n INTEGER NOT NULL,                 -- 参数 N
                k INTEGER NOT NULL,                 -- 参数 K (目标是记录 k=6)
                j INTEGER NOT NULL,                 -- 参数 J
                s INTEGER NOT NULL,                 -- 参数 S
                run_index INTEGER NOT NULL,         -- 对应参数组合的第 x 次运行
                num_results INTEGER NOT NULL,       -- 找到的集合数量 (y)
                y_condition INTEGER,                -- 输入的覆盖条件 y 值
                algorithm TEXT,                     -- 使用的求解算法 ('ILP' 或 'Greedy')
                time_taken REAL,                    -- 计算耗时 (秒)
                universe TEXT,                      -- 使用的 Universe 集合 (JSON 字符串)
                sets_found TEXT,                    -- 找到的 k 元素集合列表 (JSON 字符串)
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, -- 记录插入时间
                UNIQUE(m, n, k, j, s, run_index)    -- 确保参数组合和运行索引的唯一性
            )
        ''')
        conn.commit() # 提交更改
        print(f"数据库表 'results' 已准备就绪。")
    except sqlite3.Error as e:
        print(f"数据库设置错误: {e}")
    finally:
        if conn:
            conn.close() # 确保关闭连接

# --- 数据保存 ---
def save_result(result_data, db_file=DB_FILE):
    """
    将单次运行的结果保存到数据库中。

    Args:
        result_data (dict): 包含运行结果的字典，键应与表列名对应。
        db_file (str): 数据库文件的路径。

    Returns:
        bool: True 如果保存成功, False 如果发生错误。
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # 使用参数化查询防止 SQL 注入
        columns = ', '.join(result_data.keys())
        placeholders = ', '.join('?' * len(result_data))
        sql = f'INSERT INTO results ({columns}) VALUES ({placeholders})'

        cursor.execute(sql, list(result_data.values())) # 值需要是列表或元组
        conn.commit()
        print(f"成功将结果 (m={result_data.get('m')}, n={result_data.get('n')}, k={result_data.get('k')}, j={result_data.get('j')}, s={result_data.get('s')}, run={result_data.get('run_index')}) 保存到数据库 {db_file}")
        return True
    except sqlite3.IntegrityError:
        # 捕获唯一性约束冲突错误
        print(f"警告: 数据库中已存在相同参数组合和运行索引的结果 (m={result_data.get('m')}, n={result_data.get('n')}, k={result_data.get('k')}, j={result_data.get('j')}, s={result_data.get('s')}, run={result_data.get('run_index')})。")
        return False
    except sqlite3.Error as e:
        print(f"数据库保存错误: {e}")
        # 可以考虑在这里记录更详细的错误日志
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
        list: 包含所有结果记录的列表 (每个记录是一个元组)。
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
        print(f"数据库查询错误: {e}")
        return []
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # 作为脚本运行时，执行数据库设置（用于初始化或测试）
    print("正在直接运行 db.py 来设置数据库...")
    setup_database()
    print("\n测试查询:")
    results = get_all_results()
    if results:
        for res in results[:5]: # 打印最多5条记录
            print(res)
    else:
        print("数据库中尚无记录。")