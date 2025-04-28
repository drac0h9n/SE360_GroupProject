# main.py
# 使用 Flet 构建图形用户界面 (GUI)。
# 处理用户输入，调用 backend 计算，并显示结果。
# 当 K=6 时，会将结果保存到 SQLite 数据库。
# 包含手动输入 Universe 和手动指定覆盖条件 y 的选项。
# 移除了全局 RunCounter，使用 db.py 管理持久化 run_index。
# **新增了数据库结果管理功能，允许用户查看、显示详情和删除数据库中的记录。**
# **修改：移除了删除操作的二次确认弹窗。**

import flet as ft
import random
import time
import json  # 用于序列化/反序列化列表以便存入/读取数据库
import sys   # 用于检查命令行参数
import os    # 用于路径操作
import threading  # 用于在后台执行计算，避免UI阻塞
import functools  # 用于 functools.partial (虽然在这个版本中可能没直接用，但保留以备后用)
from datetime import datetime # 用于格式化时间戳

# 导入后端逻辑和数据库操作
from backend import Sample, comb, HAS_ORTOOLS  # 导入 Sample 类, comb 函数, HAS_ORTOOLS
import db  # 导入数据库操作模块 (db.py)

# --- 全局变量 ---
random_seed = int(time.time())  # 使用当前时间作为随机种子
random.seed(random_seed)
print(f"使用的随机种子: {random_seed}")

# --- 主应用函数 ---
def main(page: ft.Page):
    """构建并运行 Flet 应用程序的主函数"""
    page.title = "组合覆盖问题求解器 (含数据库管理)"  # 窗口标题
    page.vertical_alignment = ft.MainAxisAlignment.START  # 页面垂直对齐方式
    page.window_width = 900   # 窗口初始宽度
    page.window_height = 850  # 窗口初始高度

    # --- 状态变量 (用于跨函数共享数据) ---
    # 使用 ft.Ref 来简化跨函数更新控件或变量值的过程
    selected_db_result_id = ft.Ref[int]()  # 存储当前在数据库列表中选中的记录的ID
    db_results_list_data = ft.Ref[list]() # 存储从数据库加载的结果摘要列表 [{id:.., m:.., ...}, ...]
    db_results_list_data.current = []     # 初始化为空列表

    # --- ========================== ---
    # --- UI 控件定义 (计算部分) ---
    # --- ========================== ---

    # --- 输入控件 ---
    txt_m = ft.TextField(label="M (基础集)", hint_text="例如: 45", width=100, value="45")
    txt_n = ft.TextField(label="N (Universe)", hint_text="例如: 8", width=100, value="8")
    txt_k = ft.TextField(label="K (块大小)", hint_text="例如: 6", width=100, value="6")
    txt_j = ft.TextField(label="J (子集)", hint_text="例如: 4", width=100, value="4")
    txt_s = ft.TextField(label="S (交集>=)", hint_text="例如: 4", width=100, value="4")
    txt_timeout = ft.TextField(label="超时(s)", hint_text="例如: 60", width=100, value="60")

    # --- Universe 输入选项 ---
    chk_manual_univ = ft.Checkbox(label="手动输 Universe", value=False, on_change=None) # 复选框，初始未选中
    txt_manual_univ = ft.TextField(
        label="输入N个数字(空格分隔, 范围 1~M)",  # 手动输入 Universe 的文本框
        visible=False,  # 初始隐藏
        width=600,      # 文本框宽度
        hint_text="例如: 1 5 10 15 20 25 30 35"
    )

    # --- 覆盖条件 y 输入选项 ---
    chk_specify_y = ft.Checkbox(label="手动指定 y (覆盖次数)", value=False, on_change=None) # 复选框
    txt_specify_y = ft.TextField(
        label="输入 y 值 (范围 1 ~ C(j,s))", # 手动输入 y 的文本框
        visible=False,  # 初始隐藏
        width=250,      # 文本框宽度
        hint_text="输入正整数y"
    )

    # --- 输出/信息显示控件 ---
    theoretical_y_info = ft.Text("理论覆盖度 (C(k,s)C(n-k,j-s)): ...", size=12) # 显示理论y值
    max_single_j_coverage = ft.Text("单j集最大覆盖 (C(j,s)): ...", size=12) # 显示 C(j,s)
    # 计算结果显示区域，允许选择文本，设置最大行数，允许内容溢出以便滚动
    sample_result_info = ft.Text(
        "计算结果将显示在这里...",
        size=12,
        selectable=True,       # 允许用户选择文本
        max_lines=25,          # 限制最大显示行数（超出需滚动）
        overflow=ft.TextOverflow.VISIBLE # 内容溢出时可见（需外部容器支持滚动）
    )
    # 日志输出区域，使用 ListView 实现自动滚动
    log_output = ft.ListView(
        expand=True,           # 允许列表扩展填充空间
        spacing=5,             # 行间距
        auto_scroll=True,      # 自动滚动到底部
        height=200             # 给日志区域一个固定的高度
    )

    # --- 按钮 ---
    submit_button = ft.ElevatedButton(text="开始计算", on_click=None, icon=ft.icons.PLAY_ARROW) # 触发计算
    clear_log_button = ft.ElevatedButton(text="清空日志", on_click=lambda _: clear_log(), icon=ft.icons.CLEAR_ALL) # 清空日志区域

    # --- 进度指示器 ---
    progress_ring = ft.ProgressRing(visible=False, width=20, height=20, stroke_width = 3) # 计算时显示

    # --- ============================= ---
    # --- UI 控件定义 (数据库管理部分) ---
    # --- ============================= ---

    # --- 数据库结果列表和详情控件 ---
    db_results_list_view = ft.ListView(expand=True, spacing=5) # 显示数据库记录摘要的列表
    # 显示选中记录详情的文本区域
    db_result_details_view = ft.Text(
        "请先在上方选择一条记录，然后点击“显示详情”。",
        selectable=True,
        max_lines=20, # 限制行数
        overflow=ft.TextOverflow.VISIBLE
    )
    # 使用 RadioGroup 来管理列表的选择，确保一次只能选一个
    db_results_radio_group = ft.RadioGroup(
        content=db_results_list_view, # 将 ListView 作为 RadioGroup 的内容
        on_change=None # 选择变化的事件处理函数稍后绑定
    )

    # --- 数据库管理按钮 ---
    show_db_view_button = ft.ElevatedButton("查看/管理数据库结果", icon=ft.icons.STORAGE, on_click=None) # 切换到数据库视图
    refresh_db_button = ft.ElevatedButton("刷新列表", icon=ft.icons.REFRESH, on_click=None) # 重新加载数据库列表
    # 显示选中项详情的按钮，初始禁用，直到用户选择了某一项
    display_details_button = ft.ElevatedButton(
        "显示详情",
        icon=ft.icons.VISIBILITY,
        on_click=None,
        disabled=True # 初始禁用
    )
    # 删除选中项的按钮，初始禁用，红色以示警告
    delete_selected_button = ft.ElevatedButton(
        "删除所选",
        icon=ft.icons.DELETE_FOREVER,
        on_click=None, # <-- **修改点**: on_click 将直接绑定到 execute_delete
        color=ft.colors.RED, # 按钮文字颜色为红色
        disabled=True # 初始禁用
    )
    # 从数据库视图返回主计算界面的按钮
    back_to_main_button = ft.ElevatedButton("返回计算界面", icon=ft.icons.ARROW_BACK, on_click=None)

    # --- 数据库结果详情容器 ---
    db_details_container = ft.Container(
        content=db_result_details_view, # 包裹详情文本控件
        border=ft.border.all(1, ft.colors.BLACK26), # 添加边框
        border_radius=ft.border_radius.all(5),      # 圆角边框
        padding=10,                                # 内边距
        margin=ft.margin.only(top=10),             # 上外边距
        expand=True,                               # 允许容器扩展填充垂直空间
        # height=200 # 可以选择设置固定高度
    )

    # --- 数据库结果管理视图的整体容器 (初始隐藏) ---
    db_management_view = ft.Column(
        [
            ft.Text("数据库结果列表", size=16, weight=ft.FontWeight.BOLD), # 标题
            # 放置数据库操作按钮的行，允许换行
            ft.Row(
                [refresh_db_button, display_details_button, delete_selected_button, back_to_main_button],
                spacing=10, # 按钮间距
                wrap=True   # 允许按钮在空间不足时换行
            ),
            ft.Text("选择一条记录进行操作:", size=12), # 提示文字
            # 放置数据库结果列表的容器，设置边框和固定高度
            ft.Container(
                content=db_results_radio_group, # 包含 RadioGroup (内含 ListView)
                border=ft.border.all(1, ft.colors.BLACK12),
                border_radius=ft.border_radius.all(5),
                padding=5,
                height=300 # 给列表区域一个固定的高度
            ),
            ft.Container( # 使用 Container 来包裹 Text，并应用 margin
                content=ft.Text("选中记录详情", size=14, weight=ft.FontWeight.BOLD), # Text 不再有 margin 属性
                margin=ft.margin.only(top=10) # 将 margin 应用于包含 Text 的 Container
            ),
            db_details_container # 详情显示容器
        ],
        visible=False, # 数据库管理视图初始时隐藏
        expand=True    # 允许此视图在垂直方向上扩展
    )

    # --- ========================== ---
    # --- 辅助函数和事件处理函数 ---
    # --- ========================== ---

    # --- 日志函数 ---
    def log_message(message: str, is_error: bool = False):
        """向日志区域 (log_output) 添加一条带时间戳的消息"""
        timestamp = time.strftime("%H:%M:%S", time.localtime()) # 获取当前时间
        color = ft.colors.RED if is_error else ft.colors.BLACK87 # 错误消息用红色
        log_output.controls.append(ft.Text(f"[{timestamp}] {message}", size=11, selectable=True, color=color))
        # Flet 是响应式的，但有时显式调用 page.update() 能确保 UI 立即刷新
        page.update()

    def clear_log():
        """清空日志区域"""
        log_output.controls.clear()
        log_message("日志已清空。")
        page.update()

    # --- 显示信息/错误 的辅助函数 ---
    def show_info_message(text_control: ft.Text, message: str, is_error: bool = False):
        """更新指定的 Text 控件 (如 sample_result_info 或 db_result_details_view) 来显示信息或错误"""
        text_control.value = message # 设置文本内容
        text_control.color = ft.colors.RED if is_error else ft.colors.BLACK # 设置文本颜色
        page.update()

    # --- 控制 UI 繁忙状态的函数 ---
    def set_busy(busy: bool):
        """控制按钮和进度环的启用/禁用状态，以及输入控件的禁用状态"""
        submit_button.disabled = busy        # 禁用/启用提交按钮
        progress_ring.visible = busy         # 显示/隐藏进度环
        show_db_view_button.disabled = busy  # 计算时禁用切换数据库视图按钮

        # 禁用/启用计算相关的输入框和复选框
        for ctrl in [txt_m, txt_n, txt_k, txt_j, txt_s, txt_timeout, chk_manual_univ, txt_manual_univ, chk_specify_y, txt_specify_y]:
            ctrl.disabled = busy

        # 注意：数据库管理视图的按钮启用/禁用状态由其自身逻辑管理 (如选择后才启用详情/删除)
        # 但可以在这里统一处理，如果需要的话 (例如，计算时完全禁用数据库操作)
        # refresh_db_button.disabled = busy
        # display_details_button.disabled = busy or (selected_db_result_id.current is None) # 结合原有逻辑
        # delete_selected_button.disabled = busy or (selected_db_result_id.current is None) # 结合原有逻辑
        # back_to_main_button.disabled = busy # 计算时不允许返回

        page.update() # 应用状态更改

    # --- 输入验证和处理 ---
    def validate_and_get_int(field: ft.TextField, name: str, min_val: int = 0) -> int | None:
        """验证输入框 (field) 的值是否为整数，且不小于 min_val。
           如果有效，返回整数值；否则返回 None，记录日志，并设置输入框的错误提示。"""
        field.error_text = None # 清除之前的错误状态
        field.update()
        try:
            value = int(field.value.strip()) # 获取值并去除首尾空格，尝试转为整数
            if value < min_val: # 检查是否小于最小值
                msg = f"错误: {name} ({value}) 不能小于 {min_val}。"
                log_message(msg, is_error=True)
                show_info_message(sample_result_info, f"输入错误: {name} 不能小于 {min_val}。", is_error=True) # 在主结果区显示错误
                field.error_text = f"不能小于 {min_val}" # 设置输入框旁边的小错误提示
                field.update()
                return None # 返回 None 表示验证失败
            return value # 验证通过，返回整数值
        except ValueError: # 捕捉转换整数失败的异常
            msg = f"错误: {name} ('{field.value}') 必须是一个有效的整数。"
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, f"输入错误: {name} 必须是一个整数。", is_error=True)
            field.error_text = "必须是整数"
            field.update()
            return None

    # --- 动态更新 y 相关信息 ---
    def update_y_related_info(e=None):
        """当 N, K, J, S 输入框的值改变时，重新计算并更新理论 y 值 (理论覆盖度) 和 C(j,s) (单j集最大覆盖) 的显示。"""
        # 安全地获取输入值，如果无效则为 None
        n_str, k_str, j_str, s_str = txt_n.value, txt_k.value, txt_j.value, txt_s.value
        n, k, j, s = None, None, None, None
        try: n = int(n_str.strip()) if n_str.strip() else None
        except ValueError: pass
        try: k = int(k_str.strip()) if k_str.strip() else None
        except ValueError: pass
        try: j = int(j_str.strip()) if j_str.strip() else None
        except ValueError: pass
        try: s = int(s_str.strip()) if s_str is not None and s_str.strip() != "" else None # s=0 是有效的
        except ValueError: pass

        # --- 更新 C(j,s) 显示 ---
        max_single_cov_val_str = "等待有效 J, S..." # 默认提示
        max_single_j_coverage.color = ft.colors.GREY # 默认灰色
        max_y_limit = 0 # 存储 C(j,s) 的计算结果，用于手动 y 输入框的范围提示

        if j is not None and s is not None and j >= 0 and s >= 0: # 确保 j 和 s 是有效的非负整数
            try:
                if j < s:
                    max_single_cov_val = 0
                    max_single_cov_val_str = f"C({j},{s}): 0 (因 j < s)"
                    max_single_j_coverage.color = ft.colors.ORANGE # 橙色提示
                else:
                    max_single_cov_val = comb(j, s) # 调用 backend 的 comb 函数计算组合数
                    max_single_cov_val_str = f"C({j},{s}): {max_single_cov_val}"
                    max_single_j_coverage.color = ft.colors.BLACK # 正常黑色显示
                    max_y_limit = max_single_cov_val # 更新 y 的上限
                # 更新手动 y 输入框的标签，提示有效范围
                txt_specify_y.label = f"输入 y 值 (1 ~ {max_y_limit})" if max_y_limit > 0 else "输入 y 值 (无效或无需)"
                txt_specify_y.update()
            except ValueError as combo_err: # 捕捉 comb 函数可能抛出的参数错误
                 max_single_cov_val_str = f"C({j},{s}): 计算错误 ({combo_err})"
                 max_single_j_coverage.color = ft.colors.RED # 红色错误提示
                 txt_specify_y.label = "输入 y 值 (需有效J,S)"
                 txt_specify_y.update()
            except Exception as calc_err: # 捕捉其他可能的计算错误
                 max_single_cov_val_str = f"C({j},{s}): 未知错误 ({calc_err})"
                 max_single_j_coverage.color = ft.colors.RED
                 txt_specify_y.label = "输入 y 值 (计算错误)"
                 txt_specify_y.update()
        else: # 如果 j 或 s 无效
             txt_specify_y.label = "输入 y 值 (需有效J,S)"
             txt_specify_y.update()
        # 更新显示 C(j,s) 的文本控件
        max_single_j_coverage.value = f"单j集最大覆盖 (C(j,s)): {max_single_cov_val_str}"

        # --- 更新理论 y = C(k,s)*C(n-k,j-s) 显示 ---
        theoretical_y_val_str = "等待有效 N, K, J, S..." # 默认提示
        theoretical_y_info.color = ft.colors.GREY       # 默认灰色

        # 确保 n, k, j, s 都是有效的非负整数
        if n is not None and k is not None and j is not None and s is not None and all(v >= 0 for v in [n,k,j,s]):
            try:
                 # 检查组合数的参数是否有效
                 term1_valid = (k >= s)
                 term2_valid = (n - k >= j - s) and (j >= s) # 注意 (n-k) 和 (j-s) 都必须非负
                 params_logic_valid = (n >= k and n >= j) # 基础逻辑检查

                 if not params_logic_valid:
                      theoretical_y_val_str = "参数无效 (N需>=K且>=J)"
                      theoretical_y_info.color = ft.colors.ORANGE
                 elif not term1_valid or not term2_valid: # 如果任一组合数参数无效
                     theoretical_y_value = 0
                     invalid_term = ""
                     if not term1_valid: invalid_term += " C(k,s)"
                     if not term2_valid: invalid_term += " C(n-k,j-s)"
                     theoretical_y_val_str = f"0 (因组合项无效:{invalid_term.strip()})"
                     theoretical_y_info.color = ft.colors.BLACK # 结果是0，但原因明确，用黑色
                 else:
                    comb1 = comb(k, s)       # 计算 C(k,s)
                    comb2 = comb(n - k, j - s) # 计算 C(n-k, j-s)
                    theoretical_y_value = comb1 * comb2 # 计算理论 y 值
                    theoretical_y_val_str = f"{theoretical_y_value} (C({k},{s})={comb1} * C({n-k},{j-s})={comb2})"
                    theoretical_y_info.color = ft.colors.BLACK # 正常黑色
            except ValueError as combo_error: # 捕捉计算组合数时的错误
                 theoretical_y_val_str = f"计算错误 ({combo_error})"
                 theoretical_y_info.color = ft.colors.RED
            except Exception as calc_error:   # 捕捉其他计算错误
                 theoretical_y_val_str = f"未知错误 ({calc_error})"
                 theoretical_y_info.color = ft.colors.RED
        # 更新显示理论 y 值的文本控件
        theoretical_y_info.value = f"理论覆盖度 (C(k,s)C(n-k,j-s)): {theoretical_y_val_str}"

        page.update() # 更新页面显示

    # --- 绑定 N, K, J, S 输入框的 on_change 事件 ---
    txt_n.on_change = update_y_related_info
    txt_k.on_change = update_y_related_info
    txt_j.on_change = update_y_related_info
    txt_s.on_change = update_y_related_info

    # --- Checkbox 事件处理 ---
    def on_manual_univ_change(e):
        """当“手动输入 Universe”复选框状态改变时，切换对应文本框的可见性"""
        is_manual = chk_manual_univ.value
        txt_manual_univ.visible = is_manual # 设置文本框可见性
        if not is_manual: # 如果取消勾选
            txt_manual_univ.value = ""        # 清空文本框内容
            txt_manual_univ.error_text = None # 清除错误提示
            txt_manual_univ.update()          # 更新文本框
        # 为了更好的布局，我们将文本框放在一个 Row 中，并切换 Row 的可见性
        manual_univ_row = txt_manual_univ.parent # 获取包含文本框的父级 Row
        if manual_univ_row:
            manual_univ_row.visible = is_manual # 设置 Row 的可见性
        page.update() # 更新页面

    def on_specify_y_change(e):
        """当“手动指定 y”复选框状态改变时，切换对应文本框的可见性"""
        is_manual = chk_specify_y.value
        txt_specify_y.visible = is_manual
        if not is_manual:
            txt_specify_y.value = ""
            txt_specify_y.error_text = None
            txt_specify_y.update()
        # 同样，切换包含文本框的 Row 的可见性
        specify_y_row = txt_specify_y.parent
        if specify_y_row:
            specify_y_row.visible = is_manual
        page.update()

    # --- 绑定 Checkbox 的 on_change 事件 ---
    chk_manual_univ.on_change = on_manual_univ_change
    chk_specify_y.on_change = on_specify_y_change

    # --- ================================ ---
    # --- 数据库结果管理部分的逻辑函数 ---
    # --- ================================ ---

    def format_result_summary(result_item: dict) -> str:
        """将从数据库获取的单条结果摘要字典格式化为易于阅读的字符串，用于列表显示。"""
        # 示例输出: "ID: 123 | 45-8-6-4-4-1 | 50 组 | 2023-10-27 10:30:00"
        try:
            # 尝试解析 ISO 格式的时间戳
            timestamp_dt = datetime.fromisoformat(result_item['timestamp']) if result_item.get('timestamp') else datetime.now()
        except ValueError:
            # 如果解析失败，提供一个备用显示或记录错误
            timestamp_dt = datetime.now() # 或者 None
            print(f"警告：无法解析时间戳 '{result_item.get('timestamp')}' for ID {result_item.get('id')}")

        # 格式化时间戳
        time_str = timestamp_dt.strftime('%Y-%m-%d %H:%M') if timestamp_dt else "N/A"

        # 构建摘要字符串
        return (f"ID: {result_item.get('id', 'N/A')} | "
                f"{result_item.get('m','?')}-{result_item.get('n','?')}-{result_item.get('k','?')}-{result_item.get('j','?')}-{result_item.get('s','?')}-{result_item.get('run_index','?')} | "
                f"{result_item.get('num_results','?')} 组 | {time_str}")

    def update_db_list_view():
        """根据全局状态 `db_results_list_data.current` 更新数据库结果列表 UI (ListView)"""
        db_results_list_view.controls.clear() # 清空现有列表项
        if not db_results_list_data.current: # 如果数据为空
            db_results_list_view.controls.append(ft.Text("数据库中没有结果记录。"))
        else:
            # 遍历数据，为每条记录创建一个 Radio 按钮
            for item in db_results_list_data.current:
                summary_text = format_result_summary(item) # 获取格式化后的摘要文本
                # 每个列表项是一个 Radio 按钮，value 存储该记录的数据库 ID (转为字符串)
                db_results_list_view.controls.append(
                    ft.Radio(value=str(item['id']), label=summary_text)
                )
        # 清除之前的选择状态和详情显示
        db_results_radio_group.value = None        # 清除 RadioGroup 的选中值
        selected_db_result_id.current = None       # 清除全局存储的选中 ID
        db_result_details_view.value = "请先在上方选择一条记录，然后点击“显示详情”。" # 重置详情区提示
        display_details_button.disabled = True     # 禁用“显示详情”按钮
        delete_selected_button.disabled = True     # 禁用“删除所选”按钮
        page.update() # 更新 UI

    def load_db_results(e=None):
        """从数据库加载结果摘要列表，并更新 UI"""
        log_message("正在从数据库加载结果列表...")
        try:
            results = db.get_results_summary() # 调用 db 模块的函数获取摘要列表
            db_results_list_data.current = results # 更新全局状态变量
            update_db_list_view() # 使用新数据更新 ListView 界面
            log_message(f"成功加载 {len(results)} 条结果摘要。")
        except Exception as ex: # 捕捉可能的数据库或其他错误
            msg = f"加载数据库结果列表时出错: {ex}"
            log_message(msg, is_error=True)
            # 可以在详情区域显示错误信息
            show_info_message(db_result_details_view, f"错误: {msg}", is_error=True)
            db_results_list_data.current = [] # 出错时清空数据
            update_db_list_view() # 更新 UI 为空列表状态
        finally:
             page.update() # 确保页面最终更新

    def on_db_result_select(e):
        """当用户在数据库结果列表中选择一个 Radio 按钮时触发的回调函数"""
        selected_id_str = db_results_radio_group.value # 获取 RadioGroup 当前选中的 value (即记录ID字符串)
        if selected_id_str: # 如果有选中项
            try:
                selected_id_int = int(selected_id_str) # 将 ID 字符串转为整数
                selected_db_result_id.current = selected_id_int # 更新全局状态变量
                display_details_button.disabled = False # 启用“显示详情”按钮
                delete_selected_button.disabled = False # 启用“删除所选”按钮
                log_message(f"已选择数据库记录 ID: {selected_id_int}")
            except (ValueError, TypeError): # 捕捉转换整数失败或其他类型错误
                selected_db_result_id.current = None
                display_details_button.disabled = True
                delete_selected_button.disabled = True
                log_message(f"无效的选择值: '{selected_id_str}'。", is_error=True)
        else: # 如果没有选中项 (例如，列表为空或清除了选择)
            selected_db_result_id.current = None
            display_details_button.disabled = True
            delete_selected_button.disabled = True
        # 提示用户下一步操作
        db_result_details_view.value = "请点击“显示详情”查看所选记录。"
        page.update() # 更新按钮状态和详情区文本

    # --- 绑定数据库列表的选择事件 ---
    db_results_radio_group.on_change = on_db_result_select

    def display_selected_details(e):
        """当用户点击“显示详情”按钮时触发，获取并显示选中记录的详细信息"""
        if selected_db_result_id.current is None: # 检查是否已选中记录
            log_message("未选择任何记录以显示详情。", is_error=True)
            return

        target_id = selected_db_result_id.current
        log_message(f"正在获取 ID={target_id} 的详细信息...")
        # 更新详情区显示加载状态
        show_info_message(db_result_details_view, f"正在加载 ID={target_id} 的详情...")
        page.update() # 立即显示加载提示

        try:
            # 调用 db 模块函数获取完整详情
            details = db.get_result_details(target_id)
            if details: # 如果成功获取到详情
                # --- 格式化详情信息以便显示 ---
                params_str = f"{details.get('m','?')}-{details.get('n','?')}-{details.get('k','?')}-{details.get('j','?')}-{details.get('s','?')}-{details.get('run_index','?')}"
                # 获取解析后的 Universe 和 Sets (如果解析失败，db.py 中会存入错误字符串)
                universe_disp = details.get('universe_parsed', 'N/A')
                sets_list = details.get('sets_found_parsed', [])
                num_sets = len(sets_list) if isinstance(sets_list, list) else 0 # 确保 sets_list 是列表才计算长度

                # 构建要显示的文本
                details_text = (
                    f"ID: {details.get('id')}\n"
                    f"参数 (M-N-K-J-S-RunIdx): {params_str}\n"
                    f"时间戳: {details.get('timestamp')}\n"
                    f"算法: {details.get('algorithm', 'N/A')}\n"
                    f"耗时: {details.get('time_taken', 0):.2f} 秒\n" # 格式化浮点数
                    f"覆盖条件 (y): {details.get('y_condition', 'N/A')}\n"
                    f"Universe ({len(universe_disp) if isinstance(universe_disp, list) else 'N/A'} 个): {universe_disp}\n"
                    f"找到的集合 ({num_sets} 组):\n"
                )

                # --- 格式化显示集合列表 (sets_list) ---
                MAX_SETS_TO_DISPLAY_DB = 100 # 在详情视图中可以显示更多集合
                if num_sets > 0 and isinstance(sets_list, list):
                    sets_to_display = sets_list[:MAX_SETS_TO_DISPLAY_DB]
                    sets_lines = []
                    sets_per_line = 4 # 每行显示多少个集合
                    for i in range(0, len(sets_to_display), sets_per_line):
                         # 对每个集合排序后转为字符串，用 | 分隔
                         line = " | ".join([str(sorted(s)) for s in sets_to_display[i:i+sets_per_line]])
                         sets_lines.append(f"  {line}") # 加缩进
                    details_text += "\n".join(sets_lines) # 将所有行合并
                    if num_sets > MAX_SETS_TO_DISPLAY_DB: # 如果集合过多未完全显示
                        details_text += f"\n  ... (还有 {num_sets - MAX_SETS_TO_DISPLAY_DB} 个未显示)"
                elif isinstance(sets_list, str): # 如果 JSON 解析失败，sets_list 会是错误字符串
                     details_text += f"  (无法解析集合: {sets_list})"
                else: # 如果没有集合
                    details_text += "  (无)"

                # 更新详情显示区域
                show_info_message(db_result_details_view, details_text)
                log_message(f"已显示 ID={target_id} 的详情。")
            else: # 如果 db.get_result_details 返回 None (未找到记录)
                msg = f"未在数据库中找到 ID={target_id} 的详细信息。"
                log_message(msg, is_error=True)
                show_info_message(db_result_details_view, msg, is_error=True)
        except Exception as ex: # 捕捉获取或格式化过程中的其他错误
            msg = f"显示详情时出错 (ID={target_id}): {ex}"
            log_message(msg, is_error=True)
            show_info_message(db_result_details_view, f"错误: {msg}", is_error=True)
        finally:
             page.update() # 确保页面最终更新

    # --- 删除确认对话框 ---
    # **修改点**: 移除了 delete_confirm_dialog 的定义
    # delete_confirm_dialog = ft.AlertDialog(...) # <--- 此块代码已删除

    # **修改点**: 移除了 close_dialog 函数
    # def close_dialog(dialog: ft.AlertDialog): ... # <--- 此函数已删除

    def execute_delete(e):
        """执行实际的删除操作 (由“删除所选”按钮直接调用)"""
        # **修改点**: 移除了关闭对话框的调用
        # close_dialog(delete_confirm_dialog) # 首先关闭对话框 (不再需要)

        if selected_db_result_id.current is None: # 再次检查是否有选中项
            log_message("未选择任何记录进行删除。", is_error=True)
            return

        target_id = selected_db_result_id.current
        log_message(f"正在尝试直接删除 ID={target_id} 的记录...")
        try:
            success = db.delete_result(target_id) # 调用 db 模块的删除函数
            if success:
                log_message(f"记录 ID={target_id} 已成功从数据库删除。正在刷新列表...")
                # 删除成功后，重新加载数据库列表以反映更改
                load_db_results() # load_db_results 会自动更新 UI 并清除选择
            else:
                # 如果 db.delete_result 返回 False (可能因为未找到记录，或者sqlite错误但未抛异常)
                # db.py 内部应该已经打印了具体原因
                log_message(f"删除记录 ID={target_id} 的操作完成，但可能未实际删除（请检查日志）。", is_error=True)
        except Exception as ex: # 捕捉删除过程中的异常
            msg = f"删除记录 ID={target_id} 时发生错误: {ex}"
            log_message(msg, is_error=True)
            show_info_message(db_result_details_view, f"删除错误: {msg}", is_error=True) # 在详情区显示错误
        finally:
             page.update() # 确保 UI 更新

    # --- 将 execute_delete 绑定到确认对话框的“确定删除”按钮 ---
    # **修改点**: 移除了这一行，因为对话框不存在了
    # delete_confirm_dialog.actions[0].on_click = execute_delete

    # **修改点**: 移除了 confirm_delete 函数
    # def confirm_delete(e): ... # <--- 此函数已删除

    # --- 绑定数据库管理按钮的事件 ---
    refresh_db_button.on_click = load_db_results       # 刷新按钮 -> 加载数据
    display_details_button.on_click = display_selected_details # 显示详情按钮 -> 显示详情
    # **修改点**: “删除所选”按钮直接绑定到 execute_delete
    delete_selected_button.on_click = execute_delete     # 删除按钮 -> 直接执行删除

    # --- ================= ---
    # --- 视图切换逻辑 ---
    # --- ================= ---

    # --- 主计算视图容器 (将原有的计算界面控件组织在一个 Column 中) ---
    main_computation_view = ft.Column(
        [
            # 参数输入标题
            ft.Text("参数输入", size=16, weight=ft.FontWeight.BOLD),
            # M, N, K, J, S, Timeout 输入框行
            ft.Row(
                [txt_m, txt_n, txt_k, txt_j, txt_s, txt_timeout],
                spacing=10, alignment=ft.MainAxisAlignment.START, wrap=True
            ),
            # 复选框行
            ft.Row(
                [chk_manual_univ, chk_specify_y],
                alignment=ft.MainAxisAlignment.START, spacing=20, vertical_alignment=ft.CrossAxisAlignment.CENTER
            ),
            # 手动 Universe 输入行 (通过 Row 控制显隐)
            ft.Row([txt_manual_univ], visible=chk_manual_univ.value),
            # 手动 y 输入行 (通过 Row 控制显隐)
            ft.Row([txt_specify_y], visible=chk_specify_y.value),
            # 分隔线
            ft.Divider(height=10),
            # 理论 y 和 C(j,s) 显示行
            ft.Row([theoretical_y_info, max_single_j_coverage], alignment=ft.MainAxisAlignment.SPACE_AROUND),
            # 分隔线
            ft.Divider(height=10),
            # 操作按钮行 (计算、清日志、查看数据库)
            ft.Row(
                [submit_button, progress_ring, clear_log_button, show_db_view_button], # 添加了 show_db_view_button
                alignment=ft.MainAxisAlignment.START, spacing=15, vertical_alignment=ft.CrossAxisAlignment.CENTER
            ),
            # 日志区域标题
            ft.Text("运行日志", size=14, weight=ft.FontWeight.BOLD),
            # 日志输出容器 (固定高度)
            ft.Container(
                content=log_output,
                border=ft.border.all(1, ft.colors.BLACK12),
                border_radius=ft.border_radius.all(5),
                padding=5,
                expand=False # 日志区域不自动扩展填充空间
            ),
             # 计算结果区域标题
            ft.Text("计算结果", size=14, weight=ft.FontWeight.BOLD),
             # 计算结果显示容器 (允许扩展)
            ft.Container(
                content=sample_result_info,
                border=ft.border.all(1, ft.colors.BLACK26),
                border_radius=ft.border_radius.all(5),
                padding=10,
                margin=ft.margin.only(top=10),
                expand=True # 结果区域可以扩展填充剩余的垂直空间
            )
        ],
        visible=True, # 主计算视图默认可见
        expand=True   # 允许此视图在垂直方向上扩展
    )

    def switch_view(show_db_view: bool):
        """切换主计算视图和数据库管理视图的可见性"""
        main_computation_view.visible = not show_db_view # 如果要显示数据库视图，则隐藏主视图
        db_management_view.visible = show_db_view      # 设置数据库视图的可见性
        if show_db_view: # 如果切换到数据库视图
            load_db_results() # 自动加载/刷新数据库列表
            log_message("已切换到数据库结果管理视图。")
        else: # 如果切换回主计算视图
            log_message("已返回到主计算界面。")
        page.update() # 更新页面以应用可见性更改

    # --- 绑定视图切换按钮的事件 ---
    # 点击“查看/管理数据库结果”按钮时，调用 switch_view(True)
    show_db_view_button.on_click = lambda e: switch_view(True)
    # 点击数据库视图中的“返回计算界面”按钮时，调用 switch_view(False)
    back_to_main_button.on_click = lambda e: switch_view(False)

    # --- ========================== ---
    # --- 提交计算的核心逻辑函数 ---
    # --- ========================== ---

    def on_submit(e):
        """处理提交按钮点击事件：验证输入，获取 run_index, 启动后台计算线程。"""
        log_message("收到计算请求...")
        show_info_message(sample_result_info, "正在处理输入参数...")

        # --- 1. 清除旧错误提示 ---
        for field in [txt_m, txt_n, txt_k, txt_j, txt_s, txt_timeout, txt_manual_univ, txt_specify_y]:
            field.error_text = None
            field.update()

        error_occurred = False # 标记是否发生错误

        # --- 2. 获取并验证基本参数 M, N, K, J, S, Timeout ---
        m_processed = validate_and_get_int(txt_m, "M", 1)
        n_processed = validate_and_get_int(txt_n, "N", 1)
        k_processed = validate_and_get_int(txt_k, "K", 1)
        j_processed = validate_and_get_int(txt_j, "J", 1)
        s_processed = validate_and_get_int(txt_s, "S", 0) # S可以为0
        timeout_val = validate_and_get_int(txt_timeout, "超时时间", 1)

        if None in [m_processed, n_processed, k_processed, j_processed, s_processed, timeout_val]:
            log_message("基本参数验证失败。", is_error=True)
            error_occurred = True

        if error_occurred: return # 如果基本参数验证失败，则停止

        timeout_seconds = timeout_val

        # --- 3. 进一步检查参数间的逻辑关系 ---
        validation_errors = []
        if n_processed > m_processed: validation_errors.append(f"N ({n_processed}) 不能大于 M ({m_processed})。")
        if k_processed > n_processed: validation_errors.append(f"K ({k_processed}) 不能大于 N ({n_processed})。")
        if j_processed > n_processed: validation_errors.append(f"J ({j_processed}) 不能大于 N ({n_processed})。")
        if s_processed > k_processed: validation_errors.append(f"S ({s_processed}) 不能大于 K ({k_processed})。")
        if s_processed > j_processed: validation_errors.append(f"S ({s_processed}) 不能大于 J ({j_processed})。")
        if s_processed < 0: validation_errors.append("S 不能小于 0。")

        if validation_errors: # 如果存在逻辑错误
            error_msg = "\n".join(validation_errors)
            log_message(f"参数逻辑错误: {error_msg}", is_error=True)
            show_info_message(sample_result_info, f"错误:\n{error_msg}", is_error=True)
            return # 停止执行

        # --- 4. 处理 Universe (根据复选框状态) ---
        is_manual_univ = chk_manual_univ.value # 获取复选框状态
        log_message(f"手动 Universe 选项: {'已勾选' if is_manual_univ else '未勾选'}")
        univ = [] # 初始化 Universe 列表
        if is_manual_univ: # 如果勾选了手动输入
            manual_str = txt_manual_univ.value.strip() # 获取手动输入值
            txt_manual_univ.error_text = None # 清除旧错误
            if not manual_str: # 输入为空
                msg = "错误：勾选了手动输入 Universe，但输入框为空。"
                log_message(msg, is_error=True)
                show_info_message(sample_result_info, msg, is_error=True)
                txt_manual_univ.error_text = "输入不能为空"
                txt_manual_univ.update()
                error_occurred = True
            else: # 输入不为空，尝试解析
                try:
                    num_strs = manual_str.split() # 按空格分割
                    univ_nums_temp = []
                    for num_str in num_strs: # 逐个转换
                        try: univ_nums_temp.append(int(num_str))
                        except ValueError: raise ValueError(f"输入 '{num_str}' 不是有效整数。")

                    # 检查数量、重复、范围
                    if len(univ_nums_temp) != n_processed: raise ValueError(f"需输入 {n_processed} 个数字，实际输入 {len(univ_nums_temp)} 个。")
                    if len(set(univ_nums_temp)) != len(univ_nums_temp): raise ValueError("输入数字包含重复项。")
                    invalid_nums = [x for x in univ_nums_temp if not (1 <= x <= m_processed)]
                    if invalid_nums: raise ValueError(f"数字需在 1 到 {m_processed} 之间。无效: {invalid_nums}")

                    univ = sorted(univ_nums_temp) # 排序后存入 univ
                    log_message(f"使用手动输入的 Universe: {univ}")
                    txt_manual_univ.update() # 更新以清除可能的旧错误提示
                except ValueError as ve: # 捕捉解析或验证过程中的错误
                    log_message(f"手动 Universe 输入错误: {ve}", is_error=True)
                    show_info_message(sample_result_info, f"手动 Universe 输入错误: {ve}", is_error=True)
                    txt_manual_univ.error_text = str(ve) # 在文本框旁边显示错误
                    txt_manual_univ.update()
                    error_occurred = True
        else: # 如果未勾选手动输入，则随机生成
            try:
                univ = sorted(random.sample(range(1, m_processed + 1), n_processed)) # 随机采样
                log_message(f"随机生成的 Universe: {univ}")
            except ValueError as e: # 捕捉 M < N 等导致的采样错误
                 msg = f"无法生成随机 Universe (M={m_processed}, N={n_processed}): {e}"
                 log_message(msg, is_error=True)
                 show_info_message(sample_result_info, msg, is_error=True)
                 error_occurred = True

        if error_occurred: return # 如果处理 Universe 出错，则停止

        # --- 5. 处理覆盖条件 y (根据复选框状态) ---
        is_manual_y = chk_specify_y.value # 获取复选框状态
        log_message(f"手动 Y 选项: {'已勾选' if is_manual_y else '未勾选'}")
        condition_processed = None # 最终使用的 y 值
        max_y_single_j = -1 # 存储 C(j,s) 的计算结果
        try: # 计算 C(j,s)
            if j_processed >= s_processed >= 0: max_y_single_j = comb(j_processed, s_processed)
            else: max_y_single_j = 0
        except ValueError: # 理论上之前的验证应该阻止了这个错误，但还是加上
             msg = f"错误: 计算 C(j={j_processed}, s={s_processed}) 时参数无效。"
             log_message(msg, is_error=True)
             show_info_message(sample_result_info, msg, is_error=True)
             error_occurred = True

        if not error_occurred: # 如果计算 C(j,s) 未出错
            if is_manual_y: # 如果勾选了手动指定 y
                y_str = txt_specify_y.value.strip() # 获取输入值
                txt_specify_y.error_text = None # 清除旧错误
                if not y_str: # 输入为空
                    msg = "错误：勾选了手动指定 y，但输入框为空。"
                    log_message(msg, is_error=True)
                    show_info_message(sample_result_info, msg, is_error=True)
                    txt_specify_y.error_text = "输入不能为空"
                    txt_specify_y.update()
                    error_occurred = True
                else: # 输入不为空，尝试解析和验证
                    try:
                        specified_y = int(y_str) # 转为整数
                        # 验证 y 的范围
                        if specified_y < 1: raise ValueError("y 值必须至少为 1。")
                        # 检查 y 是否超过 C(j,s)
                        # 注意：max_y_single_j 可能为 0 (例如 j<s)，此时任何正数 y 都是无效的
                        if max_y_single_j >= 0 and specified_y > max_y_single_j:
                            raise ValueError(f"y ({specified_y}) 不能超过 C({j_processed},{s_processed})={max_y_single_j}。")
                        elif max_y_single_j == 0 and specified_y > 0:
                             raise ValueError(f"C(j,s) 为 0，无法指定正数 y ({specified_y})。")

                        condition_processed = specified_y # 验证通过，使用用户指定的值
                        log_message(f"使用手动指定的覆盖条件 y = {condition_processed}")
                        txt_specify_y.update() # 更新以清除可能的旧错误提示
                    except ValueError as ve: # 捕捉转换或范围验证错误
                        log_message(f"手动 y 值输入错误: {ve}", is_error=True)
                        show_info_message(sample_result_info, f"手动 y 值输入错误: {ve}", is_error=True)
                        txt_specify_y.error_text = str(ve)
                        txt_specify_y.update()
                        error_occurred = True
            else: # 如果未勾选手动指定 y，则自动使用 C(j,s)
                 condition_processed = max_y_single_j
                 log_message(f"使用自动计算的覆盖条件 y = C(j,s) = {condition_processed}")
                 # 后端算法应该能处理 y=0 的情况 (表示没有覆盖要求)
                 if condition_processed <= 0:
                      log_message(f"警告：计算得到的 y = {condition_processed}。后端将按此执行。")

        if error_occurred: return # 如果处理 y 出错，则停止

        # --- 所有输入参数处理和验证完毕 ---
        log_message(f"最终参数: M={m_processed}, N={n_processed}, K={k_processed}, J={j_processed}, S={s_processed}, 使用y={condition_processed}, Timeout={timeout_seconds}s")
        show_info_message(sample_result_info, f"参数验证通过。\nUniverse: {univ}\ny={condition_processed}\n正在准备启动计算 (超时: {timeout_seconds} 秒)...")

        # --- 6. 获取持久化的运行索引 (x) ---
        log_message("正在从数据库获取下一个运行索引...")
        current_run_idx = db.get_and_increment_run_index(m_processed, n_processed, k_processed, j_processed, s_processed)

        if current_run_idx is None: # 如果获取索引失败
            msg = "错误：无法从数据库获取或更新运行索引！计算取消。"
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, msg, is_error=True)
            return # 停止执行

        log_message(f"本次运行索引: {current_run_idx}")

        # --- 7. **禁用 UI**，准备执行耗时操作 ---
        set_busy(True)

        # --- 8. 创建 Sample 实例并准备运行计算 ---
        sample = None
        try:
            # 创建 Sample 实例，传入所有参数
            sample = Sample(m=m_processed, n=n_processed, k=k_processed, j=j_processed, s=s_processed,
                            y=condition_processed,    # 传递最终处理好的 y 值
                            run_idx=current_run_idx,  # 传递从数据库获取的 run_index
                            timeout=timeout_seconds,
                            rand_instance=random)     # 传递当前的随机数生成器实例
            sample.univ = univ # 将前面准备好的 Universe 设置给实例
            log_message(f"Sample 实例 (Run {current_run_idx}) 创建成功，即将启动后台计算...")

            # --- 9. 启动后台计算线程 ---
            # 将耗时的 sample.run() 方法放在一个单独的线程中执行
            computation_thread = threading.Thread(
                target=run_computation,   # 线程执行的目标函数
                args=(sample,),           # 传递 Sample 实例给目标函数
                daemon=True               # 设置为守护线程，主程序退出时该线程也会退出
            )
            computation_thread.start() # 启动线程
            # UI 保持禁用状态，计算完成后 run_computation 函数会调用 set_busy(False) 来恢复 UI

        except ValueError as e: # 捕捉 Sample 初始化时可能抛出的参数错误
            msg = f"创建计算实例时参数错误 (Run {current_run_idx}): {e}"
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, msg, is_error=True)
            set_busy(False) # 出错，恢复 UI
        except Exception as e: # 捕捉其他意外错误
            msg = f"启动计算过程中发生意外错误 (Run {current_run_idx}): {e}"
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, f"运行时错误: {e}", is_error=True)
            set_busy(False) # 出错，恢复 UI
            print(f"--- 创建或启动 Sample 实例时发生未捕获的错误 (Run {current_run_idx}) ---")
            import traceback
            traceback.print_exc() # 打印详细错误堆栈到控制台
            print(f"--- 错误结束 ---")

    # --- 用于在后台线程中运行计算并更新 UI 的函数 ---
    def run_computation(sample_instance: Sample):
        """在单独的线程中运行 Sample.run()，处理结果，更新 UI，并在结束时恢复 UI 状态。"""
        result_text = f"计算中 (Run {sample_instance.run_idx})..." # 初始结果文本
        is_final_error = False # 标记最终结果是否表示错误状态
        run_idx_local = getattr(sample_instance, 'run_idx', 'N/A') # 获取 run_index 以便在日志中使用

        try:
            # --- 执行核心计算 ---
            log_message(f"后台线程开始执行计算 (Run {run_idx_local})...")
            sample_instance.run() # 调用 Sample 对象的 run 方法，这会阻塞当前线程直到计算完成或超时
            log_message(f"后台计算执行完毕 (Run {run_idx_local})。")

            # --- 处理计算结果 ---
            log_message(f"处理计算结果 (Run {run_idx_local}). 结果标识符: {sample_instance.ans}")

            # 检查 backend 返回的结果是否有效
            if sample_instance.result and 'alg' in sample_instance.result:
                # 从结果字典中提取信息
                alg = sample_instance.result.get('alg', 'N/A')
                sets_list = sample_instance.sets if isinstance(sample_instance.sets, list) else []
                num_sets = len(sets_list)
                time_taken = sample_instance.result.get('time', 0)
                status = sample_instance.result.get('status', 'Unknown')
                final_run_idx = sample_instance.result.get('run_index', run_idx_local) # 确认 run_index
                y_used = sample_instance.result.get('coverage_target', sample_instance.y) # 获取实际使用的 y

                log_message(f"求解算法 (Run {final_run_idx}): {alg}, 状态: {status}, 耗时: {time_taken:.2f}s, 找到集合数: {num_sets}")

                # --- 构建要在主结果区显示的文本 ---
                result_text = (
                    f"结果标识: {str(sample_instance.ans)}\n"
                    f"Run Index: {final_run_idx}\n"
                    f"Universe ({len(sample_instance.univ)}个): {sample_instance.univ}\n"
                    f"算法: {alg}\n"
                    f"状态: {status}\n"
                    f"使用y: {y_used}\n"
                    f"耗时: {time_taken:.2f} 秒\n"
                    f"找到集合 ({num_sets} 个):\n"
                )
                # 根据状态判断是否是错误/失败状态
                is_final_error = status not in ('OPTIMAL', 'FEASIBLE', 'SUCCESS', 'INFEASIBLE')

                # --- 格式化显示找到的集合列表 ---
                MAX_SETS_TO_DISPLAY = 50 # 主结果区最多显示多少个集合
                if num_sets > 0 :
                    sets_to_display = sets_list[:MAX_SETS_TO_DISPLAY]
                    sets_lines = []
                    sets_per_line = 3 # 每行显示多少个集合
                    for i in range(0, len(sets_to_display), sets_per_line):
                         line = " | ".join([str(sorted(s)) for s in sets_to_display[i:i+sets_per_line]])
                         sets_lines.append(f"  {line}") # 加缩进
                    result_text += "\n".join(sets_lines)
                    if num_sets > MAX_SETS_TO_DISPLAY: # 如果集合过多
                        result_text += f"\n  ... (还有 {num_sets - MAX_SETS_TO_DISPLAY} 个未显示)"
                elif status == 'INFEASIBLE': # 如果状态是无解
                     result_text += "  (问题被证明无解)"
                else: # 其他情况 (包括成功但0集合，或错误状态)
                     result_text += "  (无)"

                # --- 数据库存储 (仅当 K=6 时) ---
                if sample_instance.k == 6:
                    log_message(f"K=6 (Run {final_run_idx})，尝试保存结果到数据库...")
                    try:
                        # 将 Universe 和 Sets 列表转换为 JSON 字符串以便存储
                        universe_str = json.dumps(sample_instance.univ)
                        found_sets_str = json.dumps(sets_list)

                        # 准备要存入数据库的数据字典
                        result_data = {
                            'm': sample_instance.m, 'n': sample_instance.n, 'k': sample_instance.k,
                            'j': sample_instance.j, 's': sample_instance.s, 'run_index': final_run_idx,
                            'num_results': num_sets, 'y_condition': y_used, 'algorithm': alg,
                            'time_taken': time_taken, 'universe': universe_str, 'sets_found': found_sets_str
                            # timestamp 会由数据库自动添加
                        }
                        # 调用 db 模块的保存函数
                        save_success = db.save_result(result_data)
                        if save_success:
                            log_message(f"结果 (Run {final_run_idx}) 已成功保存到数据库。")
                        else:
                            # save_result 内部会打印错误，这里只记录一个通用日志
                            log_message(f"保存结果 (Run {final_run_idx}) 到数据库时遇到问题（可能重复或错误，请检查日志）。", is_error=True)
                            is_final_error = True # 标记为错误状态
                    except Exception as db_err: # 捕捉保存过程中的其他异常
                        error_msg = f"错误：保存结果 (Run {final_run_idx}) 到数据库失败: {db_err}"
                        log_message(error_msg, is_error=True)
                        is_final_error = True
                        print(error_msg) # 打印到控制台
                        import traceback; traceback.print_exc() # 打印堆栈
                else:
                    log_message(f"K={sample_instance.k} != 6，结果 (Run {final_run_idx}) 未保存到数据库。")
            else: # 如果 sample_instance.result 无效或缺少 'alg' 键
                 error_msg = f"计算执行完毕 (Run {run_idx_local})，但无法获取有效的算法结果详情。"
                 log_message(error_msg, is_error=True)
                 result_text = error_msg
                 is_final_error = True

        except Exception as compute_err: # 捕捉 run_computation 过程中的顶层错误
            error_msg = f"计算执行或结果处理过程中发生错误 (Run {run_idx_local}): {compute_err}"
            log_message(error_msg, is_error=True)
            result_text = f"运行时错误 (Run {run_idx_local}):\n{compute_err}"
            is_final_error = True
            print(f"--- 计算线程中的未捕获错误 (Run {run_idx_local}) ---")
            import traceback; traceback.print_exc() # 打印堆栈到控制台
            print(f"--- 错误结束 ---")

        finally:
            # --- 无论计算成功与否，最终都需要更新 UI 并恢复控件状态 ---
             try:
                  # 更新主结果显示区域
                  show_info_message(sample_result_info, result_text, is_error=is_final_error)
                  # !! 关键：恢复 UI 控件的可用状态 !!
                  set_busy(False)
                  log_message(f"计算与结果处理流程结束 (Run {run_idx_local})。UI 已恢复。")
             except Exception as ui_update_err:
                  # 捕捉更新 UI 时可能发生的错误 (虽然 Flet 应该能处理好线程安全问题)
                  print(f"严重错误：从计算线程更新 UI 时出错: {ui_update_err}")
                  # 尝试记录到日志
                  try: log_message(f"!!! UI 更新错误: {ui_update_err}", is_error=True)
                  except: pass
                  # 尝试再次恢复按钮状态，以防万一
                  try: set_busy(False)
                  except: print("!!! 紧急：无法恢复 UI 控件状态！应用程序可能无响应。")
             finally:
                    page.update() # 确保页面得到最终更新

    # --- 将 on_submit 函数绑定到提交按钮的点击事件 ---
    submit_button.on_click = on_submit

    # --- ============ ---
    # --- 页面最终布局 ---
    # --- ============ ---
    page.add(
        ft.Container( # 使用一个顶层容器包裹所有内容
            content=ft.Column(
                [
                    main_computation_view, # 主计算视图 (初始可见)
                    db_management_view     # 数据库管理视图 (初始隐藏)
                ],
                expand=True # 让内部的 Column 能够扩展
            ),
            expand=True, # 让顶层容器填充整个页面空间
            padding = 10 # 给整体内容添加边距
        )
    )
    # 允许整个页面在内容过多时滚动
    page.scroll = ft.ScrollMode.ADAPTIVE

    # --- ============ ---
    # --- 应用初始化 ---
    # --- ============ ---
    log_message("应用程序启动...")
    log_message(f"数据库文件路径: {os.path.abspath(db.DB_FILE)}") # 显示数据库文件位置
    log_message(f"Google OR-Tools 可用: {'是' if HAS_ORTOOLS else '否'}") # 显示 OR-Tools 状态

    # 触发一次 y 相关信息的计算和显示
    update_y_related_info()

    # 尝试初始化数据库 (创建表，如果不存在)
    try:
        db.setup_database()
        log_message("数据库初始化检查完成。")
    except Exception as init_db_err:
        msg = f"严重错误：初始化数据库失败: {init_db_err}"
        log_message(msg, is_error=True)
        # 初始时 sample_result_info 可能还没完全加载好，打印到控制台更可靠
        print(msg)
        import traceback; traceback.print_exc()
        # 可以在主界面显示错误提示
        show_info_message(sample_result_info, msg, is_error=True)

    # 初始化完成后，更新一次页面
    page.update()
    log_message("应用程序界面已加载，等待用户操作。")

# --- 应用入口点 ---
if __name__ == "__main__":
     # --- 处理命令行参数以指定数据库路径 (可选) ---
    if len(sys.argv) > 1: # 如果命令行提供了参数
        custom_db_path = sys.argv[1] # 获取第一个参数
        # 简单检查是否像一个路径或文件名
        if os.sep in custom_db_path or custom_db_path.endswith(".db"):
            db_dir = os.path.dirname(os.path.abspath(custom_db_path)) # 获取目录部分
            # 如果提供的是一个已存在的目录
            if os.path.isdir(custom_db_path):
                 db.DB_FILE = os.path.join(custom_db_path, "k6_results.db") # 在该目录下使用默认文件名
                 db_dir = custom_db_path
            else: # 如果提供的是文件路径
                 db.DB_FILE = custom_db_path # 直接使用提供的路径
                 db_dir = os.path.dirname(db.DB_FILE) # 获取其目录

            print(f"信息: 使用命令行指定的数据库路径: {os.path.abspath(db.DB_FILE)}")
            # 尝试创建数据库所在的目录 (如果不存在)
            if not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True) # exist_ok=True 避免目录已存在时报错
                    print(f"信息: 已创建数据库目录: {db_dir}")
                except OSError as e:
                    print(f"错误: 无法创建数据库目录 '{db_dir}': {e}. 将使用默认路径 '{db.DB_FILE}'.")
                    db.DB_FILE = "k6_results.db" # 恢复默认值
        else:
             print(f"警告: 无效的数据库路径参数 '{custom_db_path}'. 将使用默认路径 '{db.DB_FILE}'.")

    # 启动 Flet 应用程序
    # view=ft.AppView.WEB_BROWSER 可以让应用在浏览器中打开
    ft.app(target=main)