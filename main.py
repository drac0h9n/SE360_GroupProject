# main.py
# 使用 Flet 构建图形用户界面 (GUI)，处理用户输入，调用 backend 计算，并显示结果。
# 当 K=6 时，会将结果保存到 SQLite 数据库。
# 包含手动输入 Universe 和手动指定覆盖条件 y 的选项。
# 移除了全局 RunCounter，使用 db.py 管理持久化 run_index。
# 修正了 set_busy 调用时机，以正确处理手动输入。

import flet as ft
import random
import time
import json # 用于序列化列表以便存入数据库
import sys # 用于检查命令行参数
import os # 用于路径操作
import threading # <--- 确保导入 threading

# 导入后端逻辑和数据库操作
from backend import Sample, comb, HAS_ORTOOLS # 导入 Sample 类, comb 函数, HAS_ORTOOLS
import db # 导入数据库操作模块 (db.py)

# --- 全局变量 ---
random_seed = int(time.time())
random.seed(random_seed)
print(f"使用的随机种子: {random_seed}")

# --- 主应用函数 ---
def main(page: ft.Page):
    page.title = "组合覆盖问题求解器 (持久化 Run Index)"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.window_width = 850
    page.window_height = 800

    # --- 输入控件 ---
    txt_m = ft.TextField(label="M (基础集大小)", hint_text="例如: 45", width=180, value="45")
    txt_n = ft.TextField(label="N (Universe 大小)", hint_text="例如: 8", width=180, value="8")
    txt_k = ft.TextField(label="K (覆盖块大小)", hint_text="例如: 6", width=180, value="6")
    txt_j = ft.TextField(label="J (子集大小)", hint_text="例如: 4", width=180, value="4")
    txt_s = ft.TextField(label="S (最小交集)", hint_text="例如: 4", width=180, value="4")
    txt_timeout = ft.TextField(label="超时时间 (秒)", hint_text="例如: 60", width=180, value="60")

    # --- Universe 输入选项 ---
    chk_manual_univ = ft.Checkbox(label="手动输 Universe", value=False, on_change=None)
    txt_manual_univ = ft.TextField(
        label="输入N个数字(空格分隔, 1~M)",
        visible=False,
        width=450,
        hint_text="例如: 1 5 10 15 20 25 30 35"
    )

    # --- 覆盖条件 y 输入选项 ---
    chk_specify_y = ft.Checkbox(label="手动指定 y (覆盖次数)", value=False, on_change=None)
    txt_specify_y = ft.TextField(label="输入 y 值 (1 ~ C(j,s))", visible=False, width=250, hint_text="输入正整数y")

    # --- 输出控件 ---
    theoretical_y_info = ft.Text("理论覆盖度 (C(k,s)C(n-k,j-s)): ...", size=12)
    max_single_j_coverage = ft.Text("单j集最大覆盖可能 (C(j,s)): ...", size=12)
    # 增加 selectable=True, max_lines 和 overflow
    sample_result_info = ft.Text(
        "计算结果将显示在这里...",
        size=12,
        selectable=True,
        max_lines=25, # 增加显示行数
        overflow=ft.TextOverflow.VISIBLE # 允许内容溢出容器（需要外部容器滚动）
    )
    log_output = ft.ListView(expand=True, spacing=5, auto_scroll=True)

    # --- 按钮 ---
    submit_button = ft.ElevatedButton(text="开始计算", on_click=None, icon=ft.icons.PLAY_ARROW)
    clear_log_button = ft.ElevatedButton(text="清空日志", on_click=lambda _: clear_log(), icon=ft.icons.CLEAR_ALL)

    # --- 进度指示器 ---
    progress_ring = ft.ProgressRing(visible=False, width=20, height=20, stroke_width = 3)

    # --- 日志函数 ---
    def log_message(message: str, is_error: bool = False):
        """向日志区域添加一条带时间戳的消息"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        color = ft.colors.RED if is_error else ft.colors.BLACK87
        log_output.controls.append(ft.Text(f"[{timestamp}] {message}", size=11, selectable=True, color=color))
        page.update()

    def clear_log():
        """清空日志区域"""
        log_output.controls.clear()
        log_message("日志已清空。")
        page.update()

    # --- 显示信息/错误 的辅助函数 ---
    def show_info_message(text_control: ft.Text, message: str, is_error: bool = False):
        """更新指定的 Text 控件来显示信息或错误"""
        text_control.value = message
        text_control.color = ft.Colors.RED if is_error else ft.Colors.BLACK
        page.update()

    def set_busy(busy: bool):
        """控制按钮和进度环的启用/禁用状态"""
        submit_button.disabled = busy
        progress_ring.visible = busy
        # 禁用/启用输入框和复选框
        for ctrl in [txt_m, txt_n, txt_k, txt_j, txt_s, txt_timeout, chk_manual_univ, txt_manual_univ, chk_specify_y, txt_specify_y]:
            ctrl.disabled = busy
        page.update()

    # --- 输入验证和处理 ---
    def validate_and_get_int(field: ft.TextField, name: str, min_val: int = 0) -> int | None:
        """验证输入是否为整数，并返回整数值，否则返回 None 并记录日志"""
        # 清除之前的错误状态
        field.error_text = None
        field.update()
        try:
            value = int(field.value.strip()) # 添加 strip() 去除首尾空格
            if value < min_val:
                msg = f"错误: {name} ({value}) 不能小于 {min_val}。"
                log_message(msg, is_error=True)
                show_info_message(sample_result_info, f"输入错误: {name} 不能小于 {min_val}。", is_error=True)
                field.error_text = f"不能小于 {min_val}"
                field.update()
                return None
            return value
        except ValueError:
            msg = f"错误: {name} ('{field.value}') 必须是一个有效的整数。"
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, f"输入错误: {name} 必须是一个整数。", is_error=True)
            field.error_text = "必须是整数"
            field.update()
            return None

    # --- 动态更新 y 相关信息 ---
    def update_y_related_info(e=None):
        """当 N, K, J, S 输入框的值改变时，计算并更新理论 y 值和 C(j,s)"""
        n_str, k_str, j_str, s_str = txt_n.value, txt_k.value, txt_j.value, txt_s.value
        n, k, j, s = None, None, None, None
        try: n = int(n_str.strip()) if n_str else None
        except ValueError: pass
        try: k = int(k_str.strip()) if k_str else None
        except ValueError: pass
        try: j = int(j_str.strip()) if j_str else None
        except ValueError: pass
        try: s = int(s_str.strip()) if s_str is not None and s_str.strip() != "" else None # s=0 是有效的
        except ValueError: pass

        # 更新 C(j,s) 显示
        max_single_cov_val_str = "等待有效 J, S..."
        max_single_j_coverage.color = ft.Colors.GREY
        max_y_limit = 0

        if j is not None and s is not None and j >= 0 and s >= 0:
            try:
                if j < s:
                    max_single_cov_val = 0
                    max_single_cov_val_str = f"C({j},{s}): 0 (因 j < s)"
                    max_single_j_coverage.color = ft.Colors.ORANGE
                else:
                    max_single_cov_val = comb(j, s)
                    max_single_cov_val_str = f"C({j},{s}): {max_single_cov_val}"
                    max_single_j_coverage.color = ft.Colors.BLACK
                    max_y_limit = max_single_cov_val
                txt_specify_y.label = f"输入 y 值 (1 ~ {max_y_limit})" if max_y_limit > 0 else "输入 y 值 (无效或无需)"
                txt_specify_y.update()
            except ValueError as combo_err:
                 max_single_cov_val_str = f"C({j},{s}): 计算错误 ({combo_err})"
                 max_single_j_coverage.color = ft.Colors.RED
                 txt_specify_y.label = "输入 y 值 (需有效J,S)"
                 txt_specify_y.update()
            except Exception as calc_err:
                 max_single_cov_val_str = f"C({j},{s}): 未知错误 ({calc_err})"
                 max_single_j_coverage.color = ft.Colors.RED
                 txt_specify_y.label = "输入 y 值 (计算错误)"
                 txt_specify_y.update()
        else:
             txt_specify_y.label = "输入 y 值"
             txt_specify_y.update()
        max_single_j_coverage.value = f"单j集最大覆盖可能 (C(j,s)): {max_single_cov_val_str}"

        # 更新理论 y = C(k,s)*C(n-k,j-s) 显示
        theoretical_y_val_str = "等待有效 N, K, J, S..."
        theoretical_y_info.color = ft.Colors.GREY
        if n is not None and k is not None and j is not None and s is not None and all(v >= 0 for v in [n,k,j,s]):
            try:
                 term1_valid = (k >= s)
                 term2_valid = (n - k >= j - s) and (j >= s)
                 params_logic_valid = (n >= k and n >= j)

                 if not params_logic_valid:
                      theoretical_y_val_str = "参数无效 (N需>=K且>=J)"
                      theoretical_y_info.color = ft.Colors.ORANGE
                 elif not term1_valid or not term2_valid:
                     theoretical_y_value = 0
                     invalid_term = ""
                     if not term1_valid: invalid_term += " C(k,s)"
                     if not term2_valid: invalid_term += " C(n-k,j-s)"
                     theoretical_y_val_str = f"0 (因组合项无效:{invalid_term.strip()})"
                     theoretical_y_info.color = ft.Colors.BLACK
                 else:
                    comb1 = comb(k, s)
                    comb2 = comb(n - k, j - s)
                    theoretical_y_value = comb1 * comb2
                    theoretical_y_val_str = f"{theoretical_y_value} (C({k},{s})={comb1} * C({n-k},{j-s})={comb2})"
                    theoretical_y_info.color = ft.Colors.BLACK
            except ValueError as combo_error:
                 theoretical_y_val_str = f"计算错误 ({combo_error})"
                 theoretical_y_info.color = ft.Colors.RED
            except Exception as calc_error:
                 theoretical_y_val_str = f"未知错误 ({calc_error})"
                 theoretical_y_info.color = ft.Colors.RED
        theoretical_y_info.value = f"理论覆盖度 (C(k,s)C(n-k,j-s)): {theoretical_y_val_str}"

        page.update()

    # 绑定事件到输入框
    txt_n.on_change = update_y_related_info
    txt_k.on_change = update_y_related_info
    txt_j.on_change = update_y_related_info
    txt_s.on_change = update_y_related_info

    # --- Checkbox 事件处理 ---
    def on_manual_univ_change(e):
        is_manual = chk_manual_univ.value
        txt_manual_univ.visible = is_manual
        if not is_manual:
            txt_manual_univ.value = ""
            txt_manual_univ.error_text = None
            txt_manual_univ.update()
        # 使包含文本框的 Row 的可见性与文本框一致
        manual_univ_row = txt_manual_univ.parent # 获取父级 Row
        if manual_univ_row:
            manual_univ_row.visible = is_manual
        page.update()

    def on_specify_y_change(e):
        is_manual = chk_specify_y.value
        txt_specify_y.visible = is_manual
        if not is_manual:
            txt_specify_y.value = ""
            txt_specify_y.error_text = None
            txt_specify_y.update()
        # 使包含文本框的 Row 的可见性与文本框一致
        specify_y_row = txt_specify_y.parent # 获取父级 Row
        if specify_y_row:
            specify_y_row.visible = is_manual
        page.update()

    # 绑定 Checkbox 事件
    chk_manual_univ.on_change = on_manual_univ_change
    chk_specify_y.on_change = on_specify_y_change

    # --- 提交按钮的回调函数 ---
    def on_submit(e):
        """处理提交按钮点击事件，执行计算并显示结果"""
        log_message("收到计算请求...")
        show_info_message(sample_result_info, "正在处理输入参数...")
        # --- 暫時不禁用 UI，先讀取和驗證所有輸入 ---
        # set_busy(True) # <--- 不要在这里禁用！

        # 重置之前的错误提示
        for field in [txt_m, txt_n, txt_k, txt_j, txt_s, txt_timeout, txt_manual_univ, txt_specify_y]:
            field.error_text = None
            field.update()

        error_occurred = False # 标记是否发生错误

        # 1. 获取并验证基本参数 M, N, K, J, S, Timeout
        m_processed = validate_and_get_int(txt_m, "M", 1)
        n_processed = validate_and_get_int(txt_n, "N", 1)
        k_processed = validate_and_get_int(txt_k, "K", 1)
        j_processed = validate_and_get_int(txt_j, "J", 1)
        s_processed = validate_and_get_int(txt_s, "S", 0)
        timeout_val = validate_and_get_int(txt_timeout, "超时时间", 1)

        if None in [m_processed, n_processed, k_processed, j_processed, s_processed, timeout_val]:
            log_message("基本参数验证失败。", is_error=True)
            error_occurred = True
            # 不需要 set_busy(False)，因为还没禁用

        if error_occurred:
            # set_busy(False) # 不需要
            return

        timeout_seconds = timeout_val

        # 2. 进一步检查参数间的逻辑关系
        validation_errors = []
        if n_processed > m_processed: validation_errors.append(f"N ({n_processed}) 不能大于 M ({m_processed})。")
        if k_processed > n_processed: validation_errors.append(f"K ({k_processed}) 不能大于 N ({n_processed})。")
        if j_processed > n_processed: validation_errors.append(f"J ({j_processed}) 不能大于 N ({n_processed})。")
        if s_processed > k_processed: validation_errors.append(f"S ({s_processed}) 不能大于 K ({k_processed})。")
        if s_processed > j_processed: validation_errors.append(f"S ({s_processed}) 不能大于 J ({j_processed})。")
        # 添加 s>=0 的检查，虽然 validate_and_get_int 做了，但 double check
        if s_processed < 0: validation_errors.append("S 不能小于 0。")

        if validation_errors:
            error_msg = "\n".join(validation_errors)
            log_message(f"参数逻辑错误: {error_msg}", is_error=True)
            show_info_message(sample_result_info, f"错误:\n{error_msg}", is_error=True)
            return # 不需要 set_busy(False)

        # --- 读取复选框状态 ---
        is_manual_univ = chk_manual_univ.value
        is_manual_y = chk_specify_y.value
        log_message(f"手动 Universe 选项: {'已勾选' if is_manual_univ else '未勾选'}")
        log_message(f"手动 Y 选项: {'已勾选' if is_manual_y else '未勾选'}")

        # 3. 处理 Universe (根据复选框状态)
        univ = []
        if is_manual_univ: # 手动输入模式
            manual_str = txt_manual_univ.value.strip()
            txt_manual_univ.error_text = None # 清除旧错误
            if not manual_str:
                msg = "错误：勾选了手动输入 Universe，但输入框为空。"
                log_message(msg, is_error=True)
                show_info_message(sample_result_info, msg, is_error=True)
                txt_manual_univ.error_text = "输入不能为空"
                txt_manual_univ.update()
                error_occurred = True
            else:
                try:
                    num_strs = manual_str.split()
                    univ_nums_temp = []
                    for num_str in num_strs:
                        try: univ_nums_temp.append(int(num_str))
                        except ValueError: raise ValueError(f"输入 '{num_str}' 不是有效整数。")

                    if len(univ_nums_temp) != n_processed: raise ValueError(f"需输入 {n_processed} 个数字，实际输入 {len(univ_nums_temp)} 个。")
                    if len(set(univ_nums_temp)) != len(univ_nums_temp): raise ValueError("输入数字包含重复项。")
                    invalid_nums = [x for x in univ_nums_temp if not (1 <= x <= m_processed)]
                    if invalid_nums: raise ValueError(f"数字需在 1 到 {m_processed} 之间。无效: {invalid_nums}")

                    univ = sorted(univ_nums_temp)
                    log_message(f"使用手动输入的 Universe: {univ}")
                    txt_manual_univ.update() # 更新以清除错误（如果之前有）
                except ValueError as ve:
                    log_message(f"手动 Universe 输入错误: {ve}", is_error=True)
                    show_info_message(sample_result_info, f"手动 Universe 输入错误: {ve}", is_error=True)
                    txt_manual_univ.error_text = str(ve)
                    txt_manual_univ.update()
                    error_occurred = True
        else: # 随机生成模式
            try:
                univ = sorted(random.sample(range(1, m_processed + 1), n_processed))
                log_message(f"随机生成的 Universe: {univ}")
            except ValueError as e:
                 msg = f"无法生成随机 Universe (M={m_processed}, N={n_processed}): {e}"
                 log_message(msg, is_error=True)
                 show_info_message(sample_result_info, msg, is_error=True)
                 error_occurred = True

        if error_occurred:
            return # 不需要 set_busy(False)

        # 4. 处理覆盖条件 y (根据复选框状态)
        condition_processed = None
        max_y_single_j = -1
        try:
            if j_processed >= s_processed >= 0: max_y_single_j = comb(j_processed, s_processed)
            else: max_y_single_j = 0
        except ValueError:
             msg = f"错误: 计算 C(j={j_processed}, s={s_processed}) 时参数无效。"
             log_message(msg, is_error=True)
             show_info_message(sample_result_info, msg, is_error=True)
             error_occurred = True

        if not error_occurred:
            if is_manual_y: # 手动指定 y
                y_str = txt_specify_y.value.strip()
                txt_specify_y.error_text = None # 清除旧错误
                if not y_str:
                    msg = "错误：勾选了手动指定 y，但输入框为空。"
                    log_message(msg, is_error=True)
                    show_info_message(sample_result_info, msg, is_error=True)
                    txt_specify_y.error_text = "输入不能为空"
                    txt_specify_y.update()
                    error_occurred = True
                else:
                    try:
                        specified_y = int(y_str)
                        if max_y_single_j == 0 and specified_y > 0: raise ValueError(f"C(j,s)为0，无法指定正数 y({specified_y})。")
                        elif specified_y < 1 and max_y_single_j > 0 : raise ValueError("y 值必须至少为 1。")
                        elif specified_y > max_y_single_j and max_y_single_j >= 0: raise ValueError(f"y({specified_y}) 不能超过 C({j_processed},{s_processed})={max_y_single_j}。")

                        condition_processed = specified_y
                        log_message(f"使用手动指定的覆盖条件 y = {condition_processed}")
                        txt_specify_y.update() # 更新以清除错误
                    except ValueError as ve:
                        log_message(f"手动 y 值输入错误: {ve}", is_error=True)
                        show_info_message(sample_result_info, f"手动 y 值输入错误: {ve}", is_error=True)
                        txt_specify_y.error_text = str(ve)
                        txt_specify_y.update()
                        error_occurred = True
            else: # 自动计算 y = C(j,s)
                 condition_processed = max_y_single_j
                 log_message(f"使用自动计算的覆盖条件 y = C(j,s) = {condition_processed}")
                 if condition_processed <= 0:
                      log_message(f"警告：计算得到的 y = {condition_processed}。后端将处理此情况。")

        if error_occurred:
             return # 不需要 set_busy(False)

        # --- 所有输入已处理完毕 ---
        log_message(f"最终参数: M={m_processed}, N={n_processed}, K={k_processed}, J={j_processed}, S={s_processed}, 使用y={condition_processed}, Timeout={timeout_seconds}s")
        show_info_message(sample_result_info, f"参数验证通过。\nUniverse: {univ}\ny={condition_processed}\n正在准备启动计算 (超时: {timeout_seconds} 秒)...")

        # --- 获取持久化的运行索引 (x) ---
        log_message("正在从数据库获取下一个运行索引...")
        current_run_idx = db.get_and_increment_run_index(m_processed, n_processed, k_processed, j_processed, s_processed)

        if current_run_idx is None:
            msg = "错误：无法从数据库获取或更新运行索引！"
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, msg, is_error=True)
            return # 不需要 set_busy(False)

        log_message(f"本次运行索引: {current_run_idx}")

        # --- **现在**，在执行耗时操作前，禁用 UI ---
        set_busy(True)

        # --- 创建 Sample 实例并准备运行计算 ---
        sample = None
        try:
            sample = Sample(m=m_processed, n=n_processed, k=k_processed, j=j_processed, s=s_processed,
                            y=condition_processed,
                            run_idx=current_run_idx,
                            timeout=timeout_seconds,
                            rand_instance=random)
            sample.univ = univ # 设置 Universe
            log_message(f"Sample 实例 (Run {current_run_idx}) 创建成功，即将启动后台计算...")

            # --- 启动后台计算线程 ---
            computation_thread = threading.Thread(target=run_computation, args=(sample,), daemon=True)
            computation_thread.start()
            # UI 保持禁用状态，直到 run_computation 完成并调用 set_busy(False)

        except ValueError as e:
            msg = f"创建计算实例时参数错误: {e}"
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, msg, is_error=True)
            set_busy(False) # 出错，恢复UI
        except Exception as e:
            msg = f"启动计算过程中发生错误: {e}"
            log_message(msg, is_error=True)
            show_info_message(sample_result_info, f"运行时错误: {e}", is_error=True)
            set_busy(False) # 出错，恢复UI
            print(f"--- 未捕获的创建/启动错误 ---")
            import traceback
            traceback.print_exc()
            print(f"--- 错误结束 ---")

    # --- 用于在线程中运行计算并处理结果的函数 ---
    # (run_computation 函数保持不变，它会在结束时调用 set_busy(False))
    def run_computation(sample_instance: Sample):
        """在单独的线程中运行 Sample.run() 并处理结果与UI更新"""
        result_text = f"计算中 (Run {sample_instance.run_idx})..."
        is_final_error = False # 标记最终结果是否为错误状态
        try:
            # 运行计算（阻塞当前线程）
            sample_instance.run()

            # --- 处理计算结果 (在工作线程中) ---
            log_message(f"计算 (Run {sample_instance.run_idx}) 完成。结果标识符: {sample_instance.ans}")

            if sample_instance.result and 'alg' in sample_instance.result:
                alg = sample_instance.result.get('alg', 'N/A')
                num_sets = len(sample_instance.sets) if isinstance(sample_instance.sets, list) else 0
                time_taken = sample_instance.result.get('time', 0)
                status = sample_instance.result.get('status', 'Unknown')
                run_idx = sample_instance.result.get('run_index', sample_instance.run_idx)
                y_used = sample_instance.result.get('coverage_target', sample_instance.y)

                log_message(f"求解算法 (Run {run_idx}): {alg}, 状态: {status}, 耗时: {time_taken:.2f}s, 找到集合数: {num_sets}")

                result_text = (
                    f"结果标识: {str(sample_instance.ans)}\n"
                    f"Run Index: {run_idx}\n"
                    f"Universe: {sample_instance.univ}\n"
                    f"算法: {alg}\n"
                    f"状态: {status}\n"
                    f"使用y: {y_used}\n"
                    f"耗时: {time_taken:.2f} 秒\n"
                    f"找到集合 ({num_sets} 个):\n"
                )
                is_final_error = status not in ('OPTIMAL', 'FEASIBLE', 'SUCCESS', 'INFEASIBLE') # 标记非成功/无解状态

                MAX_SETS_TO_DISPLAY = 50
                if num_sets > 0 :
                    sets_to_display = sample_instance.sets[:MAX_SETS_TO_DISPLAY]
                    sets_lines = []
                    sets_per_line = 3
                    for i in range(0, len(sets_to_display), sets_per_line):
                         line = " | ".join([str(sorted(s)) for s in sets_to_display[i:i+sets_per_line]])
                         sets_lines.append(f"  {line}") # 加缩进
                    result_text += "\n".join(sets_lines)
                    if num_sets > MAX_SETS_TO_DISPLAY:
                        result_text += f"\n  ... (还有 {num_sets - MAX_SETS_TO_DISPLAY} 个未显示)"
                elif status == 'INFEASIBLE':
                     result_text += "  (问题被证明无解)"
                else: # 其他情况 (包括成功但0集合，或错误状态)
                     result_text += "  (无)"

                # --- 数据库存储 (仅当 K=6 时) ---
                if sample_instance.k == 6:
                    log_message(f"K=6 (Run {run_idx})，尝试保存到数据库...")
                    try:
                        universe_str = json.dumps(sample_instance.univ)
                        sets_list = sample_instance.sets if isinstance(sample_instance.sets, list) else []
                        found_sets_str = json.dumps(sets_list)

                        result_data = {
                            'm': sample_instance.m, 'n': sample_instance.n, 'k': sample_instance.k,
                            'j': sample_instance.j, 's': sample_instance.s, 'run_index': run_idx,
                            'num_results': num_sets, 'y_condition': y_used, 'algorithm': alg,
                            'time_taken': time_taken, 'universe': universe_str, 'sets_found': found_sets_str
                        }
                        save_success = db.save_result(result_data)
                        if save_success:
                            log_message(f"结果 (Run {run_idx}) 已保存到数据库。")
                        else:
                            log_message(f"保存结果 (Run {run_idx}) 到数据库时遇到问题。", is_error=True)
                            is_final_error = True # 标记数据库保存也出错了
                    except Exception as db_err:
                        error_msg = f"错误：保存结果 (Run {run_idx}) 到数据库失败: {db_err}"
                        log_message(error_msg, is_error=True)
                        is_final_error = True
                        print(error_msg)
                        import traceback; traceback.print_exc()
                else:
                    log_message(f"K={sample_instance.k} != 6，结果 (Run {run_idx}) 未保存。")
            else:
                 error_msg = f"计算执行完毕 (Run {sample_instance.run_idx})，但无法获取有效结果详情。"
                 log_message(error_msg, is_error=True)
                 result_text = error_msg
                 is_final_error = True

        except Exception as compute_err:
            run_idx = getattr(sample_instance, 'run_idx', 'N/A')
            error_msg = f"计算执行 (Run {run_idx}) 过程中发生顶层错误: {compute_err}"
            log_message(error_msg, is_error=True)
            result_text = f"运行时错误 (Run {run_idx}):\n{compute_err}"
            is_final_error = True
            print(f"--- 计算线程中的未捕获错误 (Run {run_idx}) ---")
            import traceback; traceback.print_exc()
            print(f"--- 错误结束 ---")

        finally:
            # --- 安全地更新 UI 并恢复控件状态 ---
             try:
                  show_info_message(sample_result_info, result_text, is_error=is_final_error)
                  set_busy(False) # !! 关键：计算结束，恢复UI
                  log_message(f"计算流程 (Run {getattr(sample_instance, 'run_idx', 'N/A')}) 处理完毕。UI已恢复。")
             except Exception as ui_update_err:
                  print(f"错误：从计算线程更新UI时出错: {ui_update_err}")
                  # 尝试记录到日志
                  try: log_message(f"UI更新错误: {ui_update_err}", is_error=True)
                  except: pass
                  # 确保按钮无论如何都恢复（如果上面set_busy失败）
                  try: set_busy(False)
                  except: print("!!! 紧急：无法恢复 UI 控件状态！")

    # 将 on_submit 函数绑定到按钮
    submit_button.on_click = on_submit

    # --- 布局 ---
    manual_univ_row = ft.Row([txt_manual_univ], visible=False) # 将文本框放入Row以便控制显隐
    specify_y_row = ft.Row([txt_specify_y], visible=False)    # 同上

    page.add(
        ft.Container( # 使用 Container 包裹 Column 以便更好地控制滚动
            content=ft.Column(
                [
                    ft.Text("参数输入", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        [txt_m, txt_n, txt_k, txt_j],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                    ),
                    ft.Row(
                        [txt_s, txt_timeout, chk_manual_univ, chk_specify_y],
                         alignment=ft.MainAxisAlignment.START,
                         vertical_alignment=ft.CrossAxisAlignment.CENTER, # 垂直居中对齐 Checkbox 和 TextField
                         spacing=20
                    ),
                     manual_univ_row, # 手动 Universe 输入行 (初始隐藏)
                     specify_y_row,   # 手动 y 输入行 (初始隐藏)
                     ft.Divider(height=10),
                     ft.Row([theoretical_y_info, max_single_j_coverage], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                     ft.Divider(height=10),
                     ft.Row([submit_button, progress_ring, clear_log_button], alignment=ft.MainAxisAlignment.START, spacing=20, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                     ft.Text("运行日志", size=14, weight=ft.FontWeight.BOLD),
                     ft.Container(
                        content=log_output,
                        border=ft.border.all(1, ft.Colors.BLACK12),
                        border_radius=ft.border_radius.all(5),
                        padding=5,
                        expand=True, # 让日志区域占据垂直空间
                        height=250
                     ),
                     ft.Text("计算结果", size=14, weight=ft.FontWeight.BOLD),
                     ft.Container(
                         content=sample_result_info,
                         border=ft.border.all(1, ft.Colors.BLACK26),
                         border_radius=ft.border_radius.all(5),
                         padding=10,
                         margin=ft.margin.only(top=10),
                         expand=True # 让结果区域也扩展
                     )
                ],
                # 不需要对 Column 设置 expand=True，由外部 Container 控制
                # scroll=ft.ScrollMode.ADAPTIVE # Scroll 由外部 Container 或 Page 处理
            ),
            expand=True, # 让 Container 占据所有可用空间
            padding = 10 # 给整个内容区加点边距
        )
    )
    # 让页面本身可以滚动，以防内容过多
    page.scroll = ft.ScrollMode.ADAPTIVE

    # --- 应用初始化 ---
    log_message("应用程序启动。")
    log_message(f"数据库文件: {os.path.abspath(db.DB_FILE)}")
    log_message(f"OR-Tools 可用: {'是' if HAS_ORTOOLS else '否'}")
    update_y_related_info()
    try:
        db.setup_database()
        log_message("数据库初始化检查完成。")
    except Exception as init_db_err:
        msg = f"错误：初始化数据库失败: {init_db_err}"
        log_message(msg, is_error=True)
        show_info_message(sample_result_info, msg, is_error=True)
        print(msg)
        import traceback; traceback.print_exc()

# --- 应用入口 ---
if __name__ == "__main__":
    # Multiprocessing 兼容性处理 (如果需要)
    # if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
    #     try:
    #         mp.set_start_method('spawn', force=True)
    #         print("Info: Multiprocessing start method set to 'spawn'.")
    #     except RuntimeError as e:
    #         print(f"Warning: Could not set multiprocessing start method: {e}")

    # 处理命令行数据库路径参数
    if len(sys.argv) > 1:
        custom_db_path = sys.argv[1]
        if os.sep in custom_db_path or custom_db_path.endswith(".db"):
            db_dir = os.path.dirname(os.path.abspath(custom_db_path))
            if os.path.isdir(custom_db_path): # 如果提供的是目录
                 db.DB_FILE = os.path.join(custom_db_path, "k6_results.db")
                 db_dir = custom_db_path
            else: # 如果提供的是文件路径
                 db.DB_FILE = custom_db_path
                 db_dir = os.path.dirname(db.DB_FILE)

            print(f"Info: Using database path: {os.path.abspath(db.DB_FILE)}")
            if not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True) # exist_ok=True 避免目录已存在时报错
                    print(f"Info: Created database directory: {db_dir}")
                except OSError as e:
                    print(f"Error: Could not create database directory '{db_dir}': {e}. Using default.")
                    db.DB_FILE = "k6_results.db"
        else:
             print(f"Warning: Invalid DB path argument. Using default '{db.DB_FILE}'.")

    # 启动 Flet 应用
    ft.app(target=main)